"""Detect data drift between a reference dataset and a current (production) window.

    python -m drift.detect --reference data/reference.csv --current data/current.csv

Exit codes: 0 = no dataset drift, 2 = dataset drift detected (use it to trigger retraining).
"""
from __future__ import annotations

import argparse
import html
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from .metrics import ks_test, psi


@dataclass
class FeatureDrift:
    feature: str
    psi: float
    ks_stat: float
    p_value: float
    drifted: bool


@dataclass
class DriftReport:
    generated_at: str
    reference_rows: int
    current_rows: int
    drift_share: float
    dataset_drift: bool
    features: list[FeatureDrift]

    @property
    def drifted_features(self) -> list[str]:
        return [f.feature for f in self.features if f.drifted]


def detect(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    psi_threshold: float = 0.2,
    p_threshold: float = 0.01,
    share_threshold: float = 0.25,
) -> DriftReport:
    """A feature drifts when PSI > psi_threshold AND the KS test rejects at p < p_threshold.

    The dataset drifts when more than `share_threshold` of numeric features drift.
    """
    numeric = [c for c in reference.select_dtypes("number").columns if c in current.columns]
    if not numeric:
        raise ValueError("No shared numeric columns to compare")
    results = []
    for col in numeric:
        ref, cur = reference[col].dropna().to_numpy(), current[col].dropna().to_numpy()
        p = psi(ref, cur)
        ks, pv = ks_test(ref, cur)
        results.append(FeatureDrift(col, round(p, 4), round(ks, 4), pv,
                                    bool(p > psi_threshold and pv < p_threshold)))
    share = sum(r.drifted for r in results) / len(results)
    return DriftReport(
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        reference_rows=len(reference),
        current_rows=len(current),
        drift_share=round(share, 4),
        dataset_drift=share > share_threshold,
        features=sorted(results, key=lambda r: r.psi, reverse=True),
    )


def to_html(report: DriftReport) -> str:
    status = "DRIFT DETECTED" if report.dataset_drift else "No dataset drift"
    color = "#dc2626" if report.dataset_drift else "#16a34a"
    rows = "\n".join(
        f"<tr class='{'bad' if f.drifted else ''}'><td>{html.escape(f.feature)}</td>"
        f"<td>{f.psi:.3f}</td><td>{f.ks_stat:.3f}</td><td>{f.p_value:.2e}</td>"
        f"<td>{'yes' if f.drifted else 'no'}</td></tr>"
        for f in report.features
    )
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Drift report</title>
<style>body{{font-family:system-ui,sans-serif;max-width:960px;margin:2rem auto;padding:0 1rem}}
table{{border-collapse:collapse;width:100%}}td,th{{border-bottom:1px solid #e5e7eb;padding:6px;text-align:left}}
tr.bad{{background:#fef2f2}}.badge{{color:#fff;background:{color};padding:4px 10px;border-radius:6px}}</style>
</head><body><h1>Data drift report</h1>
<p><span class="badge">{status}</span> &nbsp; {report.drift_share:.0%} of features drifted ·
reference {report.reference_rows} rows · current {report.current_rows} rows · {report.generated_at}</p>
<table><tr><th>Feature</th><th>PSI</th><th>KS</th><th>p-value</th><th>Drifted</th></tr>
{rows}</table></body></html>"""


def push_metrics(report: DriftReport, gateway: str, job: str = "drift_monitor") -> None:
    """Push drift gauges to a Prometheus Pushgateway so Grafana/Alertmanager can use them."""
    from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

    reg = CollectorRegistry()
    Gauge("drift_dataset_detected", "1 if dataset drift detected", registry=reg).set(
        int(report.dataset_drift))
    Gauge("drift_share", "Share of drifted features", registry=reg).set(report.drift_share)
    g = Gauge("drift_feature_psi", "PSI per feature", ["feature"], registry=reg)
    for f in report.features:
        g.labels(f.feature).set(f.psi)
    push_to_gateway(gateway, job=job, registry=reg)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--reference", required=True, type=Path)
    ap.add_argument("--current", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=Path("reports"))
    ap.add_argument("--psi-threshold", type=float, default=0.2)
    ap.add_argument("--share-threshold", type=float, default=0.25)
    ap.add_argument("--pushgateway", default=os.getenv("PUSHGATEWAY_URL"),
                    help="e.g. localhost:9091 (optional)")
    args = ap.parse_args(argv)

    report = detect(pd.read_csv(args.reference), pd.read_csv(args.current),
                    psi_threshold=args.psi_threshold, share_threshold=args.share_threshold)
    args.out.mkdir(parents=True, exist_ok=True)
    payload = {**asdict(report), "drifted_features": report.drifted_features}
    (args.out / "drift_report.json").write_text(json.dumps(payload, indent=2))
    (args.out / "drift_report.html").write_text(to_html(report))
    if args.pushgateway:
        push_metrics(report, args.pushgateway)

    print(f"Dataset drift: {report.dataset_drift} (share={report.drift_share:.0%})")
    for f in report.features[:5]:
        print(f"  {f.feature:28s} psi={f.psi:.3f} ks={f.ks_stat:.3f} drifted={f.drifted}")
    return 2 if report.dataset_drift else 0


if __name__ == "__main__":
    sys.exit(main())
