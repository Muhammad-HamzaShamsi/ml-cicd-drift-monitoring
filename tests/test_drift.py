import json

import numpy as np
import pandas as pd

from drift.detect import detect, main
from drift.metrics import ks_test, psi
from drift.simulate import make_datasets


def test_psi_near_zero_for_same_distribution():
    rng = np.random.default_rng(0)
    assert psi(rng.normal(size=5000), rng.normal(size=5000)) < 0.02


def test_psi_large_for_shifted_distribution():
    rng = np.random.default_rng(0)
    assert psi(rng.normal(size=5000), rng.normal(loc=1.5, size=5000)) > 0.5


def test_psi_constant_feature_is_zero():
    assert psi(np.ones(100), np.ones(100)) == 0.0


def test_ks_detects_shift():
    rng = np.random.default_rng(1)
    _, p = ks_test(rng.normal(size=1000), rng.normal(loc=0.5, size=1000))
    assert p < 0.01


def test_no_drift_on_random_split():
    ref, cur = make_datasets(drift=0.0)
    report = detect(ref.drop(columns="target"), cur.drop(columns="target"))
    assert not report.dataset_drift


def test_drift_detected_when_injected():
    ref, cur = make_datasets(drift=1.0)
    report = detect(ref.drop(columns="target"), cur.drop(columns="target"))
    assert report.dataset_drift
    assert len(report.drifted_features) >= 5


def test_cli_writes_reports_and_exit_code(tmp_path):
    ref, cur = make_datasets(drift=1.0)
    ref.to_csv(tmp_path / "ref.csv", index=False)
    cur.to_csv(tmp_path / "cur.csv", index=False)
    code = main(["--reference", str(tmp_path / "ref.csv"), "--current", str(tmp_path / "cur.csv"),
                 "--out", str(tmp_path / "out")])
    assert code == 2
    data = json.loads((tmp_path / "out" / "drift_report.json").read_text())
    assert data["dataset_drift"] is True
    assert "<table>" in (tmp_path / "out" / "drift_report.html").read_text()


def test_requires_numeric_columns():
    df = pd.DataFrame({"a": ["x", "y"]})
    try:
        detect(df, df)
    except ValueError:
        return
    raise AssertionError("expected ValueError")
