"""Create reference/current CSVs from the Breast Cancer dataset, optionally with injected drift.

    python -m drift.simulate --out data            # no drift
    python -m drift.simulate --out data --drift 0.8  # strong drift on a third of features
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer


def make_datasets(drift: float = 0.0, seed: int = 42):
    rng = np.random.default_rng(seed)
    df = load_breast_cancer(as_frame=True).frame.sample(frac=1.0, random_state=seed)
    half = len(df) // 2
    reference, current = df.iloc[:half].copy(), df.iloc[half:].copy()
    if drift > 0:
        features = [c for c in df.columns if c != "target"]
        shifted = rng.choice(features, size=len(features) // 3, replace=False)
        for col in shifted:
            std = reference[col].std()
            current[col] = current[col] + drift * 2 * std + rng.normal(0, 0.1 * std, len(current))
    return reference, current


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--drift", type=float, default=0.0, help="0 = none, 1 = strong")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    ref, cur = make_datasets(args.drift)
    ref.to_csv(args.out / "reference.csv", index=False)
    cur.to_csv(args.out / "current.csv", index=False)
    print(f"Wrote {len(ref)} reference and {len(cur)} current rows to {args.out}/ (drift={args.drift})")


if __name__ == "__main__":
    main()
