"""Statistical drift metrics: Population Stability Index and Kolmogorov–Smirnov test."""
from __future__ import annotations

import numpy as np
from scipy import stats


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10, eps: float = 1e-6) -> float:
    """Population Stability Index using quantile bins of the reference distribution.

    Rule of thumb: < 0.1 no shift · 0.1–0.25 moderate · > 0.25 significant.
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:  # constant feature
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_pct = np.histogram(reference, edges)[0] / len(reference)
    cur_pct = np.histogram(current, edges)[0] / len(current)
    ref_pct = np.clip(ref_pct, eps, None)
    cur_pct = np.clip(cur_pct, eps, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def ks_test(reference: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    """Two-sample KS test. Returns (statistic, p_value)."""
    result = stats.ks_2samp(reference, current)
    return float(result.statistic), float(result.pvalue)
