from __future__ import annotations

import numpy as np


def interval_coverage(truth: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    truth = np.asarray(truth, dtype=float)
    return float(np.mean((truth >= np.asarray(lower, dtype=float)) & (truth <= np.asarray(upper, dtype=float))))


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantiles: tuple[float, ...]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    total = np.sum(w)
    if total <= 0 or not np.isfinite(total):
        return np.quantile(v, np.asarray(quantiles, dtype=float))
    cdf = (np.cumsum(w) - 0.5 * w) / total
    cdf = np.concatenate(([0.0], cdf, [1.0]))
    v = np.concatenate(([v[0]], v, [v[-1]]))
    return np.interp(np.asarray(quantiles, dtype=float), cdf, v)
