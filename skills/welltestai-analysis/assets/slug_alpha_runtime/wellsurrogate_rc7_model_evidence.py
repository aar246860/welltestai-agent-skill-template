from __future__ import annotations

import numpy as np


def logsumexp(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("-inf")
    m = float(np.max(finite))
    return float(m + np.log(np.sum(np.exp(finite - m))))


def normalize_log_evidence(log_evidence: dict[str, float]) -> dict[str, float]:
    keys = list(log_evidence)
    vals = np.asarray([log_evidence[k] for k in keys], dtype=float)
    denom = logsumexp(vals)
    if not np.isfinite(denom):
        return {k: 1.0 / len(keys) for k in keys}
    probs = np.exp(vals - denom)
    return {k: float(v) for k, v in zip(keys, probs)}

