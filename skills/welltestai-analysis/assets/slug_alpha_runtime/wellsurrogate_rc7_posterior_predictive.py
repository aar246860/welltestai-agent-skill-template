from __future__ import annotations

import numpy as np


def predictive_envelope(curves: np.ndarray, sigma: float = 0.0) -> dict[str, np.ndarray]:
    arr = np.asarray(curves, dtype=float)
    out = {
        "median": np.nanmedian(arr, axis=0),
        "lower_90": np.nanquantile(arr, 0.05, axis=0),
        "upper_90": np.nanquantile(arr, 0.95, axis=0),
    }
    z90 = 1.6448536269514722
    out["obs_lower_90"] = out["lower_90"] - z90 * float(sigma)
    out["obs_upper_90"] = out["upper_90"] + z90 * float(sigma)
    return out


def envelope_coverage(observed: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    obs = np.asarray(observed, dtype=float)
    return float(np.mean((obs >= lower) & (obs <= upper)))

