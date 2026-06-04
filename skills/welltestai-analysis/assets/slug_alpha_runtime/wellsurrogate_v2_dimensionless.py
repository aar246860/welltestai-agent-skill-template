from __future__ import annotations

import numpy as np

FLOOR = 1.0e-12


def safe_log10(values, floor: float = FLOOR):
    arr = np.maximum(np.asarray(values, dtype=float), floor)
    out = np.log10(arr)
    return float(out) if np.ndim(out) == 0 else out


def inv_log10(values):
    return 10.0 ** np.asarray(values, dtype=float)


def logit(values, eps: float = 1.0e-6):
    x = np.clip(np.asarray(values, dtype=float), eps, 1.0 - eps)
    out = np.log(x / (1.0 - x))
    return float(out) if np.ndim(out) == 0 else out


def sigmoid(values):
    x = np.asarray(values, dtype=float)
    return 1.0 / (1.0 + np.exp(-x))


def theta_to_tau(theta: np.ndarray | float, characteristic_time: np.ndarray | float):
    return np.asarray(theta, dtype=float) * np.asarray(characteristic_time, dtype=float)


def cr_time_from_tD(tD: np.ndarray, r_over_rw: np.ndarray, rw: float = 1.0, T: float = 1.0, S: float = 1.0) -> np.ndarray:
    r = np.asarray(r_over_rw, dtype=float) * rw
    return np.asarray(tD, dtype=float) * r**2 * S / T


def ch_time_from_tDw(tDw: np.ndarray, rw: float = 1.0, T: float = 1.0, S: float = 1.0) -> np.ndarray:
    return np.asarray(tDw, dtype=float) * rw**2 * S / T


def tD_at_radius(time: np.ndarray, r_over_rw: np.ndarray, rw: float = 1.0, T: float = 1.0, S: float = 1.0) -> np.ndarray:
    r = np.asarray(r_over_rw, dtype=float) * rw
    return np.asarray(time, dtype=float) * T / (r**2 * S)


def lags_from_family(prefix: str, family: str, theta_q: float, theta_h: float, characteristic_time: float) -> tuple[float, float]:
    if "classical" in family:
        return 0.0, 0.0
    if "flux_lag_only" in family:
        return float(theta_q * characteristic_time), 0.0
    if "head_lag_only" in family:
        return 0.0, float(theta_h * characteristic_time)
    if "equal_lag_collapse" in family:
        theta = float(theta_h if np.isfinite(theta_h) else theta_q)
        return theta * characteristic_time, theta * characteristic_time
    return float(theta_q * characteristic_time), float(theta_h * characteristic_time)

