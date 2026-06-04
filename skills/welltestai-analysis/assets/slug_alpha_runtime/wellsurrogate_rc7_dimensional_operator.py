from __future__ import annotations

import numpy as np


def cr_tD_from_time(*, T: float, S: float, time: np.ndarray, radius: float) -> np.ndarray:
    return float(T) * np.asarray(time, dtype=float) / (float(S) * float(radius) ** 2)


def ch_tD_from_time(*, T: float, S: float, time: np.ndarray, rw: float) -> np.ndarray:
    return float(T) * np.asarray(time, dtype=float) / (float(S) * float(rw) ** 2)


def cr_drawdown_from_sD(*, sD: np.ndarray, T: float, Q: float) -> np.ndarray:
    return float(Q) * np.asarray(sD, dtype=float) / (4.0 * np.pi * float(T))


def cr_sD_from_drawdown(drawdown: np.ndarray, *, T: float, Q: float) -> np.ndarray:
    return 4.0 * np.pi * float(T) * np.asarray(drawdown, dtype=float) / float(Q)


def ch_discharge_from_QD(*, QD: np.ndarray, T: float, sw: float) -> np.ndarray:
    return 2.0 * np.pi * float(T) * float(sw) * np.asarray(QD, dtype=float)


def ch_QD_from_discharge(discharge: np.ndarray, *, T: float, sw: float) -> np.ndarray:
    return np.asarray(discharge, dtype=float) / (2.0 * np.pi * float(T) * float(sw))


def characteristic_time_cr(*, T: float, S: float, radius: float) -> float:
    return float(S) * float(radius) ** 2 / float(T)


def characteristic_time_ch(*, T: float, S: float, rw: float) -> float:
    return float(S) * float(rw) ** 2 / float(T)


def dimensional_manifest() -> dict:
    return {
        "cr": {"tD": "T*t/(S*r^2)", "sD": "4*pi*T*s/Q"},
        "ch": {"tDw": "T*t/(S*rw^2)", "QD": "Q/(2*pi*T*sw)", "s_over_sw": "s/sw"},
        "slug": {"status": "partial-dimensional", "note": "Uses formal slug dimensionless kernel with recorded log_alpha and ar_over_a."},
    }

