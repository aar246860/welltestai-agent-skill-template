from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from scipy.special import exp1, kv

from ldl_constant_head_solution import (
    drawdown_classical,
    drawdown_ldl,
    flowrate_classical,
    flowrate_ldl,
    inverse_laplace,
    lambda_ldl,
    transmissivity_flux_correction,
)


def _as_array(time_min: np.ndarray | float) -> np.ndarray:
    return np.asarray(time_min, dtype=float)


def theis_drawdown_constant_rate(
    time_min: np.ndarray | float,
    r_cm: float,
    Q_ml_min: float,
    T_cm2_min: float,
    S: float,
) -> np.ndarray:
    """Classical line-source drawdown for fixed-discharge pumping."""

    time = np.maximum(_as_array(time_min), 1.0e-9)
    u = (r_cm**2 * S) / (4.0 * T_cm2_min * time)
    return (Q_ml_min / (4.0 * math.pi * T_cm2_min)) * exp1(u)


def _sbar_constant_rate_ldl(
    p_value: float,
    r: float,
    rw: float,
    Q: float,
    T: float,
    S: float,
    tau_q: float,
    tau_s: float,
) -> float:
    if p_value == 0.0:
        return float("nan")
    lam = lambda_ldl(p_value, T=T, S=S, tau_q=tau_q, tau_s=tau_s)
    Tq = transmissivity_flux_correction(p_value, T=T, tau_q=tau_q, tau_s=tau_s)
    denominator = 2.0 * math.pi * rw * Tq * lam * kv(1, lam * rw)
    if denominator == 0.0 or not np.isfinite(denominator):
        return float("nan")
    value = (Q / p_value) * kv(0, lam * r) / denominator
    if np.iscomplexobj(value):
        value = complex(value)
        return float(value.real)
    return float(value)


@lru_cache(maxsize=200_000)
def _constant_rate_ldl_scalar(
    time_min: float,
    r_cm: float,
    rw_cm: float,
    Q_ml_min: float,
    T_cm2_min: float,
    S: float,
    tau_q_min: float,
    tau_s_min: float,
    n_terms: int,
) -> float:
    if tau_q_min <= 1.0e-12 and tau_s_min <= 1.0e-12:
        return float(theis_drawdown_constant_rate(time_min, r_cm, Q_ml_min, T_cm2_min, S))
    return inverse_laplace(
        _sbar_constant_rate_ldl,
        time_min,
        method="stehfest",
        n_terms=n_terms,
        r=r_cm,
        rw=rw_cm,
        Q=Q_ml_min,
        T=T_cm2_min,
        S=S,
        tau_q=tau_q_min,
        tau_s=tau_s_min,
    )


def drawdown_constant_rate_ldl(
    time_min: np.ndarray | float,
    r_cm: float,
    rw_cm: float,
    Q_ml_min: float,
    T_cm2_min: float,
    S: float,
    tau_q_min: float,
    tau_s_min: float,
    n_terms: int = 8,
) -> np.ndarray:
    """Fixed-discharge lagging response from the same LDL flux law."""

    time = _as_array(time_min)
    values = [
        _constant_rate_ldl_scalar(
            float(t),
            float(r_cm),
            float(rw_cm),
            float(Q_ml_min),
            float(T_cm2_min),
            float(S),
            float(tau_q_min),
            float(tau_s_min),
            int(n_terms),
        )
        for t in time
    ]
    return np.asarray(values, dtype=float)


def drawdown_constant_head(
    time_min: np.ndarray | float,
    r_cm: float,
    rw_cm: float,
    sw_cm: float,
    params: dict[str, float],
    model: str = "lagging",
    n_terms: int = 8,
) -> np.ndarray:
    time = _as_array(time_min)
    if model == "classical":
        return drawdown_classical(
            time, r=r_cm, rw=rw_cm, T=params["T"], S=params["S"], sw=sw_cm, n_terms=n_terms
        )
    return drawdown_ldl(
        time,
        r=r_cm,
        rw=rw_cm,
        T=params["T"],
        S=params["S"],
        tau_q=params.get("tau_q", 0.0),
        tau_s=params.get("tau_s", 0.0),
        sw=sw_cm,
        n_terms=n_terms,
    )


def flowrate_constant_head(
    time_min: np.ndarray | float,
    rw_cm: float,
    sw_cm: float,
    params: dict[str, float],
    model: str = "lagging",
    n_terms: int = 8,
) -> np.ndarray:
    time = _as_array(time_min)
    if model == "classical":
        return flowrate_classical(
            time, rw=rw_cm, T=params["T"], S=params["S"], sw=sw_cm, n_terms=n_terms
        )
    return flowrate_ldl(
        time,
        rw=rw_cm,
        T=params["T"],
        S=params["S"],
        tau_q=params.get("tau_q", 0.0),
        tau_s=params.get("tau_s", 0.0),
        sw=sw_cm,
        n_terms=n_terms,
    )


def drawdown_constant_rate(
    time_min: np.ndarray | float,
    r_cm: float,
    rw_cm: float,
    Q_ml_min: float,
    params: dict[str, float],
    model: str = "lagging",
    n_terms: int = 8,
) -> np.ndarray:
    if model == "classical":
        return theis_drawdown_constant_rate(time_min, r_cm, Q_ml_min, params["T"], params["S"])
    return drawdown_constant_rate_ldl(
        time_min,
        r_cm=r_cm,
        rw_cm=rw_cm,
        Q_ml_min=Q_ml_min,
        T_cm2_min=params["T"],
        S=params["S"],
        tau_q_min=params.get("tau_q", 0.0),
        tau_s_min=params.get("tau_s", 0.0),
        n_terms=n_terms,
    )


def apply_boundary_control_lag(values: np.ndarray, time_min: np.ndarray, tau_control_min: float) -> np.ndarray:
    """First-order smoothing used only as a diagnostic measurement/control scenario."""

    values = np.asarray(values, dtype=float)
    time_min = np.asarray(time_min, dtype=float)
    if tau_control_min <= 0.0 or len(values) < 2:
        return values.copy()
    out = np.empty_like(values)
    out[0] = values[0]
    for idx in range(1, len(values)):
        dt = max(time_min[idx] - time_min[idx - 1], 0.0)
        alpha = 1.0 - math.exp(-dt / tau_control_min) if dt > 0.0 else 1.0
        out[idx] = out[idx - 1] + alpha * (values[idx] - out[idx - 1])
    return out
