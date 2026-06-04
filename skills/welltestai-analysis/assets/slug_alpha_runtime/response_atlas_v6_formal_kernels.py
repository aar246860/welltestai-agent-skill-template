from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.special import exp1, kv

from dual_control_kernels import drawdown_constant_rate_ldl
from ldl_constant_head_solution import (
    drawdown_classical,
    drawdown_classical_finite,
    drawdown_ldl,
    drawdown_ldl_finite,
    flowrate_classical,
    flowrate_classical_finite,
    flowrate_ldl,
    flowrate_ldl_finite,
    inverse_laplace,
    lambda_ldl,
    qbar_classical,
    qbar_classical_finite,
    qbar_ldl,
    qbar_ldl_finite,
    sbar_classical,
    sbar_classical_finite,
    sbar_ldl,
    sbar_ldl_finite,
    steady_finite_flowrate,
    transmissivity_flux_correction,
)


FORMAL_MODEL_FAMILIES = [
    "classical_confined",
    "ldl_full",
    "ldl_flux_lag_only",
    "ldl_storage_lag_only",
    "ldl_equal_lag_collapse",
]

REMOVED_HEURISTIC_SCENARIOS = [
    "leaky",
    "unconfined_delayed_yield",
    "skin",
    "control_lag",
    "lagging_skin",
    "lagging_boundary",
    "delayed_storage_control",
    "out_of_atlas",
]


@dataclass(frozen=True)
class FormalResponse:
    time_hr: np.ndarray
    ch_q: np.ndarray
    ch_s: np.ndarray
    ch_specific: np.ndarray
    cr_s: np.ndarray
    ch_radii_cm: np.ndarray
    cr_radii_cm: np.ndarray
    model_family: str


def _positive(values: np.ndarray, floor: float = 1.0e-10) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr[~np.isfinite(arr)] = np.nan
    return np.maximum(arr, floor)


def theis_constant_rate(
    time_min: np.ndarray | float,
    r_cm: float | np.ndarray,
    q_ml_min: float,
    t_cm2_min: float,
    storativity: float,
) -> np.ndarray:
    """Classical Theis fixed-discharge solution for a confined aquifer."""

    time = np.maximum(np.asarray(time_min, dtype=float), 1.0e-9)
    radii = np.atleast_1d(np.asarray(r_cm, dtype=float))
    rows = []
    for radius in radii:
        u = radius**2 * storativity / (4.0 * t_cm2_min * time)
        rows.append(q_ml_min * exp1(u) / (4.0 * math.pi * t_cm2_min))
    out = np.asarray(rows, dtype=float)
    return out[0] if np.ndim(r_cm) == 0 else out


def jacob_lohman_constant_head(
    time_min: np.ndarray,
    r_cm: float | np.ndarray,
    rw_cm: float,
    sw_cm: float,
    t_cm2_min: float,
    storativity: float,
    n_terms: int = 10,
    method: str = "stehfest",
) -> tuple[np.ndarray, np.ndarray]:
    """Classical Jacob-Lohman fixed-head response from formal Laplace kernels."""

    time = np.asarray(time_min, dtype=float)
    q = flowrate_classical(
        time, rw=rw_cm, T=t_cm2_min, S=storativity, sw=sw_cm, n_terms=n_terms, method=method
    )
    radii = np.atleast_1d(np.asarray(r_cm, dtype=float))
    rows = [
        drawdown_classical(
            time, r=float(radius), rw=rw_cm, T=t_cm2_min, S=storativity, sw=sw_cm, n_terms=n_terms, method=method
        )
        for radius in radii
    ]
    s = np.asarray(rows, dtype=float)
    return _positive(q), _positive(s[0] if np.ndim(r_cm) == 0 else s)


def ldl_constant_rate_laplace(
    p_value: float,
    r_cm: float,
    rw_cm: float,
    q_ml_min: float,
    t_cm2_min: float,
    storativity: float,
    tau_q_min: float,
    tau_s_min: float,
) -> float:
    """Laplace-domain fixed-discharge LDL drawdown."""

    if p_value == 0.0:
        return float("nan")
    lam = lambda_ldl(p_value, T=t_cm2_min, S=storativity, tau_q=tau_q_min, tau_s=tau_s_min)
    tq = transmissivity_flux_correction(p_value, T=t_cm2_min, tau_q=tau_q_min, tau_s=tau_s_min)
    denom = 2.0 * math.pi * rw_cm * tq * lam * kv(1, lam * rw_cm)
    if denom == 0.0 or not np.isfinite(denom):
        return float("nan")
    return float((q_ml_min / p_value) * kv(0, lam * r_cm) / denom)


def ldl_constant_rate(
    time_min: np.ndarray,
    r_cm: float | np.ndarray,
    rw_cm: float,
    q_ml_min: float,
    t_cm2_min: float,
    storativity: float,
    tau_q_min: float,
    tau_s_min: float,
    n_terms: int = 8,
) -> np.ndarray:
    """Formal fixed-discharge LDL response evaluated by inverse Laplace."""

    radii = np.atleast_1d(np.asarray(r_cm, dtype=float))
    rows = [
        drawdown_constant_rate_ldl(
            time_min,
            r_cm=float(radius),
            rw_cm=rw_cm,
            Q_ml_min=q_ml_min,
            T_cm2_min=t_cm2_min,
            S=storativity,
            tau_q_min=tau_q_min,
            tau_s_min=tau_s_min,
            n_terms=n_terms,
        )
        for radius in radii
    ]
    out = np.asarray(rows, dtype=float)
    return _positive(out[0] if np.ndim(r_cm) == 0 else out)


def ldl_constant_head_laplace(
    p_value: float,
    rw_cm: float,
    t_cm2_min: float,
    storativity: float,
    tau_q_min: float,
    tau_s_min: float,
    sw_cm: float,
) -> float:
    """Laplace-domain fixed-head LDL well flowrate with formal T_Q(p)."""

    return qbar_ldl(
        p_value,
        rw=rw_cm,
        T=t_cm2_min,
        S=storativity,
        tau_q=tau_q_min,
        tau_s=tau_s_min,
        sw=sw_cm,
    )


def ldl_constant_head(
    time_min: np.ndarray,
    r_cm: float | np.ndarray,
    rw_cm: float,
    sw_cm: float,
    t_cm2_min: float,
    storativity: float,
    tau_q_min: float,
    tau_s_min: float,
    n_terms: int = 10,
    method: str = "stehfest",
) -> tuple[np.ndarray, np.ndarray]:
    """Formal fixed-head LDL response evaluated by inverse Laplace."""

    time = np.asarray(time_min, dtype=float)
    q = flowrate_ldl(
        time,
        rw=rw_cm,
        T=t_cm2_min,
        S=storativity,
        tau_q=tau_q_min,
        tau_s=tau_s_min,
        sw=sw_cm,
        n_terms=n_terms,
        method=method,
    )
    radii = np.atleast_1d(np.asarray(r_cm, dtype=float))
    rows = [
        drawdown_ldl(
            time,
            r=float(radius),
            rw=rw_cm,
            T=t_cm2_min,
            S=storativity,
            tau_q=tau_q_min,
            tau_s=tau_s_min,
            sw=sw_cm,
            n_terms=n_terms,
            method=method,
        )
        for radius in radii
    ]
    s = np.asarray(rows, dtype=float)
    return _positive(q), _positive(s[0] if np.ndim(r_cm) == 0 else s)


def finite_boundary_constant_head(
    time_min: np.ndarray,
    r_cm: float | np.ndarray,
    rw_cm: float,
    outer_radius_cm: float,
    sw_cm: float,
    t_cm2_min: float,
    storativity: float,
    tau_q_min: float = 0.0,
    tau_s_min: float = 0.0,
    n_terms: int = 10,
    method: str = "dehoog",
) -> tuple[np.ndarray, np.ndarray]:
    """Formal finite-domain fixed-head response with s(R,t)=0."""

    time = np.asarray(time_min, dtype=float)
    if tau_q_min <= 1.0e-12 and tau_s_min <= 1.0e-12:
        q = flowrate_classical_finite(
            time, rw=rw_cm, R=outer_radius_cm, T=t_cm2_min, S=storativity, sw=sw_cm, n_terms=n_terms, method=method
        )
        drawdown_fn = drawdown_classical_finite
        extra = {}
    else:
        q = flowrate_ldl_finite(
            time,
            rw=rw_cm,
            R=outer_radius_cm,
            T=t_cm2_min,
            S=storativity,
            tau_q=tau_q_min,
            tau_s=tau_s_min,
            sw=sw_cm,
            n_terms=n_terms,
            method=method,
        )
        drawdown_fn = drawdown_ldl_finite
        extra = {"tau_q": tau_q_min, "tau_s": tau_s_min}
    radii = np.atleast_1d(np.asarray(r_cm, dtype=float))
    rows = []
    for radius in radii:
        rows.append(
            drawdown_fn(
                time,
                r=float(radius),
                rw=rw_cm,
                R=outer_radius_cm,
                T=t_cm2_min,
                S=storativity,
                sw=sw_cm,
                n_terms=n_terms,
                method=method,
                **extra,
            )
        )
    s = np.asarray(rows, dtype=float)
    return _positive(q), _positive(s[0] if np.ndim(r_cm) == 0 else s)


def first_order_measurement_operator(
    time_min: np.ndarray,
    input_signal: np.ndarray,
    tau_c_min: float,
) -> np.ndarray:
    """Formal independent measurement/control operator: tau_c dy/dt + y = x.

    This operator is not part of the aquifer response atlas unless explicitly
    used as an observation model.
    """

    time = np.asarray(time_min, dtype=float)
    values = np.asarray(input_signal, dtype=float)
    if tau_c_min <= 0.0 or values.shape[-1] < 2:
        return values.copy()
    out = np.empty_like(values)
    out[..., 0] = values[..., 0]
    for idx in range(1, values.shape[-1]):
        dt = max(time[idx] - time[idx - 1], 0.0)
        alpha = 1.0 - math.exp(-dt / tau_c_min) if dt > 0.0 else 1.0
        out[..., idx] = out[..., idx - 1] + alpha * (values[..., idx] - out[..., idx - 1])
    return out


def simulate_formal_response(
    model_family: str,
    time_hr: np.ndarray,
    ch_radii_cm: np.ndarray,
    cr_radii_cm: np.ndarray,
    parameters: dict[str, float],
    n_terms: int = 8,
) -> FormalResponse:
    """Generate paired CR/CH responses from formal kernels only."""

    if model_family not in FORMAL_MODEL_FAMILIES:
        raise ValueError(f"{model_family} is not in the formal atlas")
    time_min = np.asarray(time_hr, dtype=float) * 60.0
    t_cm2_min = 10.0 ** float(parameters["logT"])
    storativity = 10.0 ** float(parameters["logS"])
    rw_cm = float(parameters["rw_cm"])
    q0 = float(parameters["q0_ml_min"])
    sw = float(parameters["sw_cm"])
    tau_q = 10.0 ** float(parameters["log_tau_q"])
    tau_s = 10.0 ** float(parameters["log_tau_s"])
    if model_family == "classical_confined":
        tau_q = 0.0
        tau_s = 0.0
        cr_s = theis_constant_rate(time_min, cr_radii_cm, q0, t_cm2_min, storativity)
        ch_q, ch_s = jacob_lohman_constant_head(time_min, ch_radii_cm, rw_cm, sw, t_cm2_min, storativity, n_terms=n_terms)
    else:
        if model_family == "ldl_flux_lag_only":
            tau_s = 0.0
        elif model_family == "ldl_storage_lag_only":
            tau_q = 0.0
        elif model_family == "ldl_equal_lag_collapse":
            tau_s = tau_q
        cr_s = ldl_constant_rate(time_min, cr_radii_cm, rw_cm, q0, t_cm2_min, storativity, tau_q, tau_s, n_terms=n_terms)
        ch_q, ch_s = ldl_constant_head(time_min, ch_radii_cm, rw_cm, sw, t_cm2_min, storativity, tau_q, tau_s, n_terms=n_terms)
    return FormalResponse(
        time_hr=np.asarray(time_hr, dtype=float),
        ch_q=_positive(ch_q),
        ch_s=_positive(ch_s),
        ch_specific=_positive(ch_s / _positive(ch_q)[None, :]),
        cr_s=_positive(cr_s),
        ch_radii_cm=np.asarray(ch_radii_cm, dtype=float),
        cr_radii_cm=np.asarray(cr_radii_cm, dtype=float),
        model_family=model_family,
    )


__all__ = [
    "FORMAL_MODEL_FAMILIES",
    "REMOVED_HEURISTIC_SCENARIOS",
    "FormalResponse",
    "theis_constant_rate",
    "jacob_lohman_constant_head",
    "ldl_constant_rate_laplace",
    "ldl_constant_rate",
    "ldl_constant_head_laplace",
    "ldl_constant_head",
    "finite_boundary_constant_head",
    "first_order_measurement_operator",
    "simulate_formal_response",
    "qbar_classical",
    "qbar_ldl",
    "sbar_classical",
    "sbar_ldl",
    "qbar_classical_finite",
    "qbar_ldl_finite",
    "sbar_classical_finite",
    "sbar_ldl_finite",
    "steady_finite_flowrate",
]

