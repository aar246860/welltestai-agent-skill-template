from __future__ import annotations

import numpy as np
import pandas as pd

from slug_osc_v1_formal_models import (
    damped_oscillator_slug,
    fit_monotonic_initial,
    fit_oscillator_initial,
    monotonic_exponential_slug,
)
from slug_osc_v1_qc import detect_oscillation_qc

from inertial_ldl_slug_v1_schema import InertialLDLSlugAnalysis, InertialLDLSlugCase


def _aicc(n: int, rmse: float, k: int) -> float:
    n = max(int(n), 1)
    rss = max(n * float(rmse) ** 2, 1.0e-30)
    aic = n * np.log(rss / n) + 2 * k
    return float(aic + (2 * k * (k + 1)) / max(n - k - 1, 1))


def _weights(scores: pd.DataFrame) -> pd.DataFrame:
    out = scores.copy()
    delta = out["aicc"].to_numpy(float) - np.nanmin(out["aicc"].to_numpy(float))
    weight = np.exp(-0.5 * delta)
    out["delta_aicc"] = delta
    out["model_probability"] = weight / np.sum(weight)
    return out.sort_values("model_probability", ascending=False).reset_index(drop=True)


def _curve_metrics(observed: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    residual = observed - predicted
    rmse = float(np.sqrt(np.mean(residual**2)))
    mae = float(np.mean(np.abs(residual)))
    return rmse, mae


def _ldl_time_warp(time: np.ndarray, theta_q: float, theta_h: float) -> np.ndarray:
    """Low-dimensional LDL response-time coordinate used for fitting initialization.

    The formal report labels tau as effective response-time coordinates. This
    fast time-warp is used only to initialize model-family scoring; formal
    kernels provide the collapse checks.
    """
    scale = np.sqrt((1.0 + max(theta_h, 0.0)) / (1.0 + max(theta_q, 0.0)))
    return np.asarray(time, dtype=float) / max(scale, 1.0e-8)


def _fit_ldl_monotonic(time: np.ndarray, head: np.ndarray) -> dict[str, float]:
    best: dict[str, float] | None = None
    for theta_h in (0.02, 0.1, 0.5, 2.0, 8.0):
        for ratio in (0.05, 0.2, 1.0, 5.0):
            theta_q = theta_h * ratio
            warped = _ldl_time_warp(time, theta_q, theta_h)
            try:
                fit = fit_monotonic_initial(warped, head)
            except Exception:
                continue
            pred = monotonic_exponential_slug(
                warped - float(np.min(warped)),
                decay_rate=fit["decay_rate"],
                amplitude=fit["amplitude"],
                equilibrium=fit["equilibrium"],
            )
            rmse, _ = _curve_metrics(head, pred)
            row = {
                **fit,
                "theta_q": float(theta_q),
                "theta_h": float(theta_h),
                "R_tau": float(theta_q / theta_h),
                "rmse": rmse,
            }
            if best is None or row["rmse"] < best["rmse"]:
                best = row
    if best is None:
        raise RuntimeError("Unable to fit LDL monotonic branch.")
    return best


def _fit_ldl_inertial(time: np.ndarray, head: np.ndarray) -> dict[str, float]:
    best: dict[str, float] | None = None
    for theta_h in (0.02, 0.1, 0.5, 2.0, 8.0):
        for ratio in (0.05, 0.2, 1.0, 5.0):
            theta_q = theta_h * ratio
            warped = _ldl_time_warp(time, theta_q, theta_h)
            try:
                fit = fit_oscillator_initial(warped, head)
            except Exception:
                continue
            pred = damped_oscillator_slug(
                warped - float(np.min(warped)),
                zeta=fit["zeta"],
                omega0=fit["omega0"],
                amplitude=fit["amplitude"],
                equilibrium=fit["equilibrium"],
            )
            rmse, _ = _curve_metrics(head, pred)
            row = {
                **fit,
                "theta_q": float(theta_q),
                "theta_h": float(theta_h),
                "R_tau": float(theta_q / theta_h),
                "rmse": rmse,
            }
            if best is None or row["rmse"] < best["rmse"]:
                best = row
    if best is None:
        raise RuntimeError("Unable to fit LDL inertial branch.")
    return best


def _parameter_rows(model: str, fit: dict[str, float], width_scale: float = 0.25) -> list[dict[str, float | str]]:
    rows = []
    for key, value in fit.items():
        if key in {"success"} or not isinstance(value, (float, int)):
            continue
        val = float(value)
        width = max(abs(val) * width_scale, 1.0e-6)
        rows.append(
            {
                "model": model,
                "parameter": key,
                "median": val,
                "q05": val - 1.64 * width,
                "q95": val + 1.64 * width,
                "width_90": 3.28 * width,
            }
        )
    return rows


def analyze_inertial_ldl_slug_case(
    case: InertialLDLSlugCase,
    *,
    n_bootstrap: int = 20,
) -> InertialLDLSlugAnalysis:
    frame = case.frame()
    time = frame["time"].to_numpy(float)
    head = frame["normalized_head"].to_numpy(float)
    centered = time - float(np.min(time))
    qc = detect_oscillation_qc(time, head)
    qc_rows = pd.DataFrame(qc.to_rows())

    mono = fit_monotonic_initial(time, head)
    mono_pred = monotonic_exponential_slug(
        centered,
        decay_rate=mono["decay_rate"],
        amplitude=mono["amplitude"],
        equilibrium=mono["equilibrium"],
    )
    mono_rmse, mono_mae = _curve_metrics(head, mono_pred)

    osc = fit_oscillator_initial(time, head)
    osc_pred = damped_oscillator_slug(
        centered,
        zeta=osc["zeta"],
        omega0=osc["omega0"],
        amplitude=osc["amplitude"],
        equilibrium=osc["equilibrium"],
    )
    osc_rmse, osc_mae = _curve_metrics(head, osc_pred)

    ldl_mono = _fit_ldl_monotonic(time, head)
    warped_mono = _ldl_time_warp(time, ldl_mono["theta_q"], ldl_mono["theta_h"])
    ldl_mono_pred = monotonic_exponential_slug(
        warped_mono - float(np.min(warped_mono)),
        decay_rate=ldl_mono["decay_rate"],
        amplitude=ldl_mono["amplitude"],
        equilibrium=ldl_mono["equilibrium"],
    )
    ldl_mono_rmse, ldl_mono_mae = _curve_metrics(head, ldl_mono_pred)

    ldl_osc = _fit_ldl_inertial(time, head)
    warped_osc = _ldl_time_warp(time, ldl_osc["theta_q"], ldl_osc["theta_h"])
    ldl_osc_pred = damped_oscillator_slug(
        warped_osc - float(np.min(warped_osc)),
        zeta=ldl_osc["zeta"],
        omega0=ldl_osc["omega0"],
        amplitude=ldl_osc["amplitude"],
        equilibrium=ldl_osc["equilibrium"],
    )
    ldl_osc_rmse, ldl_osc_mae = _curve_metrics(head, ldl_osc_pred)

    scores = _weights(
        pd.DataFrame(
            [
                {"model": "classical_monotonic_slug", "rmse": mono_rmse, "mae": mono_mae, "aicc": _aicc(len(time), mono_rmse, 3)},
                {"model": "ldl_monotonic_slug", "rmse": ldl_mono_rmse, "mae": ldl_mono_mae, "aicc": _aicc(len(time), ldl_mono_rmse, 6)},
                {"model": "classical_inertial_slug", "rmse": osc_rmse, "mae": osc_mae, "aicc": _aicc(len(time), osc_rmse, 4)},
                {"model": "ldl_inertial_slug", "rmse": ldl_osc_rmse, "mae": ldl_osc_mae, "aicc": _aicc(len(time), ldl_osc_rmse, 7)},
            ]
        )
    )
    best = str(scores.iloc[0]["model"])
    if "inertial" in best and "ldl" in best:
        interpretation = "Oscillatory response is best screened by a coupled inertial and LDL response-time branch."
    elif "inertial" in best:
        interpretation = "Oscillatory response is best screened by a classical inertial branch; LDL coordinates are not clearly needed."
    elif "ldl" in best:
        interpretation = "Monotonic response may contain LDL response-time structure, but tau remains an effective coordinate."
    else:
        interpretation = "Classical monotonic slug recovery is the most parsimonious screened branch."

    params = []
    params.extend(_parameter_rows("classical_monotonic_slug", mono))
    params.extend(_parameter_rows("ldl_monotonic_slug", ldl_mono))
    params.extend(_parameter_rows("classical_inertial_slug", osc))
    params.extend(_parameter_rows("ldl_inertial_slug", ldl_osc))

    response_fit = pd.DataFrame(
        {
            "time": time,
            "observed_h_over_h0": head,
            "classical_monotonic": mono_pred,
            "ldl_monotonic": ldl_mono_pred,
            "classical_inertial": osc_pred,
            "ldl_inertial": ldl_osc_pred,
            "best_residual": head
            - {
                "classical_monotonic_slug": mono_pred,
                "ldl_monotonic_slug": ldl_mono_pred,
                "classical_inertial_slug": osc_pred,
                "ldl_inertial_slug": ldl_osc_pred,
            }[best],
        }
    )
    return InertialLDLSlugAnalysis(
        case_id=case.case_id,
        qc_rows=qc_rows,
        model_scores=scores,
        parameter_summary=pd.DataFrame(params),
        response_fit=response_fit,
        best_model=best,
        interpretation=interpretation,
    )
