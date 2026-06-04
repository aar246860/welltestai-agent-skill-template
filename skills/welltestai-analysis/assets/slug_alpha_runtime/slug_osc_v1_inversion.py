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
from slug_osc_v1_schema import SlugOscAnalysis, SlugOscCase


def _aicc(n: int, rmse: float, k: int) -> float:
    n = max(int(n), 1)
    rss = max((float(rmse) ** 2) * n, 1.0e-30)
    aic = n * np.log(rss / n) + 2 * k
    correction = (2 * k * (k + 1)) / max(n - k - 1, 1)
    return float(aic + correction)


def _weights_from_aicc(scores: pd.DataFrame) -> pd.DataFrame:
    values = scores["aicc"].to_numpy(float)
    delta = values - np.nanmin(values)
    weights = np.exp(-0.5 * delta)
    weights = weights / np.sum(weights)
    out = scores.copy()
    out["delta_aicc"] = delta
    out["model_probability"] = weights
    return out.sort_values("model_probability", ascending=False).reset_index(drop=True)


def _bootstrap_oscillator(
    time: np.ndarray,
    pred: np.ndarray,
    residual: np.ndarray,
    *,
    n_bootstrap: int = 60,
    seed: int = 20260604,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(int(n_bootstrap)):
        synth = pred + rng.choice(residual, size=residual.size, replace=True)
        try:
            fit = fit_oscillator_initial(time, synth)
        except Exception:
            continue
        rows.append(fit)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _bootstrap_monotonic(
    time: np.ndarray,
    pred: np.ndarray,
    residual: np.ndarray,
    *,
    n_bootstrap: int = 60,
    seed: int = 20260605,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(int(n_bootstrap)):
        synth = pred + rng.choice(residual, size=residual.size, replace=True)
        try:
            fit = fit_monotonic_initial(time, synth)
        except Exception:
            continue
        rows.append(fit)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _parameter_summary(model_name: str, samples: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in samples.columns:
        if column in {"success", "rmse"}:
            continue
        values = samples[column].to_numpy(float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        rows.append(
            {
                "model": model_name,
                "parameter": column,
                "median": float(np.nanmedian(values)),
                "q05": float(np.nanquantile(values, 0.05)),
                "q95": float(np.nanquantile(values, 0.95)),
                "width_90": float(np.nanquantile(values, 0.95) - np.nanquantile(values, 0.05)),
            }
        )
    return pd.DataFrame(rows)


def analyze_slug_case(
    case: SlugOscCase,
    *,
    equilibrium: float = 0.0,
    n_bootstrap: int = 60,
) -> SlugOscAnalysis:
    frame = case.normalized_columns().sort_values("time")
    time = frame["time"].to_numpy(float)
    head = frame["normalized_head"].to_numpy(float)
    qc = detect_oscillation_qc(time, head, equilibrium=equilibrium)
    qc_rows = pd.DataFrame(qc.to_rows())

    monotonic = fit_monotonic_initial(time, head)
    mono_pred = monotonic_exponential_slug(
        time - float(np.min(time)),
        decay_rate=monotonic["decay_rate"],
        amplitude=monotonic["amplitude"],
        equilibrium=monotonic["equilibrium"],
    )
    mono_rmse = float(np.sqrt(np.mean((mono_pred - head) ** 2)))

    oscillator = fit_oscillator_initial(time, head)
    osc_pred = damped_oscillator_slug(
        time - float(np.min(time)),
        zeta=oscillator["zeta"],
        omega0=oscillator["omega0"],
        amplitude=oscillator["amplitude"],
        equilibrium=oscillator["equilibrium"],
    )
    osc_rmse = float(np.sqrt(np.mean((osc_pred - head) ** 2)))

    scores = pd.DataFrame(
        [
            {
                "model": "monotonic_slug_reference",
                "rmse": mono_rmse,
                "aicc": _aicc(len(time), mono_rmse, 3),
                "applicability": "primary" if not qc.flags["out_of_monotonic_slug_model"].triggered else "not_primary",
            },
            {
                "model": "oscillatory_inertial_screening",
                "rmse": osc_rmse,
                "aicc": _aicc(len(time), osc_rmse, 4),
                "applicability": "primary" if qc.flags["damped_oscillation_candidate"].triggered else "screening_only",
            },
        ]
    )
    scores = _weights_from_aicc(scores)
    best_model = str(scores.iloc[0]["model"])
    if qc.flags["damped_oscillation_candidate"].triggered:
        interpretation = "Oscillatory slug response candidate; monotonic slug models should not be the primary interpretation."
    elif qc.flags["insufficient_sampling_for_oscillation"].triggered:
        interpretation = "Sampling is insufficient to support or reject oscillation; report uncertainty."
    else:
        interpretation = "No strong oscillatory signature detected; monotonic slug interpretation can be used as a reference."

    mono_samples = _bootstrap_monotonic(time, mono_pred, head - mono_pred, n_bootstrap=n_bootstrap)
    osc_samples = _bootstrap_oscillator(time, osc_pred, head - osc_pred, n_bootstrap=n_bootstrap)
    summaries = []
    if not mono_samples.empty:
        summaries.append(_parameter_summary("monotonic_slug_reference", mono_samples))
    if not osc_samples.empty:
        summaries.append(_parameter_summary("oscillatory_inertial_screening", osc_samples))
    parameter_summary = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()

    response_fit = pd.DataFrame(
        {
            "time": time,
            "observed_h_over_h0": head,
            "monotonic_median": mono_pred,
            "oscillatory_median": osc_pred,
            "monotonic_residual": head - mono_pred,
            "oscillatory_residual": head - osc_pred,
        }
    )
    return SlugOscAnalysis(
        case_id=case.case_id,
        qc_rows=qc_rows,
        model_scores=scores,
        parameter_summary=parameter_summary,
        response_fit=response_fit,
        best_model=best_model,
        interpretation=interpretation,
    )
