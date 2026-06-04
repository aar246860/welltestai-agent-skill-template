from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from wellsurrogate_rc7_calibration import weighted_quantile
from wellsurrogate_rc7_joint_case import JointCase, ObservationBlock
from wellsurrogate_rc7_model_evidence import normalize_log_evidence
from wellsurrogate_rc8_adaptive_smc import AdaptiveSMCResult, run_adaptive_smc
from wellsurrogate_rc8_posterior_predictive import predictive_envelope
from welltest_demo_io import infer_test_type, load_welltest_case
from welltest_demo_schema import RawWellTestCase


PARAMETERS = ("logT", "logS", "logD", "log_tau_q", "log_tau_h", "log_R_tau")


def _time_to_minutes(values: np.ndarray, unit: str | None) -> np.ndarray:
    unit = (unit or "min").lower()
    if unit in {"h", "hr", "hour", "hours"}:
        return values * 60.0
    if unit in {"s", "sec", "second", "seconds"}:
        return values / 60.0
    return values


def _time_to_seconds(values: np.ndarray, unit: str | None) -> np.ndarray:
    unit = (unit or "s").lower()
    if unit in {"h", "hr", "hour", "hours"}:
        return values * 3600.0
    if unit in {"min", "minute", "minutes"}:
        return values * 60.0
    return values


def _thin(df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if len(df) <= max_points:
        return df.copy()
    idx = np.unique(np.linspace(0, len(df) - 1, max_points).round().astype(int))
    return df.iloc[idx].copy()


def _logit(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 1.0e-6, 1.0 - 1.0e-6)
    return np.log(x / (1.0 - x))


def build_observation_case(
    raw_case: RawWellTestCase,
    *,
    geometry: dict[str, float] | None = None,
    forcing: dict[str, float] | None = None,
    max_points: int = 10,
    sigma: float = 0.08,
) -> JointCase:
    geometry = geometry or {}
    forcing = forcing or {}
    test_type = infer_test_type(raw_case)
    df = _thin(raw_case.data, max_points)
    raw_time = df["time"].to_numpy(float)
    time = _time_to_minutes(raw_time, raw_case.unit_config.get("time"))
    time = np.maximum(time, np.min(time[time > 0]) if np.any(time > 0) else 1.0e-3)
    if test_type == "constant_rate":
        response = np.log10(np.maximum(df["drawdown"].to_numpy(float), 1.0e-8))
        block_type = "cr"
        response_type = "drawdown"
    elif test_type == "constant_head":
        col = "discharge" if "discharge" in df.columns else "flowrate"
        response = np.log10(np.maximum(df[col].to_numpy(float), 1.0e-8))
        block_type = "ch"
        response_type = "discharge"
    elif test_type == "slug":
        col = "normalized_head" if "normalized_head" in df.columns else "h_over_h0"
        response = _logit(df[col].to_numpy(float))
        block_type = "slug"
        response_type = "normalized_head"
        time_seconds = _time_to_seconds(raw_time, raw_case.unit_config.get("time"))
        scale_seconds = float(geometry.get("slug_time_scale_seconds", np.nanmedian(time_seconds)))
        time = np.maximum(time_seconds / max(scale_seconds, 1.0e-8), 1.0e-8)
    else:
        raise ValueError(f"Unsupported test type: {test_type}")
    obs = ObservationBlock(
        case_id=raw_case.case_id,
        test_type=block_type,
        response_type=response_type,
        time=time.astype(np.float32),
        observed_response=response.astype(np.float32),
        true_response=np.full_like(response, np.nan, dtype=np.float32),
        sigma=float(sigma),
        radius=float(geometry.get("radius_cm", geometry.get("r_cm", 100.0))),
        rw=float(geometry.get("rw_cm", 5.0)),
        Q=float(forcing.get("Q_mL_min", forcing.get("Q", 600.0))),
        sw=float(forcing.get("sw_cm", forcing.get("maintained_drawdown_cm", 150.0))),
        log_alpha=float(geometry.get("log_alpha", -3.0)),
        ar_over_a=float(geometry.get("ar_over_a", 0.85)),
    )
    true = {name: 0.0 for name in PARAMETERS}
    return JointCase(case_id=raw_case.case_id, mode="unknown", true=true, observations=(obs,))


def _posterior_summary(result: AdaptiveSMCResult, mode: str) -> pd.DataFrame:
    rows = []
    for parameter in PARAMETERS:
        q05, q25, q50, q75, q95 = weighted_quantile(result.samples[parameter], result.weights, (0.05, 0.25, 0.5, 0.75, 0.95))
        width = float(q95 - q05)
        ident = "weakly_constrained" if width > 1.5 else "usable_interval"
        if parameter.startswith("log_tau") or parameter == "log_R_tau":
            ident = "effective_coordinate" if width <= 1.5 else "not_identifiable"
        rows.append({"parameter": parameter, "median": q50, "q05": q05, "q25": q25, "q75": q75, "q95": q95, "width_90": width, "best_mode": mode, "identifiability": ident})
    return pd.DataFrame(rows)


def _fit_table(case: JointCase, result: AdaptiveSMCResult) -> pd.DataFrame:
    env = predictive_envelope(result.predictions, sigma=case.observations[0].sigma)
    rows = []
    for obs_index, obs in enumerate(case.observations):
        for i, time in enumerate(obs.time):
            rows.append(
                {
                    "time": float(time),
                    "observed": float(obs.observed_response[i]),
                    "median": float(env["median"][obs_index, i]),
                    "lower_90": float(env["obs_lower_90"][obs_index, i]),
                    "upper_90": float(env["obs_upper_90"][obs_index, i]),
                    "response_type": obs.response_type,
                    "test_type": obs.test_type,
                }
            )
    return pd.DataFrame(rows)


def run_interpretation(
    obs_case: JointCase,
    *,
    modes: list[str] | None = None,
    n_particles: int = 32,
    n_terms: int = 4,
    demo_mode: bool = True,
) -> dict[str, Any]:
    modes = modes or ["classical", "ldl_full", "head_lag", "equal_lag"]
    if demo_mode:
        n_particles = min(int(n_particles), 32)
    results = {}
    evid = {}
    for mode in modes:
        res = run_adaptive_smc(obs_case, mode=mode, n_particles=n_particles, seed=20260603 + len(mode), n_terms=n_terms, max_stages=4 if demo_mode else 8)
        results[mode] = res
        evid[mode] = res.log_evidence
    probs = normalize_log_evidence(evid)
    best_mode = max(probs.items(), key=lambda item: item[1])[0]
    best = results[best_mode]
    parameter_summary = _posterior_summary(best, best_mode)
    warnings = []
    for _, row in parameter_summary.iterrows():
        if row["identifiability"] in {"not_identifiable", "weakly_constrained"}:
            warnings.append({"warning": f"{row['parameter']}_{row['identifiability']}", "interpretation": "Posterior interval is broad.", "recommended_next_action": "Use the interval rather than the median; consider additional observations."})
    model_prob = pd.DataFrame({"mode": list(probs.keys()), "probability": list(probs.values())}).sort_values("probability", ascending=False)
    fit = _fit_table(obs_case, best)
    return {
        "case_id": obs_case.case_id,
        "test_type": obs_case.observations[0].test_type,
        "observation_case": obs_case,
        "best_mode": best_mode,
        "model_probabilities": model_prob,
        "parameter_summary": parameter_summary,
        "posterior_samples": best.samples,
        "posterior_predictive_envelope": fit,
        "response_fit": fit,
        "qc_warnings": pd.DataFrame(warnings),
        "recommended_next_actions": pd.DataFrame(
            [
                {"action": "Inspect response envelope and residuals before using parameter medians."},
                {"action": "Treat tau as effective response-time coordinates unless intervals are narrow."},
            ]
        ),
    }


def summarize_posterior(result: dict[str, Any]) -> pd.DataFrame:
    return result["parameter_summary"].copy()
