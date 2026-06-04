from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_QUANTILES = np.asarray([0.0, 0.08, 0.16, 0.28, 0.42, 0.58, 0.72, 0.84, 0.92, 1.0], dtype=float)


def _safe_log(values: np.ndarray) -> np.ndarray:
    return np.log10(np.maximum(np.asarray(values, dtype=float), 1.0e-12))


def _curve_features(time: np.ndarray, response: np.ndarray) -> dict[str, float]:
    log_t = _safe_log(time)
    log_y = _safe_log(response)
    stage = np.linspace(0.0, 1.0, log_y.size)
    values = np.interp(FEATURE_QUANTILES, stage, log_y)
    slopes = np.gradient(log_y, log_t)
    slope_values = np.interp(FEATURE_QUANTILES, stage, slopes)
    features: dict[str, float] = {}
    for idx, value in enumerate(values):
        features[f"feature_log_response_q{idx:02d}"] = float(value)
    for idx, value in enumerate(slope_values):
        features[f"feature_log_slope_q{idx:02d}"] = float(value)
    early = slice(0, max(2, log_y.size // 4))
    middle = slice(max(1, log_y.size // 3), max(2, 2 * log_y.size // 3))
    late = slice(max(1, 3 * log_y.size // 4), log_y.size)
    features.update(
        {
            "feature_log_response_mean": float(np.nanmean(log_y)),
            "feature_log_response_std": float(np.nanstd(log_y)),
            "feature_log_response_range": float(np.nanmax(log_y) - np.nanmin(log_y)),
            "feature_early_slope_mean": float(np.nanmean(slopes[early])),
            "feature_middle_slope_mean": float(np.nanmean(slopes[middle])),
            "feature_late_slope_mean": float(np.nanmean(slopes[late])),
            "feature_late_minus_early_slope": float(np.nanmean(slopes[late]) - np.nanmean(slopes[early])),
            "feature_endpoint_log_ratio": float(log_y[-1] - log_y[0]),
        }
    )
    return features


def build_feature_table(cases: pd.DataFrame, observations: pd.DataFrame, splits: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build one fixed-length response-feature row per case."""

    rows: list[dict] = []
    split_map = {}
    if splits is not None and not splits.empty:
        split_map = dict(zip(splits["case_id"], splits["split"]))
    for _, case in cases.iterrows():
        obs = observations[observations["case_id"] == case["case_id"]].sort_values("obs_index")
        if obs.empty:
            continue
        features = _curve_features(obs["time_min"].to_numpy(dtype=float), obs["observed"].to_numpy(dtype=float))
        row = {
            "case_id": case["case_id"],
            "truth_family": case["truth_family"],
            "split": split_map.get(case["case_id"], "unknown"),
            "truth_parameter_name": case.get("truth_parameter_name", ""),
            "truth_parameter_value": float(case.get("truth_parameter_value", np.nan)),
            "truth_parameter_log10": float(case.get("truth_parameter_log10", np.nan)),
            "r_cm": float(case.get("r_cm", np.nan)),
            "q_ml_min": float(case.get("q_ml_min", np.nan)),
            "t_cm2_min": float(case.get("t_cm2_min", np.nan)),
            "storativity": float(case.get("storativity", np.nan)),
        }
        row.update(features)
        rows.append(row)
    table = pd.DataFrame(rows)
    feature_cols = [c for c in table.columns if c.startswith("feature_")]
    table[feature_cols] = table[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return table


def feature_columns(table: pd.DataFrame) -> list[str]:
    return [c for c in table.columns if c.startswith("feature_")]
