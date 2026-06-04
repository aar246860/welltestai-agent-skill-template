from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from formal_extension_rc6_features import build_rc6_feature_table, rc6_feature_columns


def _late_features(obs: pd.DataFrame) -> dict[str, float]:
    ordered = obs.sort_values("obs_index")
    time = np.maximum(ordered["time_min"].to_numpy(dtype=float), 1.0e-12)
    response = np.maximum(ordered["observed"].to_numpy(dtype=float), 1.0e-12)
    log_t = np.log10(time)
    log_y = np.log10(response)
    slopes = np.gradient(log_y, log_t)
    curvature = np.gradient(slopes, log_t)
    n = log_y.size
    early = slice(0, max(2, n // 4))
    middle = slice(max(1, n // 3), max(2, 2 * n // 3))
    late = slice(max(1, 3 * n // 4), n)
    early_slope = float(np.nanmean(slopes[early]))
    middle_slope = float(np.nanmean(slopes[middle]))
    late_slope = float(np.nanmean(slopes[late]))
    return {
        "feature_late_flattening_ratio": float(late_slope / max(abs(middle_slope), 1.0e-8)),
        "feature_replenishment_slope_ratio": float((middle_slope - late_slope) / max(abs(early_slope), 1.0e-8)),
        "feature_late_curvature_signature": float(np.nanmean(curvature[late])),
        "feature_late_slope_abs": abs(late_slope),
        "feature_mid_to_late_slope_drop": float(middle_slope - late_slope),
    }


def build_rc7_feature_table(cases: pd.DataFrame, observations: pd.DataFrame, splits: pd.DataFrame | None = None) -> pd.DataFrame:
    table = build_rc6_feature_table(cases, observations, splits)
    extras = []
    for case_id, obs in observations.groupby("case_id"):
        row = {"case_id": case_id}
        row.update(_late_features(obs))
        extras.append(row)
    extra_df = pd.DataFrame(extras)
    if not extra_df.empty:
        table = table.merge(extra_df, on="case_id", how="left")
    cols = rc7_feature_columns(table)
    table[cols] = table[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return table


def rc7_feature_columns(table: pd.DataFrame) -> list[str]:
    return rc6_feature_columns(table)


def write_rc7_feature_metadata(output_dir: str | Path, feature_table: pd.DataFrame) -> dict:
    metadata = {
        "feature_count": len(rc7_feature_columns(feature_table)),
        "feature_columns": rc7_feature_columns(feature_table),
        "late_time_features": [
            "feature_late_flattening_ratio",
            "feature_replenishment_slope_ratio",
            "feature_late_curvature_signature",
            "feature_late_slope_abs",
            "feature_mid_to_late_slope_drop",
        ],
    }
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / "formal_extension_rc7_feature_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata

