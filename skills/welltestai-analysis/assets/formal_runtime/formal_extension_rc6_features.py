from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from formal_extension_rc5_features import build_feature_table, feature_columns


def build_rc6_feature_table(cases: pd.DataFrame, observations: pd.DataFrame, splits: pd.DataFrame | None = None) -> pd.DataFrame:
    table = build_feature_table(cases, observations, splits)
    rows = []
    obs_by_case = {case_id: group.sort_values("obs_index") for case_id, group in observations.groupby("case_id")}
    for _, row in table.iterrows():
        obs = obs_by_case.get(row["case_id"])
        if obs is None or obs.empty:
            rows.append({})
            continue
        n = int(obs.shape[0])
        time = obs["time_min"].to_numpy(dtype=float)
        observed = np.maximum(obs["observed"].to_numpy(dtype=float), 1.0e-12)
        log_y = np.log10(observed)
        curvature = np.gradient(np.gradient(log_y, np.log10(np.maximum(time, 1.0e-12))), np.log10(np.maximum(time, 1.0e-12)))
        rows.append(
            {
                "case_id": row["case_id"],
                "feature_sparse_sampling_indicator": float(n < 20),
                "feature_missing_early_indicator": float(np.nanmin(time) > 10.0 ** -1.7),
                "feature_missing_late_indicator": float(np.nanmax(time) < 10.0 ** 2.8),
                "feature_log_curvature_mean": float(np.nanmean(curvature)),
                "feature_log_curvature_std": float(np.nanstd(curvature)),
                "feature_log_response_iqr": float(np.nanquantile(log_y, 0.75) - np.nanquantile(log_y, 0.25)),
            }
        )
    extra = pd.DataFrame(rows)
    if not extra.empty and "case_id" in extra:
        table = table.merge(extra, on="case_id", how="left")
    cols = rc6_feature_columns(table)
    table[cols] = table[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return table


def rc6_feature_columns(table: pd.DataFrame) -> list[str]:
    return feature_columns(table)


def write_feature_metadata(output_dir: str | Path, feature_table: pd.DataFrame) -> dict:
    metadata = {
        "feature_count": len(rc6_feature_columns(feature_table)),
        "feature_columns": rc6_feature_columns(feature_table),
        "feature_scaling_note": "Scaling is fitted inside each surrogate pipeline on train split only.",
    }
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / "formal_extension_rc6_feature_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata

