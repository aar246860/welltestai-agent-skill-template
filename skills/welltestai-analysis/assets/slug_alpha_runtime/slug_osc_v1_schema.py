from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "analysis_outputs" / "slug_osc_v1"
FIGURE_DIR = OUTPUT_DIR / "figures"
MODEL_DIR = OUTPUT_DIR / "models"
TRAINING_DIR = OUTPUT_DIR / "training_data"
REPORT_DIR = OUTPUT_DIR / "reports"


def ensure_slug_osc_dirs() -> None:
    for path in (OUTPUT_DIR, FIGURE_DIR, MODEL_DIR, TRAINING_DIR, REPORT_DIR):
        path.mkdir(parents=True, exist_ok=True)


@dataclass
class SlugOscMetadata:
    time_unit: str = "s"
    response_unit: str = "dimensionless"
    rw_cm: float | None = None
    slug_time_scale_seconds: float | None = None
    log_alpha: float | None = None
    ar_over_a: float | None = None
    source_path: str | None = None
    notes: dict[str, Any] = field(default_factory=dict)


@dataclass
class SlugOscCase:
    case_id: str
    data: pd.DataFrame
    metadata: SlugOscMetadata = field(default_factory=SlugOscMetadata)
    test_type: str = "slug"

    def normalized_columns(self) -> pd.DataFrame:
        frame = self.data.copy()
        if "normalized_head" not in frame.columns and "h_over_h0" in frame.columns:
            frame = frame.rename(columns={"h_over_h0": "normalized_head"})
        if "time" not in frame.columns or "normalized_head" not in frame.columns:
            raise ValueError("SlugOscCase requires time and normalized_head columns.")
        return frame[["time", "normalized_head"]].copy()


@dataclass
class SlugOscAnalysis:
    case_id: str
    qc_rows: pd.DataFrame
    model_scores: pd.DataFrame
    parameter_summary: pd.DataFrame
    response_fit: pd.DataFrame
    best_model: str
    interpretation: str
