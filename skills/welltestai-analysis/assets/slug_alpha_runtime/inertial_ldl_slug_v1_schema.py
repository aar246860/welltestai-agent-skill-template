from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class InertialLDLSlugCase:
    case_id: str
    time: np.ndarray
    normalized_head: np.ndarray
    metadata: dict | None = None

    def frame(self) -> pd.DataFrame:
        out = pd.DataFrame(
            {
                "time": np.asarray(self.time, dtype=float),
                "normalized_head": np.asarray(self.normalized_head, dtype=float),
            }
        )
        return out[np.isfinite(out["time"]) & np.isfinite(out["normalized_head"])].sort_values("time")


@dataclass
class InertialLDLSlugAnalysis:
    case_id: str
    qc_rows: pd.DataFrame
    model_scores: pd.DataFrame
    parameter_summary: pd.DataFrame
    response_fit: pd.DataFrame
    best_model: str
    interpretation: str
