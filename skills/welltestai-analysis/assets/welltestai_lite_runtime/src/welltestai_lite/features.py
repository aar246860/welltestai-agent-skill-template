from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .paths import PROJECT_SRC

if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from formal_extension_rc7_features import build_rc7_feature_table  # noqa: E402


def feature_table_from_case(case: dict, observations: pd.DataFrame) -> pd.DataFrame:
    case_id = str(case["case_id"])
    cases = pd.DataFrame(
        [
            {
                "case_id": case_id,
                "truth_family": "unknown",
                "truth_parameter_name": "",
                "truth_parameter_value": np.nan,
                "truth_parameter_log10": np.nan,
                "truth_parameter_aux": np.nan,
                "r_cm": float(case.get("radius_cm", 2.0)),
                "rw_cm": float(case.get("well_radius_cm", 2.0)),
                "q_ml_min": float(case.get("forcing_value", 100.0)),
                "t_cm2_min": float(case.get("transmissivity_scale_cm2_min", 80.0)),
                "storativity": float(case.get("storativity_scale", 1.0e-3)),
                "noise_fraction": float(case.get("noise_fraction", 0.03)),
            }
        ]
    )
    obs = observations.sort_values("time").reset_index(drop=True)
    rc_obs = pd.DataFrame(
        {
            "case_id": case_id,
            "obs_index": np.arange(obs.shape[0]),
            "time_min": obs["time"].to_numpy(dtype=float),
            "truth": obs["response"].to_numpy(dtype=float),
            "observed": np.maximum(obs["response"].to_numpy(dtype=float), 1.0e-12),
            "sigma": np.maximum(0.03 * np.abs(obs["response"].to_numpy(dtype=float)), 1.0e-6),
        }
    )
    split = pd.DataFrame({"case_id": [case_id], "split": ["query"]})
    return build_rc7_feature_table(cases, rc_obs, split)

