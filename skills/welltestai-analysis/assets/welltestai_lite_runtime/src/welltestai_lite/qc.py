from __future__ import annotations

import numpy as np
import pandas as pd


def run_qc(observations: pd.DataFrame) -> list[dict]:
    warnings: list[dict] = []
    n = int(observations.shape[0])
    time = observations["time"].to_numpy(dtype=float)
    response = observations["response"].to_numpy(dtype=float)
    if n < 16:
        warnings.append(
            {
                "warning": "sparse_sampling",
                "observed_value": n,
                "interpretation": "The response has limited sampling density.",
                "recommended_action": "Collect more points across logarithmic time before final interpretation.",
            }
        )
    if np.nanmin(time) > 0.05:
        warnings.append(
            {
                "warning": "missing_early_time",
                "observed_value": float(np.nanmin(time)),
                "interpretation": "Early-time response is not captured.",
                "recommended_action": "Add early measurements if wellbore storage, skin, or instrument response is important.",
            }
        )
    if np.nanmax(time) < 100.0:
        warnings.append(
            {
                "warning": "missing_late_time",
                "observed_value": float(np.nanmax(time)),
                "interpretation": "Late-time response is not captured.",
                "recommended_action": "Extend the test or recovery monitoring before interpreting boundary or replenishment effects.",
            }
        )
    if np.any(response <= 0):
        warnings.append(
            {
                "warning": "non_positive_response",
                "observed_value": float(np.nanmin(response)),
                "interpretation": "Log-time response features require positive values.",
                "recommended_action": "Check baseline correction, units, and sign convention.",
            }
        )
    if not warnings:
        warnings.append(
            {
                "warning": "no_major_qc_warning",
                "observed_value": n,
                "interpretation": "Basic time-series checks did not detect a blocking issue.",
                "recommended_action": "Proceed with model-family screening and review calibrated confidence.",
            }
        )
    return warnings

