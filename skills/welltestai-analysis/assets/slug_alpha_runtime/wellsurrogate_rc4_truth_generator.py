from __future__ import annotations

import numpy as np

from wellsurrogate_v2_formal_kernels import ch_kernel, cr_kernel, slug_kernel


def tau_regime(log_r_tau: float) -> str:
    if log_r_tau < -0.2:
        return "tau_q_lt_tau_h"
    if log_r_tau > 0.2:
        return "tau_q_gt_tau_h"
    return "near_equal"


def gate_curve(response: np.ndarray, max_abs: float = 60.0) -> tuple[bool, str]:
    arr = np.asarray(response, dtype=float)
    if arr.ndim != 1 or arr.size < 3:
        return False, "bad_shape"
    if not np.all(np.isfinite(arr)):
        return False, "nonfinite"
    if float(np.nanmax(np.abs(arr))) > max_abs:
        return False, "extreme_transformed_value"
    if float(np.nanstd(arr)) < 1.0e-10:
        return False, "constant_curve"
    return True, "accepted"


def generate_curve_truth(
    *,
    test_type: str,
    family: str,
    response_type: str,
    log_time: np.ndarray,
    log_radius: float,
    log_theta_q: float,
    log_theta_h: float,
    log_alpha: float,
    ar_over_a: float,
    n_terms: int = 4,
) -> np.ndarray:
    log_time = np.asarray(log_time, dtype=float)
    n = log_time.size
    if test_type == "cr":
        return cr_kernel(
            family,
            log_time,
            np.full(n, log_radius),
            np.full(n, log_theta_q),
            np.full(n, log_theta_h),
            n_terms=n_terms,
        ).astype(np.float64)
    if test_type == "ch":
        return ch_kernel(
            family,
            np.full(n, response_type),
            log_time,
            np.full(n, log_radius),
            np.full(n, log_theta_q),
            np.full(n, log_theta_h),
            n_terms=n_terms,
        ).astype(np.float64)
    if test_type == "slug":
        return slug_kernel(
            family,
            log_time,
            np.full(n, log_alpha),
            np.full(n, log_theta_q),
            np.full(n, log_theta_h),
            np.full(n, ar_over_a),
        ).astype(np.float64)
    raise ValueError(f"Unknown test_type: {test_type}")


def inverse_transformed_response(test_type: str, transformed: np.ndarray) -> np.ndarray:
    arr = np.asarray(transformed, dtype=float)
    if test_type in {"cr", "ch"}:
        return 10.0 ** arr
    if test_type == "slug":
        return 1.0 / (1.0 + np.exp(-np.clip(arr, -80.0, 80.0)))
    raise ValueError(f"Unknown test_type: {test_type}")

