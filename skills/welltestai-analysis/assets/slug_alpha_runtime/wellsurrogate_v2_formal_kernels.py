from __future__ import annotations

import numpy as np

from response_atlas_v6_formal_kernels import jacob_lohman_constant_head, ldl_constant_head, ldl_constant_rate, theis_constant_rate
from wellpfn_v2_slug_bridge import classical_slug_response, ldl_slug_response
from wellsurrogate_v2_dimensionless import ch_time_from_tDw, cr_time_from_tD, lags_from_family, logit, safe_log10, sigmoid, tD_at_radius


def cr_kernel(family: str, log_tD: np.ndarray, log_r_over_rw: np.ndarray, log_theta_q: np.ndarray, log_theta_h: np.ndarray, n_terms: int = 6) -> np.ndarray:
    tD = 10.0 ** np.asarray(log_tD, dtype=float)
    r_over_rw = 10.0 ** np.asarray(log_r_over_rw, dtype=float)
    theta_q = 10.0 ** np.asarray(log_theta_q, dtype=float)
    theta_h = 10.0 ** np.asarray(log_theta_h, dtype=float)
    out = np.empty_like(tD, dtype=float)
    for idx in range(tD.size):
        time = cr_time_from_tD(np.asarray([tD[idx]]), np.asarray([r_over_rw[idx]]))
        r = np.asarray([r_over_rw[idx]])
        tc = r_over_rw[idx] ** 2
        tau_q, tau_h = lags_from_family("cr", family, theta_q[idx], theta_h[idx], tc)
        if "classical" in family or "equal_lag_collapse" in family:
            s = theis_constant_rate(time, r, 4.0 * np.pi, 1.0, 1.0)[0, 0]
        else:
            s = ldl_constant_rate(time, r, 1.0, 4.0 * np.pi, 1.0, 1.0, tau_q, tau_h, n_terms=n_terms)[0, 0]
        out[idx] = max(float(s), 1.0e-12)
    return safe_log10(out)


def ch_kernel(
    family: str,
    response_type: np.ndarray,
    log_tDw: np.ndarray,
    log_r_over_rw: np.ndarray,
    log_theta_q: np.ndarray,
    log_theta_h: np.ndarray,
    n_terms: int = 6,
) -> np.ndarray:
    tDw = 10.0 ** np.asarray(log_tDw, dtype=float)
    r_over_rw = 10.0 ** np.asarray(log_r_over_rw, dtype=float)
    theta_q = 10.0 ** np.asarray(log_theta_q, dtype=float)
    theta_h = 10.0 ** np.asarray(log_theta_h, dtype=float)
    response_type = np.asarray(response_type).astype(str)
    out = np.empty_like(tDw, dtype=float)
    for idx in range(tDw.size):
        time = ch_time_from_tDw(np.asarray([tDw[idx]]))
        r = np.asarray([r_over_rw[idx]])
        tau_q, tau_h = lags_from_family("ch", family, theta_q[idx], theta_h[idx], 1.0)
        if "classical" in family or "equal_lag_collapse" in family:
            q, s = jacob_lohman_constant_head(time, r, 1.0, 1.0, 1.0, 1.0, n_terms=n_terms)
        else:
            q, s = ldl_constant_head(time, r, 1.0, 1.0, 1.0, 1.0, tau_q, tau_h, n_terms=n_terms)
        if response_type[idx] == "discharge":
            value = float(q[0]) / (2.0 * np.pi)
        else:
            value = float(s[0, 0])
        out[idx] = max(value, 1.0e-12)
    return safe_log10(out)


def slug_kernel(family: str, log_td: np.ndarray, log_alpha: np.ndarray, log_theta_q: np.ndarray, log_theta_h: np.ndarray, ar_over_a: np.ndarray) -> np.ndarray:
    td = 10.0 ** np.asarray(log_td, dtype=float)
    alpha = 10.0 ** np.asarray(log_alpha, dtype=float)
    theta_q = 10.0 ** np.asarray(log_theta_q, dtype=float)
    theta_h = 10.0 ** np.asarray(log_theta_h, dtype=float)
    ar_over_a = np.asarray(ar_over_a, dtype=float)
    out = np.empty_like(td, dtype=float)
    grouped: dict[tuple[float, float, float, float], list[int]] = {}
    for idx in range(td.size):
        tq, th = lags_from_family("slug", family, theta_q[idx], theta_h[idx], 1.0)
        key = (float(alpha[idx]), float(tq), float(th), float(ar_over_a[idx]))
        grouped.setdefault(key, []).append(idx)
    for (a_value, tq, th, ar_value), indices in grouped.items():
        idx_array = np.asarray(indices, dtype=int)
        if "classical" in family or "equal_lag_collapse" in family:
            values = classical_slug_response(td[idx_array], alpha=a_value, ar_over_a=ar_value)
        else:
            values = ldl_slug_response(td[idx_array], alpha=a_value, theta_q=tq, theta_h=th, ar_over_a=ar_value)
        out[idx_array] = np.asarray(values, dtype=float)
    return logit(out)


def inverse_cr_response(log_sD: np.ndarray) -> np.ndarray:
    return 10.0 ** np.asarray(log_sD, dtype=float)


def inverse_ch_response(log_value: np.ndarray) -> np.ndarray:
    return 10.0 ** np.asarray(log_value, dtype=float)


def inverse_slug_response(logit_value: np.ndarray) -> np.ndarray:
    return sigmoid(logit_value)
