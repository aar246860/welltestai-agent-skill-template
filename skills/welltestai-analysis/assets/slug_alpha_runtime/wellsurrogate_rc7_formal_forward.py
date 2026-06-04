from __future__ import annotations

import numpy as np

from wellsurrogate_rc4_truth_generator import generate_curve_truth, inverse_transformed_response
from wellsurrogate_rc7_dimensional_operator import (
    ch_QD_from_discharge,
    ch_discharge_from_QD,
    ch_tD_from_time,
    cr_drawdown_from_sD,
    cr_sD_from_drawdown,
    cr_tD_from_time,
)
from wellsurrogate_rc7_schema import MODE_TO_FAMILY


def _candidate_scalars(theta: dict | np.void) -> tuple[float, float, float, float, float]:
    logT = float(theta["logT"])
    logS = float(theta["logS"])
    log_tau_h = float(theta["log_tau_h"])
    log_tau_q = float(theta["log_tau_q"])
    log_R_tau = float(theta["log_R_tau"])
    return logT, logS, log_tau_h, log_tau_q, log_R_tau


def predict_observation(obs, mode: str, theta: dict | np.void, n_terms: int = 4) -> np.ndarray:
    logT, logS, log_tau_h, log_tau_q, _ = _candidate_scalars(theta)
    T = 10.0**logT
    S = 10.0**logS
    family = MODE_TO_FAMILY[mode][obs.test_type]
    if obs.test_type == "cr":
        tD = cr_tD_from_time(T=T, S=S, time=obs.time, radius=max(obs.radius, obs.rw))
        tc = S * max(obs.radius, obs.rw) ** 2 / T
        log_theta_h = log_tau_h - np.log10(tc)
        log_theta_q = log_tau_q - np.log10(tc)
        transformed = generate_curve_truth(
            test_type="cr",
            family=family,
            response_type="drawdown",
            log_time=np.log10(np.maximum(tD, 1.0e-12)),
            log_radius=np.log10(max(obs.radius / obs.rw, 1.0)),
            log_theta_q=log_theta_q,
            log_theta_h=log_theta_h,
            log_alpha=obs.log_alpha,
            ar_over_a=obs.ar_over_a,
            n_terms=n_terms,
        )
        sD = inverse_transformed_response("cr", transformed)
        return np.log10(np.maximum(cr_drawdown_from_sD(sD=sD, T=T, Q=obs.Q), 1.0e-30))
    if obs.test_type == "ch":
        tDw = ch_tD_from_time(T=T, S=S, time=obs.time, rw=obs.rw)
        tc = S * obs.rw**2 / T
        log_theta_h = log_tau_h - np.log10(tc)
        log_theta_q = log_tau_q - np.log10(tc)
        log_radius = 0.0 if obs.response_type == "discharge" else np.log10(max(obs.radius / obs.rw, 1.0))
        transformed = generate_curve_truth(
            test_type="ch",
            family=family,
            response_type=obs.response_type,
            log_time=np.log10(np.maximum(tDw, 1.0e-12)),
            log_radius=log_radius,
            log_theta_q=log_theta_q,
            log_theta_h=log_theta_h,
            log_alpha=obs.log_alpha,
            ar_over_a=obs.ar_over_a,
            n_terms=n_terms,
        )
        value = inverse_transformed_response("ch", transformed)
        if obs.response_type == "discharge":
            return np.log10(np.maximum(ch_discharge_from_QD(QD=value, T=T, sw=obs.sw), 1.0e-30))
        return np.log10(np.maximum(obs.sw * value, 1.0e-30))
    # Partial-dimensional slug: time is already dimensionless for RC7.
    transformed = generate_curve_truth(
        test_type="slug",
        family=family,
        response_type="normalized_head",
        log_time=np.log10(np.maximum(obs.time, 1.0e-12)),
        log_radius=0.0,
        log_theta_q=log_tau_q,
        log_theta_h=log_tau_h,
        log_alpha=obs.log_alpha,
        ar_over_a=obs.ar_over_a,
        n_terms=n_terms,
    )
    return transformed


def predict_joint_case(case, mode: str, theta: dict | np.void, n_terms: int = 4) -> np.ndarray:
    return np.asarray([predict_observation(obs, mode, theta, n_terms=n_terms) for obs in case.observations], dtype=float)

