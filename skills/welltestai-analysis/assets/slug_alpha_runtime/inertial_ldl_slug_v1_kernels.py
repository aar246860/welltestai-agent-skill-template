from __future__ import annotations

import numpy as np

from inertial_ldl_slug_v1_admittance import (
    classical_slug_admittance,
    ldl_slug_admittance,
)
from inertial_ldl_slug_v1_laplace import inverse_laplace_stehfest_safe


def coupled_slug_laplace(
    p: complex,
    *,
    alpha: float,
    theta_q: float = 0.0,
    theta_h: float = 0.0,
    inertance: float = 0.0,
    damping: float = 1.0,
    stiffness: float = 0.0,
) -> complex:
    """Coupled wellbore-inertia and aquifer-admittance slug kernel.

    The dimensionless form is
    H(p) = (M p + C) / (M p^2 + C p + K + A_a(p)).
    With M=0, C=1, and K=0, it collapses to the monotonic aquifer response.
    """
    pc = complex(p)
    m = max(float(inertance), 0.0)
    c = max(float(damping), 1.0e-12)
    k = max(float(stiffness), 0.0)
    admittance = ldl_slug_admittance(
        pc,
        alpha=alpha,
        theta_q=theta_q,
        theta_h=theta_h,
    )
    denominator = m * pc * pc + c * pc + k + admittance
    numerator = m * pc + c
    if abs(denominator) < 1.0e-300:
        return complex(np.nan)
    return complex(numerator / denominator)


def classical_slug_monotonic(times: np.ndarray, *, alpha: float) -> np.ndarray:
    times = np.asarray(times, dtype=float)
    values = inverse_laplace_stehfest_safe(
        lambda p: 1.0 / (p + classical_slug_admittance(p, alpha=alpha)),
        times,
    )
    return np.clip(values, 0.0, 1.05)


def ldl_slug_monotonic(
    times: np.ndarray,
    *,
    alpha: float,
    theta_q: float,
    theta_h: float,
) -> np.ndarray:
    times = np.asarray(times, dtype=float)
    values = inverse_laplace_stehfest_safe(
        lambda p: 1.0
        / (p + ldl_slug_admittance(p, alpha=alpha, theta_q=theta_q, theta_h=theta_h)),
        times,
    )
    return np.clip(values, 0.0, 1.05)


def classical_inertial_slug_time(
    times: np.ndarray,
    *,
    alpha: float,
    inertance: float,
    damping: float,
    stiffness: float,
) -> np.ndarray:
    times = np.asarray(times, dtype=float)
    if inertance <= 1.0e-14 and stiffness <= 1.0e-14:
        return classical_slug_monotonic(times, alpha=alpha)
    return inverse_laplace_stehfest_safe(
        lambda p: coupled_slug_laplace(
            p,
            alpha=alpha,
            theta_q=0.0,
            theta_h=0.0,
            inertance=inertance,
            damping=damping,
            stiffness=stiffness,
        ),
        times,
    )


def ldl_inertial_slug_time(
    times: np.ndarray,
    *,
    alpha: float,
    theta_q: float,
    theta_h: float,
    inertance: float,
    damping: float,
    stiffness: float,
) -> np.ndarray:
    times = np.asarray(times, dtype=float)
    if inertance <= 1.0e-14 and stiffness <= 1.0e-14:
        return ldl_slug_monotonic(times, alpha=alpha, theta_q=theta_q, theta_h=theta_h)
    return inverse_laplace_stehfest_safe(
        lambda p: coupled_slug_laplace(
            p,
            alpha=alpha,
            theta_q=theta_q,
            theta_h=theta_h,
            inertance=inertance,
            damping=damping,
            stiffness=stiffness,
        ),
        times,
    )
