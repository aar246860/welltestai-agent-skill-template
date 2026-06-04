from __future__ import annotations

import math

import numpy as np
from mpmath import invertlaplace, mp
from scipy.optimize import brentq
from scipy.special import kv


def _stehfest_weights(n_terms: int) -> np.ndarray:
    if n_terms % 2 != 0:
        raise ValueError("Stehfest inversion requires an even number of terms.")

    half = n_terms // 2
    weights = np.zeros(n_terms, dtype=float)

    for i in range(1, n_terms + 1):
        total = 0.0
        k_min = (i + 1) // 2
        k_max = min(i, half)
        for k in range(k_min, k_max + 1):
            numerator = k**half * math.factorial(2 * k)
            denominator = (
                math.factorial(half - k)
                * math.factorial(k)
                * math.factorial(k - 1)
                * math.factorial(i - k)
                * math.factorial(2 * k - i)
            )
            total += numerator / denominator
        weights[i - 1] = total * ((-1) ** (half + i))
    return weights


_STEHFEST_12 = _stehfest_weights(12)


def inverse_laplace_stehfest(laplace_function, times: np.ndarray, n_terms: int = 12) -> np.ndarray:
    """Invert a Laplace-domain function at positive times."""
    weights = _STEHFEST_12 if n_terms == 12 else _stehfest_weights(n_terms)
    times = np.asarray(times, dtype=float)
    if np.any(times <= 0.0):
        raise ValueError("Stehfest inversion requires strictly positive times.")

    ln2 = math.log(2.0)
    values = np.empty_like(times)
    for idx, time in np.ndenumerate(times):
        samples = np.array(
            [laplace_function((j + 1) * ln2 / time) for j in range(len(weights))],
            dtype=float,
        )
        if np.any(~np.isfinite(samples)):
            values[idx] = np.nan
            continue
        with np.errstate(all="ignore"):
            values[idx] = ln2 / time * np.dot(weights, samples)
    return values


def inverse_laplace_dehoog(laplace_function, times: np.ndarray) -> np.ndarray:
    """Independent inverse-Laplace evaluation using de Hoog."""
    times = np.asarray(times, dtype=float)
    if np.any(times <= 0.0):
        raise ValueError("de Hoog inversion requires strictly positive times.")

    mp.dps = 50
    values = []
    for time in times:
        def wrapped(p):
            try:
                return laplace_function(p)
            except Exception:
                return laplace_function(complex(p))

        result = invertlaplace(wrapped, time, method="dehoog")
        values.append(float(np.real(result)))
    return np.asarray(values, dtype=float)


def _classical_laplace_response(p: float, alpha: float) -> float:
    z = math.sqrt(p)
    with np.errstate(all="ignore"):
        ratio = kv(1, z) / kv(0, z)
        value = 1.0 / (p + 2.0 * alpha * z * ratio)
    if not np.isfinite(value):
        return np.inf
    return float(np.real(value))


def _dpl_laplace_response(p: float, alpha: float, theta_q: float, theta_h: float) -> float:
    mu = math.sqrt(p * (1.0 + p * theta_q) / (1.0 + p * theta_h))
    boundary_factor = math.sqrt(p * (1.0 + p * theta_h) / (1.0 + p * theta_q))
    with np.errstate(all="ignore"):
        ratio = kv(1, mu) / kv(0, mu)
        value = 1.0 / (p + 2.0 * alpha * boundary_factor * ratio)
    if not np.isfinite(value):
        return np.inf
    return float(np.real(value))


def classical_step_response(times: np.ndarray, alpha: float) -> np.ndarray:
    """Dimensionless well response after a full instantaneous slug."""
    response = inverse_laplace_stehfest(
        lambda p: _classical_laplace_response(p, alpha),
        np.asarray(times, dtype=float),
    )
    return np.clip(response, 0.0, 1.0)


def find_pressurization_time(alpha: float, ar_over_a: float) -> float:
    """Recover dimensionless pressurization time from Ar/A for partial tests."""
    if ar_over_a >= 1.0:
        return math.inf
    if ar_over_a <= 0.0:
        return 0.0

    target = 1.0 - ar_over_a

    def residual(time_value: float) -> float:
        return classical_step_response(np.array([time_value]), alpha=alpha)[0] - target

    grid = np.logspace(-8, 8, 81)
    values = np.array([residual(value) for value in grid], dtype=float)
    finite = np.isfinite(values)
    grid = grid[finite]
    values = values[finite]
    if len(grid) == 0:
        raise ValueError("Unable to bracket the classical pressurization time.")

    sign_changes = np.where(np.sign(values[:-1]) != np.sign(values[1:]))[0]
    if len(sign_changes) == 0:
        if values[0] < 0.0:
            return float(grid[0])
        return float(grid[-1])

    idx = int(sign_changes[0])
    return brentq(residual, float(grid[idx]), float(grid[idx + 1]), maxiter=200)


def classical_recovery_curve(times: np.ndarray, alpha: float, ar_over_a: float = 1.0) -> np.ndarray:
    """Dimensionless recovery curve for full or partially terminated air-pressurized tests."""
    times = np.asarray(times, dtype=float)
    if ar_over_a >= 1.0:
        return classical_step_response(times, alpha=alpha)

    tr = find_pressurization_time(alpha=alpha, ar_over_a=ar_over_a)
    response = classical_step_response(times, alpha=alpha) - classical_step_response(times + tr, alpha=alpha)
    return np.clip(response, 0.0, 1.0)


def dpl_step_response(times: np.ndarray, alpha: float, theta_q: float, theta_h: float) -> np.ndarray:
    """Dimensionless well response for the lagging-theory slug-test model."""
    response = inverse_laplace_stehfest(
        lambda p: _dpl_laplace_response(p, alpha, theta_q, theta_h),
        np.asarray(times, dtype=float),
    )
    return np.clip(response, 0.0, 1.0)


def find_dpl_pressurization_time(alpha: float, theta_q: float, theta_h: float, ar_over_a: float) -> float:
    if ar_over_a >= 1.0:
        return math.inf
    if ar_over_a <= 0.0:
        return 0.0

    target = 1.0 - ar_over_a

    def residual(time_value: float) -> float:
        return dpl_step_response(np.array([time_value]), alpha=alpha, theta_q=theta_q, theta_h=theta_h)[0] - target

    grid = np.logspace(-8, 8, 81)
    values = np.array([residual(value) for value in grid], dtype=float)
    finite = np.isfinite(values)
    grid = grid[finite]
    values = values[finite]
    if len(grid) == 0:
        raise ValueError("Unable to bracket the lagging-theory pressurization time.")

    sign_changes = np.where(np.sign(values[:-1]) != np.sign(values[1:]))[0]
    if len(sign_changes) == 0:
        if values[0] < 0.0:
            return float(grid[0])
        return float(grid[-1])

    idx = int(sign_changes[0])
    return brentq(residual, float(grid[idx]), float(grid[idx + 1]), maxiter=200)


def dpl_recovery_curve(
    times: np.ndarray,
    alpha: float,
    theta_q: float,
    theta_h: float,
    ar_over_a: float = 1.0,
) -> np.ndarray:
    """Dimensionless recovery curve for the lagging-theory air-pressurized slug-test model."""
    times = np.asarray(times, dtype=float)
    if ar_over_a >= 1.0:
        return dpl_step_response(times, alpha=alpha, theta_q=theta_q, theta_h=theta_h)

    tr = find_dpl_pressurization_time(
        alpha=alpha,
        theta_q=theta_q,
        theta_h=theta_h,
        ar_over_a=ar_over_a,
    )
    response = dpl_step_response(times, alpha=alpha, theta_q=theta_q, theta_h=theta_h) - dpl_step_response(
        times + tr,
        alpha=alpha,
        theta_q=theta_q,
        theta_h=theta_h,
    )
    return np.clip(response, 0.0, 1.0)
