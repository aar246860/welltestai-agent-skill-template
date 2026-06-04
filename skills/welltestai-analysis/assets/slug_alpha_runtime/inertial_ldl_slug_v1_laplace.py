from __future__ import annotations

import math
from typing import Callable

import numpy as np
from mpmath import invertlaplace, mp


def stehfest_weights(n_terms: int = 12) -> np.ndarray:
    if n_terms % 2:
        raise ValueError("Stehfest inversion requires an even number of terms.")
    half = n_terms // 2
    weights = np.zeros(n_terms, dtype=float)
    for i in range(1, n_terms + 1):
        total = 0.0
        for k in range((i + 1) // 2, min(i, half) + 1):
            total += (
                k**half
                * math.factorial(2 * k)
                / (
                    math.factorial(half - k)
                    * math.factorial(k)
                    * math.factorial(k - 1)
                    * math.factorial(i - k)
                    * math.factorial(2 * k - i)
                )
            )
        weights[i - 1] = total * ((-1) ** (half + i))
    return weights


_STEHFEST_12 = stehfest_weights(12)


def inverse_laplace_stehfest_safe(
    laplace_function: Callable[[complex], complex],
    times: np.ndarray,
    *,
    n_terms: int = 12,
) -> np.ndarray:
    """Fast real-axis inverse Laplace evaluation for type-curve generation."""
    times = np.asarray(times, dtype=float)
    values = np.empty_like(times, dtype=float)
    weights = _STEHFEST_12 if n_terms == 12 else stehfest_weights(n_terms)
    ln2 = math.log(2.0)
    for idx, time in np.ndenumerate(times):
        if time <= 0.0:
            values[idx] = 1.0
            continue
        samples = []
        for j, weight in enumerate(weights, start=1):
            p = j * ln2 / time
            try:
                samples.append(weight * laplace_function(p))
            except Exception:
                samples.append(np.nan)
        arr = np.asarray(samples, dtype=complex)
        if np.any(~np.isfinite(arr.real)) or np.any(~np.isfinite(arr.imag)):
            values[idx] = np.nan
        else:
            values[idx] = float(np.real(ln2 / time * np.sum(arr)))
    return values


def inverse_laplace_dehoog_safe(
    laplace_function: Callable[[complex], complex],
    times: np.ndarray,
    *,
    dps: int = 80,
) -> np.ndarray:
    """Independent complex-safe inverse Laplace evaluation using de Hoog."""
    times = np.asarray(times, dtype=float)
    mp.dps = int(dps)
    values: list[float] = []
    for time in times:
        if time <= 0.0:
            values.append(1.0)
            continue

        def wrapped(p):
            try:
                return laplace_function(p)
            except Exception:
                return laplace_function(complex(p))

        result = invertlaplace(wrapped, float(time), method="dehoog")
        values.append(float(np.real(complex(result))))
    return np.asarray(values, dtype=float)
