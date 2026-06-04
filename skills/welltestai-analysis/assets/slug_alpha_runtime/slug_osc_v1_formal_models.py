from __future__ import annotations

import numpy as np
from scipy.optimize import differential_evolution, least_squares


def monotonic_exponential_slug(
    time: np.ndarray,
    *,
    decay_rate: float,
    amplitude: float = 1.0,
    equilibrium: float = 0.0,
) -> np.ndarray:
    """Simple monotonic reference used for QC and fitting initialization."""
    time = np.asarray(time, dtype=float)
    decay = max(float(decay_rate), 1.0e-12)
    return float(equilibrium) + float(amplitude) * np.exp(-decay * np.maximum(time, 0.0))


def damped_oscillator_slug(
    time: np.ndarray,
    *,
    zeta: float,
    omega0: float,
    amplitude: float = 1.0,
    equilibrium: float = 0.0,
) -> np.ndarray:
    """Dimensionless inertial slug screening response.

    The model solves h'' + 2*zeta*omega0*h' + omega0^2*h = 0 with
    h(0)=amplitude and h'(0)=0, then adds the equilibrium offset.
    """
    time = np.asarray(time, dtype=float)
    zeta = float(zeta)
    omega0 = max(float(omega0), 1.0e-12)
    amp = float(amplitude)
    eq = float(equilibrium)
    t = np.maximum(time, 0.0)

    if zeta < 1.0:
        wd = omega0 * np.sqrt(max(1.0 - zeta**2, 1.0e-12))
        coeff = zeta / np.sqrt(max(1.0 - zeta**2, 1.0e-12))
        response = amp * np.exp(-zeta * omega0 * t) * (np.cos(wd * t) + coeff * np.sin(wd * t))
        return eq + response
    if np.isclose(zeta, 1.0):
        return eq + amp * (1.0 + omega0 * t) * np.exp(-omega0 * t)

    root = np.sqrt(zeta**2 - 1.0)
    r1 = -omega0 * (zeta - root)
    r2 = -omega0 * (zeta + root)
    denominator = r2 - r1
    response = amp * (r2 * np.exp(r1 * t) - r1 * np.exp(r2 * t)) / denominator
    return eq + response


def _oscillator_residual(params: np.ndarray, time: np.ndarray, head: np.ndarray) -> np.ndarray:
    zeta, omega0, amplitude, equilibrium = params
    pred = damped_oscillator_slug(
        time,
        zeta=zeta,
        omega0=omega0,
        amplitude=amplitude,
        equilibrium=equilibrium,
    )
    return pred - head


def fit_oscillator_initial(time: np.ndarray, head: np.ndarray) -> dict[str, float]:
    """Robust initial underdamped oscillator fit for screening and reports."""
    time = np.asarray(time, dtype=float)
    head = np.asarray(head, dtype=float)
    finite = np.isfinite(time) & np.isfinite(head)
    time = time[finite]
    head = head[finite]
    order = np.argsort(time)
    time = time[order]
    head = head[order]
    if time.size < 5:
        raise ValueError("At least five finite observations are required.")
    t_span = max(float(time[-1] - time[0]), 1.0e-8)
    centered_t = time - float(time[0])
    amp0 = float(head[0] - np.nanmedian(head[-max(3, time.size // 5) :]))
    eq0 = float(np.nanmedian(head[-max(3, time.size // 5) :]))
    if abs(amp0) < 1.0e-6:
        amp0 = float(np.nanmax(head) - np.nanmin(head)) or 1.0

    bounds = (
        np.array([0.01, 0.1 / t_span, -2.5 * abs(amp0), eq0 - 2.0 * abs(amp0)]),
        np.array([0.99, 80.0 / t_span, 2.5 * abs(amp0), eq0 + 2.0 * abs(amp0)]),
    )

    def objective(x: np.ndarray) -> float:
        residual = _oscillator_residual(x, centered_t, head)
        return float(np.mean(residual**2))

    de = differential_evolution(
        objective,
        bounds=list(zip(bounds[0], bounds[1])),
        seed=20260604,
        maxiter=50,
        popsize=8,
        polish=False,
        updating="immediate",
    )
    result = least_squares(
        _oscillator_residual,
        de.x,
        args=(centered_t, head),
        bounds=bounds,
        max_nfev=3000,
    )
    params = result.x
    pred = damped_oscillator_slug(
        centered_t,
        zeta=params[0],
        omega0=params[1],
        amplitude=params[2],
        equilibrium=params[3],
    )
    residual = pred - head
    zeta = float(params[0])
    omega0 = float(params[1])
    omega_d = omega0 * np.sqrt(max(1.0 - zeta**2, 0.0))
    period = float(2.0 * np.pi / omega_d) if omega_d > 0.0 else float("inf")
    damping_time = float(1.0 / max(zeta * omega0, 1.0e-12))
    return {
        "zeta": zeta,
        "omega0": omega0,
        "omega_d": float(omega_d),
        "period": period,
        "damping_time": damping_time,
        "amplitude": float(params[2]),
        "equilibrium": float(params[3]),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "success": bool(result.success),
    }


def fit_monotonic_initial(time: np.ndarray, head: np.ndarray) -> dict[str, float]:
    """Fit a monotonic exponential reference for model comparison."""
    time = np.asarray(time, dtype=float)
    head = np.asarray(head, dtype=float)
    finite = np.isfinite(time) & np.isfinite(head)
    time = time[finite]
    head = head[finite]
    order = np.argsort(time)
    time = time[order]
    head = head[order]
    centered_t = time - float(time[0])
    tail = head[-max(3, head.size // 5) :]
    eq0 = float(np.nanmedian(tail))
    amp0 = float(head[0] - eq0)
    t_span = max(float(centered_t[-1] - centered_t[0]), 1.0e-8)
    lower = np.array([1.0e-8, -2.5 * abs(amp0), eq0 - 2.0 * abs(amp0)])
    upper = np.array([80.0 / t_span, 2.5 * abs(amp0), eq0 + 2.0 * abs(amp0)])

    def residual(params: np.ndarray) -> np.ndarray:
        return monotonic_exponential_slug(
            centered_t,
            decay_rate=params[0],
            amplitude=params[1],
            equilibrium=params[2],
        ) - head

    result = least_squares(
        residual,
        np.array([1.0 / t_span, amp0, eq0]),
        bounds=(lower, upper),
        max_nfev=2000,
    )
    pred = monotonic_exponential_slug(
        centered_t,
        decay_rate=result.x[0],
        amplitude=result.x[1],
        equilibrium=result.x[2],
    )
    return {
        "decay_rate": float(result.x[0]),
        "amplitude": float(result.x[1]),
        "equilibrium": float(result.x[2]),
        "rmse": float(np.sqrt(np.mean((pred - head) ** 2))),
        "success": bool(result.success),
    }
