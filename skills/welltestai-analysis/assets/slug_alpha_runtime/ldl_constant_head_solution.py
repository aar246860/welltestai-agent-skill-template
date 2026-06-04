from __future__ import annotations

import math
from functools import lru_cache
from typing import Callable

import mpmath as mp
import numpy as np
from scipy.special import iv, k0, k1, kv, kve


def _real_if_close(value: complex | float, tolerance: float = 1.0e-10) -> complex | float:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, complex) or np.iscomplexobj(value):
        value = complex(value)
        if abs(value.imag) <= tolerance * max(1.0, abs(value.real)):
            return float(value.real)
    return value


def _is_zero(value: complex | float) -> bool:
    return abs(value) == 0.0


def _k0(value: complex | float) -> complex | float:
    return _real_if_close(kv(0, value))


def _k1(value: complex | float) -> complex | float:
    return _real_if_close(kv(1, value))


def _k1_over_k0(value: complex | float) -> complex | float:
    """Return K1(value)/K0(value) without large-argument underflow."""

    denominator = kve(0, value)
    if _is_zero(denominator) or not np.isfinite(denominator):
        return float("nan")
    return _real_if_close(kve(1, value) / denominator)


def _k0_ratio(numerator_arg: complex | float, denominator_arg: complex | float) -> complex | float:
    """Return K0(numerator_arg)/K0(denominator_arg) with scaled Bessel functions.

    Direct K0 values underflow at large arguments, which is common for
    early-time observation-well drawdown under flux-lag coordinates. Because
    kve(v, z) = exp(z) K_v(z), the ratio can be evaluated as
    kve(0, a) / kve(0, b) * exp(-(a - b)).
    """

    numerator = kve(0, numerator_arg)
    denominator = kve(0, denominator_arg)
    if _is_zero(denominator) or not np.isfinite(denominator):
        return float("nan")
    with np.errstate(all="ignore"):
        ratio = (numerator / denominator) * np.exp(-(numerator_arg - denominator_arg))
    if not np.isfinite(ratio):
        return float("nan")
    return _real_if_close(ratio)


def _i0(value: complex | float) -> complex | float:
    return _real_if_close(iv(0, value))


def _i1(value: complex | float) -> complex | float:
    return _real_if_close(iv(1, value))


def transmissivity_flux_correction(
    p_value: float, T: float, tau_q: float, tau_s: float
) -> float:
    """Return the Laplace-domain flux transmissivity T_Q(p).

    The adopted LDL constitutive law is

        (1 + p*tau_q) * ubar = T * (1 + p*tau_s) * grad(sbar).

    Solving for flux gives T_Q(p) = T*(1+p*tau_s)/(1+p*tau_q).
    """

    denominator = 1.0 + p_value * tau_q
    if not np.iscomplexobj(denominator) and denominator <= 0.0:
        return float("nan")
    return _real_if_close(T * (1.0 + p_value * tau_s) / denominator)


def lambda_classical(p_value: float, T: float, S: float) -> float:
    """Classical confined-aquifer radial propagation coefficient."""

    argument = p_value * S / T
    if not np.iscomplexobj(argument) and argument < 0.0:
        return float("nan")
    return _real_if_close(np.sqrt(argument))


def lambda_ldl(p_value: float, T: float, S: float, tau_q: float, tau_s: float) -> float:
    """LDL radial propagation coefficient for the constant-head problem."""

    denominator = T * (1.0 + p_value * tau_s)
    if not np.iscomplexobj(denominator) and denominator <= 0.0:
        return float("nan")
    argument = p_value * S * (1.0 + p_value * tau_q) / denominator
    if not np.iscomplexobj(argument) and argument < 0.0:
        return float("nan")
    return _real_if_close(np.sqrt(argument))


def sbar_classical(
    p_value: float, r: float, rw: float, T: float, S: float, sw: float
) -> float:
    """Infinite-domain classical constant-head drawdown in Laplace domain."""

    if p_value == 0.0:
        return float("nan")
    lam = lambda_classical(p_value, T=T, S=S)
    ratio = _k0_ratio(lam * r, lam * rw)
    if not np.isfinite(ratio):
        return float("nan")
    return _real_if_close((sw / p_value) * ratio)


def sbar_ldl(
    p_value: float,
    r: float,
    rw: float,
    T: float,
    S: float,
    tau_q: float,
    tau_s: float,
    sw: float,
) -> float:
    """Infinite-domain LDL constant-head drawdown in Laplace domain."""

    if p_value == 0.0:
        return float("nan")
    lam = lambda_ldl(p_value, T=T, S=S, tau_q=tau_q, tau_s=tau_s)
    ratio = _k0_ratio(lam * r, lam * rw)
    if not np.isfinite(ratio):
        return float("nan")
    return _real_if_close((sw / p_value) * ratio)


def qbar_classical(p_value: float, rw: float, T: float, S: float, sw: float) -> float:
    """Positive abstraction-rate magnitude for classical constant-head flow."""

    if p_value == 0.0:
        return float("nan")
    lam = lambda_classical(p_value, T=T, S=S)
    ratio = _k1_over_k0(lam * rw)
    if not np.isfinite(ratio):
        return float("nan")
    numerator = 2.0 * math.pi * rw * T * sw * lam * ratio
    return _real_if_close(numerator / p_value)


def qbar_ldl(
    p_value: float,
    rw: float,
    T: float,
    S: float,
    tau_q: float,
    tau_s: float,
    sw: float,
) -> float:
    """Positive abstraction-rate magnitude for LDL constant-head flow.

    The well-wall flux is computed with T_Q(p), not the scalar transmissivity T.
    """

    if p_value == 0.0:
        return float("nan")
    lam = lambda_ldl(p_value, T=T, S=S, tau_q=tau_q, tau_s=tau_s)
    ratio = _k1_over_k0(lam * rw)
    if not np.isfinite(ratio):
        return float("nan")
    Tq = transmissivity_flux_correction(p_value, T=T, tau_q=tau_q, tau_s=tau_s)
    numerator = 2.0 * math.pi * rw * Tq * sw * lam * ratio
    return _real_if_close(numerator / p_value)


def steady_finite_flowrate(T: float, sw: float, rw: float, R: float) -> float:
    """Steady positive flowrate for a finite Dirichlet outer boundary."""

    if R <= rw:
        raise ValueError("R must be greater than rw")
    return float(2.0 * math.pi * T * sw / math.log(R / rw))


def _finite_dirichlet_shape(lam: complex | float, r: float, R: float) -> complex | float:
    return _k0(lam * r) * _i0(lam * R) - _i0(lam * r) * _k0(lam * R)


def _finite_dirichlet_flux_shape(
    lam: complex | float, rw: float, R: float
) -> complex | float:
    return _k1(lam * rw) * _i0(lam * R) + _i1(lam * rw) * _k0(lam * R)


def sbar_classical_finite(
    p_value: float, r: float, rw: float, R: float, T: float, S: float, sw: float
) -> float:
    """Finite-domain classical constant-head drawdown with s(R,t)=0."""

    if p_value == 0.0:
        return float("nan")
    if R <= rw:
        raise ValueError("R must be greater than rw")
    lam = lambda_classical(p_value, T=T, S=S)
    denominator = _finite_dirichlet_shape(lam, rw, R)
    if _is_zero(denominator) or not np.isfinite(denominator):
        return float("nan")
    value = (sw / p_value) * _finite_dirichlet_shape(lam, r, R) / denominator
    return _real_if_close(value)


def sbar_ldl_finite(
    p_value: float,
    r: float,
    rw: float,
    R: float,
    T: float,
    S: float,
    tau_q: float,
    tau_s: float,
    sw: float,
) -> float:
    """Finite-domain LDL constant-head drawdown with s(R,t)=0."""

    if p_value == 0.0:
        return float("nan")
    if R <= rw:
        raise ValueError("R must be greater than rw")
    lam = lambda_ldl(p_value, T=T, S=S, tau_q=tau_q, tau_s=tau_s)
    denominator = _finite_dirichlet_shape(lam, rw, R)
    if _is_zero(denominator) or not np.isfinite(denominator):
        return float("nan")
    value = (sw / p_value) * _finite_dirichlet_shape(lam, r, R) / denominator
    return _real_if_close(value)


def qbar_classical_finite(
    p_value: float, rw: float, R: float, T: float, S: float, sw: float
) -> float:
    """Positive finite-domain classical constant-head flowrate magnitude."""

    if p_value == 0.0:
        return float("nan")
    if R <= rw:
        raise ValueError("R must be greater than rw")
    lam = lambda_classical(p_value, T=T, S=S)
    denominator = _finite_dirichlet_shape(lam, rw, R)
    if _is_zero(denominator) or not np.isfinite(denominator):
        return float("nan")
    numerator = 2.0 * math.pi * rw * T * sw * lam * _finite_dirichlet_flux_shape(
        lam, rw, R
    )
    return _real_if_close(numerator / (p_value * denominator))


def qbar_ldl_finite(
    p_value: float,
    rw: float,
    R: float,
    T: float,
    S: float,
    tau_q: float,
    tau_s: float,
    sw: float,
) -> float:
    """Positive finite-domain LDL constant-head flowrate magnitude."""

    if p_value == 0.0:
        return float("nan")
    if R <= rw:
        raise ValueError("R must be greater than rw")
    lam = lambda_ldl(p_value, T=T, S=S, tau_q=tau_q, tau_s=tau_s)
    denominator = _finite_dirichlet_shape(lam, rw, R)
    if _is_zero(denominator) or not np.isfinite(denominator):
        return float("nan")
    Tq = transmissivity_flux_correction(p_value, T=T, tau_q=tau_q, tau_s=tau_s)
    numerator = 2.0 * math.pi * rw * Tq * sw * lam * _finite_dirichlet_flux_shape(
        lam, rw, R
    )
    return _real_if_close(numerator / (p_value * denominator))


@lru_cache(maxsize=16)
def stehfest_coefficients(n_terms: int) -> np.ndarray:
    """Return Stehfest coefficients for an even number of terms."""

    if n_terms <= 0 or n_terms % 2 != 0:
        raise ValueError("n_terms must be a positive even integer")

    coeffs = np.zeros(n_terms, dtype=float)
    half = n_terms // 2
    for i in range(1, n_terms + 1):
        k_min = math.floor((i + 1) / 2)
        k_max = min(i, half)
        total = 0.0
        for k in range(k_min, k_max + 1):
            numerator = (k**half) * math.factorial(2 * k)
            denominator = (
                math.factorial(half - k)
                * math.factorial(k)
                * math.factorial(k - 1)
                * math.factorial(i - k)
                * math.factorial(2 * k - i)
            )
            total += numerator / denominator
        coeffs[i - 1] = ((-1) ** (half + i)) * total
    return coeffs


def inverse_laplace_stehfest(
    function_handle: Callable[..., float],
    time: float,
    n_terms: int = 12,
    **kwargs,
) -> float:
    """Numerically invert a Laplace-domain function with Stehfest weights."""

    if time <= 0.0:
        return float("nan")

    coeffs = stehfest_coefficients(n_terms)
    ln2_over_t = math.log(2.0) / time
    total = 0.0

    for idx in range(1, n_terms + 1):
        p_value = idx * ln2_over_t
        function_value = function_handle(p_value, **kwargs)
        if not np.isfinite(function_value):
            return float("nan")
        total += coeffs[idx - 1] * function_value

    return float(ln2_over_t * total)


def _mp_tq(p_value, T: float, tau_q: float, tau_s: float):
    return mp.mpf(T) * (1 + p_value * mp.mpf(tau_s)) / (1 + p_value * mp.mpf(tau_q))


def _mp_lambda_classical(p_value, T: float, S: float):
    return mp.sqrt(p_value * mp.mpf(S) / mp.mpf(T))


def _mp_lambda_ldl(p_value, T: float, S: float, tau_q: float, tau_s: float):
    return mp.sqrt(
        p_value
        * mp.mpf(S)
        * (1 + p_value * mp.mpf(tau_q))
        / (mp.mpf(T) * (1 + p_value * mp.mpf(tau_s)))
    )


def _mp_finite_shape(lam, r: float, R: float):
    return mp.besselk(0, lam * r) * mp.besseli(0, lam * R) - mp.besseli(
        0, lam * r
    ) * mp.besselk(0, lam * R)


def _mp_finite_flux_shape(lam, rw: float, R: float):
    return mp.besselk(1, lam * rw) * mp.besseli(0, lam * R) + mp.besseli(
        1, lam * rw
    ) * mp.besselk(0, lam * R)


def _evaluate_laplace_mpmath(function_handle: Callable[..., float], p_value, kwargs):
    name = getattr(function_handle, "__name__", "")

    if name == "qbar_classical":
        lam = _mp_lambda_classical(p_value, kwargs["T"], kwargs["S"])
        return (
            2
            * mp.pi
            * kwargs["rw"]
            * kwargs["T"]
            * kwargs["sw"]
            * lam
            * mp.besselk(1, lam * kwargs["rw"])
            / (p_value * mp.besselk(0, lam * kwargs["rw"]))
        )

    if name == "qbar_ldl":
        lam = _mp_lambda_ldl(
            p_value, kwargs["T"], kwargs["S"], kwargs["tau_q"], kwargs["tau_s"]
        )
        return (
            2
            * mp.pi
            * kwargs["rw"]
            * _mp_tq(p_value, kwargs["T"], kwargs["tau_q"], kwargs["tau_s"])
            * kwargs["sw"]
            * lam
            * mp.besselk(1, lam * kwargs["rw"])
            / (p_value * mp.besselk(0, lam * kwargs["rw"]))
        )

    if name == "qbar_classical_finite":
        lam = _mp_lambda_classical(p_value, kwargs["T"], kwargs["S"])
        denominator = _mp_finite_shape(lam, kwargs["rw"], kwargs["R"])
        return (
            2
            * mp.pi
            * kwargs["rw"]
            * kwargs["T"]
            * kwargs["sw"]
            * lam
            * _mp_finite_flux_shape(lam, kwargs["rw"], kwargs["R"])
            / (p_value * denominator)
        )

    if name == "qbar_ldl_finite":
        lam = _mp_lambda_ldl(
            p_value, kwargs["T"], kwargs["S"], kwargs["tau_q"], kwargs["tau_s"]
        )
        denominator = _mp_finite_shape(lam, kwargs["rw"], kwargs["R"])
        return (
            2
            * mp.pi
            * kwargs["rw"]
            * _mp_tq(p_value, kwargs["T"], kwargs["tau_q"], kwargs["tau_s"])
            * kwargs["sw"]
            * lam
            * _mp_finite_flux_shape(lam, kwargs["rw"], kwargs["R"])
            / (p_value * denominator)
        )

    if name == "sbar_classical":
        lam = _mp_lambda_classical(p_value, kwargs["T"], kwargs["S"])
        return (
            (kwargs["sw"] / p_value)
            * mp.besselk(0, lam * kwargs["r"])
            / mp.besselk(0, lam * kwargs["rw"])
        )

    if name == "sbar_ldl":
        lam = _mp_lambda_ldl(
            p_value, kwargs["T"], kwargs["S"], kwargs["tau_q"], kwargs["tau_s"]
        )
        return (
            (kwargs["sw"] / p_value)
            * mp.besselk(0, lam * kwargs["r"])
            / mp.besselk(0, lam * kwargs["rw"])
        )

    if name == "sbar_classical_finite":
        lam = _mp_lambda_classical(p_value, kwargs["T"], kwargs["S"])
        return (
            (kwargs["sw"] / p_value)
            * _mp_finite_shape(lam, kwargs["r"], kwargs["R"])
            / _mp_finite_shape(lam, kwargs["rw"], kwargs["R"])
        )

    if name == "sbar_ldl_finite":
        lam = _mp_lambda_ldl(
            p_value, kwargs["T"], kwargs["S"], kwargs["tau_q"], kwargs["tau_s"]
        )
        return (
            (kwargs["sw"] / p_value)
            * _mp_finite_shape(lam, kwargs["r"], kwargs["R"])
            / _mp_finite_shape(lam, kwargs["rw"], kwargs["R"])
        )

    value = function_handle(complex(p_value), **kwargs)
    if isinstance(value, complex) or np.iscomplexobj(value):
        return mp.mpc(value.real, value.imag)
    return mp.mpf(value)


def inverse_laplace_mpmath(
    function_handle: Callable[..., float],
    time: float,
    method: str = "dehoog",
    degree: int | None = None,
    dps: int = 40,
    **kwargs,
) -> float:
    """Invert a Laplace-domain function with an mpmath method."""

    if time <= 0.0:
        return float("nan")
    if method not in {"dehoog", "talbot"}:
        raise ValueError("mpmath method must be 'dehoog' or 'talbot'")

    previous_dps = mp.mp.dps
    mp.mp.dps = dps
    try:
        options = {"method": method}
        if degree is not None:
            options["degree"] = degree

        def wrapped(p_value):
            return _evaluate_laplace_mpmath(function_handle, p_value, kwargs)

        inverted = mp.invertlaplace(wrapped, time, **options)
    finally:
        mp.mp.dps = previous_dps

    if abs(mp.im(inverted)) <= mp.mpf("1e-8") * max(mp.mpf("1"), abs(mp.re(inverted))):
        return float(mp.re(inverted))
    return float(mp.re(inverted))


def inverse_laplace(
    function_handle: Callable[..., float],
    time: float,
    method: str = "stehfest",
    n_terms: int = 12,
    **kwargs,
) -> float:
    """Invert a Laplace-domain function with the requested method."""

    if method == "stehfest":
        return inverse_laplace_stehfest(function_handle, time, n_terms=n_terms, **kwargs)
    if method in {"dehoog", "talbot"}:
        return inverse_laplace_mpmath(function_handle, time, method=method, **kwargs)
    raise ValueError("method must be 'stehfest', 'dehoog', or 'talbot'")


def _evaluate_series(
    function_handle: Callable[..., float],
    times: np.ndarray | list[float] | tuple[float, ...],
    n_terms: int = 12,
    method: str = "stehfest",
    **kwargs,
) -> np.ndarray:
    time_values = np.asarray(times, dtype=float)
    return np.asarray(
        [
            inverse_laplace(
                function_handle,
                float(time_value),
                method=method,
                n_terms=n_terms,
                **kwargs,
            )
            for time_value in time_values
        ],
        dtype=float,
    )


def drawdown_classical(
    times: np.ndarray | list[float] | tuple[float, ...],
    r: float,
    rw: float,
    T: float,
    S: float,
    sw: float,
    n_terms: int = 12,
    method: str = "stehfest",
) -> np.ndarray:
    return _evaluate_series(
        sbar_classical,
        times,
        n_terms=n_terms,
        method=method,
        r=r,
        rw=rw,
        T=T,
        S=S,
        sw=sw,
    )


def drawdown_ldl(
    times: np.ndarray | list[float] | tuple[float, ...],
    r: float,
    rw: float,
    T: float,
    S: float,
    tau_q: float,
    tau_s: float,
    sw: float,
    n_terms: int = 12,
    method: str = "stehfest",
) -> np.ndarray:
    return _evaluate_series(
        sbar_ldl,
        times,
        n_terms=n_terms,
        method=method,
        r=r,
        rw=rw,
        T=T,
        S=S,
        tau_q=tau_q,
        tau_s=tau_s,
        sw=sw,
    )


def flowrate_classical(
    times: np.ndarray | list[float] | tuple[float, ...],
    rw: float,
    T: float,
    S: float,
    sw: float,
    n_terms: int = 12,
    method: str = "stehfest",
) -> np.ndarray:
    return _evaluate_series(
        qbar_classical,
        times,
        n_terms=n_terms,
        method=method,
        rw=rw,
        T=T,
        S=S,
        sw=sw,
    )


def flowrate_ldl(
    times: np.ndarray | list[float] | tuple[float, ...],
    rw: float,
    T: float,
    S: float,
    tau_q: float,
    tau_s: float,
    sw: float,
    n_terms: int = 12,
    method: str = "stehfest",
) -> np.ndarray:
    return _evaluate_series(
        qbar_ldl,
        times,
        n_terms=n_terms,
        method=method,
        rw=rw,
        T=T,
        S=S,
        tau_q=tau_q,
        tau_s=tau_s,
        sw=sw,
    )


def drawdown_classical_finite(
    times: np.ndarray | list[float] | tuple[float, ...],
    r: float,
    rw: float,
    R: float,
    T: float,
    S: float,
    sw: float,
    n_terms: int = 12,
    method: str = "stehfest",
) -> np.ndarray:
    return _evaluate_series(
        sbar_classical_finite,
        times,
        n_terms=n_terms,
        method=method,
        r=r,
        rw=rw,
        R=R,
        T=T,
        S=S,
        sw=sw,
    )


def drawdown_ldl_finite(
    times: np.ndarray | list[float] | tuple[float, ...],
    r: float,
    rw: float,
    R: float,
    T: float,
    S: float,
    tau_q: float,
    tau_s: float,
    sw: float,
    n_terms: int = 12,
    method: str = "stehfest",
) -> np.ndarray:
    return _evaluate_series(
        sbar_ldl_finite,
        times,
        n_terms=n_terms,
        method=method,
        r=r,
        rw=rw,
        R=R,
        T=T,
        S=S,
        tau_q=tau_q,
        tau_s=tau_s,
        sw=sw,
    )


def flowrate_classical_finite(
    times: np.ndarray | list[float] | tuple[float, ...],
    rw: float,
    R: float,
    T: float,
    S: float,
    sw: float,
    n_terms: int = 12,
    method: str = "stehfest",
) -> np.ndarray:
    return _evaluate_series(
        qbar_classical_finite,
        times,
        n_terms=n_terms,
        method=method,
        rw=rw,
        R=R,
        T=T,
        S=S,
        sw=sw,
    )


def flowrate_ldl_finite(
    times: np.ndarray | list[float] | tuple[float, ...],
    rw: float,
    R: float,
    T: float,
    S: float,
    tau_q: float,
    tau_s: float,
    sw: float,
    n_terms: int = 12,
    method: str = "stehfest",
) -> np.ndarray:
    return _evaluate_series(
        qbar_ldl_finite,
        times,
        n_terms=n_terms,
        method=method,
        rw=rw,
        R=R,
        T=T,
        S=S,
        tau_q=tau_q,
        tau_s=tau_s,
        sw=sw,
    )


def classical_type_curve(
    time_d: np.ndarray | list[float] | tuple[float, ...],
    n_terms: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    """Return dimensionless infinite-domain classical CHT flowrate curve."""

    time_values = np.asarray(time_d, dtype=float)
    rw = 1.0
    T = 1.0
    S = 1.0
    sw = 1.0
    q_values = flowrate_classical(
        time_values, rw=rw, T=T, S=S, sw=sw, n_terms=n_terms
    )
    return time_values, q_values / (2.0 * math.pi * T * sw)


def ldl_type_curve(
    time_d: np.ndarray | list[float] | tuple[float, ...],
    tau_q_d: float,
    tau_s_d: float,
    n_terms: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    """Return dimensionless infinite-domain LDL CHT flowrate curve.

    The dimensionless setup uses rw = T = S = sw = 1, so tc = S*rw^2/T = 1.
    The returned flowrate is normalized by 2*pi*T*sw.
    """

    time_values = np.asarray(time_d, dtype=float)
    rw = 1.0
    T = 1.0
    S = 1.0
    sw = 1.0
    q_values = flowrate_ldl(
        time_values,
        rw=rw,
        T=T,
        S=S,
        tau_q=tau_q_d,
        tau_s=tau_s_d,
        sw=sw,
        n_terms=n_terms,
    )
    return time_values, q_values / (2.0 * math.pi * T * sw)
