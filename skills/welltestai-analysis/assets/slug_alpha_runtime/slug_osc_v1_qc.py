from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class QCFlag:
    code: str
    triggered: bool
    severity: str
    observed_value: str
    interpretation: str
    recommended_next_action: str


@dataclass(frozen=True)
class OscillationQCReport:
    flags: dict[str, QCFlag]
    metrics: dict[str, float]

    def triggered_flags(self) -> list[QCFlag]:
        return [flag for flag in self.flags.values() if flag.triggered]

    def to_rows(self) -> list[dict[str, Any]]:
        return [
            {
                "code": flag.code,
                "triggered": flag.triggered,
                "severity": flag.severity,
                "observed_value": flag.observed_value,
                "interpretation": flag.interpretation,
                "recommended_next_action": flag.recommended_next_action,
            }
            for flag in self.flags.values()
        ]


def _flag(
    code: str,
    triggered: bool,
    severity: str,
    observed: object,
    interpretation: str,
    action: str,
) -> QCFlag:
    return QCFlag(
        code=code,
        triggered=bool(triggered),
        severity=severity if triggered else "none",
        observed_value=str(observed),
        interpretation=interpretation,
        recommended_next_action=action,
    )


def _moving_average(values: np.ndarray, window: int = 5) -> np.ndarray:
    if values.size < window:
        return values.copy()
    kernel = np.ones(window, dtype=float) / float(window)
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _compress_signs(values: np.ndarray, tol: float) -> np.ndarray:
    signs: list[int] = []
    for value in values:
        if abs(float(value)) <= tol:
            continue
        sign = 1 if value > 0 else -1
        if not signs or signs[-1] != sign:
            signs.append(sign)
    return np.asarray(signs, dtype=int)


def _local_extrema(time: np.ndarray, values: np.ndarray, tol: float) -> tuple[np.ndarray, np.ndarray]:
    maxima: list[int] = []
    minima: list[int] = []
    for idx in range(1, values.size - 1):
        prev_v = values[idx - 1]
        this_v = values[idx]
        next_v = values[idx + 1]
        if this_v > prev_v + tol and this_v > next_v + tol:
            maxima.append(idx)
        if this_v < prev_v - tol and this_v < next_v - tol:
            minima.append(idx)
    return np.asarray(maxima, dtype=int), np.asarray(minima, dtype=int)


def detect_oscillation_qc(
    time: np.ndarray,
    normalized_head: np.ndarray,
    *,
    equilibrium: float = 0.0,
    min_points_for_oscillation: int = 12,
) -> OscillationQCReport:
    """Detect oscillation risks in slug/recovery data before model fitting."""

    time = np.asarray(time, dtype=float)
    head = np.asarray(normalized_head, dtype=float)
    finite = np.isfinite(time) & np.isfinite(head)
    time = time[finite]
    head = head[finite]
    if time.size:
        order = np.argsort(time)
        time = time[order]
        head = head[order]

    metrics: dict[str, float] = {
        "n_points": float(time.size),
        "time_span": float(np.nanmax(time) - np.nanmin(time)) if time.size else 0.0,
    }

    flags: dict[str, QCFlag] = {}
    sparse = time.size < min_points_for_oscillation
    flags["insufficient_sampling_for_oscillation"] = _flag(
        "insufficient_sampling_for_oscillation",
        sparse,
        "warning",
        time.size,
        "Sampling is too sparse to claim an oscillatory slug response.",
        "Collect higher-frequency early-time data or treat oscillation classification as inconclusive.",
    )
    if time.size < 4:
        for code in (
            "nonmonotonic_recovery",
            "repeated_sign_change",
            "equilibrium_crossing",
            "damped_oscillation_candidate",
            "sensor_or_airline_oscillation_candidate",
            "uncertain_equilibrium_level",
            "out_of_monotonic_slug_model",
        ):
            flags[code] = _flag(code, False, "none", "insufficient data", "", "")
        metrics.update({"slope_sign_changes": 0.0, "equilibrium_crossings": 0.0, "n_extrema": 0.0})
        return OscillationQCReport(flags=flags, metrics=metrics)

    centered = head - float(equilibrium)
    smoothed = _moving_average(centered, window=5)
    response_range = float(np.nanmax(smoothed) - np.nanmin(smoothed))
    tol = max(0.005, 0.02 * response_range)

    slope = np.diff(smoothed) / np.maximum(np.diff(time), 1.0e-12)
    slope_signs = _compress_signs(slope, tol=max(tol / max(metrics["time_span"], 1.0e-12), 1.0e-6))
    slope_sign_changes = max(int(slope_signs.size) - 1, 0)
    value_signs = _compress_signs(smoothed, tol=tol)
    equilibrium_crossings = max(int(value_signs.size) - 1, 0)
    maxima, minima = _local_extrema(time, smoothed, tol=tol)
    extrema = np.sort(np.concatenate([maxima, minima]))

    metrics.update(
        {
            "response_range": response_range,
            "tolerance": tol,
            "slope_sign_changes": float(slope_sign_changes),
            "equilibrium_crossings": float(equilibrium_crossings),
            "n_extrema": float(extrema.size),
        }
    )

    nonmonotonic = slope_sign_changes >= 1
    repeated_sign = equilibrium_crossings >= 2
    equilibrium_crossing = equilibrium_crossings >= 1
    extrema_amplitudes = np.abs(smoothed[extrema]) if extrema.size else np.asarray([], dtype=float)
    if extrema_amplitudes.size >= 3:
        envelope_declines = bool(extrema_amplitudes[-1] < 0.8 * extrema_amplitudes[0])
    elif extrema_amplitudes.size >= 2:
        envelope_declines = bool(extrema_amplitudes[-1] < 0.95 * extrema_amplitudes[0])
    else:
        envelope_declines = False
    oscillatory_candidate = (
        (not sparse)
        and response_range > 5.0 * tol
        and (
            repeated_sign
            or (equilibrium_crossing and extrema.size >= 2)
            or (extrema.size >= 2 and envelope_declines)
        )
    )
    sensor_candidate = nonmonotonic and not oscillatory_candidate and response_range <= 6.0 * tol
    uncertain_equilibrium = equilibrium_crossing and not repeated_sign and extrema.size < 2
    out_of_monotonic = oscillatory_candidate or repeated_sign

    flags["nonmonotonic_recovery"] = _flag(
        "nonmonotonic_recovery",
        nonmonotonic,
        "warning",
        slope_sign_changes,
        "Recovery contains local reversals and should not be assumed monotonic.",
        "Inspect baseline correction, sensor noise, and early-time sampling before monotonic fitting.",
    )
    flags["repeated_sign_change"] = _flag(
        "repeated_sign_change",
        repeated_sign,
        "warning",
        equilibrium_crossings,
        "The response crosses the equilibrium level multiple times.",
        "Evaluate an underdamped or inertial slug response branch.",
    )
    flags["equilibrium_crossing"] = _flag(
        "equilibrium_crossing",
        equilibrium_crossing,
        "warning",
        equilibrium_crossings,
        "The response crosses the assumed equilibrium level.",
        "Verify the equilibrium water level and normalization before parameter reporting.",
    )
    flags["damped_oscillation_candidate"] = _flag(
        "damped_oscillation_candidate",
        oscillatory_candidate,
        "warning",
        f"extrema={extrema.size}, crossings={equilibrium_crossings}",
        "The record is consistent with a damped oscillatory slug response.",
        "Fit an inertial/oscillatory branch and do not force a monotonic slug model.",
    )
    flags["sensor_or_airline_oscillation_candidate"] = _flag(
        "sensor_or_airline_oscillation_candidate",
        sensor_candidate,
        "info",
        f"range={response_range:.4g}, tol={tol:.4g}",
        "Small reversals may reflect measurement or air-line behavior rather than aquifer response.",
        "Check transducer sampling, air-line setup, and raw water-level preprocessing.",
    )
    flags["uncertain_equilibrium_level"] = _flag(
        "uncertain_equilibrium_level",
        uncertain_equilibrium,
        "warning",
        equilibrium_crossings,
        "The assumed equilibrium level may be uncertain.",
        "Re-check static water level and baseline correction.",
    )
    flags["out_of_monotonic_slug_model"] = _flag(
        "out_of_monotonic_slug_model",
        out_of_monotonic,
        "error",
        f"crossings={equilibrium_crossings}",
        "A monotonic slug recovery model is not an adequate primary model.",
        "Use oscillatory screening or report the case as out of the monotonic model family.",
    )
    return OscillationQCReport(flags=flags, metrics=metrics)
