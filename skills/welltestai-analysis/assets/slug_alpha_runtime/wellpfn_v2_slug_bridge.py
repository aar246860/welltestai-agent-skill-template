from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SLUG_PROJECT = PROJECT_ROOT.parent / "LDL_slug test"
SLUG_SRC = SLUG_PROJECT / "src"
if SLUG_SRC.exists() and str(SLUG_SRC) not in sys.path:
    sys.path.insert(0, str(SLUG_SRC))


def _load_models():
    try:
        from dpl_slug.models import classical_recovery_curve, dpl_recovery_curve
    except Exception as exc:  # pragma: no cover - exercised when sibling project is absent.
        raise ImportError(f"Unable to import LDL slug-test models from {SLUG_SRC}") from exc
    return classical_recovery_curve, dpl_recovery_curve


def classical_slug_response(times: np.ndarray, alpha: float, ar_over_a: float = 1.0) -> np.ndarray:
    classical_recovery_curve, _ = _load_models()
    values = classical_recovery_curve(np.asarray(times, dtype=float), alpha=float(alpha), ar_over_a=float(ar_over_a))
    return np.clip(np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)


def ldl_slug_response(
    times: np.ndarray,
    alpha: float,
    theta_q: float,
    theta_h: float,
    ar_over_a: float = 1.0,
) -> np.ndarray:
    _, dpl_recovery_curve = _load_models()
    values = dpl_recovery_curve(
        np.asarray(times, dtype=float),
        alpha=float(alpha),
        theta_q=float(theta_q),
        theta_h=float(theta_h),
        ar_over_a=float(ar_over_a),
    )
    return np.clip(np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)


def slug_equal_lag_error(times: np.ndarray, alpha: float, theta: float, ar_over_a: float = 1.0) -> float:
    classical = classical_slug_response(times, alpha=alpha, ar_over_a=ar_over_a)
    lagged = ldl_slug_response(times, alpha=alpha, theta_q=theta, theta_h=theta, ar_over_a=ar_over_a)
    return float(np.max(np.abs(classical - lagged)))

