from __future__ import annotations

import numpy as np
from scipy.special import kv


def _sqrt(value: complex) -> complex:
    return complex(np.sqrt(complex(value)))


def _k_ratio(z: complex) -> complex:
    zc = complex(z)
    if abs(zc) < 1.0e-14:
        zc = 1.0e-14 + 0.0j
    k0 = kv(0, zc)
    k1 = kv(1, zc)
    if abs(k0) < 1.0e-300:
        k0 = 1.0e-300 + 0.0j
    return complex(k1 / k0)


def classical_slug_admittance(p: complex, *, alpha: float) -> complex:
    """Classical confined slug-test aquifer admittance in dimensionless form."""
    pc = complex(p)
    z = _sqrt(pc)
    return complex(2.0 * float(alpha) * z * _k_ratio(z))


def ldl_slug_admittance(
    p: complex,
    *,
    alpha: float,
    theta_q: float,
    theta_h: float,
) -> complex:
    """LDL/DPL slug-test aquifer admittance.

    The expression follows the existing local DPL slug kernel and reduces to the
    classical admittance when theta_q = theta_h = 0.
    """
    tq = max(float(theta_q), 0.0)
    th = max(float(theta_h), 0.0)
    pc = complex(p)
    if tq <= 1.0e-14 and th <= 1.0e-14:
        return classical_slug_admittance(pc, alpha=alpha)
    numerator = 1.0 + pc * tq
    denominator = 1.0 + pc * th
    mu = _sqrt(pc * numerator / denominator)
    boundary_factor = _sqrt(pc * denominator / numerator)
    return complex(2.0 * float(alpha) * boundary_factor * _k_ratio(mu))
