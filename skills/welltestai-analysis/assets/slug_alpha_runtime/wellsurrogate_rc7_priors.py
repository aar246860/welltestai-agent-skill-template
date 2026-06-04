from __future__ import annotations

import numpy as np


def sample_prior(n: int, rng: np.random.Generator) -> np.ndarray:
    out = np.empty(int(n), dtype=[("logT", "f8"), ("logS", "f8"), ("logD", "f8"), ("log_tau_q", "f8"), ("log_tau_h", "f8"), ("log_R_tau", "f8")])
    out["logT"] = rng.uniform(-1.0, 3.0, int(n))
    out["logS"] = rng.uniform(-5.0, -1.0, int(n))
    out["logD"] = out["logT"] - out["logS"]
    out["log_tau_h"] = rng.uniform(-3.5, 1.5, int(n))
    out["log_R_tau"] = rng.uniform(-2.0, 2.0, int(n))
    out["log_tau_q"] = out["log_tau_h"] + out["log_R_tau"]
    return out


def jitter_particles(samples: np.ndarray, rng: np.random.Generator, scale: float = 0.15) -> np.ndarray:
    out = samples.copy()
    for name, lo, hi in (("logT", -1.2, 3.2), ("logS", -5.2, -0.8), ("log_tau_h", -4.0, 2.0), ("log_R_tau", -2.5, 2.5)):
        out[name] = np.clip(out[name] + rng.normal(0.0, scale, out.shape[0]), lo, hi)
    out["logD"] = out["logT"] - out["logS"]
    out["log_tau_q"] = out["log_tau_h"] + out["log_R_tau"]
    return out

