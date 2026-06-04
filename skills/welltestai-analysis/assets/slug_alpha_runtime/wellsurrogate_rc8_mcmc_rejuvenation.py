from __future__ import annotations

from typing import Callable

import numpy as np


PARAMS = ("logT", "logS", "log_tau_h", "log_R_tau")


def _as_dict(row: np.void) -> dict[str, float]:
    return {name: float(row[name]) for name in row.dtype.names}


def _update_derived(samples: np.ndarray) -> np.ndarray:
    samples["logD"] = samples["logT"] - samples["logS"]
    samples["log_tau_q"] = samples["log_tau_h"] + samples["log_R_tau"]
    return samples


def _proposal(row: np.void, rng: np.random.Generator, proposal_scale: float) -> dict[str, float]:
    theta = _as_dict(row)
    for name in PARAMS:
        theta[name] += float(rng.normal(0.0, proposal_scale))
    theta["logT"] = float(np.clip(theta["logT"], -1.2, 3.2))
    theta["logS"] = float(np.clip(theta["logS"], -5.2, -0.8))
    theta["log_tau_h"] = float(np.clip(theta["log_tau_h"], -4.0, 2.0))
    theta["log_R_tau"] = float(np.clip(theta["log_R_tau"], -2.5, 2.5))
    theta["logD"] = theta["logT"] - theta["logS"]
    theta["log_tau_q"] = theta["log_tau_h"] + theta["log_R_tau"]
    return theta


def rejuvenate_particles(
    samples: np.ndarray,
    loglik: np.ndarray,
    score_fn: Callable[[dict[str, float]], float],
    rng: np.random.Generator,
    *,
    proposal_scale: float = 0.12,
    n_steps: int = 2,
) -> tuple[np.ndarray, dict[str, float]]:
    moved = samples.copy()
    current = np.asarray(loglik, dtype=float).copy()
    accepted = 0
    attempted = 0
    for _ in range(int(n_steps)):
        for idx in range(moved.shape[0]):
            proposal = _proposal(moved[idx], rng, proposal_scale)
            prop_score = float(score_fn(proposal))
            cur_score = float(current[idx])
            if not np.isfinite(cur_score) or np.log(rng.uniform()) < prop_score - cur_score:
                for name in moved.dtype.names:
                    moved[name][idx] = proposal[name]
                current[idx] = prop_score
                accepted += 1
            attempted += 1
    _update_derived(moved)
    return moved, {
        "acceptance_rate": float(accepted / max(attempted, 1)),
        "attempted_moves": int(attempted),
        "accepted_moves": int(accepted),
        "proposal_scale": float(proposal_scale),
    }
