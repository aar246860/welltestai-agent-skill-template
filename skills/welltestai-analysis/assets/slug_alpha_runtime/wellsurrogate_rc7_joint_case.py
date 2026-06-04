from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wellsurrogate_rc7_schema import ACTIVE_MODES


@dataclass(frozen=True)
class ObservationBlock:
    case_id: str
    test_type: str
    response_type: str
    time: np.ndarray
    observed_response: np.ndarray
    true_response: np.ndarray
    sigma: float
    radius: float
    rw: float
    Q: float
    sw: float
    log_alpha: float
    ar_over_a: float


@dataclass(frozen=True)
class JointCase:
    case_id: str
    mode: str
    true: dict[str, float]
    observations: tuple[ObservationBlock, ...]


def _sample_true(rng: np.random.Generator) -> dict[str, float]:
    logT = float(rng.uniform(-1.0, 3.0))
    logS = float(rng.uniform(-5.0, -1.0))
    log_tau_h = float(rng.uniform(-3.0, 1.0))
    log_R_tau = float(rng.uniform(-1.5, 1.5))
    return {
        "logT": logT,
        "logS": logS,
        "logD": logT - logS,
        "log_tau_h": log_tau_h,
        "log_R_tau": log_R_tau,
        "log_tau_q": log_tau_h + log_R_tau,
    }


def _design_for(test_type: str, rng: np.random.Generator) -> dict[str, float | str]:
    if test_type == "cr":
        return {"response_type": "drawdown", "radius": float(10.0 ** rng.uniform(0.0, 1.2)), "rw": float(10.0 ** rng.uniform(-0.2, 0.3)), "Q": float(10.0 ** rng.uniform(1.0, 3.0)), "sw": 1.0, "log_alpha": -3.0, "ar_over_a": 0.85}
    if test_type == "ch":
        response_type = "discharge" if rng.uniform() < 0.55 else "drawdown"
        radius = 0.0 if response_type == "discharge" else float(10.0 ** rng.uniform(0.0, 1.2))
        return {"response_type": response_type, "radius": radius, "rw": float(10.0 ** rng.uniform(-0.2, 0.3)), "Q": 1.0, "sw": float(10.0 ** rng.uniform(0.0, 2.0)), "log_alpha": -3.0, "ar_over_a": 0.85}
    return {"response_type": "normalized_head", "radius": 1.0, "rw": 1.0, "Q": 1.0, "sw": 1.0, "log_alpha": float(rng.uniform(-5.0, -1.0)), "ar_over_a": float(rng.uniform(0.65, 1.0))}


def make_joint_case(
    *,
    case_id: str,
    mode: str,
    test_combo: tuple[str, ...],
    seed: int,
    n_time: int,
    noise_std: float = 0.03,
) -> JointCase:
    if mode not in ACTIVE_MODES:
        raise ValueError(f"Unsupported RC7 mode: {mode}")
    rng = np.random.default_rng(seed)
    true = _sample_true(rng)
    observations: list[ObservationBlock] = []
    for test_type in test_combo:
        design = _design_for(test_type, rng)
        if test_type == "slug":
            time = 10.0 ** np.linspace(-4.0, 4.0, n_time)
        else:
            time = 10.0 ** np.linspace(-3.0, 3.0, n_time)
        # Fill response later in synthetic_cases/formal_forward; zeros are acceptable for construction tests.
        zeros = np.zeros(n_time, dtype=np.float32)
        observations.append(
            ObservationBlock(
                case_id=case_id,
                test_type=test_type,
                response_type=str(design["response_type"]),
                time=time.astype(np.float32),
                observed_response=zeros.copy(),
                true_response=zeros.copy(),
                sigma=float(noise_std),
                radius=float(design["radius"]),
                rw=float(design["rw"]),
                Q=float(design["Q"]),
                sw=float(design["sw"]),
                log_alpha=float(design["log_alpha"]),
                ar_over_a=float(design["ar_over_a"]),
            )
        )
    return JointCase(case_id=case_id, mode=mode, true=true, observations=tuple(observations))


def replace_observation_responses(case: JointCase, true_responses: list[np.ndarray], observed_responses: list[np.ndarray]) -> JointCase:
    blocks = []
    for obs, true, observed in zip(case.observations, true_responses, observed_responses):
        blocks.append(
            ObservationBlock(
                case_id=obs.case_id,
                test_type=obs.test_type,
                response_type=obs.response_type,
                time=obs.time,
                observed_response=np.asarray(observed, dtype=np.float32),
                true_response=np.asarray(true, dtype=np.float32),
                sigma=obs.sigma,
                radius=obs.radius,
                rw=obs.rw,
                Q=obs.Q,
                sw=obs.sw,
                log_alpha=obs.log_alpha,
                ar_over_a=obs.ar_over_a,
            )
        )
    return JointCase(case_id=case.case_id, mode=case.mode, true=case.true, observations=tuple(blocks))

