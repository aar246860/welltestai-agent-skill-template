from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wellsurrogate_rc8_formal_forward import predict_joint_case
from wellsurrogate_rc8_likelihood import effective_sample_size, joint_gaussian_loglik_batch, normalized_weights
from wellsurrogate_rc8_mcmc_rejuvenation import rejuvenate_particles
from wellsurrogate_rc8_priors import jitter_particles, sample_prior


@dataclass
class AdaptiveSMCResult:
    samples: np.ndarray
    weights: np.ndarray
    loglik: np.ndarray
    predictions: np.ndarray
    ess: float
    log_evidence: float
    stage_summary: list[dict]
    rejuvenation_summary: list[dict]


def _observed(case) -> np.ndarray:
    return np.asarray([obs.observed_response for obs in case.observations], dtype=float)


def _sigma(case) -> np.ndarray:
    return np.asarray([np.full(obs.observed_response.size, obs.sigma, dtype=float) for obs in case.observations], dtype=float)


def _predict(case, mode: str, samples: np.ndarray, n_terms: int) -> np.ndarray:
    rows = []
    for theta in samples:
        try:
            pred = predict_joint_case(case, mode, theta, n_terms=n_terms)
        except Exception:
            pred = np.full((len(case.observations), case.observations[0].observed_response.size), np.nan)
        rows.append(pred if np.all(np.isfinite(pred)) else np.full_like(pred, np.nan))
    return np.asarray(rows, dtype=float)


def _tempered_ess(loglik: np.ndarray, previous_temp: float, candidate_temp: float) -> float:
    return effective_sample_size(normalized_weights((candidate_temp - previous_temp) * loglik))


def _next_temperature(loglik: np.ndarray, current: float, target_ess: float) -> float:
    if current >= 1.0:
        return 1.0
    if _tempered_ess(loglik, current, 1.0) >= target_ess:
        return 1.0
    lo, hi = current, 1.0
    for _ in range(20):
        mid = 0.5 * (lo + hi)
        if _tempered_ess(loglik, current, mid) < target_ess:
            hi = mid
        else:
            lo = mid
    return float(max(lo, min(hi, 1.0)))


def _log_evidence(loglik: np.ndarray) -> float:
    finite = loglik[np.isfinite(loglik)]
    if finite.size == 0:
        return float("-inf")
    m = float(np.max(finite))
    return float(m + np.log(np.mean(np.exp(finite - m))))


def run_adaptive_smc(
    case,
    *,
    mode: str,
    n_particles: int = 128,
    seed: int = 20260603,
    n_terms: int = 4,
    target_ess_fraction: float = 0.65,
    max_stages: int = 8,
    mcmc_steps: int = 1,
) -> AdaptiveSMCResult:
    rng = np.random.default_rng(seed)
    samples = sample_prior(int(n_particles), rng)
    obs = _observed(case)
    sig = _sigma(case)
    stage_summary: list[dict] = []
    rejuvenation_summary: list[dict] = []
    temperature = 0.0
    weights = np.full(int(n_particles), 1.0 / int(n_particles))
    predictions = _predict(case, mode, samples, n_terms)
    loglik = joint_gaussian_loglik_batch(obs, predictions, sig)
    target_ess = max(4.0, float(target_ess_fraction) * int(n_particles))

    for stage in range(1, int(max_stages) + 1):
        next_temp = _next_temperature(loglik, temperature, target_ess)
        weights = normalized_weights((next_temp - temperature) * loglik)
        ess = effective_sample_size(weights)
        stage_summary.append(
            {
                "stage": int(stage),
                "temperature": float(next_temp),
                "ess": float(ess),
                "finite_particle_fraction": float(np.mean(np.isfinite(loglik))),
                "max_loglik": float(np.nanmax(loglik)) if np.any(np.isfinite(loglik)) else float("-inf"),
            }
        )
        temperature = next_temp
        if temperature >= 1.0:
            break
        if ess < 0.85 * n_particles:
            idx = rng.choice(np.arange(int(n_particles)), size=int(n_particles), replace=True, p=weights)
            samples = jitter_particles(samples[idx], rng, scale=0.08)
            predictions = _predict(case, mode, samples, n_terms)
            loglik = joint_gaussian_loglik_batch(obs, predictions, sig)

            def score(theta: dict[str, float]) -> float:
                pred_case = predict_joint_case(case, mode, theta, n_terms=n_terms)
                return float(joint_gaussian_loglik_batch(obs, pred_case[None, :, :], sig)[0])

            samples, stats = rejuvenate_particles(samples, loglik, score, rng, proposal_scale=0.08, n_steps=mcmc_steps)
            predictions = _predict(case, mode, samples, n_terms)
            loglik = joint_gaussian_loglik_batch(obs, predictions, sig)
            rejuvenation_summary.append({"stage": int(stage), **stats})
    if temperature < 1.0:
        temperature = 1.0
        weights = normalized_weights(loglik)
        stage_summary.append(
            {
                "stage": int(len(stage_summary) + 1),
                "temperature": 1.0,
                "ess": float(effective_sample_size(weights)),
                "finite_particle_fraction": float(np.mean(np.isfinite(loglik))),
                "max_loglik": float(np.nanmax(loglik)) if np.any(np.isfinite(loglik)) else float("-inf"),
                "forced_final_temperature": True,
            }
        )
    weights = normalized_weights(loglik)
    ess = effective_sample_size(weights)
    return AdaptiveSMCResult(
        samples=samples,
        weights=weights,
        loglik=loglik,
        predictions=predictions,
        ess=ess,
        log_evidence=_log_evidence(loglik),
        stage_summary=stage_summary,
        rejuvenation_summary=rejuvenation_summary,
    )
