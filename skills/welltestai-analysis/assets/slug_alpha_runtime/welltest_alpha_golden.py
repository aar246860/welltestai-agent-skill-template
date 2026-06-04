from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from wellsurrogate_rc7_joint_case import JointCase, ObservationBlock
from wellsurrogate_rc8_formal_forward import predict_joint_case
from welltest_alpha_schema import CaseMetadata, RawAlphaCase


@dataclass(frozen=True)
class GoldenCase:
    raw_case: RawAlphaCase
    expected_best_mode: str
    truth: dict[str, float]
    source_model: str


def _case(case_id: str, mode: str, block: ObservationBlock, true: dict[str, float]) -> np.ndarray:
    joint = JointCase(case_id=case_id, mode=mode, true=true, observations=(block,))
    return predict_joint_case(joint, mode, true, n_terms=4)[0]


def _true() -> dict[str, float]:
    return {
        "logT": 1.0,
        "logS": -3.0,
        "logD": 4.0,
        "log_tau_h": -2.0,
        "log_R_tau": 1.0,
        "log_tau_q": -1.0,
    }


def _logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def make_golden_cases() -> dict[str, GoldenCase]:
    true = _true()
    times = np.asarray([0.002, 0.01, 0.05, 0.2, 1.0, 5.0, 20.0, 100.0], dtype=np.float32)
    cr_block = ObservationBlock(
        case_id="golden_constant_rate",
        test_type="cr",
        response_type="drawdown",
        time=times,
        observed_response=np.zeros_like(times),
        true_response=np.zeros_like(times),
        sigma=0.03,
        radius=87.0,
        rw=5.0,
        Q=599.0,
        sw=1.0,
        log_alpha=0.0,
        ar_over_a=0.85,
    )
    ch_block = ObservationBlock(
        case_id="golden_constant_head",
        test_type="ch",
        response_type="discharge",
        time=times,
        observed_response=np.zeros_like(times),
        true_response=np.zeros_like(times),
        sigma=0.03,
        radius=0.0,
        rw=5.0,
        Q=1.0,
        sw=150.0,
        log_alpha=0.0,
        ar_over_a=0.85,
    )
    slug_time = np.asarray([0.002, 0.017, 0.035, 0.059, 0.104, 0.204, 0.46, 2.52], dtype=np.float32)
    slug_block = ObservationBlock(
        case_id="golden_slug_classical",
        test_type="slug",
        response_type="normalized_head",
        time=slug_time,
        observed_response=np.zeros_like(slug_time),
        true_response=np.zeros_like(slug_time),
        sigma=0.03,
        radius=1.0,
        rw=5.0,
        Q=1.0,
        sw=1.0,
        log_alpha=0.0,
        ar_over_a=0.85,
    )
    cr = _case("golden_constant_rate", "classical", cr_block, true)
    ch = _case("golden_constant_head", "classical", ch_block, true)
    slug = _case("golden_slug_classical", "classical", slug_block, true)
    slug_h = _logistic(slug)
    # Enforce strict decline for the golden regression guard if numerical tails are flat at machine precision.
    slug_h = np.maximum.accumulate(slug_h[::-1])[::-1]
    slug_h = np.minimum(slug_h, np.linspace(0.98, 0.02, slug_h.size))

    return {
        "golden_constant_rate": GoldenCase(
            raw_case=RawAlphaCase(
                case_id="golden_constant_rate",
                test_type="constant_rate",
                data=pd.DataFrame({"time": times, "drawdown": 10.0 ** cr}),
                metadata=CaseMetadata(time_unit="min", response_unit="cm", forcing_unit="mL/min", rw_cm=5.0, radius_cm=87.0, q_ml_min=599.0),
            ),
            expected_best_mode="classical",
            truth=true,
            source_model="classical Theis through RC8 formal kernel",
        ),
        "golden_constant_head": GoldenCase(
            raw_case=RawAlphaCase(
                case_id="golden_constant_head",
                test_type="constant_head",
                data=pd.DataFrame({"time": times, "discharge": 10.0 ** ch}),
                metadata=CaseMetadata(time_unit="min", response_unit="mL/min", forcing_unit="cm", rw_cm=5.0, maintained_drawdown_cm=150.0),
            ),
            expected_best_mode="classical",
            truth=true,
            source_model="classical Jacob-Lohman through RC8 formal kernel",
        ),
        "golden_slug_classical": GoldenCase(
            raw_case=RawAlphaCase(
                case_id="golden_slug_classical",
                test_type="slug",
                data=pd.DataFrame({"time": slug_time, "normalized_head": slug_h}),
                metadata=CaseMetadata(time_unit="s", response_unit="dimensionless", rw_cm=5.0, slug_time_scale_seconds=1.0, log_alpha=0.0, ar_over_a=0.85),
            ),
            expected_best_mode="classical",
            truth=true,
            source_model="classical slug through RC8 formal kernel",
        ),
    }


def write_golden_cases(output_dir) -> list:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, golden in make_golden_cases().items():
        path = output_dir / f"{name}.csv"
        golden.raw_case.data.to_csv(path, index=False)
        paths.append(path)
    return paths
