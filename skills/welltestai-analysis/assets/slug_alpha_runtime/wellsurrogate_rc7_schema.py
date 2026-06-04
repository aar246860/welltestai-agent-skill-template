from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "analysis_outputs" / "wellsurrogate_rc7"
FIGURE_DIR = OUTPUT_DIR / "figures"
POSTERIOR_DIR = OUTPUT_DIR / "posteriors"
SMC_DIR = OUTPUT_DIR / "smc_runs"
CALIBRATION_DIR = OUTPUT_DIR / "calibration"
JOINT_DIR = OUTPUT_DIR / "joint_inversion"
CROSS_TEST_DIR = OUTPUT_DIR / "cross_test_transfer"
LOG_DIR = OUTPUT_DIR / "logs"

ACTIVE_MODES = ("classical", "ldl_full", "head_lag", "equal_lag")
EXCLUDED_FAMILIES = ("cr_ldl_flux_lag_only",)
PARAMETER_NAMES = ("logT", "logS", "logD", "log_tau_q", "log_tau_h", "log_R_tau")

MODE_TO_FAMILY = {
    "classical": {"cr": "cr_classical_theis", "ch": "ch_classical_jacob_lohman", "slug": "slug_classical"},
    "ldl_full": {"cr": "cr_ldl_full", "ch": "ch_ldl_full", "slug": "slug_dpl_full"},
    "head_lag": {"cr": "cr_ldl_head_lag_only", "ch": "ch_ldl_head_lag_only", "slug": "slug_dpl_head_lag_only"},
    "equal_lag": {"cr": "cr_ldl_equal_lag_collapse", "ch": "ch_ldl_equal_lag_collapse", "slug": "slug_dpl_equal_lag_collapse"},
}


def ensure_output_dirs(base_dir: Path | None = None) -> None:
    root = Path(base_dir) if base_dir is not None else OUTPUT_DIR
    for path in (root, root / "figures", root / "posteriors", root / "smc_runs", root / "calibration", root / "joint_inversion", root / "cross_test_transfer", root / "logs"):
        path.mkdir(parents=True, exist_ok=True)


def schema_dictionary() -> dict[str, Any]:
    return {
        "method": "WellSurrogate-RC7",
        "role": "full dimensional joint Bayesian inversion for CR, CH, and slug tests",
        "field_data_used_for_training": False,
        "full_dimensional_joint_inversion": True,
        "sampled_parameters": list(PARAMETER_NAMES),
        "active_modes": list(ACTIVE_MODES),
        "excluded_families": list(EXCLUDED_FAMILIES),
        "claim_boundary": [
            "RC7 samples logT, logS, and lag coordinates jointly.",
            "Slug remains partial-dimensional where full geometry is unavailable.",
            "Field data are not used for training or posterior calibration.",
        ],
    }


def write_schema_json(path: Path | None = None) -> Path:
    ensure_output_dirs(path.parent if path is not None else None)
    out = path or OUTPUT_DIR / "wellsurrogate_rc7_schema.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(schema_dictionary(), indent=2), encoding="utf-8")
    return out

