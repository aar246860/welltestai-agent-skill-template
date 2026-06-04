from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


SUPPORTED_TEST_TYPES = ("constant_rate", "constant_head", "slug")
DEMO_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "demo_outputs" / "welltest_interpreter_demo"
DEMO_FIGURE_DIR = DEMO_OUTPUT_DIR / "figures"
DEMO_TABLE_DIR = DEMO_OUTPUT_DIR / "tables"


@dataclass
class RawWellTestCase:
    case_id: str
    test_type: str | None
    data: pd.DataFrame
    unit_config: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    source_path: Path | None = None


def ensure_demo_dirs() -> None:
    for path in (DEMO_OUTPUT_DIR, DEMO_FIGURE_DIR, DEMO_TABLE_DIR):
        path.mkdir(parents=True, exist_ok=True)


def demo_schema() -> dict[str, Any]:
    return {
        "method": "WellTest Interpreter Demo",
        "supported_test_types": list(SUPPORTED_TEST_TYPES),
        "field_data_used_for_training": False,
        "field_data_used_for_calibration": False,
        "result_keys": [
            "case_id",
            "test_type",
            "model_probabilities",
            "parameter_summary",
            "posterior_predictive_envelope",
            "response_fit",
            "qc_warnings",
            "recommended_next_actions",
        ],
    }
