from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


ALPHA_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "analysis_outputs" / "welltest_alpha"
ALPHA_FIGURE_DIR = ALPHA_OUTPUT_DIR / "figures"
ALPHA_TABLE_DIR = ALPHA_OUTPUT_DIR / "tables"
ALPHA_REPORT_DIR = ALPHA_OUTPUT_DIR / "reports"

SUPPORTED_ALPHA_TEST_TYPES = ("slug", "recovery")


@dataclass
class CaseMetadata:
    time_unit: str = ""
    response_unit: str = ""
    forcing_unit: str = ""
    rw_cm: float | None = None
    radius_cm: float | None = None
    q_ml_min: float | None = None
    maintained_drawdown_cm: float | None = None
    slug_time_scale_seconds: float | None = None
    log_alpha: float | None = None
    ar_over_a: float | None = None
    source_path: str | None = None
    notes: dict[str, Any] = field(default_factory=dict)

    def geometry_dict(self) -> dict[str, float]:
        values: dict[str, float] = {}
        if self.rw_cm is not None:
            values["rw_cm"] = float(self.rw_cm)
        if self.radius_cm is not None:
            values["radius_cm"] = float(self.radius_cm)
        if self.slug_time_scale_seconds is not None:
            values["slug_time_scale_seconds"] = float(self.slug_time_scale_seconds)
        if self.log_alpha is not None:
            values["log_alpha"] = float(self.log_alpha)
        if self.ar_over_a is not None:
            values["ar_over_a"] = float(self.ar_over_a)
        return values

    def forcing_dict(self) -> dict[str, float]:
        values: dict[str, float] = {}
        if self.q_ml_min is not None:
            values["Q_mL_min"] = float(self.q_ml_min)
        if self.maintained_drawdown_cm is not None:
            values["sw_cm"] = float(self.maintained_drawdown_cm)
        return values

    def unit_config(self) -> dict[str, str]:
        config: dict[str, str] = {}
        if self.time_unit:
            config["time"] = self.time_unit
        if self.response_unit:
            config["response"] = self.response_unit
        if self.forcing_unit:
            config["forcing"] = self.forcing_unit
        return config


@dataclass
class RawAlphaCase:
    case_id: str
    test_type: str
    data: pd.DataFrame
    metadata: CaseMetadata = field(default_factory=CaseMetadata)

    def copy_with_data(self, data: pd.DataFrame) -> "RawAlphaCase":
        return RawAlphaCase(case_id=self.case_id, test_type=self.test_type, data=data.copy(), metadata=self.metadata)


def ensure_alpha_dirs(base_dir: Path | None = None) -> Path:
    root = Path(base_dir) if base_dir is not None else ALPHA_OUTPUT_DIR
    for path in (root, root / "figures", root / "tables", root / "reports", root / "golden_cases"):
        path.mkdir(parents=True, exist_ok=True)
    return root
