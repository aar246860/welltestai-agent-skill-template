from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml


REQUIRED_CASE_FIELDS = ("case_id", "test_type", "data_file", "time_column", "response_column")
SUPPORTED_TEST_TYPES = {"constant_rate", "constant_head", "finite_boundary", "late_time_ambiguity", "slug"}


def load_case(path: str | Path) -> dict[str, Any]:
    case_path = Path(path)
    with case_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Case file did not parse to a mapping: {case_path}")
    data["_case_path"] = str(case_path)
    return data


def data_path_for_case(case: dict[str, Any], *, base_dir: str | Path | None = None) -> Path:
    base = Path(base_dir) if base_dir is not None else Path(case.get("_case_path", ".")).parent
    data_path = Path(str(case["data_file"]))
    return data_path if data_path.is_absolute() else base / data_path


def load_observations(case: dict[str, Any], *, base_dir: str | Path | None = None) -> pd.DataFrame:
    path = data_path_for_case(case, base_dir=base_dir)
    frame = pd.read_csv(path)
    time_col = str(case["time_column"])
    response_col = str(case["response_column"])
    if time_col not in frame.columns or response_col not in frame.columns:
        raise ValueError(f"Data file must contain columns `{time_col}` and `{response_col}`: {path}")
    out = frame[[time_col, response_col]].copy().rename(columns={time_col: "time", response_col: "response"})
    out["time"] = pd.to_numeric(out["time"], errors="coerce")
    out["response"] = pd.to_numeric(out["response"], errors="coerce")
    return out.dropna().reset_index(drop=True)


def validate_case(case: dict[str, Any], *, base_dir: str | Path | None = None) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_CASE_FIELDS:
        if not case.get(field):
            errors.append(f"Missing required field: {field}")
    if case.get("test_type") and case["test_type"] not in SUPPORTED_TEST_TYPES:
        errors.append(f"Unsupported test_type: {case['test_type']}")
    if "data_file" in case:
        path = data_path_for_case(case, base_dir=base_dir)
        if not path.exists():
            errors.append(f"Data file does not exist: {path}")
        else:
            try:
                obs = load_observations(case, base_dir=base_dir)
                if obs.empty:
                    errors.append("Data file contains no valid numeric observations.")
            except Exception as exc:
                errors.append(str(exc))
    return errors
