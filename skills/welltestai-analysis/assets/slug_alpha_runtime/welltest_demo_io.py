from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from welltest_demo_schema import RawWellTestCase


def _looks_like_numeric_header(columns) -> bool:
    for col in columns:
        try:
            float(str(col).strip())
            return True
        except ValueError:
            continue
    return False


def _standardize_columns(df: pd.DataFrame, test_type: str | None) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "time" in df.columns:
        return df
    response = "value"
    if test_type == "constant_rate":
        response = "drawdown"
    elif test_type == "constant_head":
        response = "discharge"
    elif test_type == "slug":
        response = "normalized_head"
    if df.shape[1] >= 2:
        df = df.iloc[:, :2]
        df.columns = ["time", response]
    return df


def load_welltest_case(
    path: str | Path,
    *,
    test_type: str | None = None,
    unit_config: dict[str, str] | None = None,
    case_id: str | None = None,
) -> RawWellTestCase:
    path = Path(path)
    df = pd.read_csv(path)
    if _looks_like_numeric_header(df.columns) or "time" not in [str(c).strip().lower() for c in df.columns]:
        df = pd.read_csv(path, header=None)
    df = _standardize_columns(df, test_type)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    return RawWellTestCase(
        case_id=case_id or path.stem,
        test_type=test_type,
        data=df,
        unit_config=unit_config or {},
        source_path=path,
    )


def infer_test_type(case: RawWellTestCase) -> str:
    if case.test_type:
        return case.test_type
    cols = set(case.data.columns)
    if "discharge" in cols or "flowrate" in cols:
        return "constant_head"
    if "drawdown" in cols:
        return "constant_rate"
    if "normalized_head" in cols or "h_over_h0" in cols:
        return "slug"
    raise ValueError("Cannot infer test type from columns.")


def write_example_schema_files(output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    examples = {
        "example_constant_rate_schema.csv": pd.DataFrame({"time": [0.1, 1.0], "drawdown": [0.2, 1.1], "well_id": ["obs1", "obs1"], "radius": [100.0, 100.0]}),
        "example_constant_head_schema.csv": pd.DataFrame({"time": [0.1, 1.0], "discharge": [1000.0, 700.0], "drawdown": [150.0, 150.0]}),
        "example_slug_schema.csv": pd.DataFrame({"time": [1.0, 10.0], "normalized_head": [0.9, 0.4], "well_id": ["well1", "well1"]}),
    }
    paths = []
    for name, df in examples.items():
        path = output_dir / name
        df.to_csv(path, index=False)
        paths.append(path)
    return paths
