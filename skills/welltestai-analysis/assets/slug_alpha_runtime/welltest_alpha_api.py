from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from welltest_alpha_schema import ALPHA_OUTPUT_DIR, RawAlphaCase, ensure_alpha_dirs
from welltest_alpha_validation import ValidationReport, validate_case
from welltest_demo_api import build_observation_case, run_interpretation
from welltest_demo_plotting import plot_model_probabilities, plot_parameter_intervals, plot_qc_warnings, plot_response_fit
from welltest_demo_schema import RawWellTestCase


def _to_demo_case(case: RawAlphaCase) -> RawWellTestCase:
    unit_config = case.metadata.unit_config()
    if case.test_type == "constant_rate":
        unit_config["drawdown"] = case.metadata.response_unit
    elif case.test_type == "constant_head":
        unit_config["discharge"] = case.metadata.response_unit
    elif case.test_type == "slug":
        unit_config["normalized_head"] = case.metadata.response_unit
    return RawWellTestCase(
        case_id=case.case_id,
        test_type=case.test_type,
        data=case.data.copy(),
        unit_config=unit_config,
        metadata=case.metadata.notes.copy(),
        source_path=Path(case.metadata.source_path) if case.metadata.source_path else None,
    )


def _validation_df(report: ValidationReport) -> pd.DataFrame:
    return pd.DataFrame(report.to_rows(), columns=["code", "severity", "observed_value", "interpretation", "recommended_next_action"])


def fit_case(
    case: RawAlphaCase,
    *,
    n_particles: int = 24,
    sigma: float = 0.10,
    max_points: int = 8,
    modes: list[str] | None = None,
) -> dict[str, Any]:
    validation = validate_case(case)
    if validation.has_blocking_errors:
        raise ValueError(f"Blocking validation errors for {case.case_id}: {[item.code for item in validation.items if item.severity == 'error']}")
    obs = build_observation_case(
        _to_demo_case(case),
        geometry=case.metadata.geometry_dict(),
        forcing=case.metadata.forcing_dict(),
        max_points=max_points,
        sigma=sigma,
    )
    result = run_interpretation(obs, modes=modes or ["classical", "ldl_full", "head_lag", "equal_lag"], n_particles=n_particles, demo_mode=True)
    result["validation"] = validation
    validation_rows = _validation_df(validation)
    model_warnings = result["qc_warnings"].copy()
    max_model_probability = float(result["model_probabilities"]["probability"].max()) if not result["model_probabilities"].empty else 0.0
    if max_model_probability < 0.75:
        model_warnings = pd.concat(
            [
                model_warnings,
                pd.DataFrame(
                    [
                        {
                            "warning": "model_family_ambiguity",
                            "interpretation": "No single formal model family dominates the posterior evidence.",
                            "recommended_next_action": "Report model probabilities and inspect residuals instead of using only the best mode.",
                            "source": "posterior",
                        }
                    ]
                ),
            ],
            ignore_index=True,
            sort=False,
        )
    if not validation_rows.empty:
        validation_rows = validation_rows.rename(columns={"code": "warning"})
        validation_rows["source"] = "input_validation"
        model_warnings["source"] = "posterior"
        result["qc_warnings"] = pd.concat([model_warnings, validation_rows[["warning", "interpretation", "recommended_next_action", "source"]]], ignore_index=True, sort=False)
    else:
        result["qc_warnings"] = model_warnings
    return result


def make_report(result: dict[str, Any], output_dir: str | Path = ALPHA_OUTPUT_DIR) -> dict[str, Any]:
    output_root = ensure_alpha_dirs(Path(output_dir))
    table_dir = output_root / "tables"
    figure_dir = output_root / "figures"
    report_dir = output_root / "reports"
    case_id = result["case_id"]
    tables = {
        "parameter_summary": table_dir / f"{case_id}_parameter_summary.csv",
        "model_probabilities": table_dir / f"{case_id}_model_probabilities.csv",
        "response_fit": table_dir / f"{case_id}_response_fit.csv",
        "qc_warnings": table_dir / f"{case_id}_qc_warnings.csv",
    }
    for key, path in tables.items():
        frame_key = "model_probabilities" if key == "model_probabilities" else key
        result[frame_key].to_csv(path, index=False)
    figures = {
        "response_fit": plot_response_fit(result, figure_dir),
        "parameter_intervals": plot_parameter_intervals(result, figure_dir),
        "model_probabilities": plot_model_probabilities(result, figure_dir),
        "qc_warnings": plot_qc_warnings(result, figure_dir),
    }
    summary_path = report_dir / f"{case_id}_summary.md"
    warnings = result["qc_warnings"]
    warning_text = "No QC warnings." if warnings.empty else "\n".join(f"- {row['warning']}: {row['recommended_next_action']}" for _, row in warnings.iterrows())
    summary_path.write_text(
        f"# {case_id}\n\n"
        f"Best mode: `{result['best_mode']}`\n\n"
        f"Field data used for training: `False`\n\n"
        f"## QC warnings\n\n{warning_text}\n",
        encoding="utf-8",
    )
    manifest_path = report_dir / f"{case_id}_manifest.json"
    manifest_path.write_text(json.dumps({"case_id": case_id, "best_mode": result["best_mode"], "field_data_used_for_training": False}, indent=2), encoding="utf-8")
    return {"tables": tables, "figures": figures, "summary": summary_path, "manifest": manifest_path}


def validate_and_fit(case: RawAlphaCase, output_dir: str | Path = ALPHA_OUTPUT_DIR, **kwargs) -> tuple[dict[str, Any], dict[str, Any]]:
    result = fit_case(case, **kwargs)
    report = make_report(result, output_dir)
    return result, report
