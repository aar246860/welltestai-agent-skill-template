from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from welltest_alpha_schema import RawAlphaCase, SUPPORTED_ALPHA_TEST_TYPES


@dataclass(frozen=True)
class ValidationItem:
    code: str
    severity: str
    observed_value: str
    interpretation: str
    recommended_next_action: str


@dataclass(frozen=True)
class ValidationReport:
    case_id: str
    items: tuple[ValidationItem, ...]

    @property
    def has_blocking_errors(self) -> bool:
        return any(item.severity == "error" for item in self.items)

    def to_rows(self) -> list[dict[str, str]]:
        return [item.__dict__.copy() for item in self.items]


def _item(code: str, severity: str, observed: object, interpretation: str, action: str) -> ValidationItem:
    return ValidationItem(
        code=code,
        severity=severity,
        observed_value=str(observed),
        interpretation=interpretation,
        recommended_next_action=action,
    )


def validate_case(case: RawAlphaCase) -> ValidationReport:
    items: list[ValidationItem] = []
    if case.test_type not in SUPPORTED_ALPHA_TEST_TYPES:
        items.append(_item("ambiguous_test_type", "error", case.test_type, "Unsupported or missing test type.", "Set test_type explicitly."))
    if "time" not in case.data.columns:
        items.append(_item("missing_time_column", "error", list(case.data.columns), "No time column was found.", "Provide a time column."))
    if case.metadata.time_unit == "":
        items.append(_item("missing_time_unit", "error", "", "Time unit is required for safe interpretation.", "Set time_unit to s, min, or h."))
    if case.metadata.rw_cm is None:
        items.append(_item("missing_geometry", "error", "rw_cm=None", "Well radius is required.", "Provide rw_cm."))
    if case.test_type == "constant_rate":
        if "drawdown" not in case.data.columns:
            items.append(_item("wrong_response_type", "error", list(case.data.columns), "Constant-rate tests require drawdown.", "Provide a drawdown column."))
        if case.metadata.radius_cm is None:
            items.append(_item("missing_geometry", "error", "radius_cm=None", "Observation radius is required for constant-rate tests.", "Provide radius_cm."))
        if case.metadata.q_ml_min is None:
            items.append(_item("missing_forcing", "warning", "q_ml_min=None", "Pumping rate is missing; a demo default would be unsafe.", "Provide q_ml_min."))
    if case.test_type == "constant_head":
        if "discharge" not in case.data.columns and "flowrate" not in case.data.columns:
            items.append(_item("wrong_response_type", "error", list(case.data.columns), "Constant-head tests require discharge or flowrate.", "Provide discharge column."))
        if case.metadata.maintained_drawdown_cm is None:
            items.append(_item("missing_forcing", "warning", "maintained_drawdown_cm=None", "Maintained drawdown is missing.", "Provide maintained_drawdown_cm."))
    if case.test_type == "slug":
        if "normalized_head" not in case.data.columns and "h_over_h0" not in case.data.columns:
            items.append(_item("wrong_response_type", "error", list(case.data.columns), "Slug tests require normalized_head or h_over_h0.", "Provide normalized_head."))
        if case.metadata.slug_time_scale_seconds is None:
            items.append(_item("missing_geometry", "error", "slug_time_scale_seconds=None", "Slug interpretation requires an explicit time scale.", "Provide slug_time_scale_seconds."))
        if case.metadata.log_alpha is None:
            items.append(_item("missing_geometry", "error", "log_alpha=None", "Slug geometry alpha is required.", "Provide log_alpha or justify its prior."))
        col = "normalized_head" if "normalized_head" in case.data.columns else "h_over_h0" if "h_over_h0" in case.data.columns else None
        if col is not None:
            values = case.data[col].to_numpy(float)
            if np.any(values <= 0.0) or np.any(values >= 1.0):
                items.append(_item("suspicious_slug_normalization", "warning", (float(np.nanmin(values)), float(np.nanmax(values))), "Normalized slug heads should usually lie between 0 and 1.", "Check H/H0 normalization and baseline."))
            if np.any(np.diff(values) > 0.05):
                items.append(_item("nonmonotonic_slug_recovery", "warning", "positive local increase", "Slug recovery is not monotonically declining.", "Check noise, baseline, or use a segment with reliable recovery."))
    if "time" in case.data.columns:
        t = case.data["time"].to_numpy(float)
        if np.any(~np.isfinite(t)) or np.any(t <= 0.0):
            items.append(_item("negative_or_zero_time", "error", "time<=0", "Time values must be finite and positive.", "Remove nonpositive time values or shift to elapsed time."))
        if len(t) < 5:
            items.append(_item("sparse_sampling", "warning", len(t), "Fewer than five observations limits uncertainty estimates.", "Provide more time points if available."))
    return ValidationReport(case_id=case.case_id, items=tuple(items))
