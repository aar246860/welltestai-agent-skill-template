from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from inertial_ldl_slug_v1_schema import InertialLDLSlugAnalysis


COLORS = {
    "observed": "#222222",
    "classical_monotonic": "#0072B2",
    "ldl_monotonic": "#009E73",
    "classical_inertial": "#D55E00",
    "ldl_inertial": "#CC79A7",
}


def _write_fit_figure(analysis: InertialLDLSlugAnalysis, fig_path: Path) -> None:
    fit = analysis.response_fit
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.0), sharex=True, constrained_layout=True)
    ax = axes[0]
    ax.scatter(fit["time"], fit["observed_h_over_h0"], s=20, color=COLORS["observed"], label="Observed")
    for col, label in [
        ("classical_monotonic", "Classical monotonic"),
        ("ldl_monotonic", "LDL monotonic"),
        ("classical_inertial", "Classical inertial"),
        ("ldl_inertial", "Inertial-LDL"),
    ]:
        ax.plot(fit["time"], fit[col], lw=2.0, color=COLORS[col], label=label)
    ax.axhline(0.0, color="#777777", lw=1.0)
    ax.set_ylabel("Normalized head, H/H0")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.25)
    axes[1].plot(fit["time"], fit["best_residual"], color="#555555", lw=1.8)
    axes[1].axhline(0.0, color="#777777", lw=1.0)
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Best-model residual")
    axes[1].grid(True, alpha=0.25)
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)


def _table_html(frame: pd.DataFrame, *, max_rows: int = 40) -> str:
    if frame.empty:
        return "<p>No rows.</p>"
    return frame.head(max_rows).to_html(index=False, classes="data-table", float_format=lambda v: f"{v:.4g}")


def write_inertial_ldl_slug_report(
    analysis: InertialLDLSlugAnalysis,
    out_html: str | Path,
) -> Path:
    out = Path(out_html)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig_dir = out.parent / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / f"{analysis.case_id}_inertial_ldl_slug_fit.png"
    _write_fit_figure(analysis, fig_path)
    rel = fig_path.relative_to(out.parent).as_posix()
    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Inertial-LDL Slug Report - {analysis.case_id}</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 32px; color: #1f2933; }}
h1, h2 {{ color: #102a43; }}
.panel {{ border: 1px solid #d9e2ec; border-radius: 8px; padding: 16px; margin: 16px 0; }}
.data-table {{ border-collapse: collapse; font-size: 13px; }}
.data-table th, .data-table td {{ border: 1px solid #bcccdc; padding: 6px 8px; }}
.warning {{ color: #9b2c2c; }}
img {{ max-width: 100%; }}
</style>
</head>
<body>
<h1>Inertial-LDL Slug Report</h1>
<div class="panel">
<p><strong>Case:</strong> {analysis.case_id}</p>
<p><strong>Best screened branch:</strong> {analysis.best_model}</p>
<p><strong>Interpretation:</strong> {analysis.interpretation}</p>
<p><strong>Claim boundary:</strong> This report separates wellbore inertia from LDL response-time coordinates. It does not by itself prove a unique aquifer mechanism or final hydraulic conductivity.</p>
</div>
<h2>Response Fit</h2>
<img src="{rel}" alt="Inertial LDL slug fit">
<h2>Model Scores</h2>
{_table_html(analysis.model_scores)}
<h2>Parameter Intervals</h2>
{_table_html(analysis.parameter_summary)}
<h2>QC Warnings</h2>
{_table_html(analysis.qc_rows)}
</body>
</html>
"""
    out.write_text(html, encoding="utf-8")
    return out
