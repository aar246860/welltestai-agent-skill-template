from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from slug_osc_v1_schema import FIGURE_DIR, REPORT_DIR, SlugOscAnalysis, ensure_slug_osc_dirs


CMC_BLUE = "#0C7BDC"
CMC_ORANGE = "#D55E00"
CMC_GREEN = "#009E73"
CMC_PURPLE = "#785EF0"
BLACK = "#222222"


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=260)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return path.with_suffix(".png")


def plot_slug_osc_fit(analysis: SlugOscAnalysis, figure_dir: Path | None = None) -> Path:
    out = Path(figure_dir) if figure_dir is not None else FIGURE_DIR
    frame = analysis.response_fit.copy()
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 7.4), sharex=True, height_ratios=[2.2, 1.0])
    ax = axes[0]
    ax.scatter(frame["time"], frame["observed_h_over_h0"], color=BLACK, s=42, label="observed H/H0", zorder=4)
    ax.plot(frame["time"], frame["monotonic_median"], color=CMC_GREEN, lw=2.2, label="monotonic reference")
    ax.plot(frame["time"], frame["oscillatory_median"], color=CMC_BLUE, lw=2.2, label="oscillatory screening")
    ax.axhline(0.0, color="#555555", lw=1.1, ls="--", label="equilibrium")
    ax.set_xscale("log")
    ax.set_ylabel("normalized head, H/H0")
    ax.legend(loc="best", frameon=True, framealpha=0.94)
    ax.grid(True, which="both", color="#d9e0e6", lw=0.6, alpha=0.7)

    axes[1].scatter(frame["time"], frame["monotonic_residual"], color=CMC_GREEN, s=32, label="monotonic residual")
    axes[1].scatter(frame["time"], frame["oscillatory_residual"], color=CMC_BLUE, s=32, label="oscillatory residual")
    axes[1].axhline(0.0, color="#333333", lw=1.1)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("elapsed time")
    axes[1].set_ylabel("residual")
    axes[1].legend(loc="best", frameon=True, framealpha=0.94)
    axes[1].grid(True, which="both", color="#d9e0e6", lw=0.6, alpha=0.7)
    return _save(fig, out / f"{analysis.case_id}_slug_osc_fit")


def plot_model_scores(analysis: SlugOscAnalysis, figure_dir: Path | None = None) -> Path:
    out = Path(figure_dir) if figure_dir is not None else FIGURE_DIR
    scores = analysis.model_scores.sort_values("model_probability")
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    colors = [CMC_GREEN if "monotonic" in model else CMC_BLUE for model in scores["model"]]
    ax.barh(scores["model"], scores["model_probability"], color=colors)
    ax.set_xlim(0, 1)
    ax.set_xlabel("posterior-style model probability")
    ax.set_title("Slug model comparison")
    ax.grid(True, axis="x", color="#d9e0e6", lw=0.6)
    return _save(fig, out / f"{analysis.case_id}_slug_osc_model_scores")


def write_slug_osc_report(analysis: SlugOscAnalysis, report_dir: Path | None = None) -> Path:
    ensure_slug_osc_dirs()
    out = Path(report_dir) if report_dir is not None else REPORT_DIR
    out.mkdir(parents=True, exist_ok=True)
    fig_dir = out / "figures"
    fit_fig = plot_slug_osc_fit(analysis, fig_dir)
    score_fig = plot_model_scores(analysis, fig_dir)

    tables_dir = out / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    analysis.qc_rows.to_csv(tables_dir / f"{analysis.case_id}_qc.csv", index=False)
    analysis.model_scores.to_csv(tables_dir / f"{analysis.case_id}_model_scores.csv", index=False)
    analysis.parameter_summary.to_csv(tables_dir / f"{analysis.case_id}_parameters.csv", index=False)
    analysis.response_fit.to_csv(tables_dir / f"{analysis.case_id}_response_fit.csv", index=False)

    triggered = analysis.qc_rows[analysis.qc_rows["triggered"] == True] if not analysis.qc_rows.empty else pd.DataFrame()
    qc_html = "<p>No triggered QC warnings.</p>" if triggered.empty else triggered.to_html(index=False, escape=True)
    params_html = "<p>No parameter intervals available.</p>" if analysis.parameter_summary.empty else analysis.parameter_summary.to_html(index=False, escape=True)
    scores_html = analysis.model_scores.to_html(index=False, escape=True)

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Slug Oscillation V1 Report - {analysis.case_id}</title>
  <style>
    body {{ font-family: Arial, Helvetica, sans-serif; margin: 0; background: #f7f7f5; color: #222; }}
    main {{ max-width: 1080px; margin: 0 auto; padding: 28px 24px 56px; background: #fff; }}
    h1 {{ color: #17324d; font-size: 30px; }}
    h2 {{ color: #17324d; border-bottom: 2px solid #d9e2ec; padding-bottom: 6px; margin-top: 32px; }}
    img {{ max-width: 100%; border: 1px solid #dce2e8; display: block; margin-top: 10px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border: 1px solid #dfe4ea; padding: 7px 8px; text-align: left; }}
    th {{ background: #edf3f8; }}
    .note {{ background: #fff7e6; border-left: 4px solid #d99000; padding: 10px 12px; }}
  </style>
</head>
<body><main>
<h1>Slug Oscillation V1 Report</h1>
<p class="note">{analysis.interpretation}</p>
<p>Best current screening model: <strong>{analysis.best_model}</strong>. This report is a screening result, not proof of a unique aquifer mechanism.</p>
<h2>Raw Recovery and Candidate Fits</h2>
<img src="{fit_fig.relative_to(out).as_posix()}" alt="Slug oscillation fit" />
<h2>Model Comparison</h2>
<img src="{score_fig.relative_to(out).as_posix()}" alt="Model scores" />
{scores_html}
<h2>QC Warnings</h2>
{qc_html}
<h2>Parameter Intervals</h2>
{params_html}
<h2>Claim Boundary</h2>
<p>The oscillatory branch is an inertial well-response screening model. Do not report aquifer K from this branch unless a formal aquifer-impedance transformation is implemented and validated.</p>
</main></body></html>
"""
    path = out / f"{analysis.case_id}_slug_osc_report.html"
    path.write_text(html, encoding="utf-8")
    return path
