"""Run a local WellTestAI slug/recovery alpha analysis from an Agent Skill."""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


def _as_float(value: Any, *, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    return float(value)


def _rel(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _copy_plot_with_slug_label(response_csv: Path) -> str:
    return (
        "<p class=\"note\">The response-fit figure uses the runtime's transformed-response "
        "coordinate for posterior scoring. The input slug data are still read as normalized "
        "head H/H0. Use the CSV tables for numeric review.</p>"
    )


def _data_table(frame: pd.DataFrame, max_rows: int = 10) -> str:
    if frame.empty:
        return "<p>No rows.</p>"
    display = frame.head(max_rows).copy()
    return display.to_html(index=False, escape=True, classes="data-table")


def _inverse_logit(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(values, -80.0, 80.0)))


def _write_raw_slug_figure(response_fit: pd.DataFrame, figure_dir: Path, case_id: str) -> Path:
    figure_dir.mkdir(parents=True, exist_ok=True)
    fit = response_fit.copy()
    fit = fit.sort_values("time")
    time = fit["time"].to_numpy(float)
    observed = _inverse_logit(fit["observed"].to_numpy(float))
    median = _inverse_logit(fit["median"].to_numpy(float))
    lower = _inverse_logit(fit["lower_90"].to_numpy(float))
    upper = _inverse_logit(fit["upper_90"].to_numpy(float))

    fig, axes = plt.subplots(2, 1, figsize=(8.8, 7.2), sharex=True, height_ratios=[2.2, 1.0])
    ax = axes[0]
    ax.fill_between(time, lower, upper, color="#0C7BDC", alpha=0.18, label="90% posterior envelope")
    ax.plot(time, median, color="#0C7BDC", lw=2.4, label="posterior median")
    ax.scatter(time, observed, color="#222222", s=42, label="observed H/H0", zorder=3)
    ax.set_xscale("log")
    ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel("normalized head, H/H0")
    ax.legend(loc="upper right", frameon=True, framealpha=0.92)
    ax.grid(True, which="both", color="#d8dde3", lw=0.6, alpha=0.7)

    residual = observed - median
    axes[1].scatter(time, residual, color="#D55E00", s=36)
    axes[1].axhline(0.0, color="#333333", lw=1.1)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("dimensionless or scaled elapsed time")
    axes[1].set_ylabel("residual")
    axes[1].grid(True, which="both", color="#d8dde3", lw=0.6, alpha=0.7)

    fig.tight_layout()
    png = figure_dir / f"{case_id}_raw_slug_recovery.png"
    pdf = figure_dir / f"{case_id}_raw_slug_recovery.pdf"
    fig.savefig(png, dpi=260)
    fig.savefig(pdf)
    plt.close(fig)
    return png


def _write_html_report(
    *,
    case_id: str,
    result: dict[str, Any],
    report_manifest: dict[str, Any],
    html_path: Path,
) -> None:
    tables = {key: Path(path) for key, path in report_manifest["tables"].items()}
    figures = {key: Path(path) for key, path in report_manifest["figures"].items()}

    model_prob = pd.read_csv(tables["model_probabilities"])
    params = pd.read_csv(tables["parameter_summary"])
    warnings = pd.read_csv(tables["qc_warnings"])
    response_fit = pd.read_csv(tables["response_fit"])
    raw_slug_figure = _write_raw_slug_figure(
        response_fit,
        html_path.parent / "figures",
        case_id,
    )

    out_root = html_path.parent.resolve()
    figure_blocks = []
    if raw_slug_figure.exists():
        figure_blocks.append(
            f"<section><h2>Slug Recovery on Original H/H0 Scale</h2>"
            f"<img src=\"{html.escape(_rel(raw_slug_figure, out_root))}\" alt=\"Slug recovery on H/H0 scale\" />"
            f"</section>"
        )
    ordered_figures = [
        ("response_fit", "Slug recovery fit and residuals"),
        ("parameter_intervals", "Parameter intervals"),
        ("model_probabilities", "Formal mode probabilities"),
        ("qc_warnings", "QC warning summary"),
    ]
    for key, title in ordered_figures:
        path = figures.get(key)
        if path and path.exists():
            figure_blocks.append(
                f"<section><h2>{html.escape(title)}</h2>"
                f"<img src=\"{html.escape(_rel(path, out_root))}\" alt=\"{html.escape(title)}\" />"
                f"</section>"
            )

    warning_text = (
        "<p>No QC warnings were generated.</p>"
        if warnings.empty
        else _data_table(warnings, max_rows=30)
    )

    best_mode = html.escape(str(result.get("best_mode", "")))
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(
        f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>WellTestAI Slug Alpha Report - {html.escape(case_id)}</title>
  <style>
    body {{
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      color: #222;
      background: #f7f7f5;
      line-height: 1.55;
    }}
    main {{
      max-width: 1080px;
      margin: 0 auto;
      padding: 28px 22px 54px;
      background: #fff;
    }}
    h1, h2, h3 {{ color: #17324d; }}
    h1 {{ font-size: 30px; margin-bottom: 4px; }}
    h2 {{ font-size: 22px; margin-top: 34px; border-bottom: 2px solid #d9e2ec; padding-bottom: 6px; }}
    .lede {{ font-size: 16px; color: #3f4d5a; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin: 20px 0; }}
    .card {{ border: 1px solid #d6dde4; border-radius: 6px; padding: 12px 14px; background: #fbfcfd; }}
    .label {{ color: #5b6773; font-size: 13px; }}
    .value {{ font-size: 20px; font-weight: 700; color: #17324d; }}
    img {{ max-width: 100%; height: auto; display: block; margin: 10px 0 4px; border: 1px solid #e1e5e8; }}
    .data-table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    .data-table th, .data-table td {{ border: 1px solid #dfe4ea; padding: 7px 8px; text-align: left; }}
    .data-table th {{ background: #edf3f8; }}
    .note {{ padding: 10px 12px; background: #fff7e6; border-left: 4px solid #d99000; }}
    code {{ background: #eef2f6; padding: 1px 4px; border-radius: 3px; }}
  </style>
</head>
<body>
<main>
  <h1>WellTestAI Slug Alpha Report</h1>
  <p class=\"lede\">Case <code>{html.escape(case_id)}</code>. This local report screens slug/recovery response against formal classical and lagging-coordinate modes. It is not a proof of a unique aquifer mechanism.</p>

  <div class=\"cards\">
    <div class=\"card\"><div class=\"label\">Best current mode</div><div class=\"value\">{best_mode}</div></div>
    <div class=\"card\"><div class=\"label\">Field data used for training</div><div class=\"value\">False</div></div>
    <div class=\"card\"><div class=\"label\">Report type</div><div class=\"value\">Slug alpha</div></div>
  </div>

  <h2>How to Read This Report</h2>
  <p>The input is a normalized slug recovery curve, usually H/H0 versus elapsed time. The tool validates units and key geometry metadata, estimates posterior-style parameter intervals, compares formal response modes, and reports QC warnings. Use the result as a screening layer before final engineering judgment.</p>
  {_copy_plot_with_slug_label(tables["response_fit"])}

  {''.join(figure_blocks)}

  <section>
    <h2>Model Probabilities</h2>
    {_data_table(model_prob)}
  </section>

  <section>
    <h2>Parameter Intervals</h2>
    {_data_table(params, max_rows=30)}
  </section>

  <section>
    <h2>QC Warnings and Next Actions</h2>
    {warning_text}
  </section>

  <section>
    <h2>Response Fit Table Preview</h2>
    {_data_table(response_fit)}
  </section>

  <section>
    <h2>Claim Boundary</h2>
    <p>This alpha release supports local screening of slug/recovery data. Lagging parameters are effective response-time coordinates unless additional identifiability evidence supports a stronger interpretation. Do not use this report alone to claim a unique aquifer mechanism or final design parameter set.</p>
  </section>
</main>
</body>
</html>
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and report a slug/recovery case.")
    parser.add_argument("--case", required=True, help="Path to slug case.yaml")
    parser.add_argument("--out", help="Output HTML report path")
    parser.add_argument("--output-dir", help="Output directory for tables, figures, and report")
    parser.add_argument("--particles", type=int, default=24, help="Number of posterior particles")
    parser.add_argument("--max-points", type=int, default=8, help="Maximum observations used by the alpha scorer")
    args = parser.parse_args()

    skill_root = Path(__file__).resolve().parents[1]
    runtime_dir = skill_root / "assets" / "slug_alpha_runtime"
    if not runtime_dir.exists():
        raise SystemExit(f"Missing bundled slug alpha runtime: {runtime_dir}")
    sys.path.insert(0, str(runtime_dir))

    from welltest_alpha_api import fit_case, make_report
    from welltest_alpha_schema import CaseMetadata, RawAlphaCase

    case_path = Path(args.case).expanduser().resolve()
    if not case_path.exists():
        raise SystemExit(f"Missing case file: {case_path}")
    with case_path.open("r", encoding="utf-8") as handle:
        case_data = yaml.safe_load(handle) or {}

    test_type = str(case_data.get("test_type", "")).strip().lower()
    if test_type not in {"slug", "recovery"}:
        raise SystemExit(
            "This public Skill release is slug/recovery only. "
            "Set test_type: slug or test_type: recovery. Constant-rate and "
            "constant-head workflows are intentionally not exposed in this release."
        )

    data_path = Path(str(case_data.get("data_file", "")))
    if not data_path.is_absolute():
        data_path = case_path.parent / data_path
    if not data_path.exists():
        raise SystemExit(f"Missing observations CSV: {data_path}")

    time_column = str(case_data.get("time_column", "time"))
    response_column = str(case_data.get("response_column", "normalized_head"))
    raw = pd.read_csv(data_path)
    missing_cols = [col for col in (time_column, response_column) if col not in raw.columns]
    if missing_cols:
        raise SystemExit(f"Missing required CSV columns {missing_cols}; found {list(raw.columns)}")
    data = raw[[time_column, response_column]].rename(
        columns={time_column: "time", response_column: "normalized_head"}
    )

    case_id = str(case_data.get("case_id") or case_path.stem)
    metadata = CaseMetadata(
        time_unit=str(case_data.get("time_unit", "")),
        response_unit=str(case_data.get("response_unit", "dimensionless")),
        rw_cm=_as_float(case_data.get("rw_cm", case_data.get("well_radius_cm"))),
        slug_time_scale_seconds=_as_float(case_data.get("slug_time_scale_seconds")),
        log_alpha=_as_float(case_data.get("log_alpha")),
        ar_over_a=_as_float(case_data.get("ar_over_a")),
        source_path=str(data_path),
        notes={
            "case_yaml": str(case_path),
            "alpha_release_scope": "slug/recovery only",
        },
    )
    case = RawAlphaCase(case_id=case_id, test_type="slug", data=data, metadata=metadata)

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None
    html_path = Path(args.out).expanduser().resolve() if args.out else None
    if output_dir is None:
        if html_path is not None:
            output_dir = html_path.parent / f"{case_id}_assets"
        else:
            output_dir = Path.cwd() / "outputs" / case_id
    if html_path is None:
        html_path = output_dir / "reports" / f"{case_id}_slug_report.html"

    result = fit_case(case, n_particles=args.particles, max_points=args.max_points)
    report_manifest = make_report(result, output_dir)
    _write_html_report(
        case_id=case_id,
        result=result,
        report_manifest=report_manifest,
        html_path=html_path,
    )

    manifest_out = html_path.with_suffix(".manifest.json")
    manifest_out.write_text(
        json.dumps(
            {
                "case_id": case_id,
                "html_report": str(html_path),
                "output_dir": str(output_dir),
                "best_mode": result.get("best_mode"),
                "field_data_used_for_training": False,
                "release_scope": "slug/recovery only",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"WellTestAI slug report written to: {html_path}")
    print(f"Report assets written to: {output_dir}")


if __name__ == "__main__":
    main()
