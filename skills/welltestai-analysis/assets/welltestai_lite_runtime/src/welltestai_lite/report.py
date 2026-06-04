from __future__ import annotations

import html
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CMC_VIK = ["#0072B2", "#D55E00", "#56B4E9", "#CC79A7", "#009E73", "#666666"]


def _plot_response(result: dict, output_dir: Path) -> Path:
    obs = pd.DataFrame(result["observations"])
    path = output_dir / f"{result['case_id']}_response.png"
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    axes[0].plot(obs["time"], obs["response"], color=CMC_VIK[0], lw=2.0)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("time")
    axes[0].set_ylabel("response")
    axes[0].set_title("Response curve")
    log_t = np.log10(np.maximum(obs["time"].to_numpy(float), 1.0e-12))
    log_y = np.log10(np.maximum(obs["response"].to_numpy(float), 1.0e-12))
    slope = np.gradient(log_y, log_t)
    axes[1].plot(obs["time"], slope, color=CMC_VIK[1], lw=2.0)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("d log(response) / d log(time)")
    axes[1].set_title("Log-time derivative")
    for ax in axes:
        ax.grid(True, color="#E8E8E8", lw=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=280)
    plt.close(fig)
    return path


def _probability_table(result: dict) -> str:
    rows = []
    for item in result["prediction"]["family_probabilities"]:
        role = "response family"
        if item["family"] in {"wellbore_storage", "skin"}:
            role = "early-time / near-well effect"
        rows.append(f"<tr><td>{html.escape(item['family'])}</td><td>{html.escape(role)}</td><td>{item['probability']:.3f}</td></tr>")
    return "<table><thead><tr><th>Screening label</th><th>Interpretation layer</th><th>Calibrated probability</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"


def _interpretation_layers(result: dict) -> str:
    prediction = result["prediction"]
    primary = prediction["primary_aquifer_candidate"]
    effect = prediction["top_effect_candidate"]
    return f"""
<h2>Response-family interpretation</h2>
<div class="read-grid">
<div class="card"><strong>Primary response candidate</strong><p>{html.escape(primary['display_name'])} ({primary['probability']:.2f}). This is the aquifer-test response family that should be checked by formal inversion.</p></div>
<div class="card"><strong>Early-time / near-well effect candidate</strong><p>{html.escape(effect['display_name'])} ({effect['probability']:.2f}). Wellbore storage is treated as an early-time effect, not a standalone aquifer family.</p></div>
<div class="card"><strong>Raw top screening label</strong><p>{html.escape(prediction['top_screening_label'])} ({prediction['top_screening_probability']:.2f}). This label is useful for triage, but it may describe an effect rather than a conceptual aquifer model.</p></div>
<div class="card"><strong>Finite-boundary interpretation</strong><p>Boundary labels are finite-domain hypotheses. A no-flow boundary tends to strengthen late-time drawdown, while a recharge or replenishment boundary tends to flatten late-time response.</p></div>
</div>
"""


def _warnings_table(title: str, warnings: list[dict]) -> str:
    rows = []
    for warning in warnings:
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(warning['warning']))}</td>"
            f"<td>{html.escape(str(warning['interpretation']))}</td>"
            f"<td>{html.escape(str(warning['recommended_action']))}</td>"
            "</tr>"
        )
    return f"<h2>{title}</h2><table><thead><tr><th>Warning</th><th>Interpretation</th><th>Recommended action</th></tr></thead><tbody>{''.join(rows)}</tbody></table>"


def _case_example(result: dict) -> str:
    case = result["case"]
    case_id = html.escape(str(result["case_id"]))
    data_file = html.escape(str(case.get("data_file", "observations.csv")))
    time_col = html.escape(str(case.get("time_column", "time")))
    response_col = html.escape(str(case.get("response_column", "response")))
    test_type = html.escape(str(case.get("test_type", "unknown")))
    time_unit = html.escape(str(case.get("time_unit", "")))
    response_unit = html.escape(str(case.get("response_unit", "")))
    radius = html.escape(str(case.get("radius_cm", "")))
    forcing = html.escape(str(case.get("forcing_value", "")))
    yaml_text = html.escape(
        "\n".join(
            [
                f"case_id: {case_id}",
                f"test_type: {test_type}",
                f"data_file: {data_file}",
                f"time_column: {time_col}",
                f"response_column: {response_col}",
                f"time_unit: {time_unit}",
                f"response_unit: {response_unit}",
                f"radius_cm: {radius}",
                f"forcing_value: {forcing}",
            ]
        )
    )
    return f"""
<h2>Case example</h2>
<p>This demo case is intentionally simple. A consultant only needs a small case file and a CSV time series.</p>
<div class="example-grid">
<div>
<h3>Case file</h3>
<pre>{yaml_text}</pre>
</div>
<div>
<h3>Observation CSV</h3>
<pre>{time_col},{response_col}
0.02,0.033
0.04,0.052
...
2560.0,2.635</pre>
</div>
</div>
"""


def _step_by_step(result: dict) -> str:
    case_path_hint = f"examples/{result['case'].get('test_type', 'constant_rate')}/case.yaml"
    return f"""
<h2>Step-by-step pilot run</h2>
<ol class="steps">
<li><strong>Prepare one case file and one CSV.</strong> Keep client names and coordinates out of the pilot file unless sharing permission is explicit.</li>
<li><strong>Validate the file format.</strong><pre>python -m welltestai_lite.cli validate {html.escape(case_path_hint)}</pre></li>
<li><strong>Generate a screening report.</strong><pre>python -m welltestai_lite.cli report {html.escape(case_path_hint)} --model-path models/formal_extension_rc7_calibrated_surrogate.joblib --out report.html</pre></li>
<li><strong>Review the output with engineering judgment.</strong> Treat the model-family probabilities as screening evidence, then decide whether formal inversion or extra observations are needed.</li>
</ol>
"""


def _read_result_guidance(result: dict) -> str:
    primary = result["prediction"]["primary_aquifer_candidate"]
    effect = result["prediction"]["top_effect_candidate"]
    top = html.escape(str(primary["display_name"]))
    prob = float(primary["probability"])
    confidence = html.escape(str(result["prediction"]["confidence_level"]))
    return f"""
<h2>How to read this result</h2>
<div class="read-grid">
<div class="card"><strong>1. Start with QC.</strong><p>If early or late time is missing, do not overinterpret skin, wellbore storage, leakage, or boundary behavior.</p></div>
<div class="card"><strong>2. Read the response shape.</strong><p>The left panel checks whether the response grows, flattens, or has obvious data problems. The right panel summarizes log-time slope.</p></div>
<div class="card"><strong>3. Separate response family from effects.</strong><p>This case ranks <strong>{top}</strong> as the primary response candidate with probability <strong>{prob:.2f}</strong>. The strongest near-well effect flag is <strong>{html.escape(effect['display_name'])}</strong>.</p></div>
<div class="card"><strong>4. Use warnings to plan the next step.</strong><p>Ambiguity warnings mean the available curve is not enough to identify a unique physical mechanism.</p></div>
</div>
"""


def _case_specific_note(result: dict) -> str:
    test_type = str(result["case"].get("test_type", ""))
    if test_type != "finite_boundary":
        return ""
    return """
<h2>Finite-boundary stress example</h2>
<div class="claim"><strong>Boundary claim boundary.</strong> This demo includes a late-time finite-boundary style response shape. It does not prove that the classifier can uniquely identify a finite boundary. Treat boundary labels as screening hypotheses and confirm them with formal finite-boundary inversion, late-time data, spatial observation wells, and site evidence.</div>
"""


def write_html_report(result: dict, output_path: str | Path) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig_path = _plot_response(result, out.parent)
    rel_fig = fig_path.name
    html_text = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>WellTestAI Local Lite Report - {html.escape(result['case_id'])}</title>
<style>
body {{ font-family: Arial, Helvetica, sans-serif; margin: 28px; color: #1f2933; }}
h1 {{ font-size: 28px; margin-bottom: 4px; }}
h2 {{ font-size: 20px; margin-top: 28px; border-bottom: 1px solid #d8dee9; padding-bottom: 6px; }}
h3 {{ font-size: 16px; margin-bottom: 8px; }}
.summary {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 18px 0; }}
.card {{ border: 1px solid #d8dee9; border-radius: 6px; padding: 12px; background: #fbfcfd; }}
.example-grid, .read-grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 14px; }}
.steps li {{ margin: 10px 0; }}
.value {{ font-size: 20px; font-weight: 700; color: #0072B2; }}
pre {{ white-space: pre-wrap; background: #17212b; color: #f5f7fa; padding: 12px; border-radius: 6px; font-size: 13px; overflow-x: auto; }}
table {{ border-collapse: collapse; width: 100%; margin-top: 10px; }}
th, td {{ text-align: left; border-bottom: 1px solid #e5e9f0; padding: 8px; font-size: 14px; }}
img {{ max-width: 100%; margin: 10px 0; }}
.claim {{ background: #f7f7f7; border-left: 4px solid #D55E00; padding: 12px; margin-top: 16px; }}
@media (max-width: 860px) {{ .summary, .example-grid, .read-grid {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<h1>WellTestAI Local Lite</h1>
<p>Engineering screening report for <strong>{html.escape(result['case_id'])}</strong>.</p>
<div class="summary">
<div class="card"><div>Response candidate</div><div class="value">{html.escape(result['prediction']['primary_aquifer_candidate']['family'])}</div></div>
<div class="card"><div>Probability</div><div class="value">{result['prediction']['top_probability']:.3f}</div></div>
<div class="card"><div>Effect flag</div><div class="value">{html.escape(result['prediction']['top_effect_candidate']['effect'])}</div></div>
<div class="card"><div>Observations</div><div class="value">{result['observation_count']}</div></div>
</div>
{_step_by_step(result)}
{_case_example(result)}
{_read_result_guidance(result)}
{_interpretation_layers(result)}
{_case_specific_note(result)}
<h2>Response diagnostics</h2>
<img src="{html.escape(rel_fig)}" alt="Response diagnostics">
<h2>Model-family probability</h2>
{_probability_table(result)}
{_warnings_table("Input QC", result["qc_warnings"])}
{_warnings_table("Model and ambiguity warnings", result["model_warnings"])}
<h2>Recommended next action</h2>
<p>{html.escape(result['recommended_next_action'])}</p>
<div class="claim"><strong>Claim boundary.</strong> {html.escape(result['claim_boundary'])} Field data are not used for training by this local runtime.</div>
</body>
</html>
"""
    out.write_text(html_text, encoding="utf-8")
    return out
