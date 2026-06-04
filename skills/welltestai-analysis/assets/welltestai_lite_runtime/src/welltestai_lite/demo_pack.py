from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .model_runtime import analyze_case, write_result_json
from .paths import MARKETING_OUTPUT, PACKAGE_ROOT, RC7_MODEL_PATH
from .report import write_html_report


CLAIM_BOUNDARY = (
    "Local Lite is a screening and communication aid. It reports calibrated model-family "
    "probabilities and QC warnings from the current RC7 runtime, but final design decisions "
    "still require formal inversion and professional hydrogeologic review."
)


DEMO_CASES = (
    ("constant_rate", "demo_constant_rate"),
    ("late_time_boundary", "demo_late_time_boundary"),
    ("finite_boundary", "demo_finite_no_flow_boundary"),
    ("replenishment_ambiguity", "demo_replenishment_ambiguity"),
)


def example_case_paths() -> list[Path]:
    paths = []
    for folder, _case_id in DEMO_CASES:
        paths.append(PACKAGE_ROOT / "examples" / folder / "case.yaml")
    return paths


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")
    return path


def _write_one_pager(results: list[dict[str, Any]], output_dir: Path) -> Path:
    rows = []
    for result in results:
        rows.append(
            "<tr>"
            f"<td>{result['case_id']}</td>"
            f"<td>{result['case'].get('test_type', 'unknown')}</td>"
            f"<td>{result['prediction']['top_family']}</td>"
            f"<td>{result['prediction']['top_probability']:.2f}</td>"
            f"<td>{result['prediction']['confidence_level']}</td>"
            "</tr>"
        )
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>WellTestAI Local Lite One-Pager</title>
<style>
body {{ font-family: Arial, Helvetica, sans-serif; color: #17212b; margin: 34px; max-width: 1040px; }}
h1 {{ font-size: 34px; margin: 0 0 6px; }}
h2 {{ font-size: 22px; margin-top: 26px; }}
p, li {{ font-size: 16px; line-height: 1.45; }}
.lead {{ font-size: 19px; color: #3a4652; }}
.grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin: 22px 0; }}
.card {{ border: 1px solid #d8dee6; border-radius: 8px; padding: 15px; background: #fbfcfd; }}
.number {{ color: #0072B2; font-size: 30px; font-weight: 700; }}
table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
th, td {{ border-bottom: 1px solid #e4e8ee; text-align: left; padding: 9px; }}
.note {{ border-left: 4px solid #D55E00; padding: 12px; background: #f8f8f8; }}
</style>
</head>
<body>
<h1>WellTestAI Local Lite</h1>
<p class="lead">A local screening package for well-test QC, response-family triage, and consultant-ready reporting.</p>
<div class="grid">
<div class="card"><div class="number">1</div><strong>Standardize input</strong><p>Read a simple case file and time-series CSV without exposing client data to a cloud service.</p></div>
<div class="card"><div class="number">2</div><strong>Screen response family</strong><p>Use the existing calibrated RC7 runtime to rank formal response families and ambiguity warnings.</p></div>
<div class="card"><div class="number">3</div><strong>Report next actions</strong><p>Generate a local HTML report with QC flags, model-family probabilities, and recommended follow-up.</p></div>
</div>
<h2>Demo cases</h2>
<table><thead><tr><th>Case</th><th>Input type</th><th>Top family</th><th>Probability</th><th>Confidence</th></tr></thead><tbody>
{''.join(rows)}
</tbody></table>
<h2>Pilot value for consultants</h2>
<ul>
<li>Rapid first-pass interpretation before full manual curve matching.</li>
<li>Consistent QC warnings across projects and engineers.</li>
<li>Explicit uncertainty language for ambiguous late-time replenishment, boundary, or leakage-like behavior.</li>
<li>Local execution keeps client data inside the consultant workspace during pilot testing.</li>
</ul>
<p class="note"><strong>Claim boundary.</strong> {CLAIM_BOUNDARY}</p>
</body>
</html>
"""
    return _write_text(output_dir / "one_pager" / "welltestai_local_lite_one_pager.html", html)


def _write_deck(results: list[dict[str, Any]], output_dir: Path) -> Path:
    report_links = []
    for result in results:
        report_links.append(
            f"<li><a href='../reports/{result['case_id']}_report.html'>{result['case_id']}</a>: "
            f"{result['prediction']['top_family']} ({result['prediction']['top_probability']:.2f})</li>"
        )
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>WellTestAI Consultant Pitch Deck</title>
<style>
body {{ margin: 0; font-family: Arial, Helvetica, sans-serif; background: #eef2f5; color: #15202b; }}
.slide {{ width: 1120px; min-height: 630px; margin: 28px auto; background: white; box-shadow: 0 10px 30px rgba(0,0,0,0.12); padding: 52px 60px; box-sizing: border-box; }}
h1 {{ font-size: 44px; margin: 0 0 18px; }}
h2 {{ font-size: 34px; margin: 0 0 22px; }}
p, li {{ font-size: 22px; line-height: 1.38; }}
.eyebrow {{ color: #0072B2; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; }}
.two {{ display: grid; grid-template-columns: 1fr 1fr; gap: 34px; align-items: start; }}
.card {{ border: 1px solid #dce2e8; border-radius: 10px; padding: 20px; background: #fbfcfd; }}
.metric {{ font-size: 52px; font-weight: 700; color: #D55E00; }}
table {{ width: 100%; border-collapse: collapse; margin-top: 18px; }}
th, td {{ text-align: left; border-bottom: 1px solid #e4e8ee; padding: 12px; font-size: 20px; }}
.footer {{ margin-top: 34px; color: #56616d; font-size: 16px; }}
</style>
</head>
<body>
<section class="slide">
<div class="eyebrow">Local pilot package</div>
<h1>WellTestAI Local Lite</h1>
<p>Automated well-test QC, response-family screening, and consultant-facing reporting using the current RC7 calibrated runtime.</p>
<div class="footer">{CLAIM_BOUNDARY}</div>
</section>
<section class="slide">
<h2>Consultant pain point</h2>
<div class="two">
<div class="card"><div class="metric">Hours</div><p>Manual curve matching and report drafting slow down first-pass interpretation.</p></div>
<div class="card"><div class="metric">Risk</div><p>Boundary, leakage-like, or delayed responses can be overinterpreted from a single curve.</p></div>
</div>
</section>
<section class="slide">
<h2>What the pilot does</h2>
<ol>
<li>Reads a case YAML and observation CSV locally.</li>
<li>Runs QC and calibrated response-family screening.</li>
<li>Creates an HTML report with probabilities, warnings, and next actions.</li>
</ol>
</section>
<section class="slide">
<h2>Demo outputs</h2>
<ul>{''.join(report_links)}</ul>
<p>Each demo includes a response curve, derivative diagnostic, calibrated model-family probability table, and claim boundary.</p>
</section>
<section class="slide">
<h2>Pilot request</h2>
<table><thead><tr><th>Need</th><th>Minimum input</th></tr></thead><tbody>
<tr><td>One pumping or constant-head test</td><td>Time, response, forcing, radius, units</td></tr>
<tr><td>Engineering review</td><td>Original interpretation and known site constraints</td></tr>
<tr><td>Validation feedback</td><td>Which warnings are useful, confusing, or missing</td></tr>
</tbody></table>
</section>
</body>
</html>
"""
    return _write_text(output_dir / "deck" / "welltestai_consultant_pitch_deck.html", html)


def _write_sales_materials(output_dir: Path) -> list[Path]:
    outreach = """# Pilot outreach email

Subject: Local screening tool for pumping-test QC and first-pass interpretation

Dear Dr. Ko and Engineer Huang,

We are preparing a local pilot package for well-test screening. The tool reads a case file and time-series CSV, runs basic QC, ranks response-family candidates using a calibrated RC7 runtime, and generates a consultant-facing HTML report with ambiguity warnings and suggested next actions.

The goal is not to replace your professional interpretation. The pilot is intended to reduce repetitive first-pass checks, make QC warnings consistent, and identify cases where leakage-like, boundary-like, or delayed-response behavior may need additional evidence.

For a first pilot, one de-identified pumping or constant-head test with units, well radius, observation radius, and pumping schedule is sufficient.
"""
    script = """# Discovery call script

1. Ask which well-test reports take the most manual time.
2. Confirm the minimum data they can share for a private local pilot.
3. Show the demo report and explain the claim boundary.
4. Ask whether QC warnings, model-family probabilities, or next-test recommendations are most useful.
5. Agree on one blind case and define what would count as a useful pilot result.
"""
    data_request = """# Pilot data request

Minimum private pilot package:

- CSV with time and response columns.
- Test type: constant-rate, constant-head, or slug/recovery.
- Units for time, response, and forcing.
- Pumping or maintained-head schedule.
- Pumping well radius and observation well radius.
- Existing consultant interpretation, if they are willing to provide it after the blind run.

Do not include client names, addresses, or coordinates unless sharing permission is explicit.
"""
    return [
        _write_text(output_dir / "sales" / "pilot_outreach_email.md", outreach),
        _write_text(output_dir / "sales" / "discovery_call_script.md", script),
        _write_text(output_dir / "sales" / "pilot_data_request.md", data_request),
    ]


def _write_storyboard(results: list[dict[str, Any]], output_dir: Path) -> Path:
    lines = [
        "# WellTestAI Local Lite demo storyboard",
        "",
        "1. Open the project folder and show that all analysis runs locally.",
        "2. Open `welltestai_lite/examples/constant_rate/case.yaml` and explain the required fields.",
        "3. Run `python -m welltestai_lite.cli validate <case.yaml>`.",
        "4. Run `python -m welltestai_lite.cli report <case.yaml> --out <report.html>`.",
        "5. Open the HTML report and point to the response curve and log-time derivative.",
        "6. Explain model-family probabilities as screening output, not final design proof.",
        "7. Show the QC and ambiguity warnings.",
        "8. Ask the consultant for one de-identified blind test to evaluate report usefulness.",
        "",
        "Demo report files:",
    ]
    for result in results:
        lines.append(f"- `reports/{result['case_id']}_report.html`")
    return _write_text(output_dir / "storyboard" / "welltestai_demo_storyboard.md", "\n".join(lines))


def _copy_dir(src: Path, dst: Path) -> None:
    if src.exists():
        shutil.copytree(src, dst, dirs_exist_ok=True)


def _build_release_bundle(output_dir: Path) -> Path:
    release_root = output_dir / "release"
    package_dir = release_root / "welltestai_local_lite_package"
    package_dir.mkdir(parents=True, exist_ok=True)
    _copy_dir(PACKAGE_ROOT / "src", package_dir / "src")
    _copy_dir(PACKAGE_ROOT / "examples", package_dir / "examples")
    shutil.copy2(PACKAGE_ROOT / "README.md", package_dir / "README.md")
    shutil.copy2(PACKAGE_ROOT / "pyproject.toml", package_dir / "pyproject.toml")
    model_dir = package_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(RC7_MODEL_PATH, model_dir / RC7_MODEL_PATH.name)
    _copy_dir(output_dir / "reports", package_dir / "demo_reports")
    _copy_dir(output_dir / "sales", package_dir / "sales")
    _copy_dir(output_dir / "one_pager", package_dir / "one_pager")
    _copy_dir(output_dir / "deck", package_dir / "deck")
    _copy_dir(output_dir / "storyboard", package_dir / "storyboard")
    _write_text(
        package_dir / "RUN_DEMO.ps1",
        """
$env:PYTHONPATH = "src"
$model = "models/formal_extension_rc7_calibrated_surrogate.joblib"
python -m welltestai_lite.cli validate examples/constant_rate/case.yaml
python -m welltestai_lite.cli report examples/constant_rate/case.yaml --model-path $model --out demo_constant_rate_report.html
""",
    )
    _write_text(
        package_dir / "CLAIM_BOUNDARY.md",
        f"""
# Claim Boundary

{CLAIM_BOUNDARY}

The release candidate includes only the calibrated RC7 screening artifact. It does not include the raw 129 MB training surrogate, old manuscript folders, submission packages, or client field data.
""",
    )
    _write_text(
        package_dir / "QUICK_START_ZH.md",
        """
# WellTestAI Local Lite 顧問試用快速開始

## 這包可以做什麼

這是本機執行的抽水試驗初判工具。它可以讀取一個 `case.yaml` 和一個觀測資料 CSV，做基本 QC、response-family screening、早期近井效應提示、有限邊界/補注 ambiguity 提醒，並產生 HTML 報告。

## 這包不能宣稱什麼

- 不能取代正式水文地質判釋。
- 不能證明 LDL、邊界、補注、skin 或 wellbore storage 是唯一機制。
- `wellbore_storage` 和 `skin` 是早期或近井效應提示，不是獨立含水層 family。
- 有限邊界結果目前只能當 screening hypothesis，仍需要 formal inversion、晚期資料、多口觀測井或場址證據確認。

## 第一次執行

在解壓縮後的資料夾執行：

```powershell
$env:PYTHONPATH = "src"
$model = "models/formal_extension_rc7_calibrated_surrogate.joblib"
python -m welltestai_lite.cli validate examples/constant_rate/case.yaml
python -m welltestai_lite.cli report examples/constant_rate/case.yaml --model-path $model --out demo_constant_rate_report.html
```

執行後用瀏覽器開啟 `demo_constant_rate_report.html`。

## 顧問提供資料時需要什麼

最低限度：

- time column
- response column，例如 drawdown 或 discharge
- time unit
- response unit
- test type，例如 constant-rate 或 constant-head
- pumping rate 或 maintained drawdown
- pumping well radius
- observation radius

請先移除案主名稱、地址、座標或其他敏感資訊。

## 建議試用方式

1. 顧問提供一組去識別化案例。
2. 先不要提供原始人工判釋。
3. 用 WellTestAI Local Lite 產生初判報告。
4. 再和顧問原本判釋比較。
5. 記錄哪些 QC warning、response-family candidate、effect flag、next action 對工程師有幫助。
""",
    )
    zip_base = release_root / "welltestai_local_lite_package"
    zip_path = Path(shutil.make_archive(str(zip_base), "zip", root_dir=release_root, base_dir="welltestai_local_lite_package"))
    final_named_zip = release_root / "WellTestAI_Local_Lite_Consultant_Pilot_20260604.zip"
    shutil.copy2(zip_path, final_named_zip)
    return zip_path


def build_marketing_pack(output_dir: str | Path = MARKETING_OUTPUT) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    reports: list[str] = []
    for case_path in example_case_paths():
        result = analyze_case(case_path)
        results.append(result)
        report_path = write_html_report(result, out / "reports" / f"{result['case_id']}_report.html")
        write_result_json(result, out / "reports" / f"{result['case_id']}_result.json")
        reports.append(str(report_path))
    one_pager = _write_one_pager(results, out)
    deck = _write_deck(results, out)
    sales = _write_sales_materials(out)
    storyboard = _write_storyboard(results, out)
    release_zip = _build_release_bundle(out)
    manifest = {
        "package": "WellTestAI Local Lite",
        "demo_case_count": len(results),
        "reports": reports,
        "one_pager": str(one_pager),
        "deck": str(deck),
        "sales_materials": [str(path) for path in sales],
        "storyboard": str(storyboard),
        "release_zip": str(release_zip),
        "field_data_used_for_training": False,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (out / "marketing_pack_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
