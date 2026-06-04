# WellTestAI Agent Skill Template

This repository provides a Codex-compatible Agent Skill for natural-language assisted well-test analysis using the local `welltestai_lite` package.

The skill is designed for consultant-facing pilot use. It helps an AI coding agent validate input files, run a local WellTestAI analysis, interpret QC warnings, and generate an HTML report without uploading field data to a cloud service.

## What This Repository Contains

- `skills/welltestai-analysis/SKILL.md`: the Agent Skill.
- `skills/welltestai-analysis/references/`: concise domain references loaded only when needed.
- `skills/welltestai-analysis/scripts/run_welltestai_analysis.py`: deterministic helper for validate/analyze/report runs.
- `examples/`: a minimal synthetic case template.

This repository includes a small calibrated demonstration model so the Skill can run immediately after dependencies are available. It does not include unpublished field datasets or manuscript files.

## Install the Skill

Copy the skill folder into your Codex skills directory:

```powershell
$src = "skills\welltestai-analysis"
$dst = "$env:USERPROFILE\.codex\skills\welltestai-analysis"
New-Item -ItemType Directory -Force -Path (Split-Path $dst) | Out-Null
Copy-Item -Recurse -Force $src $dst
```

Restart Codex after copying the skill.

## Use With WellTestAI Local Lite

The Skill includes a bundled WellTestAI Lite runtime and calibrated model. Point the agent to:

- a `case.yaml` file;
- an observations CSV file referenced by the case file;

Python dependencies are still required:

```powershell
python -m pip install numpy pandas scikit-learn joblib PyYAML matplotlib
```

## Current Model Scope

The bundled calibrated model currently supports constant-rate, constant-head, and finite-boundary screening workflows. Its active labels are:

- `boundary_no_flow`
- `boundary_recharge`
- `ldl_full`
- `skin`
- `wellbore_storage`

Slug/recovery interpretation is not yet included in this bundled model. The Skill will not silently classify slug data with the bundled model. A validated slug-capable model should be added in a future release.

Example prompt:

```text
Use the WellTestAI analysis skill to validate this case.yaml, run the local report,
explain the response-family candidate, QC warnings, and recommended next action.
```

Manual command:

```powershell
python skills\welltestai-analysis\scripts\run_welltestai_analysis.py `
  --case examples\constant_rate_minimal\case.yaml `
  --out outputs\constant_rate_report.html
```

## Claim Boundary

The skill supports screening and report generation. It should not be used to claim that a field dataset proves a unique aquifer mechanism. Near-well effects such as wellbore storage and skin are interpreted as effects, not standalone aquifer families.
