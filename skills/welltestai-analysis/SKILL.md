---
name: welltestai-analysis
description: Use when a user wants an AI agent to validate, analyze, interpret, or report constant-rate pumping-test, constant-head-test, or finite-boundary screening CSV/case.yaml data with the bundled WellTestAI Lite runtime and calibrated model. The skill guides local-only consultant workflows, QC warnings, response-family screening, near-well effect flags, and cautious claim boundaries. The bundled model does not yet support slug/recovery interpretation unless the user provides a separately validated slug-capable model.
---

# WellTestAI Analysis

Use this skill to help a user run local WellTestAI Lite analysis from natural language.

## Core Rule

Treat WellTestAI output as a screening and reporting aid. Do not state that a field case proves a unique aquifer mechanism. Report model candidates, effect flags, uncertainty, residual behavior, QC warnings, and recommended next actions.

## Current Bundled Model Scope

The bundled calibrated model supports these screening labels:

- `boundary_no_flow`
- `boundary_recharge`
- `ldl_full`
- `skin`
- `wellbore_storage`

It does not yet contain a validated slug/recovery model family. If a user provides slug-test data, stop and explain that slug support requires a separately validated slug-capable model or the next training release.

## Expected Inputs

Ask for or locate:

- `case.yaml`
- referenced observations CSV
- bundled WellTestAI Lite runtime in this Skill, or a local `welltestai_lite` installation
- bundled calibrated model, or a user-provided local model path
- desired output report path

If input schema is unclear, read `references/input_schema.md`.

## Workflow

1. Validate the case file.
2. Confirm that field data remain local.
3. Run the local report command or helper script.
4. Read the result JSON or HTML summary.
5. Separate response-family candidates from effect flags.
6. Explain QC warnings and recommended next actions.
7. Give the user the report path and any blocked items.

Preferred helper:

```powershell
python skills/welltestai-analysis/scripts/run_welltestai_analysis.py `
  --case path\to\case.yaml `
  --out path\to\report.html
```

Use `--model path\to\model.joblib` only when overriding the bundled calibrated model.

Fallback direct commands when using an external installation:

```powershell
python -m welltestai_lite.cli validate path\to\case.yaml
python -m welltestai_lite.cli report path\to\case.yaml --model-path path\to\model.joblib --out path\to\report.html
```

## Interpretation Rules

- `wellbore_storage` and `skin` are near-well or early-time effects, not aquifer families.
- Boundary labels are screening candidates unless confirmed by independent geometry, late-time coverage, or formal inversion.
- LDL labels are effective response-time coordinates unless identifiability evidence is strong.
- Poor CR to CH or CH to CR transfer indicates transformation uncertainty or missing physics, not automatic model failure.
- If the report gives low confidence, recommend additional observations instead of forcing a single answer.

Read `references/claim_boundary.md` before drafting external-facing conclusions.

## Output Style

Return a concise consultant-readable summary:

- input files checked;
- report path;
- primary response candidate;
- near-well or boundary effect flags;
- important QC warnings;
- parameter interval or uncertainty statement if available;
- next recommended test or data requirement.

Do not write software-engineering jargon into the user-facing interpretation.
