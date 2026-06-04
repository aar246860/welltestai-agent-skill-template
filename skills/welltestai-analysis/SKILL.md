---
name: welltestai-analysis
description: Use when a user wants an AI agent to validate, analyze, interpret, or report slug-test or recovery-test CSV/case.yaml data with the bundled local WellTestAI Slug Alpha runtime. The skill supports local-only consultant workflows, QC warnings, formal classical and lagging-coordinate slug response screening, posterior-style parameter intervals, and cautious report generation. This public release intentionally does not expose constant-rate pumping or constant-head-test workflows.
---

# WellTestAI Slug Analysis

Use this skill to help a user run local slug/recovery test screening from natural language.

## Core Rule

Treat output as an alpha screening and reporting aid. Do not state that a single slug curve proves a unique aquifer mechanism or final design parameter set. Report candidate response modes, parameter intervals, QC warnings, and recommended next actions.

## Current Bundled Scope

This release is for slug/recovery data only. It accepts normalized head recovery, usually `H/H0`, against elapsed time. Constant-rate pumping and constant-head workflows are intentionally not exposed in this Skill release.

Supported public modes are:

- classical slug/recovery reference mode;
- lagging-coordinate slug/recovery mode;
- head/storage-lag slug coordinate;
- equal-lag collapse check.
- oscillation QC and Inertial-LDL slug screening branch for damped oscillatory records.

## Expected Inputs

Ask for or locate:

- `case.yaml`;
- referenced observations CSV;
- elapsed time column;
- normalized head column;
- well radius `rw_cm`;
- slug time-scale metadata `slug_time_scale_seconds`;
- slug geometry coordinate `log_alpha`;
- desired output report path or folder.

If input schema is unclear, read `references/input_schema.md`.

## Workflow

1. Confirm that the user wants slug/recovery analysis.
2. Confirm field data remain local.
3. Validate the `case.yaml` and CSV.
4. Run the deterministic helper script. It will route oscillatory records to the Slug-Osc report.
5. Read the HTML report, tables, and QC warnings.
6. Explain candidate modes, intervals, warnings, and next recommended action.
7. Avoid overclaiming mechanism uniqueness.

Preferred helper:

```powershell
python skills/welltestai-analysis/scripts/run_slug_analysis.py `
  --case path\to\case.yaml `
  --out path\to\slug_report.html
```

Example:

```powershell
python skills/welltestai-analysis/scripts/run_slug_analysis.py `
  --case examples\slug_golden\case.yaml `
  --out outputs\slug_golden_report.html
```

## Interpretation Rules

- Slug results depend strongly on geometry metadata and early-time quality.
- Oscillatory records should not be forced into monotonic slug models.
- The oscillatory branch reports damping and frequency screening coordinates, plus weakly identifiable LDL response-time coordinates when supported. It does not report final aquifer K.
- Lagging parameters are effective response-time coordinates unless identifiability evidence is strong.
- Broad intervals or mode ambiguity should be reported as uncertainty, not forced into one answer.
- Nonmonotonic recovery, sparse sampling, missing time scale, or uncertain normalization should trigger QC warnings.
- A good slug screening result can guide whether to run formal inversion, collect more early-time data, or request better well-construction metadata.

Read `references/claim_boundary.md` before drafting external-facing conclusions.

## Output Style

Return a concise consultant-readable summary:

- input files checked;
- report path;
- primary candidate mode;
- key parameter intervals;
- important QC warnings;
- recommended next action.

Do not write software-engineering jargon into the user-facing interpretation.
