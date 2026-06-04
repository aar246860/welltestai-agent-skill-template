# WellTestAI Slug Agent Skill Template

This repository provides a Codex-compatible Agent Skill for natural-language assisted slug-test and recovery-test screening using a bundled local WellTestAI Slug Alpha runtime.

The current public release is intentionally narrow: it supports slug/recovery analysis only. Constant-rate pumping and constant-head-test workflows are not exposed in this Skill.

## What This Repository Contains

- `skills/welltestai-analysis/SKILL.md`: the Agent Skill.
- `skills/welltestai-analysis/scripts/run_slug_analysis.py`: deterministic local helper.
- `skills/welltestai-analysis/assets/slug_alpha_runtime/`: bundled slug alpha runtime.
- `skills/welltestai-analysis/references/`: input schema and claim boundary.
- `examples/slug_golden/`: minimal monotonic slug/recovery example.
- `examples/slug_oscillatory/`: damped oscillatory slug example.

The repository does not include unpublished field datasets or manuscript files.

## Install the Skill

Copy the skill folder into your Codex skills directory:

```powershell
.\INSTALL_SKILL.ps1 -InstallDependencies
```

If PowerShell blocks local scripts, run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\INSTALL_SKILL.ps1 -InstallDependencies
```

Restart Codex after copying the skill.

The installer copies the Skill only. Field data remain on your machine.

## Manual Test

Run the bundled slug example:

```powershell
python skills\welltestai-analysis\scripts\run_slug_analysis.py `
  --case examples\slug_golden\case.yaml `
  --out outputs\slug_golden_report.html
```

Open `outputs\slug_golden_report.html` in a browser.

Run the bundled oscillatory slug example:

```powershell
python skills\welltestai-analysis\scripts\run_slug_analysis.py `
  --case examples\slug_oscillatory\case.yaml `
  --out outputs\slug_oscillatory_report.html
```

## Minimum Slug Case File

```yaml
case_id: example_slug
test_type: slug
data_file: observations.csv
time_column: time
response_column: normalized_head
time_unit: s
response_unit: dimensionless
rw_cm: 5.0
slug_time_scale_seconds: 1.0
log_alpha: 0.0
```

The CSV should contain elapsed time and normalized head recovery:

```csv
time,normalized_head
0.002,0.755
0.017,0.603
0.035,0.520
```

## Current Claim Boundary

This alpha release supports local screening and report generation for slug/recovery data, including QC routing for damped oscillatory records. It should not be used alone to claim a unique aquifer mechanism or final design parameter set. Lagging parameters are effective response-time coordinates unless additional identifiability evidence supports a stronger interpretation. Oscillatory reports provide damping and frequency screening coordinates, not final aquifer K.
