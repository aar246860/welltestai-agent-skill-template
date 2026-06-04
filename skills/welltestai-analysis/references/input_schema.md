# WellTestAI Slug Input Schema

Minimum `case.yaml` fields:

- `case_id`: readable case identifier.
- `test_type`: `slug` or `recovery`.
- `data_file`: path to CSV relative to the case file.
- `time_column`: elapsed-time field in the CSV.
- `response_column`: normalized head field in the CSV.
- `time_unit`: for example `s`, `min`, or `h`.
- `response_unit`: usually `dimensionless`.
- `rw_cm`: well radius in cm.
- `slug_time_scale_seconds`: explicit slug time scale used for dimensionless interpretation.
- `log_alpha`: slug geometry coordinate used by the bundled alpha runtime.

Optional metadata:

- `ar_over_a`: casing or pressurization area ratio if available;
- sensor resolution;
- baseline correction notes;
- screen interval and well construction notes;
- preprocessing notes.

CSV expectations:

- one row per observation;
- positive elapsed time;
- normalized head recovery usually between 0 and 1;
- no mixed units inside a single column;
- blank or missing rows removed before analysis.

Example:

```yaml
case_id: slug_demo
test_type: slug
data_file: observations.csv
time_column: time
response_column: normalized_head
time_unit: s
response_unit: dimensionless
rw_cm: 5.0
slug_time_scale_seconds: 1.0
log_alpha: 0.0
ar_over_a: 0.85
```

If units, geometry, or normalization are uncertain, report a QC risk instead of guessing silently.
