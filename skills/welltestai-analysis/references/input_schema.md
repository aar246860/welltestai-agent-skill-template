# WellTestAI Input Schema

Minimum case file fields:

- `case_id`: readable case identifier.
- `test_type`: `constant_rate`, `constant_head`, or `finite_boundary` for screening demos.
- `data_file`: path to CSV relative to the case file.
- `time_column`: time field in the CSV.
- `response_column`: response field in the CSV.
- `time_unit`: for example `min`, `hour`, or `s`.
- `response_unit`: for example `cm`, `m`, `mL/min`, or normalized response.

Recommended metadata:

- `well_radius_cm`: pumping well radius;
- `radius_cm`: observation radius or representative radius;
- `forcing_value`: pumping rate for constant-rate tests or maintained drawdown scale for constant-head tests;
- `forcing_unit`: forcing unit;
- aquifer thickness if known;
- sensor resolution and preprocessing notes;
- recovery or late-time coverage flag.

CSV expectations:

- one row per observation;
- numeric time values;
- numeric response values;
- no mixed units inside a single column;
- blank or missing rows removed before analysis.

If units are uncertain, stop and report an `inconsistent_units` risk instead of guessing silently.
