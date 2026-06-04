# WellTestAI Input Schema

Minimum case file fields:

- `case_id`: readable case identifier.
- `test_type`: `constant_rate`, `constant_head`, or `finite_boundary` for screening demos.
- `observations`: path to CSV relative to the case file.
- `time_column`: time field in the CSV.
- `response_column`: response field in the CSV.
- `time_unit`: for example `min`, `hour`, or `s`.
- `response_unit`: for example `cm`, `m`, `mL/min`, or normalized response.

Recommended metadata:

- pumping well radius;
- observation radius or observation well list;
- pumping rate for constant-rate tests;
- maintained drawdown for constant-head tests;
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

