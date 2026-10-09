# Navigation input and gate-refresh evaluation

Exploratory matched-seed sample. Prior and current input are rerun concurrently to reduce historical/API-time confounding. Fields-only arm uses the previous prompt to isolate the added fields; expanded also changes prompt and transition context. Expanded-refresh additionally interrupts a maneuver when a gate is passed. Physics pauses for API calls. No claim of statistical significance or internal model understanding. Input additions are a bundle, not per-feature tests.

| Architecture | Condition | Valid runs | Gates per seed | Total gates | Mean gates | Completed |
|---|---|---:|---|---:|---:|---:|
| single | previous | 4/4 | 42:0, 43:0, 44:0, 45:0 | 0 | 0.00 | 0 |
| single | fields_only | 4/4 | 42:1, 43:0, 44:0, 45:0 | 1 | 0.25 | 0 |
| single | expanded | 4/4 | 42:0, 43:2, 44:2, 45:2 | 6 | 1.50 | 0 |
| single | expanded_refresh | 4/4 | 42:0, 43:1, 44:3, 45:2 | 6 | 1.50 | 0 |
| parallel | previous | 4/4 | 42:1, 43:0, 44:0, 45:0 | 1 | 0.25 | 0 |
| parallel | fields_only | 4/4 | 42:0, 43:0, 44:0, 45:0 | 0 | 0.00 | 0 |
| parallel | expanded | 4/4 | 42:0, 43:1, 44:3, 45:1 | 5 | 1.25 | 0 |
| parallel | expanded_refresh | 4/4 | 42:2, 43:1, 44:1, 45:1 | 5 | 1.25 | 0 |

## Paired effects

| Architecture | Difference | Pairs | Mean extra gates |
|---|---|---:|---:|
| single | fields_only_minus_previous | 4 | +0.25 |
| single | expanded_minus_previous | 4 | +1.50 |
| single | expanded_minus_fields_only | 4 | +1.25 |
| single | expanded_refresh_minus_expanded | 4 | +0.00 |
| parallel | fields_only_minus_previous | 4 | -0.25 |
| parallel | expanded_minus_previous | 4 | +1.00 |
| parallel | expanded_minus_fields_only | 4 | +1.25 |
| parallel | expanded_refresh_minus_expanded | 4 | +0.00 |

## Flights

| Seed | Architecture | Condition | Gates | Status |
|---|---|---|---:|---|
| 42 | parallel | expanded | 0/5 | gate_missed |
| 42 | parallel | expanded_refresh | 2/5 | gate_missed |
| 42 | parallel | fields_only | 0/5 | gate_missed |
| 42 | parallel | previous | 1/5 | gate_missed |
| 42 | single | expanded | 0/5 | gate_missed |
| 42 | single | expanded_refresh | 0/5 | gate_missed |
| 42 | single | fields_only | 1/5 | out_of_bounds |
| 42 | single | previous | 0/5 | gate_missed |
| 43 | parallel | expanded | 1/5 | gate_missed |
| 43 | parallel | expanded_refresh | 1/5 | gate_missed |
| 43 | parallel | fields_only | 0/5 | gate_missed |
| 43 | parallel | previous | 0/5 | gate_missed |
| 43 | single | expanded | 2/5 | gate_missed |
| 43 | single | expanded_refresh | 1/5 | gate_missed |
| 43 | single | fields_only | 0/5 | gate_missed |
| 43 | single | previous | 0/5 | gate_missed |
| 44 | parallel | expanded | 3/5 | out_of_bounds |
| 44 | parallel | expanded_refresh | 1/5 | gate_missed |
| 44 | parallel | fields_only | 0/5 | gate_missed |
| 44 | parallel | previous | 0/5 | gate_missed |
| 44 | single | expanded | 2/5 | gate_missed |
| 44 | single | expanded_refresh | 3/5 | out_of_bounds |
| 44 | single | fields_only | 0/5 | gate_missed |
| 44 | single | previous | 0/5 | gate_missed |
| 45 | parallel | expanded | 1/5 | gate_missed |
| 45 | parallel | expanded_refresh | 1/5 | gate_missed |
| 45 | parallel | fields_only | 0/5 | gate_missed |
| 45 | parallel | previous | 0/5 | gate_missed |
| 45 | single | expanded | 2/5 | gate_missed |
| 45 | single | expanded_refresh | 2/5 | collision |
| 45 | single | fields_only | 0/5 | gate_missed |
| 45 | single | previous | 0/5 | gate_missed |
