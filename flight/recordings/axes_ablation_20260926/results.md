# JEV flight: axes and telemetry comparison

Small matched-seed exploratory sample; no statistical significance or mechanistic understanding claim. Full telemetry tests a bundle, not individual features. API responses are not deterministic. Two axes see the same state but do not negotiate; aerodynamic coupling remains. API/physics errors excluded from navigation averages and shown separately. Failed-request costs may be unavailable.

All groups: identical seeded slalom courses, aircraft physics, 15 joint actions, stabilizer and decision interval.

| Architecture | Input | Valid runs | Mean gates | Finished | Decision latency (s) | Cost/run ($) | Request bytes/decision |
|---|---|---:|---:|---:|---:|---:|---:|
| single | basic | 4/4 | 0.00 | 0 | 0.652 | 0.00105 | 5683 |
| single | full | 4/4 | 0.00 | 0 | 0.701 | 0.00170 | 9559 |
| parallel | basic | 4/4 | 0.00 | 0 | 0.741 | 0.00158 | 7900 |
| parallel | full | 4/4 | 0.50 | 0 | 0.663 | 0.00400 | 15668 |

## Matched effects

| Comparison | Pairs | Mean additional gates |
|---|---:|---:|
| parallel_minus_single_basic | 4 | +0.00 |
| parallel_minus_single_full | 4 | +0.50 |
| full_minus_basic_single | 4 | +0.00 |
| full_minus_basic_parallel | 4 | +0.50 |

## Individual flights

| Seed | Architecture | Input | Gates | Status | API calls |
|---|---|---|---:|---|---:|
| 42 | parallel | basic | 0/5 | out_of_bounds | 24 |
| 42 | parallel | full | 0/5 | gate_missed | 20 |
| 42 | single | basic | 0/5 | gate_missed | 11 |
| 42 | single | full | 0/5 | gate_missed | 10 |
| 43 | parallel | basic | 0/5 | out_of_bounds | 22 |
| 43 | parallel | full | 0/5 | collision | 20 |
| 43 | single | basic | 0/5 | out_of_bounds | 12 |
| 43 | single | full | 0/5 | gate_missed | 11 |
| 44 | parallel | basic | 0/5 | gate_missed | 24 |
| 44 | parallel | full | 1/5 | gate_missed | 42 |
| 44 | single | basic | 0/5 | out_of_bounds | 12 |
| 44 | single | full | 0/5 | gate_missed | 11 |
| 45 | parallel | basic | 0/5 | out_of_bounds | 24 |
| 45 | parallel | full | 1/5 | out_of_bounds | 36 |
| 45 | single | basic | 0/5 | gate_missed | 12 |
| 45 | single | full | 0/5 | gate_missed | 11 |
