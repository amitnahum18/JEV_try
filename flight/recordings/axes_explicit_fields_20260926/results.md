# JEV flight: axes and telemetry comparison

Small matched-seed exploratory sample; no statistical significance or mechanistic understanding claim. Full telemetry tests a bundle, not individual features. API responses are not deterministic. Two axes see the same state but do not negotiate; aerodynamic coupling remains. API/physics errors excluded from navigation averages and shown separately. Failed-request costs may be unavailable.

All groups: identical seeded slalom courses, aircraft physics, 15 joint actions, stabilizer and decision interval.

| Architecture | Input | Valid runs | Mean gates | Finished | Decision latency (s) | Cost/run ($) | Request bytes/decision |
|---|---|---:|---:|---:|---:|---:|---:|
| single | basic | 4/4 | 0.00 | 0 | 0.658 | 0.00109 | 6368 |
| single | full | 4/4 | 0.00 | 0 | 0.633 | 0.00181 | 10247 |
| parallel | basic | 4/4 | 0.00 | 0 | 0.672 | 0.00160 | 9253 |
| parallel | full | 4/4 | 0.25 | 0 | 0.706 | 0.00399 | 17007 |

## Matched effects

| Comparison | Pairs | Mean additional gates |
|---|---:|---:|
| parallel_minus_single_basic | 4 | +0.00 |
| parallel_minus_single_full | 4 | +0.25 |
| full_minus_basic_single | 4 | +0.00 |
| full_minus_basic_parallel | 4 | +0.25 |

## Individual flights

| Seed | Architecture | Input | Gates | Status | API calls |
|---|---|---|---:|---|---:|
| 42 | parallel | basic | 0/5 | gate_missed | 20 |
| 42 | parallel | full | 1/5 | out_of_bounds | 50 |
| 42 | single | basic | 0/5 | gate_missed | 12 |
| 42 | single | full | 0/5 | gate_missed | 12 |
| 43 | parallel | basic | 0/5 | gate_missed | 22 |
| 43 | parallel | full | 0/5 | gate_missed | 20 |
| 43 | single | basic | 0/5 | gate_missed | 11 |
| 43 | single | full | 0/5 | gate_missed | 11 |
| 44 | parallel | basic | 0/5 | gate_missed | 22 |
| 44 | parallel | full | 0/5 | gate_missed | 20 |
| 44 | single | basic | 0/5 | gate_missed | 11 |
| 44 | single | full | 0/5 | gate_missed | 10 |
| 45 | parallel | basic | 0/5 | gate_missed | 22 |
| 45 | parallel | full | 0/5 | gate_missed | 22 |
| 45 | single | basic | 0/5 | gate_missed | 11 |
| 45 | single | full | 0/5 | gate_missed | 11 |
