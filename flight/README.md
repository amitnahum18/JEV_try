# JEV Flight Lab — PyFlyt obstacle course

A fixed-wing aircraft flies through randomly placed static rectangular gates.
PyFlyt 0.29.0 supplies the aerodynamic flight physics; the gate frames are solid
PyBullet collision bodies. The task is to pass the openings in order without
colliding. It produces a live simulation, decision logs and optional native
flight recordings at normal simulation speed.

## Open the live simulation

From the workspace root:

```powershell
.\.venv\Scripts\python.exe flight\run_course.py
```

The default is now the eight-frame `vertical_challenge` course on seed 44,
with no walls, heights about 16–58 m, a 120-second limit, one JEV, focused numerical
input and prompt v8. It uses 45 joint pitch/bank choices. JEV receives gate data regardless of the
display camera; no image is sent. Use `--policy jev` for the original six-action
baseline, or `--policy single --telemetry full --prompt-version 3` for the earlier
joint policy. The two-JEV policy still defaults to full input/version 3; the new
focused input has only been evaluated with one JEV.

The previous five-gate course remains available with `--layout slalom`; its
default focused prompt is v5.

### Vertical control verification (2026-09-27)

All three latest real-JEV trials (seeds 44, 45, 46) completed 8/8. The original
failed seed 44 course is selected for video. This is an exploratory sample,
not a guarantee for future runs. Every intermediate failed trial is preserved.

There are 9 nose-up pitch targets: +2, +4, +6, +12, +20, −3, −6, −10, −16 degrees,
combined with 5 bank targets: 0, ±25, ±40 degrees. Prompt v8 adds a geometric
center-pitch estimate and explicit numerical choice criteria. The estimate
uses the current opening's climb window and measured nose-pitch minus
flight-path-angle difference. It is approximate and does not simulate actions.
JEV selects the action through the API; no code selects or overrides it.

Adding angles alone did not fix the observed miss. LEVEL is +2-degree nose
trim, which can arrest a descent. The updated criteria distinguish maintaining
a safe descent from selecting nominal trim and reevaluate the required pitch.
Course geometry, physics, actual scoring margins and 0.5-second decisions remain
unchanged. The gate-update audit verifies exact request states and transitions.

- Report: [findings_he.md](recordings/pitch_v8_20260927/findings_he.md).
- Video: [verified flight at 1x](videos/jev-vertical-verified/jev_vertical_verified_normal_speed.mp4).
- Replay: `python flight/run_course.py --replay flight/recordings/pitch_v8_20260927/single_focused_seed44/episode.json`.
- Previous open course: `--layout hard_turns` (v6, 15 choices).
- Historical coarse vertical mode: `--layout vertical_challenge --prompt-version 6 --pitch-options coarse`.

### Sharp-turn correction: latest results

The old displayed seed-44 flight knew both remaining gates. At the gate-3 exit,
motion pointed 44.556 degrees left and gate 4 lay 14.958 degrees right: a 59.514
degree right correction. It initially commanded a strong right bank, then
alternated left/right despite continued large right alignment errors. Just before
passing gate 3, center angles also grew rapidly while the constant-motion crossing
remained inside its opening. Neither event is a camera visibility filter.

Focused input supplies the safe track/climb angular windows, signed corrections
to the opening, actual recent command-response measurements, and the remaining
side-boundary distance/time. Version 5 adds a preferred 1m interior buffer to
avoid waiting until a small edge miss becomes urgent. The real gate dimensions,
scoring margins, physics, course, 15 actions and 0.5s interval are unchanged.
These fields do not select, override, simulate or filter JEV's actions.

| Level-1 seed | Previous full/prompt 3 | Focused/prompt 4 | Focused + preferred buffer/prompt 5 |
|---|---:|---:|---:|
| 42 | 0/5 | 1/5 | 5/5 |
| 43 | 0/5 | 5/5 | 5/5 |
| 44 | 3/5 | 4/5 | 5/5 |
| 45 | 3/5 | 5/5 | 5/5 |
| Mean gates | 1.50 | 3.75 | 5.00 |
| Complete flights | 0/4 | 2/4 | 4/4 |

All 12 flights made real JEV API calls. Four reused course seeds are an exploratory
sample, not proof of reliability on new seeds or higher levels. Prompt and input
changed together, so individual causal effects were not isolated. Original
records remain preserved under `recordings/sharp_turn_20260926/`.

Replay the new complete flight on the same seed that previously failed:

```powershell
.\.venv\Scripts\python.exe flight\run_course.py --replay flight\recordings\sharp_turn_20260926\buffered_v5\v5\single_focused_seed44\episode.json
```

### Normal-speed video

The successful seed-44 flight is exported at **1x simulated time**, 30 fps,
1440 by 860, with a two-second completion hold. The flight lasts 29.6167 seconds;
the encoded video lasts about 31.667 seconds including the hold. Rendering can
take longer than playback; API waiting time is not included in the video.

The MP4 is `videos/jev-passed-normal-speed/jev_flight_5_gates_normal_speed.mp4`.
The HyperFrames project wraps the actual native simulation footage unchanged
at playback rate 1. The footage contains numerical state, simultaneous pitch/bank
commands, actual actuator deflections and the flight trail.

The replay exporter advances every recorded physical step; it validates the
outcome and final position against the original episode and writes export timing
and source provenance beside the source MP4. It makes no model requests.

```powershell
.\.venv\Scripts\python.exe flight\export_replay.py flight\recordings\sharp_turn_20260926\buffered_v5\v5\single_focused_seed44\episode.json --output flight\videos\source_footage\new_export.mp4 --fps 30
```

`imageio-ffmpeg` supplies a local FFmpeg executable if one is not on PATH.

The detailed Hebrew report is [findings_he.md](recordings/sharp_turn_20260926/findings_he.md).

The window displays a close chase view, actual aircraft bank/heading/pitch,
speed and climb rate, gate-center errors, predicted crossing errors, measured
control-surface deflections, and a map with the actual flight trace. It stays
open when an attempt ends and shows a prominent retry notice.

| Key | Action |
|---|---|
| R | Retry the same course with the current policy |
| N | New random course |
| J | Switch to the original six-action JEV policy |
| B | Start a new attempt with the random baseline |
| 1 / 2 / 3 | Restart at the selected difficulty |
| C | Switch between close chase and overview cameras |
| I | Pause and inspect the exact JSON input sent to JEV; scroll with the wheel or PageUp/PageDown |
| Space | Pause or resume |
| Escape | Exit |

Existing shared `.venv` and root `.env` are used. JEV needs the existing
`OPENROUTER_API_KEY`. If setting up a new environment, install
`flight/requirements.txt`. PyFlyt must remain on the documented 0.29.0 version
for exact reproducibility of the stored episodes.

## Original baseline: full input and six outputs (`--policy jev`)

JEV receives:

| Group | Data |
|---|---|
| Aircraft | World position and velocity, body velocity, speed, horizontal speed, climb rate, roll/pitch/yaw, explicit left-bank and nose-up pitch angles, angular rates, heading, actual velocity-track heading and climb angle |
| Dynamics | World/body linear acceleration, specific-force load in g, body reference angle of attack and sideslip |
| Actuators | Aileron/elevator/rudder/throttle commands, measured surface deflections, measured motor throttle, and stabilization targets |
| Next three gates | Center position, opening dimensions, world/body relative coordinates, center distance, distance/time to the gate plane, signed left/up center errors, normalized errors, bearing/elevation, heading/velocity-track/climb alignment errors and closing speed |
| Crossing estimate | Where constant straight world velocity would cross the gate plane; predicted left/up center errors and remaining opening clearances |
| Context | Previous action and the latest four actions with observed changes in attitude, velocity heading, climb rate and gate-center errors; decision interval, physical coordinate conventions and zero-wind setting |

The crossing estimate is a geometric extrapolation of current velocity, not a
navigation policy or a promise that the aircraft will follow that path. Angle
of attack is evaluated at the aircraft body reference; individual lifting
surfaces have their own aerodynamic state inside PyFlyt.

`latest_jev_input.json` in each JEV run directory contains the actual request
payload excluding authentication. Press I to inspect it in the window.
Each decision is stored together with the complete input, confidence, probability
distribution, latency, and known API cost.

Exactly six choices are available:

1. `up` — pitch up to climb.
2. `down` — pitch down to descend.
3. `left` — left coordinated turn with a 25° bank.
4. `right` — right coordinated turn with a 25° bank.
5. `roll_left` — stronger left coordinated turn with a 40° bank.
6. `roll_right` — stronger right coordinated turn with a 40° bank.

Up/down return the wings toward level. Requested attitude targets are translated
to control-surface commands; actual angles depend on the aerodynamic response.
The aircraft is never moved along a scripted path or teleported to a gate.

There is no hidden navigation policy, route planner, or alternative model advising
JEV. A shared basic controller converts a selected maneuver to aileron, elevator,
rudder and throttle commands and stabilizes attitude/rates and speed. **This
controller has no access to gate locations.** Both JEV and random use the same
controller and the same six actions. Low-level stabilization is therefore part of
the experiment, not an ability attributed to JEV itself.

The installed PyFlyt version's direct surface mode `-1` is used to keep the
elevator and rudder independently controlled. Each choice lasts at most 0.5 simulated
seconds by default, with a new decision immediately after passing a gate. Physics pauses while waiting for the API; this evaluates
decisions on the exact supplied state and does not measure performance under
real-time network delays. The window remains responsive while JEV thinks.

## Expanded navigation features (version 3)

Full input now includes a prominent `navigation` block. This block is sent to
the original six-action JEV, the joint-action single JEV, and both parallel JEVs
when `--telemetry full` is selected. Basic input retains its smaller field set
for input-ablation comparisons. The new full feature schema is
`flight_navigation_v3`; `--telemetry full` uses prompt version 3 unless specified.

| Group | Added explicit navigation features |
|---|---|
| Altitude | Current world altitude and height over the flat ground; gate-center altitude; signed height difference; allowed altitude band including the aircraft margin; climb rate and vertical acceleration |
| Attitude and motion | Nose-up pitch and left bank; bank/pitch angle rates; actual heading rate; actual velocity-track turn rate; flight-path-angle rate; observed horizontal turn radius when measurable |
| Gate alignment | Motion-direction and climb-angle errors, rates of change of those errors, gate-bearing rate, lateral/height-error rates, horizontal range and time to the gate plane |
| Opening geometry | Feasible constant-direction track and climb angle windows, with inside-window checks; opening width/height and margins remain in the existing gate fields |
| Crossing | Existing constant-velocity crossing errors/clearances are repeated in the navigation summary; required lateral and vertical velocity to reach the center assuming constant current world-x speed |
| Bank release | Estimated time to return within 3 degrees of wings level, additional change in velocity-track direction while leveling, estimated final track, and alignment error assuming the gate bearing stays fixed |
| Next gate | Center-to-center lateral/altitude changes, bearing and elevation to describe the upcoming transition |

Positive means left/up. Every field name includes its unit. The geometry uses
the fixed x-plane rectangular gates and the existing 1.25 m lateral / 0.5 m
vertical body-center margins. Height over ground assumes this simulation's flat
world-z=0 ground; it is not a terrain sensor. Being inside the opening's altitude
or angular band alone does not mean a gate has been passed.

The live display highlights **velocity-track error** rather than only nose
heading error, and shows actual track turn rate plus bank-leveling estimates in
the bottom navigation panel. I still exposes all exact model inputs. The prompt
explicitly distinguishes the gate being left of the aircraft's position from
the gate being left of its current motion direction.

### Estimate assumptions and provenance

- Angular attitude rates are computed from body angular velocity and Euler
  attitude. Actual track/climb angle rates use world velocity and the most recent
  1/120-second finite-difference acceleration. These rates are not smoothed and
  can be noisy during surface transitions; acceleration-derived rates are null
  before the first simulated step or where direction is undefined.
- The constant-velocity crossing forecast and constant-rate one-interval
  alignment estimate are geometric extrapolations, not predictions of the
  action JEV is about to choose. Angular opening windows assume an unchanged
  direction of motion. Required center-reaching velocities assume unchanged
  current world-x speed; they are features, not commands.
- The bank-release response uses an empirical first-order bank-decay fit from
  **66 recorded intervals in 7 earlier banked_v2 episodes** during commands
  targeting zero bank. The median time constant is approximately **0.243 s**;
  sample p10/p90 constants are approximately **0.213 / 0.469 s**. The complete
  source intervals are retained in `artifacts/bank_level_response_fit.json`.
- To estimate additional turn, the current measured velocity-track turn rate is
  assumed to decay exponentially with the same time constant as bank. This
  approximation omits roll inertia, yaw lag, sideslip, and pitch/lift coupling.
  The sample p10/p90 range describes fitted response variation, not a prediction
  confidence interval. The estimate is disabled for the legacy control profile
  or when measurements/calibration are unavailable. It has not been validated
  on independent flights. A three-case diagnostic on the earlier seed-42 flight
  measured leveling-time underestimates of about 0.10-0.12 s and additional-turn
  errors from about -1.28 to +2.88 degrees. See
  `artifacts/leveling_estimator_audit.json`; this is not full estimator validation.
- Every estimate carries its assumption in the actual model payload. No field
  selects an action, overrides JEV, or advances a hidden physics rollout.

The additions have now been evaluated in the 32-flight experiment below.
The earlier tables elsewhere in this document describe older feature sets; stored historical requests and
decisions remain the evidence for those runs. Fresh runs log the new feature
schema and prompt version. Physical replay executes old recorded actions;
its current telemetry can contain new features without changing what the model
originally received (both are distinguished in the input inspector).

### Evaluation: input changes and gate transitions

`recordings/navigation_evaluation_20260926/` contains a fresh 32-flight matched
experiment: four seeds (42-45), two architectures, and four conditions. Level,
gate geometry, physics, 15 joint actions, and nominal 0.5-second interval are
identical. The conditions separate added fields, prompt guidance, and refreshing
the decision at a gate crossing:

| Condition | Single JEV mean gates / 5 | Parallel JEVs mean gates / 5 |
|---|---:|---:|
| Previous full input and prompt 2 | 0.00 | 0.25 |
| Added fields, previous prompt 2 | 0.25 | 0.00 |
| Added fields and updated prompt 3 | 1.50 | 1.25 |
| Also refresh the decision immediately after passing a gate | 1.50 | 1.25 |

All 32 flights were valid navigation attempts; there were no API/physics errors.
No run passed all five gates; the maximum was three. Input plus prompt guidance
improved this small sample, whereas fields alone did not improve consistently.
The sample does not establish statistical significance, isolate individual
useful fields, or show that either architecture is generally better.

A physical replay audit checked **36,200 physics steps**, **24 gate passages**,
and **925 exact request payloads** across all 32 episodes. Gate number, center,
relative position, altitude difference, navigation target and per-axis snapshots
matched the simulator. Targets update in the same physics step as a passage.

The earlier displayed flight continued its previous-gate action for **0.2083 s**
after gate 1 while waiting for the next fixed decision boundary. This was a
decision-scheduling delay, not stale gate coordinates. Fresh runs now interrupt
the action and request a new decision immediately after a successful passage;
the expanded-refresh condition measured zero simulated delay at all crossings.
This scheduling change did not improve the aggregate gate count in this sample.
Physics still pauses while waiting for JEV, so network-time robustness is untested.

Full input also includes `navigation.target_update`: active gate, previous
decision's gate, target-change flag, last passage time and elapsed time on the
current target. Stored decision rows include `applied_steps`, allowing exact
replay of maneuvers shortened by a gate passage. Old recordings without that
field retain their original fixed-interval replay behavior.

Remaining failures in the four parallel/expanded-refresh flights were lateral
misses of approximately 23-41 m; their vertical errors were within 2.1 m and the
usable height band was +/-5.5 m. More altitude fields do not explain those four
failures. Short histories of actual maneuver responses and a more accurate
bank-release model are candidate improvements, not benefits established here.

Run the evaluation and audits:

```powershell
.\.venv\Scripts\python.exe flight\evaluate_navigation.py --seeds 4 --workers 3
.\.venv\Scripts\python.exe flight\test_navigation.py
.\.venv\Scripts\python.exe flight\audit_navigation.py flight\recordings\navigation_evaluation_20260926
```

Nine local regression tests cover angle windows, units/signs, body-to-heading
rates, unavailable measurements, input separation, target changes, and repeated
live-driver decisions (with mocked API replies). The scored flights use real
JEV calls. Detailed scores and per-seed differences are in `results.md` and
`comparison.json`; the replay audit is in `gate_update_audit.json`.

## Difficulty

| Level | Speed target | Gate opening |
|---|---|---|
| 1 | 20 m/s (72 km/h) | 16 m wide × 12 m high |
| 2 | 25 m/s (90 km/h) | 13 m wide × 10 m high |
| 3 | 30 m/s (108 km/h) | 10 m wide × 8 m high |

The default layout is a slalom: gate centers alternate between left and right
by about 14, 18 or 22 meters depending on level, with changing altitude and small
seeded jitter. The first gate is 95 meters ahead and subsequent gates are spaced
110 meters apart. Successive openings do not overlap laterally, so straight flight
cannot complete the course. Use `--layout legacy` to view the earlier gentle layout.

Actual speed
is logged and can differ slightly from the target. The standard task has five
gates. A passage is scored when the aircraft crosses a gate's x plane inside the
opening, with center margins of 1.25 m laterally and 0.5 m vertically. Solid-frame
or ground contact ends the attempt. A missed gate, excessive bank, boundary exit,
time limit, or technical failure is recorded separately.

## Repeatable comparison

```powershell
.\.venv\Scripts\python.exe flight\run_course.py --headless --policy jev --compare-random --episodes 3 --seed 42 --level 1
```

This runs JEV and random on the same seeds and gate geometry, with identical
physics/controller settings. Outputs under `flight/recordings/` include:

- Per-episode JSON: complete inputs, choices, trajectory, gate events, and outcome.
- `comparison.json`: compact results for all episodes.
- `results.md`: readable table.
- Interactive runs also save `latest_view.png`.

Score: +100 per passed gate, −50 for a missed opening, −100 for a collision.
API failures are technical failures, never fabricated flight results. No baseline
replaces a failed JEV call.

The initial comparison on the older gentle layout and original controls at
`recordings/first_comparison/` produced the following. These results do not describe
the new slalom:

| Seed | JEV gates | Random gates | JEV outcome |
|---|---|---|---|
| 42 | 3/5 | 1/5 | Missed gate 4 |
| 43 | 1/5 | 1/5 | Collision |
| 44 | 2/5 | 2/5 | Missed gate 3 |

Mean gates passed: JEV **2.00**, random **1.33**. Neither completed a course.
Three paired attempts are a preliminary sample, not evidence of general flight
competence.

The first slalom comparison with expanded inputs at `recordings/rich_slalom_v2/`
tested seeds 42 and 43. JEV passed 0/5 and 1/5 gates; random passed 1/5 and 0/5.
Neither completed a course. The two-attempt sample does not establish an advantage.

The actual aerodynamic control response was also measured over 1.5 simulated
seconds: moderate left/right turns reached about 25° bank and 14° heading change;
stronger bank turns reached about 40° bank and 23° heading change. Full measurements
are in `artifacts/control_response_v2.json`.

## Replay an actual recorded attempt

```powershell
.\.venv\Scripts\python.exe flight\run_course.py --replay flight\recordings\first_comparison\jev_level1_seed42_161216_303912.json
```

Replay reconstructs the course and executes the recorded discrete decisions in
the same PyFlyt physics. It makes no new model requests. Press J or R to start a
fresh JEV attempt. Press B for a random attempt. The previous recording is preserved.
Older recordings automatically select the legacy layout and control profile.

## One JEV versus two parallel JEVs; basic versus full input

Run a matched 2 x 2 experiment:

```powershell
.\.venv\Scripts\python.exe flight\compare_axes.py --seeds 4 --seed 42 --workers 2
```

The four groups are `single/basic`, `single/full`, `parallel/basic` and
`parallel/full`. Each flies the exact same seeded slalom courses at level 1.
Two independent simulator processes run at once. The two JEV axis calls within
a parallel decision also run concurrently, using the same frozen state snapshot.

For a fair comparison, **all four groups have the same 15 joint maneuvers**:

- Pitch: `up` (nose up 6 degrees), `down` (nose down 6 degrees), or `level`
  (nominal nose-up trim 2 degrees; it does not hold altitude).
- Bank: `straight` (wings level), `left`/`right` (25 degrees), or
  `roll_left`/`roll_right` (40 degrees). Wings level does not lock heading.
- Single JEV selects one combination, such as `up__left`.
- Parallel JEVs independently select pitch and bank; their choices are composed
  without voting, negotiation, another model, or a route planner. Bank affects
  vertical lift, so independence can introduce conflicting decisions.

These joint controls are a separate experiment from the original six-action
JEV/random baseline above; their numerical results should not be mixed.

| Input | Included |
|---|---|
| Basic | Position, world velocity, Euler attitude; the next three gates' centers, opening sizes, world-relative offsets and distances; shared control definitions, conventions, environment and last four maneuver/time pairs |
| Full | Everything in basic, plus body velocity, angular rates, explicit bank/pitch/track/climb angles, acceleration, load, attack/sideslip angles, actuator measurements, body-relative gate coordinates, alignment errors, time to plane, normalized errors, closing speed and constant-velocity crossing estimates |

Both groups see the same three upcoming gates and action history. Rich result
feedback is recorded for analysis but is not leaked into basic model input.
The experiment tests the expanded parameter bundle; it cannot identify which
individual field helps. API calls are nondeterministic and physics pauses during
calls, so this is neither a real-time latency tolerance test nor proof of an
internal understanding of navigation.

Each output directory contains `plan.json`, `comparison.json`, `results.md`,
and one subdirectory per condition and seed. Each `episode.json` stores exact
requests, individual axis choices/probabilities/confidences, joint action,
physical response, trajectory and outcome. Latency is measured for the complete
decision (including both calls for parallel); known cost is the sum of calls.
Technical errors are listed separately and excluded from navigation averages.
Outputs must use a fresh directory to preserve previous results.

Replay a recorded two-JEV attempt in the existing viewer, without API calls:

```powershell
.\.venv\Scripts\python.exe flight\run_course.py --replay flight\recordings\axes_explicit_fields_20260926\parallel_full_seed42\episode.json
```

The recorded requests for each axis can be inspected in the episode JSON.

Start a fresh live attempt with either architecture:

```powershell
.\.venv\Scripts\python.exe flight\run_course.py --policy parallel --telemetry full
.\.venv\Scripts\python.exe flight\run_course.py --policy single --telemetry basic
```

R keeps the selected architecture. J explicitly switches to the original
six-action JEV policy. I displays the exact request for each axis. An API error
on either axis ends the attempt; it is not replaced with a default decision.

### Findings on 2026-09-26

The initial 16 flights are preserved under `recordings/axes_ablation_20260926/`.
An additional diagnostic reflected one gate left/right while keeping the initial
aircraft state fixed. The original prompt chose left in all 16 decisions, giving
8/16 steering signs toward the gate. Explicit references to the relevant nested
state fields, following the official [TypeSafe question guidance](https://docs.typesafe.ai/primitives#reference-specific-fields),
gave 15/16 correct initial steering signs on the repeated diagnostic. This
compares two prompt formulations; it does not prove an internal cause or optimal
flight. Exact calls are saved in `mirror_diagnostic/` and `mirror_explicit_fields/`.

The corrected prompt (version 2) was then used for a fresh
matched 16-flight comparison under `recordings/axes_explicit_fields_20260926/`:

| Architecture | Input | Gates per run (seeds 42, 43, 44, 45) | Mean gates | Full completions |
|---|---|---|---:|---:|
| Single JEV | Basic | 0, 0, 0, 0 | 0.00 | 0/4 |
| Single JEV | Full | 0, 0, 0, 0 | 0.00 | 0/4 |
| Two parallel JEVs | Basic | 0, 0, 0, 0 | 0.00 | 0/4 |
| Two parallel JEVs | Full | 1, 0, 0, 0 | 0.25 | 0/4 |

There were no technical failures in these 16 flights. This is a small possible
advantage for the parallel/full combination, insufficient to establish that
splitting control is worthwhile or that richer input reliably improves flight.
The expanded input is a bundle, so individual useful fields remain unidentified.
Version 1 produced a different result (parallel/full passed one gate on two of
four seeds), which further limits any conclusion. All results are preserved.
Use `--prompt-version 1` with `compare_axes.py` or `probe_axes.py` to run the
earlier prompt. The six-action live prompt also now references the immediate
gate and aircraft fields explicitly.

Mean decision latency in the corrected experiment was 0.63-0.71 seconds across
groups, with physics paused during calls. Known mean cost per episode was about
$0.00181 for single/full and $0.00399 for parallel/full; episode lengths differed,
so this is not a fixed-length cost comparison. Full records include per-decision
cost and the probability/confidence values returned by the API without alteration.

Physical replay of the recorded parallel/full seed-42 attempt reproduced its
one passed gate, final outcome and final position within 1 mm.

## Open frames with harder turns

The default in `run_maze.py` is now `hard_turns`: eight open frames, no
surrounding walls, larger lateral reversals and altitude changes. Seed 44 has
centerline direction changes up to 45.7 degrees and frame heights around
19-35 metres. The original 15 actions, physics and passing margins are retained.

```powershell
.\.venv\Scripts\python.exe flight\run_maze.py --layout hard_turns --seeds 44 45 46 --workers 3 --prompt-version 6 --output-dir flight\recordings\open_turns_new_attempt
```

The three seeded courses completed 8/8 frames on 2026-09-26. One first attempt
had a transport timeout at 3.5 seconds; its separate retry completed. This
means four attempts total, three completions and one technical stop, with no
recorded navigation failure. See `recordings/open_hard_turns_20260926/results.md`.

The normal-speed completed video is at
`videos/open_turns_source/jev_8of8_open_frames_1x.mp4`.

For an interactive replay with no API calls:

```powershell
.\.venv\Scripts\python.exe flight\run_course.py --replay flight\recordings\open_hard_turns_20260926\v6\single_focused_seed46\episode.json
```

For a fresh live window:

```powershell
.\.venv\Scripts\python.exe flight\run_course.py --layout hard_turns --gates 8 --seconds 85
```

The default single-JEV focused input uses prompt v6 for `hard_turns`.

### Extreme height preview

`vertical_challenge` has open frames at approximately 26, 48, 17, 58, 20,
54, 16 and 42 metres. It adds two pitch choices: strong climb at +20 degrees
and strong descent at -16 degrees. Gentle climb/descent remain +6/-6 degrees,
with nominal trim +2 degrees. Five pitch choices times five bank choices
give 25 joint choices for one JEV. The actual pitch targets are included in
the input and every candidate description.

```powershell
.\.venv\Scripts\python.exe flight\run_course.py --layout vertical_challenge --preview-quality fast
```

This opens a live preview. JSON state and choices are saved for inspection;
video export is a separate operation. The user subsequently requested the
video of the previewed attempt. It passed two frames and missed the third;
the actual 1x recording is at
`videos/jev-vertical-25-choices/jev_vertical_25_choices_normal_speed.mp4`.
The source and measured outcome are documented in
`recordings/vertical_25_choices_preview_20260926/video_export.md`. Fast preview
reduces only the live viewport rendering resolution; recorded aircraft physics
and the numerical HUD remain unchanged.

## Earlier eight-wall maze

The `maze` layout adds real collision walls around eight winding openings,
with alternating climbs, descents and turns. Run a fresh attempt with one JEV
and the same 15 joint actions:

```powershell
.\.venv\Scripts\python.exe flight\run_maze.py --layout maze --seeds 44 --workers 1 --prompt-version 6 --output-dir flight\recordings\maze_new_attempt
```

The recorded 2026-09-26 attempts with prompt v6 each passed 4/8 gates before
OpenRouter returned HTTP 402 for insufficient credits. None completed the
new maze. See `recordings/maze_20260926/results.md` for every attempt and the
prompt change. A normal-speed partial MP4 is saved at
`videos/jev-maze-normal-speed/jev_maze_partial_normal_speed.mp4`.

The recording export uses only existing choices and native physics:

```powershell
.\.venv\Scripts\python.exe flight\export_parallel.py flight\recordings\maze_20260926\criteria_v6\single_focused_seed44\episode.json --output flight\videos\maze_partial_reexport.mp4 --fps 30 --workers 4
```

The four workers render frames offline; they do not call JEV or control the
aircraft as a team. The final technical stop is shown explicitly.

## Code and reference

- `course.py`: gate geometry, physics, state input, action translation and scoring.
- `run_course.py`: JEV calls, random baseline, live window, replay and persistence.
- `axis_policy.py`: matched joint and parallel-axis policies; input ablation.
- `compare_axes.py`: paired-seed experiment, exact recordings and results table.
- `probe_axes.py`: mirrored static-gate input sensitivity diagnostic.
- `navigation.py`: explicit navigation measurements, geometry and bank-release estimates.
- `evaluate_navigation.py`: controlled before/after and gate-refresh comparison.
- `audit_navigation.py`: per-step physics replay and exact input/target audit.
- `test_navigation.py`: local navigation and driver regression tests.
- `maneuver_context.py`: compact opening windows, measured response history and optional interior buffer.
- `investigate_turns.py`: recorded-state sharp-turn probes and complete matched flights.
- `export_replay.py`: offscreen video export at simulation speed 1x.
- `export_parallel.py`: isolated offline rendering of disjoint frame ranges.
- `run_maze.py`: real single-JEV attempts on the open hard-turn course; retains the earlier wall layout.
- Official fixed-wing documentation:
  https://taijunjet.com/PyFlyt/documentation/core/drones/fixedwing.html
