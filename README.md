# JEV experiments

Experiments with `typesafe/jev-1.13` through OpenRouter. JEV receives a structured
state or question and chooses among explicitly defined answers or actions.
The repository contains runnable code, saved decisions, evaluation reports and
videos for four experiments:

| Folder | Experiment | Documentation |
|---|---|---|
| `flight/` | Aircraft navigation through obstacles in PyFlyt | [Flight guide](flight/README.md) |
| `chess/` | JEV chess policies against Maia and Stockfish | [Chess guide](chess/README.md) |
| `car/` | Highway driving and DQN comparisons | [Driving guide](car/README.md) |
| `knowledge/` | A 50-question knowledge and reasoning exam | [Exam guide](knowledge/README.md) |

## Setup

The commands below use Windows PowerShell. The recorded experiments used Python
3.12. Install Python, Git and Git LFS first. Videos, chess engines and model
weights are stored with Git LFS; clone the repository to download those assets.

```powershell
git lfs install
git clone https://github.com/amitnahum18/JEV_try.git
cd JEV_try
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r flight\requirements.txt -r chess\requirements.txt
Copy-Item .env.example .env
notepad .env
```

Put your OpenRouter key in `.env` as `OPENROUTER_API_KEY=your_key_here`.
Live JEV runs make API requests and require OpenRouter API access and sufficient
credits. Recorded-flight replay uses local data and makes no model requests.
The `.env` file, Python environment and installed JavaScript dependencies are
excluded from Git.

The installation above covers flight, chess and the knowledge exam. For the
JEV-only driving run below, DQN training or driving comparisons, also install:

```powershell
.\.venv\Scripts\python.exe -m pip install -r car\requirements-rl.txt
```

Run the commands below from the repository root. Interactive simulations need
a desktop display. The bundled chess executables are for Windows; other systems
need compatible engines and updated engine paths.

## Flight: PyFlyt obstacle course

Open the live aircraft simulation:

```powershell
.\.venv\Scripts\python.exe flight\run_course.py
```

By default, one JEV selects from 45 simultaneous pitch/bank maneuvers using
focused numerical input and prompt v8. The `vertical_challenge` course has eight
open frames, changing heights, seed 44 and a 120-second limit. The window shows
the aircraft, decisions and score. There are no surrounding walls.
See [flight/README.md](flight/README.md) for controls and comparison commands.

On 2026-09-27, the v8 policy completed all eight frames on seeds 44, 45 and 46.
These three trials reused previously tested seeds and are an exploratory sample;
they do not establish reliability on new courses. See
[the v8 findings](flight/recordings/pitch_v8_20260927/findings_he.md) and
[the verified 1x flight video](flight/videos/jev-vertical-verified/jev_vertical_verified_normal_speed.mp4).

Replay the recorded seed-44 flight without model calls:

```powershell
.\.venv\Scripts\python.exe flight\run_course.py --replay flight\recordings\pitch_v8_20260927\single_focused_seed44\episode.json
```

### Earlier flight evaluations (2026-09-26)

The flight experiment also compares a single JEV choosing joint pitch/bank
commands with two parallel JEVs dividing those axes, each with basic and expanded
telemetry. Run `flight/compare_axes.py` for the matched four-condition experiment.
The navigation update was evaluated in 32 matched flights. Scores, transition
audits and conclusions are in
[the navigation evaluation](flight/recordings/navigation_evaluation_20260926/findings_he.md).
The earlier sharp-turn correction uses opening-angle windows, measured maneuver
response history and a preferred 1m interior buffer. It completed all five gates
on four tested level-1 courses (seeds 42–45); the paired previous version completed
none. This is a small exploratory sample. Details and replay are in
[the sharp-turn findings](flight/recordings/sharp_turn_20260926/findings_he.md).
The earlier `hard_turns` course uses prompt v6 and 15 choices. It completed all
eight frames on seeds 44, 45 and 46. One initial transport timeout was retried successfully;
all four attempts are preserved in
[the open-frame results](flight/recordings/open_hard_turns_20260926/results.md).
The full 1x recording is
[the eight-frame MP4](flight/videos/open_turns_source/jev_8of8_open_frames_1x.mp4).

## Chess: JEV vs Stockfish

After setup, launch the model-vs-model desktop match:

```powershell
.\.venv\Scripts\python.exe chess\chess_app.py
```

The chess experiment is finished. The final comparison against Maia 1100,
Maia 1500, Maia 1900, and Stockfish 19 is in
`chess/recordings/non_llm_four_20260926/`. Earlier recordings and video projects
are preserved under `chess/archive/`. The reusable chess code and installed
opponents remain in `chess/`.

See [chess/README.md](chess/README.md) for the folder index and run commands.

## Car simulation

The highway simulator and its saved results are in `car/`. Run the JEV-only
driving simulation with:

```powershell
.\.venv\Scripts\python.exe car\jev_only_sim.py --seed 104 --vehicles 50
```

See [car/README.md](car/README.md) for driving results, safety logic, and DQN
training and comparison commands. All driving and flight agents operate in
simulation.

## Knowledge and reasoning exam

The custom exam contains 50 questions covering mathematics, probability,
physics and logic. JEV chooses from four answers plus `unsure`. Model questions
are in English; Hebrew questions and reports are also included.

```powershell
.\.venv\Scripts\python.exe knowledge\run_exam.py
```

This starts a new API-backed exam run. Saved runs are in `knowledge/runs/`.
See [knowledge/README.md](knowledge/README.md) for local answer-key checks,
evaluation methodology and output files. This is an exploratory exam, and its
difficulty has not been calibrated with human participants.
