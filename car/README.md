# JEV Five-Lane Highway Simulator

The project explores `typesafe/jev-1.13` as a highway-driving policy in HighwayEnv. The original `car_sim.py` mode gives JEV all-lane gap, speed and time-to-collision summaries and lets it select safe lane and speed actions.

## Grid-based parallel experiment

`compare_sim.py` runs JEV and Stable-Baselines3 DQN side by side on the same seeded traffic. Both policies receive a 5×15 occupancy grid: rows go from physical left/up to right/down; columns span 140 m behind through 140 m ahead in 20 m cells; `0` is empty, `1` is another car, and `2` is the ego car in the center column. DQN receives the 75 flattened cell values; JEV receives the same matrix as JSON.

The lane-only action set is `move_up`, `stay`, and `move_down`. The grid DQN baseline keeps a fixed 44 m/s target and has no longitudinal actions. The JEV-only run adds a separate speed supervisor that can lower its target when dense traffic leaves no safe passing lane. The grid DQN wrapper retains its 20 m occupancy gate. The display also draws each model's current input matrix under its road view.

## JEV-only long-lookahead run

`jev_only_sim.py` runs JEV without the DQN panel. It uses a 5×31 grid at the same 20 m resolution (300 m ahead and behind), per-action front/rear gap and TTC values, and constant-speed projected grids through 4.5 seconds. JEV chooses routine lane changes. The safety gate checks destination front/rear gaps, the origin-lane sweep, and whether any adjacent escape remains open in 3 seconds; it rejects moves below 12 m at the destination, 8 m in the origin lane during the transition, or into a projected dead end. If current-lane TTC reaches 3 seconds and JEV has not selected a safe exit, an urgent safety fallback takes the best route-viable adjacent lane; below 1.5 seconds it can take the best immediately safe lane if no viable route remains. A speed supervisor keeps the 44 m/s cruise target when an overtake is viable; if no safe exit exists and a slower lead vehicle is closing within 7 seconds, it lowers the target toward that vehicle's speed plus a small passing margin. During a lane change, new requests are held and any JEV response based on the old traffic snapshot is discarded. The report records safety holds, urgent exits, speed-limited steps, and per-step traffic facts. Run it from the project root:

```powershell
.\.venv\Scripts\python.exe car\jev_only_sim.py --seed 104 --vehicles 50
```

`jev_prompt_eval.py` tests the prompt over multiple traffic seeds. In the current ten-seed, 50-car evaluation, the car made lane changes but still collided in every run. With no braking action, prompt tuning alone has not made dense traffic collision-free.

The grid DQN needs its own training because its observation and action spaces differ from the earlier model. Train it on the same 50-car environment with:

```powershell
.\.venv\Scripts\python.exe car\rl_baseline.py train --grid --vehicles 50 --steps 10000 --n-envs 4 --seed 42
```

Evaluate it and run the live comparison with:

```powershell
.\.venv\Scripts\python.exe car\rl_baseline.py evaluate --grid --vehicles 50 --episodes 8 --seed-start 101
.\.venv\Scripts\python.exe car\compare_sim.py --seed 101 --vehicles 50
```

The model, checkpoints, DQN evaluation, and paired comparison JSON are saved under `car/artifacts/`. JEV calls OpenRouter during the live simulation; `.env` should contain `OPENROUTER_API_KEY` and must stay private.

## Standard DQN experiment

The non-grid DQN remains available with normalized Kinematics observations and five actions:

```powershell
.\.venv\Scripts\python.exe car\rl_baseline.py train --steps 20000 --n-envs 4
.\.venv\Scripts\python.exe car\rl_baseline.py evaluate --episodes 8 --seed-start 101
```

Install its optional dependency with `python -m pip install -r car\requirements-rl.txt`. To run the original JEV display, use `.\.venv\Scripts\python.exe car\car_sim.py`. `car\run_sky_relay.py` keeps the earlier PyFlyt drone experiment. All agents operate only in simulation.


