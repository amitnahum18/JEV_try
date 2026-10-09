"""Evaluate the current JEV prompt over multiple fixed traffic seeds."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from grid_jev import GridJevPolicy, describe_lane_options
from grid_world import (
    FIXED_TARGET_SPEED_M_S,
    GRID_ROWS,
    lane_action_choices,
    lane_change_is_safe,
    lane_change_in_progress,
    lane_kinematics_summary,
    lane_change_safety_summary,
    occupancy_grid,
    traffic_forecast,
    target_speed_for_scene,
    urgent_overtake_action,
)
from rl_baseline import make_env


GRID_COLUMNS = 31
SIM_STEP_S = 1.0 / 3.0
OUTPUT_PATH = Path(__file__).resolve().parent / "artifacts" / "jev_prompt_eval_gridwide.json"


def evaluate_seed(seed: int, vehicles: int, max_steps: int) -> dict:
    env = make_env(
        render_mode="rgb_array",
        vehicles_count=vehicles,
        grid_observation=False,
        lane_only=True,
        simulation_frequency=9,
    )
    env.reset(seed=seed)
    env.unwrapped.vehicle.target_speed = FIXED_TARGET_SPEED_M_S
    driver = GridJevPolicy()
    speeds: list[float] = []
    targets: list[float] = []
    previous_lane = int(env.unwrapped.vehicle.lane_index[2])
    lane_changes = 0
    safety_rejections = 0
    urgent_overtakes = 0
    terminal_info: dict = {}
    done = False

    for _ in range(max_steps):
        grid = occupancy_grid(env, columns=GRID_COLUMNS)
        actions = lane_action_choices(grid)
        changing_lanes = lane_change_in_progress(env)
        if changing_lanes:
            actions = {"stay": "Hold position and finish the current lane change before starting another."}
        lane_safety = lane_change_safety_summary(env, actions)
        lane_kinematics = lane_kinematics_summary(env, grid, actions)
        urgent_action = urgent_overtake_action(lane_kinematics, lane_safety)
        desired_speed, speed_limited = target_speed_for_scene(
            float(env.unwrapped.vehicle.speed), FIXED_TARGET_SPEED_M_S,
            lane_kinematics, lane_safety, changing_lanes,
            current_target_speed_m_s=float(env.unwrapped.vehicle.target_speed),
        )
        env.unwrapped.vehicle.target_speed = desired_speed
        state = {
            "occupancy_grid": grid.tolist(),
            "grid_shape": [GRID_ROWS, GRID_COLUMNS],
            "meters_per_cell": 20,
            "ego_cell": [
                int(np.argwhere(grid == 2)[0][0]) if np.any(grid == 2) else GRID_ROWS // 2,
                GRID_COLUMNS // 2,
            ],
            "fixed_target_speed_m_s": FIXED_TARGET_SPEED_M_S,
            "current_target_speed_m_s": round(desired_speed, 1),
            "speed_mode": "traffic_limited" if speed_limited else "cruise",
            "available_actions": list(actions),
            "lane_change_in_progress": changing_lanes,
            "lane_kinematics": lane_kinematics,
            "lane_change_safety": lane_safety,
            "traffic_forecast": traffic_forecast(env, columns=GRID_COLUMNS),
        }
        new_decision, action_name = driver.update()
        driver.request(state, describe_lane_options(actions, state["lane_kinematics"], lane_safety))
        action_id = {"move_down": 0, "stay": 1, "move_up": 2}.get(action_name, 1) if new_decision else 1
        if changing_lanes:
            action_id = 1
        if action_id != 1 and not lane_change_is_safe(env, action_id):
            safety_rejections += 1
            action_id = 1
        urgent = action_id == 1 and urgent_action is not None and not changing_lanes
        if urgent:
            action_id = {"move_down": 0, "move_up": 2}[urgent_action]
            urgent_overtakes += 1
        _obs, _reward, terminated, truncated, terminal_info = env.step(action_id)
        ego = env.unwrapped.vehicle
        speeds.append(float(ego.speed))
        targets.append(float(ego.target_speed))
        lane = int(ego.lane_index[2])
        lane_changes += int(lane != previous_lane)
        previous_lane = lane
        done = bool(terminated or truncated)
        if done:
            break
        time.sleep(SIM_STEP_S)

    result = {
        "seed": seed,
        "steps": len(speeds),
        "mean_speed_m_s": round(float(np.mean(speeds)), 2) if speeds else 0.0,
        "minimum_target_speed_m_s": round(min(targets), 1) if targets else FIXED_TARGET_SPEED_M_S,
        "speed_limited_steps": sum(target < FIXED_TARGET_SPEED_M_S - 0.1 for target in targets),
        "lane_changes": lane_changes,
        "safety_rejections": safety_rejections,
        "urgent_overtakes": urgent_overtakes,
        "collision": bool(terminal_info.get("crashed", False)),
        "completed": done and not terminal_info.get("crashed", False),
        "jev_decisions": driver.decisions,
    }
    env.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-start", type=int, default=101)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--vehicles", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=120)
    args = parser.parse_args()

    results = [evaluate_seed(args.seed_start + i, args.vehicles, args.max_steps) for i in range(args.seeds)]
    report = {
        "agent": "JEV",
        "grid_shape": [GRID_ROWS, GRID_COLUMNS],
        "lookahead_m": (GRID_COLUMNS // 2) * 20,
        "traffic_vehicles": args.vehicles,
        "seed_start": args.seed_start,
        "episodes": results,
        "collision_rate": round(sum(r["collision"] for r in results) / len(results), 2),
        "mean_speed_m_s": round(float(np.mean([r["mean_speed_m_s"] for r in results])), 2),
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
