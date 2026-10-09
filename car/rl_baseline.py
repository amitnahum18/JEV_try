"""Train and evaluate a DQN baseline on the same five-lane HighwayEnv task."""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import gymnasium as gym
import highway_env  # noqa: F401 - registers the HighwayEnv tasks
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from grid_world import FIXED_TARGET_SPEED_M_S, GridObservationWrapper


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "artifacts" / "dqn_five_lane"
GRID_MODEL_PATH = ROOT / "artifacts" / "dqn_grid_five_lane"
TARGET_SPEEDS_M_S = [10.0, 15.666667, 21.333333, 27.0, 32.666667, 38.333333, 44.0]


def make_env(
    render_mode: str | None = None,
    screen_size: tuple[int, int] | None = None,
    vehicles_count: int = 24,
    grid_observation: bool = False,
    lane_only: bool = False,
    simulation_frequency: int | None = None,
) -> gym.Env:
    """Use identical traffic, observations, actions, and reward in train/eval."""
    config = {
        "lanes_count": 5,
        "vehicles_count": int(vehicles_count),
        "vehicles_density": 1.0,
        "initial_lane_id": 2,
        "duration": 60,
        "simulation_frequency": simulation_frequency or (9 if grid_observation else 15),
        "policy_frequency": 3,
        "action": {
            "type": "DiscreteMetaAction",
            "longitudinal": not lane_only,
            "lateral": True,
            "target_speeds": TARGET_SPEEDS_M_S,
        },
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 15,
            "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
            "absolute": False,
            "normalize": True,
        },
        "reward_speed_range": [20, 44],
        "collision_reward": -10.0,
        "right_lane_reward": 0.0,
        "offroad_terminal": False,
    }
    if screen_size:
        config.update({
            "screen_width": int(screen_size[0]),
            "screen_height": int(screen_size[1]),
            "centering_position": [0.5, 0.5],
            "scaling": 5.5,
        })
    env = gym.make(
        "highway-v0",
        render_mode=render_mode,
        config=config,
    )
    return GridObservationWrapper(env) if grid_observation else env


def train(
    steps: int,
    seed: int,
    n_envs: int = 4,
    grid: bool = False,
    vehicles_count: int = 24,
    resume: Path | None = None,
    simulation_frequency: int | None = None,
) -> Path:
    # Separate simulator processes remove most Python-side stepping overhead.
    # Keep this factory at module scope so Windows' spawn mode can import it.
    model_path = GRID_MODEL_PATH if grid else MODEL_PATH
    env_factory = partial(
        make_env,
        grid_observation=grid,
        lane_only=grid,
        vehicles_count=vehicles_count,
        simulation_frequency=simulation_frequency or (9 if grid else 15),
    )
    env = SubprocVecEnv([env_factory for _ in range(n_envs)], start_method="spawn")
    try:
        checkpoint_callback = CheckpointCallback(
            save_freq=max(1, 5_000 // n_envs),
            save_path=str(model_path.parent / "checkpoints" / ("grid" if grid else "standard")),
            name_prefix=model_path.name,
        )
        if resume is not None:
            model = DQN.load(str(resume), env=env, device="auto")
        else:
            model = DQN(
                "MlpPolicy",
                env,
                policy_kwargs={"net_arch": [256, 256]},
                learning_rate=5e-4,
                buffer_size=50_000,
                learning_starts=500,
                batch_size=64,
                gamma=0.95,
                train_freq=1,
                gradient_steps=1,
                target_update_interval=500,
                exploration_fraction=0.2,
                exploration_final_eps=0.05,
                verbose=1,
                seed=seed,
                device="auto",
            )
        env.seed(seed)
        model.learn(
            total_timesteps=steps,
            callback=checkpoint_callback,
            progress_bar=False,
            reset_num_timesteps=resume is None,
        )
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(model_path))
        metadata = {
            "algorithm": "Stable-Baselines3 DQN",
            "environment": "highway-v0",
            "policy_input": "5x15 occupancy matrix (0 empty, 1 traffic, 2 ego)" if grid else "15x7 normalized Kinematics",
            "grid_observation": grid,
            "lane_only_actions": grid,
            "fixed_target_speed_m_s": FIXED_TARGET_SPEED_M_S if grid else None,
            "training_traffic_vehicles": vehicles_count,
            "matrix_rows": "physical left-to-right, top-to-bottom",
            "matrix_columns": "140m behind to 140m ahead in 20m cells; ego centered",
            "seed": seed,
            "training_steps": model.num_timesteps,
            "n_envs": n_envs,
            "simulation_frequency": simulation_frequency or (9 if grid else 15),
            "lanes_count": 5,
            "vehicles_count": vehicles_count,
            "vehicles_density": 1.0,
            "observation": "5x15 occupancy grid, flattened; codes 0/1/2" if grid else "Kinematics, 15 vehicles, 7 normalized relative features",
            "action": "DiscreteMetaAction: lane left/idle/lane right; fixed 44 m/s target" if grid else "DiscreteMetaAction with seven target speeds",
            "reward_speed_range_m_s": [20, 44],
            "collision_reward": -10.0,
        }
        model_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return model_path.with_suffix(".zip")
    finally:
        env.close()


def evaluate(
    model_path: Path,
    episodes: int,
    seed_start: int,
    render_mode: str | None = None,
    grid: bool = False,
    vehicles_count: int = 24,
    simulation_frequency: int | None = None,
) -> list[dict]:
    model = DQN.load(str(model_path))
    results = []
    for seed in range(seed_start, seed_start + episodes):
        env = make_env(
            render_mode=render_mode,
            vehicles_count=vehicles_count,
            grid_observation=grid,
            lane_only=grid,
            simulation_frequency=simulation_frequency,
        )
        try:
            _obs, _info = env.reset(seed=seed)
            speeds = []
            rewards = 0.0
            lane_changes = 0
            previous_lane = int(env.unwrapped.vehicle.lane_index[2])
            terminated = truncated = False
            while not (terminated or truncated):
                action, _state = model.predict(_obs, deterministic=True)
                _obs, reward, terminated, truncated, info = env.step(action)
                ego = env.unwrapped.vehicle
                speeds.append(float(ego.speed))
                rewards += float(reward)
                lane = int(ego.lane_index[2])
                lane_changes += int(lane != previous_lane)
                previous_lane = lane
            results.append({
                "seed": seed,
                "mean_speed_m_s": round(float(np.mean(speeds)), 2) if speeds else 0.0,
                "mean_speed_km_h": round(float(np.mean(speeds)) * 3.6, 1) if speeds else 0.0,
                "reward": round(rewards, 2),
                "lane_changes": lane_changes,
                "collision": bool(info.get("crashed", False)),
                "completed": bool(truncated and not info.get("crashed", False)),
                "steps": len(speeds),
            })
            print(json.dumps(results[-1]))
        finally:
            env.close()
    speeds = [row["mean_speed_m_s"] for row in results]
    collisions = sum(row["collision"] for row in results)
    summary = {
        "episodes": len(results),
        "mean_speed_m_s": round(float(np.mean(speeds)), 2) if speeds else 0.0,
        "collision_rate": collisions / len(results) if results else 0.0,
        "completion_rate": sum(row["completed"] for row in results) / len(results) if results else 0.0,
        "total_lane_changes": sum(row["lane_changes"] for row in results),
    }
    print("SUMMARY " + json.dumps(summary))
    (ROOT / "artifacts").mkdir(parents=True, exist_ok=True)
    (ROOT / "artifacts" / f"dqn_eval_{seed_start}_{len(results)}.json").write_text(
        json.dumps({"summary": summary, "episodes": results}, indent=2), encoding="utf-8"
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train", help="train a DQN baseline")
    train_parser.add_argument("--steps", type=int, default=20_000)
    train_parser.add_argument("--seed", type=int, default=1)
    train_parser.add_argument("--n-envs", type=int, default=4)
    train_parser.add_argument("--grid", action="store_true", help="train on occupancy grid and lane-only actions")
    train_parser.add_argument("--vehicles", type=int, default=None, help="traffic vehicles (default: 50 for grid, 24 otherwise)")
    train_parser.add_argument("--resume", type=Path, default=None, help="resume from a saved DQN checkpoint")
    train_parser.add_argument("--simulation-frequency", type=int, default=None)
    eval_parser = subparsers.add_parser("evaluate", help="evaluate the saved DQN")
    eval_parser.add_argument("--grid", action="store_true", help="evaluate the grid/lane-only model")
    eval_parser.add_argument("--model", type=Path, default=None)
    eval_parser.add_argument("--vehicles", type=int, default=None)
    eval_parser.add_argument("--simulation-frequency", type=int, default=None)
    eval_parser.add_argument("--episodes", type=int, default=8)
    eval_parser.add_argument("--seed-start", type=int, default=101)
    eval_parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    if args.command == "train":
        grid = args.grid
        vehicles_count = args.vehicles if args.vehicles is not None else (50 if grid else 24)
        simulation_frequency = args.simulation_frequency if args.simulation_frequency is not None else (9 if grid else 15)
        model_path = GRID_MODEL_PATH if grid else MODEL_PATH
        print(f"Training DQN for {args.steps:,} steps across {args.n_envs} environments; model output: {model_path}.zip")
        print(f"Saved model: {train(args.steps, args.seed, args.n_envs, grid, vehicles_count, args.resume, simulation_frequency)}")
    else:
        model_path = args.model or (GRID_MODEL_PATH if args.grid else MODEL_PATH).with_suffix(".zip")
        vehicles_count = args.vehicles if args.vehicles is not None else (50 if args.grid else 24)
        simulation_frequency = args.simulation_frequency if args.simulation_frequency is not None else (9 if args.grid else 15)
        evaluate(model_path, args.episodes, args.seed_start, "human" if args.render else None, args.grid, vehicles_count, simulation_frequency)


if __name__ == "__main__":
    main()
