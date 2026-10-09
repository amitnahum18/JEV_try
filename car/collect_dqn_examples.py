"""Collect compact state/action demonstrations from the trained DQN policy."""

from __future__ import annotations

import argparse
import json

import numpy as np
from stable_baselines3 import DQN

from car_sim import highway_state
from rl_baseline import MODEL_PATH, make_env
from rl_imitation import EXAMPLES_PATH, state_vector


ACTION_NAMES = {
    0: "lane_index_delta=-1",
    1: "hold_lane_and_speed",
    2: "lane_index_delta=+1",
    3: "accelerate",
    4: "slow_down",
}


def compact_summary(state: dict) -> dict:
    lanes = []
    for lane in state["lane_context"]:
        lanes.append([
            round(float(lane["front_gap_m"]), 1) if isinstance(lane["front_gap_m"], (int, float)) else None,
            round(float(lane["front_speed_m_s"]), 1) if isinstance(lane["front_speed_m_s"], (int, float)) else None,
            round(float(lane["rear_gap_m"]), 1) if isinstance(lane["rear_gap_m"], (int, float)) else None,
            round(float(lane["rear_speed_m_s"]), 1) if isinstance(lane["rear_speed_m_s"], (int, float)) else None,
        ])
    return {
        "lane_index": int(state["lane_index_from_left"]),
        "ego_speed_m_s": round(float(state["ego_speed_m_s"]), 1),
        "lanes_front_gap_speed_rear_gap_speed": lanes,
    }


def collect(model_path: str, steps: int, seed: int, sample_every: int) -> int:
    model = DQN.load(model_path)
    env = make_env()
    records = []
    try:
        observation, _info = env.reset(seed=seed)
        sim_time = 0.0
        for step in range(steps):
            state = highway_state(env, sim_time)
            if step % sample_every == 0:
                action, _hidden = model.predict(observation, deterministic=True)
                action_id = int(np.asarray(action).item())
                records.append({
                    "features": state_vector(state).tolist(),
                    "lane": int(state["lane_index_from_left"]),
                    "summary": compact_summary(state),
                    "action": ACTION_NAMES[action_id],
                    "seed": seed,
                    "step": step,
                })
            else:
                action, _hidden = model.predict(observation, deterministic=True)
                action_id = int(np.asarray(action).item())
            observation, _reward, terminated, truncated, _info = env.step(action_id)
            sim_time += 1.0 / float(env.unwrapped.config["policy_frequency"])
            if terminated or truncated:
                observation, _info = env.reset(seed=seed + 1 + step)
                sim_time = 0.0
    finally:
        env.close()
    EXAMPLES_PATH.parent.mkdir(parents=True, exist_ok=True)
    EXAMPLES_PATH.write_text(json.dumps(records, separators=(",", ":")), encoding="utf-8")
    print(f"Saved {len(records)} DQN examples to {EXAMPLES_PATH}")
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(MODEL_PATH.with_suffix(".zip")))
    parser.add_argument("--steps", type=int, default=2_400)
    parser.add_argument("--seed", type=int, default=401)
    parser.add_argument("--sample-every", type=int, default=3)
    args = parser.parse_args()
    collect(args.model, args.steps, args.seed, args.sample_every)


if __name__ == "__main__":
    main()
