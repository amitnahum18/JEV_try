"""Retrieve nearby DQN examples to guide, but not replace, JEV decisions."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent
EXAMPLES_PATH = ROOT / "artifacts" / "dqn_examples.json"
GAP_CAP_M = 250.0
SPEED_DELTA_CAP_M_S = 20.0


def _number(value: Any, fallback: float) -> float:
    return float(value) if isinstance(value, (int, float)) else fallback


def state_vector(state: dict[str, Any]) -> np.ndarray:
    """Compact scale-normalized representation shared by collection and lookup."""
    current_lane = int(state["lane_index_from_left"])
    vector = [current_lane / 4.0, _number(state.get("ego_speed_m_s"), 0.0) / 44.0]
    for lane in state["lane_context"]:
        front_gap = min(GAP_CAP_M, max(0.0, _number(lane.get("front_gap_m"), GAP_CAP_M)))
        rear_gap = min(GAP_CAP_M, max(0.0, _number(lane.get("rear_gap_m"), GAP_CAP_M)))
        ego_speed = _number(state.get("ego_speed_m_s"), 0.0)
        front_speed = _number(lane.get("front_speed_m_s"), ego_speed)
        rear_speed = _number(lane.get("rear_speed_m_s"), ego_speed)
        front_delta = np.clip(front_speed - ego_speed, -SPEED_DELTA_CAP_M_S, SPEED_DELTA_CAP_M_S)
        rear_delta = np.clip(rear_speed - ego_speed, -SPEED_DELTA_CAP_M_S, SPEED_DELTA_CAP_M_S)
        vector.extend((front_gap / GAP_CAP_M, front_delta / SPEED_DELTA_CAP_M_S,
                       rear_gap / GAP_CAP_M, rear_delta / SPEED_DELTA_CAP_M_S))
    return np.asarray(vector, dtype=np.float32)


@lru_cache(maxsize=1)
def _load_records() -> list[dict[str, Any]]:
    if not EXAMPLES_PATH.exists():
        return []
    return json.loads(EXAMPLES_PATH.read_text(encoding="utf-8"))


def nearest_demonstrations(state: dict[str, Any], count: int = 3) -> list[dict[str, Any]]:
    """Return a few concise, same-lane DQN examples near the current traffic state."""
    try:
        records = _load_records()
        query = state_vector(state)
        lane = int(state["lane_index_from_left"])
        candidates = [record for record in records if record.get("lane") == lane]
        if not candidates:
            candidates = records
        ranked = sorted(candidates, key=lambda row: float(np.linalg.norm(query - row["features"])))[:count]
        return [
            {
                "similar_situation": row["summary"],
                "dqn_action": row["action"],
                "similarity": round(1.0 / (1.0 + float(np.linalg.norm(query - row["features"]))), 2),
            }
            for row in ranked
        ]
    except (OSError, ValueError, KeyError, TypeError):
        return []
