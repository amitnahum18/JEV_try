"""Shared top-down occupancy grid and fixed-speed lane-only action setup."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box


GRID_ROWS = 5
GRID_COLUMNS = 15
METERS_PER_CELL = 20.0
EGO_COLUMN = GRID_COLUMNS // 2
FIXED_TARGET_SPEED_M_S = 44.0
LANE_CHANGE_PREDICTION_S = 1.5
LANE_CHANGE_SWEEP_S = 0.4
MIN_PREDICTED_LANE_GAP_M = 12.0
MIN_ORIGIN_LANE_GAP_M = 8.0
URGENT_LANE_CHANGE_TTC_S = 3.0
TRAFFIC_SPEED_LIMIT_TTC_S = 7.0
MIN_TRAFFIC_LIMIT_SPEED_M_S = 20.0
TARGET_SPEED_RECOVERY_M_S_PER_STEP = 1.0


def occupancy_grid(env: gym.Env, columns: int = GRID_COLUMNS) -> np.ndarray:
    """Return 5x15 cells: 0 empty, 1 other vehicle, 2 ego.

    Rows are physical left-to-right (screen top-to-bottom); columns are
    distance behind-to-ahead, centered on the ego. Positive distance is ahead.
    """
    base = env.unwrapped
    ego = base.vehicle
    lanes_count = int(base.config["lanes_count"])
    lane_key = tuple(ego.lane_index[:2])
    heading = np.array([np.cos(ego.heading), np.sin(ego.heading)], dtype=float)
    left_axis = np.array([-heading[1], heading[0]], dtype=float)

    row_for_lane = getattr(base, "_occupancy_grid_row_for_lane", None)
    if row_for_lane is None:
        lateral_position = {}
        for lane_id in range(lanes_count):
            lane = base.road.network.get_lane((*lane_key, lane_id))
            lane_position = np.asarray(lane.position(0.0, 0.0), dtype=float)
            lateral_position[lane_id] = float(np.dot(lane_position, left_axis))
        ordered_lanes = sorted(lateral_position, key=lateral_position.get, reverse=True)
        row_for_lane = {lane_id: row for row, lane_id in enumerate(ordered_lanes)}
        base._occupancy_grid_row_for_lane = row_for_lane

    columns = max(3, int(columns))
    ego_column = columns // 2
    grid = np.zeros((GRID_ROWS, columns), dtype=np.int8)
    ego_row = row_for_lane.get(int(ego.lane_index[2]), GRID_ROWS // 2)
    grid[ego_row, ego_column] = 2
    for vehicle in base.road.vehicles:
        if vehicle is ego:
            continue
        lane_id = int(vehicle.lane_index[2])
        row = row_for_lane.get(lane_id)
        if row is None:
            continue
        delta = np.asarray(vehicle.position, dtype=float) - np.asarray(ego.position, dtype=float)
        longitudinal_distance = float(np.dot(delta, heading))
        column = int(round(longitudinal_distance / METERS_PER_CELL)) + ego_column
        if 0 <= column < columns:
            grid[row, column] = 1
    return grid


def grid_action_allowed(grid: np.ndarray, action: int) -> bool:
    """Keep the same small adjacent-lane safety gate for both policies."""
    if action == 1:
        return True
    ego_rows = np.argwhere(grid == 2)
    if len(ego_rows) != 1:
        return False
    ego_row = int(ego_rows[0][0])
    target_row = ego_row + 1 if action == 0 else ego_row - 1 if action == 2 else -1
    if not 0 <= target_row < GRID_ROWS:
        return False
    ego_column = grid.shape[1] // 2
    nearby = grid[target_row, max(0, ego_column - 1):min(grid.shape[1], ego_column + 2)]
    return not bool(np.any(nearby == 1))


def lane_action_choices(grid: np.ndarray) -> dict[str, str]:
    """List physically available adjacent lanes; safety is checked separately."""
    ego_cells = np.argwhere(grid == 2)
    if len(ego_cells) != 1:
        return {"stay": "Keep the current lane; the ego cell is unavailable."}
    ego_row = int(ego_cells[0][0])
    actions = {"stay": "Stay in your current lane at the fixed target speed."}
    if ego_row > 0:
        actions["move_up"] = "Request a fast lane change one lane left/up."
    if ego_row < grid.shape[0] - 1:
        actions["move_down"] = "Request a fast lane change one lane right/down."
    return actions


def traffic_forecast(
    env: gym.Env,
    columns: int = GRID_COLUMNS,
    horizons_s: tuple[float, ...] = (0.75, 1.5, 3.0, 4.5),
) -> dict[str, object]:
    """Project the occupancy grid forward at constant current vehicle speeds."""
    base = env.unwrapped
    ego = base.vehicle
    row_for_lane = getattr(base, "_occupancy_grid_row_for_lane", {})
    ego_row = row_for_lane.get(int(ego.lane_index[2]), GRID_ROWS // 2)
    heading = np.array([np.cos(ego.heading), np.sin(ego.heading)], dtype=float)
    ego_column = int(columns) // 2
    ego_speed = float(ego.speed)
    projections: dict[str, list[list[int]]] = {}
    for horizon_s in horizons_s:
        forecast = np.zeros((GRID_ROWS, int(columns)), dtype=np.int8)
        forecast[ego_row, ego_column] = 2
        for other in base.road.vehicles:
            if other is ego:
                continue
            row = row_for_lane.get(int(other.lane_index[2]))
            if row is None:
                continue
            delta = np.asarray(other.position, dtype=float) - np.asarray(ego.position, dtype=float)
            longitudinal = float(np.dot(delta, heading))
            future_gap = longitudinal + (float(other.speed) - ego_speed) * float(horizon_s)
            column = int(round(future_gap / METERS_PER_CELL)) + ego_column
            if 0 <= column < int(columns):
                forecast[row, column] = 1
        projections[f"{horizon_s:.2f}s"] = forecast.tolist()
    return {
        "assumption": "constant current speed and lane for every traffic vehicle",
        "horizons_s": list(horizons_s),
        "occupancy_grids": projections,
    }


def lane_change_safety_summary(
    env: gym.Env,
    available_actions: dict[str, str],
    horizon_s: float = LANE_CHANGE_PREDICTION_S,
    min_gap_m: float = MIN_PREDICTED_LANE_GAP_M,
) -> dict[str, dict[str, float | bool | None]]:
    """Predict front/rear gaps during lane entry without changing vehicle speed."""
    base = env.unwrapped
    ego = base.vehicle
    row_for_lane = getattr(base, "_occupancy_grid_row_for_lane", {})
    ego_row = row_for_lane.get(int(ego.lane_index[2]))
    if ego_row is None:
        return {}

    heading = np.array([np.cos(ego.heading), np.sin(ego.heading)], dtype=float)
    ego_speed = float(ego.speed)
    row_for_action = {"stay": ego_row, "move_up": ego_row - 1, "move_down": ego_row + 1}
    summaries: dict[str, dict[str, float | bool | None]] = {}
    route_horizon_s = 3.0
    projected_by_row: dict[int, tuple[list[float], list[float]]] = {
        row: ([], []) for row in range(GRID_ROWS)
    }
    for other in base.road.vehicles:
        if other is ego:
            continue
        other_row = row_for_lane.get(int(other.lane_index[2]))
        if other_row is None:
            continue
        delta = np.asarray(other.position, dtype=float) - np.asarray(ego.position, dtype=float)
        longitudinal = float(np.dot(delta, heading))
        future_longitudinal = longitudinal + (float(other.speed) - ego_speed) * route_horizon_s
        if longitudinal * future_longitudinal <= 0:
            predicted_gap = 0.0
            side = 0 if longitudinal >= 0 else 1
        else:
            predicted_gap = abs(future_longitudinal)
            side = 0 if future_longitudinal > 0 else 1
        projected_by_row[other_row][side].append(predicted_gap)
    future_open_rows: set[int] = set()
    for row, (front_values, rear_values) in projected_by_row.items():
        front_gap = min(front_values, default=None)
        rear_gap = min(rear_values, default=None)
        if (front_gap is None or front_gap >= min_gap_m) and (rear_gap is None or rear_gap >= min_gap_m):
            future_open_rows.add(row)
    for action in available_actions:
        target_row = row_for_action.get(action, -1)
        if action == "stay":
            summaries[action] = {"safe": True, "front_predicted_gap_m": None,
                                 "rear_predicted_gap_m": None, "horizon_s": horizon_s}
            continue
        if not 0 <= target_row < GRID_ROWS:
            summaries[action] = {"safe": False, "front_predicted_gap_m": None,
                                 "rear_predicted_gap_m": None, "horizon_s": horizon_s}
            continue

        target_lanes = {lane_id for lane_id, row in row_for_lane.items() if row == target_row}
        origin_lanes = {lane_id for lane_id, row in row_for_lane.items() if row == ego_row}
        front_gaps: list[float] = []
        rear_gaps: list[float] = []
        origin_front_gaps: list[float] = []
        for other in base.road.vehicles:
            if other is ego:
                continue
            other_lane = int(other.lane_index[2])
            delta = np.asarray(other.position, dtype=float) - np.asarray(ego.position, dtype=float)
            longitudinal = float(np.dot(delta, heading))
            if abs(longitudinal) < 1e-6:
                continue
            gap = abs(longitudinal)
            other_speed = float(other.speed)
            closing_speed = ego_speed - other_speed if longitudinal > 0 else other_speed - ego_speed
            if other_lane in target_lanes:
                predicted_gap = max(0.0, gap - max(0.0, closing_speed) * horizon_s)
                (front_gaps if longitudinal > 0 else rear_gaps).append(predicted_gap)
            if other_lane in origin_lanes and longitudinal > 0:
                origin_front_gaps.append(max(0.0, gap - max(0.0, closing_speed) * LANE_CHANGE_SWEEP_S))

        nearest_front = min(front_gaps, default=None)
        nearest_rear = min(rear_gaps, default=None)
        nearest_origin_front = min(origin_front_gaps, default=None)
        immediate_safe = (
            all(gap is None or gap >= min_gap_m for gap in (nearest_front, nearest_rear))
            and (nearest_origin_front is None or nearest_origin_front >= MIN_ORIGIN_LANE_GAP_M)
        )
        route_rows = {target_row}
        if target_row > 0:
            route_rows.add(target_row - 1)
        if target_row < GRID_ROWS - 1:
            route_rows.add(target_row + 1)
        future_open_lane_count = len(route_rows & future_open_rows)
        route_viable = future_open_lane_count > 0
        summaries[action] = {
            "safe": immediate_safe and route_viable,
            "immediate_safe": immediate_safe,
            "route_viable": route_viable,
            "future_open_lane_count": future_open_lane_count,
            "route_horizon_s": route_horizon_s,
            "front_predicted_gap_m": round(nearest_front, 1) if nearest_front is not None else None,
            "rear_predicted_gap_m": round(nearest_rear, 1) if nearest_rear is not None else None,
            "origin_front_predicted_gap_m": round(nearest_origin_front, 1)
            if nearest_origin_front is not None else None,
            "horizon_s": horizon_s,
            "sweep_horizon_s": LANE_CHANGE_SWEEP_S,
        }
    return summaries


def lane_change_is_safe(env: gym.Env, action: int) -> bool:
    """Hard safety gate for a proposed lane change, evaluated on the live scene."""
    if action == 1:
        return True
    grid = occupancy_grid(env)
    action_name = {0: "move_down", 2: "move_up"}.get(int(action))
    if action_name is None:
        return False
    choices = lane_action_choices(grid)
    if action_name not in choices:
        return False
    return bool(lane_change_safety_summary(env, {action_name: choices[action_name]})
                .get(action_name, {}).get("safe", False))


def urgent_overtake_action(
    lane_kinematics: dict[str, dict[str, float | None]],
    lane_safety: dict[str, dict[str, float | bool | None]],
    threshold_s: float = URGENT_LANE_CHANGE_TTC_S,
) -> str | None:
    """Provide a last-moment safe lane change if a slow JEV response says to stay."""
    current_ttc = lane_kinematics.get("stay", {}).get("ahead_ttc_s")
    if current_ttc is None or current_ttc > threshold_s:
        return None
    safe_actions = [
        action for action in ("move_up", "move_down")
        if lane_safety.get(action, {}).get("immediate_safe", lane_safety.get(action, {}).get("safe")) is True
    ]
    if not safe_actions:
        return None

    viable_actions = [action for action in safe_actions if lane_safety[action].get("route_viable") is True]
    if viable_actions:
        safe_actions = viable_actions
    elif current_ttc > 1.5:
        return None

    def score(action: str) -> tuple[float, float]:
        facts = lane_safety[action]
        front = facts.get("front_predicted_gap_m")
        rear = facts.get("rear_predicted_gap_m")
        front_score = float(front) if front is not None else 300.0
        rear_score = float(rear) if rear is not None else 300.0
        return min(front_score, rear_score), front_score

    return max(safe_actions, key=score)


def target_speed_for_scene(
    ego_speed_m_s: float,
    cruise_speed_m_s: float,
    lane_kinematics: dict[str, dict[str, float | None]],
    lane_safety: dict[str, dict[str, float | bool | None]],
    lane_change_active: bool = False,
    current_target_speed_m_s: float | None = None,
) -> tuple[float, bool]:
    """Cruise fast when an exit is viable; match a slower lead car when trapped."""
    if lane_change_active or any(
        lane_safety.get(action, {}).get("safe") is True for action in ("move_up", "move_down")
    ):
        target = float(cruise_speed_m_s)
    else:
        current = lane_kinematics.get("stay", {})
        gap = current.get("nearest_ahead_m")
        closing = current.get("ahead_closing_speed_m_s")
        ttc = current.get("ahead_ttc_s")
        if gap is None or closing is None or ttc is None or ttc > TRAFFIC_SPEED_LIMIT_TTC_S or closing <= 0:
            target = float(cruise_speed_m_s)
        else:
            lead_speed = max(0.0, float(ego_speed_m_s) - float(closing))
            following_margin = min(8.0, max(2.0, float(gap) / TRAFFIC_SPEED_LIMIT_TTC_S))
            target = min(float(cruise_speed_m_s), max(MIN_TRAFFIC_LIMIT_SPEED_M_S, lead_speed + following_margin))

    if current_target_speed_m_s is not None and target > current_target_speed_m_s:
        target = min(target, float(current_target_speed_m_s) + TARGET_SPEED_RECOVERY_M_S_PER_STEP)
    return target, target < cruise_speed_m_s - 0.1


def lane_change_in_progress(env: gym.Env) -> bool:
    """True while the ego vehicle is still transitioning to its requested lane."""
    vehicle = env.unwrapped.vehicle
    current = tuple(vehicle.lane_index)
    target = tuple(vehicle.target_lane_index)
    return bool(current[:2] == target[:2] and current[2] != target[2])


def lane_gap_summary(grid: np.ndarray, available_actions: dict[str, str]) -> dict[str, dict[str, int | None]]:
    """Make near/far gaps easy to compare while keeping the full grid as input."""
    ego_cells = np.argwhere(grid == 2)
    if len(ego_cells) != 1:
        return {}
    ego_row, ego_column = (int(value) for value in ego_cells[0])
    row_for_action = {"stay": ego_row, "move_up": ego_row - 1, "move_down": ego_row + 1}
    result: dict[str, dict[str, int | None]] = {}
    for action in available_actions:
        row = row_for_action.get(action, -1)
        if not 0 <= row < grid.shape[0]:
            continue
        ahead = np.flatnonzero(grid[row, ego_column + 1:] == 1)
        behind = np.flatnonzero(grid[row, :ego_column] == 1)
        result[action] = {
            "nearest_ahead_m": int((ahead[0] + 1) * 20) if len(ahead) else None,
            "nearest_behind_m": int((ego_column - behind[-1]) * 20) if len(behind) else None,
        }
    return result


def lane_kinematics_summary(
    env: gym.Env, grid: np.ndarray, available_actions: dict[str, str]
) -> dict[str, dict[str, float | None]]:
    """Summarize closest visible vehicle gaps and closing times by lane action."""
    base = env.unwrapped
    ego = base.vehicle
    ego_cells = np.argwhere(grid == 2)
    row_for_lane = getattr(base, "_occupancy_grid_row_for_lane", {})
    if len(ego_cells) != 1:
        return {}
    ego_row, ego_column = (int(value) for value in ego_cells[0])
    ego_speed = float(ego.speed)
    heading = np.array([np.cos(ego.heading), np.sin(ego.heading)], dtype=float)
    row_for_action = {"stay": ego_row, "move_up": ego_row - 1, "move_down": ego_row + 1}
    summaries: dict[str, dict[str, float | None]] = {}

    for action in available_actions:
        target_row = row_for_action.get(action, -1)
        if not 0 <= target_row < grid.shape[0]:
            continue
        target_lanes = {lane_id for lane_id, row in row_for_lane.items() if row == target_row}
        ahead: list[tuple[float, float]] = []
        behind: list[tuple[float, float]] = []
        for other in base.road.vehicles:
            if other is ego or int(other.lane_index[2]) not in target_lanes:
                continue
            delta = np.asarray(other.position, dtype=float) - np.asarray(ego.position, dtype=float)
            longitudinal = float(np.dot(delta, heading))
            column = int(round(longitudinal / METERS_PER_CELL)) + ego_column
            if not 0 <= column < grid.shape[1]:
                continue
            gap = abs(longitudinal)
            if longitudinal > 0:
                ahead.append((gap, float(other.speed)))
            elif longitudinal < 0:
                behind.append((gap, float(other.speed)))

        nearest_ahead = min(ahead, default=None, key=lambda value: value[0])
        nearest_behind = min(behind, default=None, key=lambda value: value[0])
        ahead_closing = ego_speed - nearest_ahead[1] if nearest_ahead else 0.0
        rear_closing = nearest_behind[1] - ego_speed if nearest_behind else 0.0
        summaries[action] = {
            "nearest_ahead_m": round(nearest_ahead[0], 1) if nearest_ahead else None,
            "ahead_closing_speed_m_s": round(ahead_closing, 1) if nearest_ahead else None,
            "ahead_ttc_s": round(nearest_ahead[0] / ahead_closing, 1)
            if nearest_ahead and ahead_closing > 0.5
            else None,
            "nearest_behind_m": round(nearest_behind[0], 1) if nearest_behind else None,
            "rear_closing_speed_m_s": round(rear_closing, 1) if nearest_behind else None,
            "rear_ttc_s": round(nearest_behind[0] / rear_closing, 1)
            if nearest_behind and rear_closing > 0.5
            else None,
        }
    return summaries


class GridObservationWrapper(gym.Wrapper):
    """Replace HighwayEnv's feature list with a flat occupancy matrix for DQN."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._last_grid: np.ndarray | None = None
        self.observation_space = Box(
            low=0.0,
            high=2.0,
            shape=(GRID_ROWS * GRID_COLUMNS,),
            dtype=np.float32,
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        _observation, info = self.env.reset(seed=seed, options=options)
        self.env.unwrapped.vehicle.target_speed = FIXED_TARGET_SPEED_M_S
        self._last_grid = occupancy_grid(self.env)
        return self._last_grid.astype(np.float32).reshape(-1), info

    def step(self, action):
        current_grid = self._last_grid if self._last_grid is not None else occupancy_grid(self.env)
        if not grid_action_allowed(current_grid, int(action)):
            action = 1
        _observation, reward, terminated, truncated, info = self.env.step(action)
        self._last_grid = occupancy_grid(self.env)
        return (
            self._last_grid.astype(np.float32).reshape(-1),
            reward,
            terminated,
            truncated,
            info,
        )
