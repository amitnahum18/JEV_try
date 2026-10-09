"""JEV chooses continuous driving and overtaking actions on a five-lane road."""

from __future__ import annotations

import queue
import argparse
import json
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import highway_env  # noqa: F401 - registers HighwayEnv tasks with Gymnasium
import numpy as np

from jev_client import Decision, choose_action
from rl_imitation import nearest_demonstrations


ACTION_INDEX = {
    "hold_lane": 1,
    "accelerate": 3,
    "slow_down": 4,
}
LANE_ACTION_PREFIX = "move_to_lane_"
TARGET_SPEEDS_M_S = [10.0, 15.666667, 21.333333, 27.0, 32.666667, 38.333333, 44.0]
LANE_SPEED_LOOKAHEAD_S = 8.0
LANE_SPEED_CLEAR_GAP_M = 250.0
BASE_ACTIONS = {
    "hold_lane": "Keep the current lane and target speed.",
    "accelerate": "Increase target speed one step toward 44 m/s; choose only when the current lane has room to accelerate.",
    "slow_down": "Reduce target speed one step. Prefer a safe overtake instead when a slower lead vehicle blocks you and a passing lane is open. Brake when no safe pass is available or the gap/TTC is critical.",
}
POLICY_INSTRUCTIONS = (
    "You are the autonomous driving policy in a five-lane, one-way highway simulation. "
    "You choose the driving action. Your objective is to maximize safe average speed by comparing every lane, "
    "not just the lane immediately beside you. Use lane_context's estimated_speed_m_s, speed_gain_m_s, "
    "front/rear gaps, and relative speeds. Speed estimates predict sustainable speed over the next 8 seconds. "
    "recent_history shows whether each gap is opening or closing. Pass a slower lead proactively when a safe "
    "lane offers a meaningful gain; do not wait for TTC to become critical before selecting an overtake. "
    "When a move_to_lane_N action is available, it means every lane change on the route to lane N has a "
    "checked safe gap. Choose the lane with the highest estimated speed if its gain is material; the executor "
    "will move one lane at a time and hold that target until reached. Do not change lanes for a small gain. "
    "If no lane has a clear advantage, accelerate on open road; hold behind slower traffic until a safe route "
    "opens; brake only when TTC is below 8 s or the front gap is below 30 m. "
    "similar_dqn_demonstrations contains the DQN's action in nearby situations; use these examples as a "
    "behavioral reference when relevant, but reject an example if its maneuver is unsafe or unavailable now. "
    "Only select an action present in available_actions. A committed lane target is shown in the state; keep it "
    "unless the route becomes unsafe. The current and target speed are in m/s and km/h."
)


@dataclass
class PolicyResult:
    decision: Decision | None = None
    error: str | None = None


def action_options(state: dict) -> dict[str, str]:
    """Build a safe all-lane candidate set with estimated speed benefit."""
    lead_gap = state["lead_gap_m"]
    lead_relative_speed = state["closing_speed_m_s"]
    ego_speed = state["ego_speed_m_s"]
    front_ttc = state["front_time_to_collision_s"]
    following_risk = (
        isinstance(lead_gap, (float, int)) and lead_gap < 30
    ) or (isinstance(front_ttc, (float, int)) and front_ttc < 8.0)
    lane_context = state["lane_context"]
    current_lane = state["lane_index_from_left"]
    current_target = state.get("committed_target_lane")

    def safe_lane(lane: dict) -> bool:
        front_speed = lane["front_speed_m_s"]
        rear_speed = lane["rear_speed_m_s"]
        front_closing = max(0.0, ego_speed - front_speed) if isinstance(front_speed, (int, float)) else 0.0
        rear_closing = max(0.0, rear_speed - ego_speed) if isinstance(rear_speed, (int, float)) else 0.0
        front_minimum = max(32 + 2 * front_closing, 18 + 6 * front_closing)
        rear_minimum = max(35 + 3 * rear_closing, 25 + 6 * rear_closing)
        front_gap = lane["front_gap_m"]
        rear_gap = lane["rear_gap_m"]
        front_ok = front_gap == "clear" or isinstance(front_gap, (int, float)) and front_gap >= front_minimum
        rear_ok = rear_gap == "clear" or isinstance(rear_gap, (int, float)) and rear_gap >= rear_minimum
        return front_ok and rear_ok

    def route_is_safe(target_lane: int) -> bool:
        step = 1 if target_lane > current_lane else -1
        return all(safe_lane(lane_context[index]) for index in range(current_lane + step, target_lane + step, step))

    options: dict[str, str] = {"hold_lane": BASE_ACTIONS["hold_lane"]}

    # Finish an already selected route instead of asking JEV to reconsider it
    # on each delayed API response. The simulator executes this committed
    # maneuver while JEV waits for the next real choice point.
    if isinstance(current_target, int) and current_target != current_lane:
        name = f"{LANE_ACTION_PREFIX}{current_target}"
        if route_is_safe(current_target):
            options[name] = f"Continue the previously selected safe route to lane {current_target + 1}."
            return options
        if following_risk and state["current_target_speed_m_s"] > state["min_target_speed_m_s"]:
            options["slow_down"] = BASE_ACTIONS["slow_down"]
        return options

    def lane_speed_potential(lane: dict) -> float:
        gap = lane["front_gap_m"]
        speed = lane["front_speed_m_s"]
        maximum = state["max_target_speed_m_s"]
        if not isinstance(gap, (int, float)) or not isinstance(speed, (int, float)) or gap >= LANE_SPEED_CLEAR_GAP_M:
            return maximum
        usable_gap = max(0.0, float(gap) - 30.0)
        return min(maximum, float(speed) + usable_gap / LANE_SPEED_LOOKAHEAD_S)

    current_potential = lane_speed_potential(lane_context[current_lane])
    pass_urgency = (
        isinstance(lead_gap, (float, int))
        and lead_gap < LANE_SPEED_CLEAR_GAP_M
        and lead_relative_speed > 1.2
    )
    minimum_gain = 2.0 if pass_urgency else 3.0
    for lane_index, lane in enumerate(lane_context):
        if lane_index == current_lane or not route_is_safe(lane_index):
            continue
        potential = lane_speed_potential(lane)
        gain = potential - current_potential
        if gain >= minimum_gain:
            options[f"{LANE_ACTION_PREFIX}{lane_index}"] = (
                f"Select lane {lane_index + 1}; estimated speed potential {potential:.1f} m/s, "
                f"gain {gain:.1f} m/s over this lane. Full route gaps are checked."
                + (" Passing now is preferred over waiting behind the slower lead." if pass_urgency else "")
            )
    if following_risk and state["current_target_speed_m_s"] > state["min_target_speed_m_s"]:
        options["slow_down"] = BASE_ACTIONS["slow_down"]

    lead_is_close = isinstance(lead_gap, (float, int)) and (
        lead_gap < 55 or lead_gap / max(ego_speed, 0.1) < 2.5
    )
    lead_is_closing = (
        isinstance(lead_gap, (float, int)) and lead_gap < 150 and lead_relative_speed > 1.5
    )
    lead_blocking = lead_is_close or lead_is_closing
    if not following_risk and not lead_blocking and state["current_target_speed_m_s"] < state["max_target_speed_m_s"]:
        options["accelerate"] = BASE_ACTIONS["accelerate"]
    return options


class ContinuousJevDriver:
    """Ask JEV asynchronously while HighwayEnv continues to advance."""

    def __init__(self) -> None:
        self.results: queue.Queue[PolicyResult] = queue.Queue()
        self.pending = False
        self.lock = threading.Lock()
        self.action = "hold_lane"
        self.confidence = 0.0
        self.probabilities: dict[str, float] = {}
        self.model_decision = False
        self.latency = 0.0
        self.next_retry_at = 0.0
        self.lane_change_ready_at = 0.0
        self.speed_action_ready_at = 0.0
        self.decisions = 0
        self.state_history: deque[dict] = deque(maxlen=5)
        self.action_history: deque[str] = deque(maxlen=5)
        self.last_decision_at = 0.0
        self.target_lane: int | None = None

    def observe(self, state: dict) -> None:
        """Attach short-term traffic trends and the active maneuver to this decision."""
        point = {
            "time_s": state["simulation_time_seconds"],
            "lane": state["lane_index_from_left"],
            "speed_m_s": state["ego_speed_m_s"],
            "lead_gap_m": state["lead_gap_m"],
            "closing_speed_m_s": state["closing_speed_m_s"],
            "ttc_s": state["front_time_to_collision_s"],
        }
        if not self.state_history or self.state_history[-1]["time_s"] != point["time_s"]:
            self.state_history.append(point)
        state["recent_history"] = list(self.state_history)
        state["recent_jev_actions"] = list(self.action_history)
        state["previous_jev_action"] = self.action
        state["committed_target_lane"] = self.target_lane
        state["maneuver_phase"] = "changing_lane" if self.target_lane is not None else "cruise"
        state["seconds_since_last_decision"] = (
            round(time.monotonic() - self.last_decision_at, 2)
            if self.last_decision_at else "first decision"
        )

    def set_lane_target(self, action: str) -> bool:
        if not action.startswith(LANE_ACTION_PREFIX):
            return False
        try:
            self.target_lane = int(action.removeprefix(LANE_ACTION_PREFIX))
            return True
        except ValueError:
            return False

    def sync_lane_target(self, current_lane: int) -> None:
        if self.target_lane == current_lane:
            self.target_lane = None

    def request(self, state: dict) -> None:
        self.observe(state)
        demonstrations = nearest_demonstrations(state)
        if demonstrations:
            state["similar_dqn_demonstrations"] = demonstrations
        state["lane_change_cooldown"] = time.monotonic() < self.lane_change_ready_at
        if self.target_lane is not None and self.target_lane != state["lane_index_from_left"]:
            return

        options = action_options(state)
        state["available_actions"] = list(options)
        state["action_guidance"] = {
            "move_to_lane_N": "Select the destination lane with the best safe speed potential; the executor traverses one lane at a time.",
            "slow_down": "Reduce the target speed one level to restore headway; choose a safe lane change if one offers a better path.",
            "accelerate": "Raise the target speed one level when the current lane has room.",
            "hold_lane": "Keep the current target speed and lane; compare this against every safe lane and speed option.",
        }
        if len(options) == 1:
            self.action = next(iter(options))
            self.model_decision = False
            return

        with self.lock:
            if self.pending or not self.results.empty() or time.monotonic() < self.next_retry_at:
                return
            self.pending = True

        def ask() -> None:
            try:
                decision = choose_action(
                    state=state,
                    actions=options,
                    instructions=POLICY_INSTRUCTIONS,
                    timeout_seconds=6.0,
                )
                self.results.put(PolicyResult(decision=decision))
            except Exception as exc:
                self.results.put(PolicyResult(error=str(exc)))
            finally:
                with self.lock:
                    self.pending = False

        threading.Thread(target=ask, name="jev-highway-policy", daemon=True).start()

    def update(self) -> bool:
        changed = False
        while True:
            try:
                result = self.results.get_nowait()
            except queue.Empty:
                return changed
            if result.decision:
                self.action = result.decision.choice
                self.confidence = result.decision.confidence
                self.probabilities = result.decision.probabilities
                self.model_decision = True
                self.latency = result.decision.latency_seconds
                self.decisions += 1
                self.last_decision_at = time.monotonic()
                self.action_history.append(self.action)
                changed = True
                probability_text = ", ".join(
                    f"{name}={probability:.0%}"
                    for name, probability in sorted(
                        self.probabilities.items(), key=lambda item: item[1], reverse=True
                    )[:3]
                )
                print(
                    f"JEV #{self.decisions}: {self.action} "
                    f"(confidence {self.confidence:.0%}; {probability_text}; {self.latency:.2f}s)"
                )
            elif result.error:
                self.next_retry_at = time.monotonic() + 4.0
                self.action = "slow_down"
                changed = True
                print(f"JEV unavailable; using slow_down and retrying shortly: {result.error}")


def highway_state(env: gym.Env, sim_time: float) -> dict:
    base = env.unwrapped
    ego = base.vehicle
    lane_index = int(ego.lane_index[2])
    lane_count = int(base.config["lanes_count"])
    target_speeds = [float(speed) for speed in base.action_type.target_speeds]
    heading = np.array([np.cos(ego.heading), np.sin(ego.heading)])
    lane_gaps: dict[int, dict[str, float | str]] = {
        index: {
            "front": float("inf"),
            "rear": float("inf"),
            "front_speed": "clear",
            "rear_speed": "clear",
        }
        for index in range(lane_count)
    }

    for other in base.road.vehicles:
        if other is ego:
            continue
        other_lane = int(other.lane_index[2])
        if other_lane not in lane_gaps:
            continue
        delta = np.asarray(other.position) - np.asarray(ego.position)
        longitudinal_distance = float(np.dot(delta, heading))
        slot = lane_gaps[other_lane]
        if longitudinal_distance >= 0 and longitudinal_distance < slot["front"]:
            slot["front"] = longitudinal_distance
            slot["front_speed"] = round(float(other.speed), 1)
        elif longitudinal_distance < 0 and -longitudinal_distance < slot["rear"]:
            slot["rear"] = -longitudinal_distance
            slot["rear_speed"] = round(float(other.speed), 1)

    for gaps in lane_gaps.values():
        for side in ("front", "rear"):
            if not np.isfinite(gaps[side]):
                gaps[side] = "clear"

    current = lane_gaps[lane_index]
    lead_gap = current["front"]
    lead_speed = current["front_speed"]
    closing_speed = (
        round(float(ego.speed) - float(lead_speed), 1)
        if isinstance(lead_speed, (float, int))
        else 0.0
    )
    front_ttc = (
        round(float(lead_gap) / closing_speed, 1)
        if isinstance(lead_gap, (float, int)) and closing_speed > 0.1
        else "not closing"
    )

    def neighbor_gaps(index: int) -> dict | None:
        if index < 0 or index >= lane_count:
            return None
        return {
            key: round(float(value), 1) if isinstance(value, (float, int)) else value
            for key, value in lane_gaps[index].items()
            if key in {"front", "rear", "front_speed", "rear_speed"}
        }

    maximum_speed = max(target_speeds)

    def speed_potential(gaps: dict) -> float:
        gap = gaps["front"]
        speed = gaps["front_speed"]
        if not isinstance(gap, (int, float)) or not isinstance(speed, (int, float)) or gap >= LANE_SPEED_CLEAR_GAP_M:
            return maximum_speed
        usable_gap = max(0.0, float(gap) - 30.0)
        return min(maximum_speed, float(speed) + usable_gap / LANE_SPEED_LOOKAHEAD_S)

    current_potential = speed_potential(lane_gaps[lane_index])
    lane_context = [
        {
            "lane_index_from_left": index,
            "is_current_lane": index == lane_index,
            "front_gap_m": lane_gaps[index]["front"],
            "front_speed_m_s": lane_gaps[index]["front_speed"],
            "front_closing_speed_m_s": (
                round(float(ego.speed) - float(lane_gaps[index]["front_speed"]), 1)
                if isinstance(lane_gaps[index]["front_speed"], (float, int)) else 0.0
            ),
            "rear_gap_m": lane_gaps[index]["rear"],
            "rear_speed_m_s": lane_gaps[index]["rear_speed"],
            "rear_closing_speed_m_s": (
                round(float(lane_gaps[index]["rear_speed"]) - float(ego.speed), 1)
                if isinstance(lane_gaps[index]["rear_speed"], (float, int)) else 0.0
            ),
            "estimated_speed_m_s": round(speed_potential(lane_gaps[index]), 1),
            "speed_gain_m_s": round(speed_potential(lane_gaps[index]) - current_potential, 1),
        }
        for index in range(lane_count)
    ]

    return {
        "scenario": "five-lane straight highway with moving traffic",
        "simulation_time_seconds": round(sim_time, 1),
        "lane_index_from_left": lane_index,
        "lanes_count": lane_count,
        "ego_speed_m_s": round(float(ego.speed), 1),
        "ego_speed_km_h": round(float(ego.speed) * 3.6, 1),
        "current_target_speed_m_s": round(float(ego.target_speed), 1),
        "current_target_speed_km_h": round(float(ego.target_speed) * 3.6, 1),
        "target_speed_steps_m_s": target_speeds,
        "min_target_speed_m_s": min(target_speeds),
        "max_target_speed_m_s": max(target_speeds),
        "lane_speed_estimate_horizon_s": LANE_SPEED_LOOKAHEAD_S,
        "lead_gap_m": round(float(lead_gap), 1) if isinstance(lead_gap, (float, int)) else "clear",
        "lead_vehicle_speed_m_s": lead_speed,
        "closing_speed_m_s": closing_speed,
        "front_time_to_collision_s": front_ttc,
        "current_lane_time_headway_s": (
            round(float(lead_gap) / max(float(ego.speed), 0.1), 1)
            if isinstance(lead_gap, (float, int)) else "clear"
        ),
        "lane_context": lane_context,
        "left_lane_gaps_m": neighbor_gaps(lane_index - 1),
        "right_lane_gaps_m": neighbor_gaps(lane_index + 1),
        "vehicles_in_scene": len(base.road.vehicles),
        "collision_already_occurred": bool(getattr(ego, "crashed", False)),
    }


def overlay_hud(state: dict, driver: ContinuousJevDriver) -> None:
    try:
        import pygame

        surface = pygame.display.get_surface()
        if surface is None:
            return
        title = pygame.font.Font(None, 28)
        small = pygame.font.Font(None, 22)
        panel = pygame.Rect(12, 12, 600, 142)
        pygame.draw.rect(surface, (12, 20, 29), panel, border_radius=8)
        pygame.draw.rect(surface, (48, 210, 170), panel, 2, border_radius=8)
        lead = state["lead_gap_m"]
        lead_label = f"{lead} m" if isinstance(lead, (float, int)) else str(lead)
        rows = [
            ("JEV DRIVER", title, (240, 246, 250), 18),
            (f"Lane {state['lane_index_from_left'] + 1}/5   {state['ego_speed_m_s']:.1f}/{state['current_target_speed_m_s']:.1f} m/s   Lead {lead_label}", small, (196, 208, 218), 48),
            (f"JEV: {driver.action.replace('_', ' ')}   {'model ' + format(driver.confidence, '.0%') if driver.model_decision else 'only safe choice'}   target {driver.target_lane + 1 if driver.target_lane is not None else '—'}", small, (91, 230, 190), 74),
        ]
        for label, font, color, y in rows:
            surface.blit(font.render(label, True, color), (24, y))
        available = action_options(dict(state))
        available_lanes = {
            int(name.removeprefix(LANE_ACTION_PREFIX))
            for name in available
            if name.startswith(LANE_ACTION_PREFIX)
        }
        for index, lane in enumerate(state["lane_context"]):
            current = lane["is_current_lane"]
            open_for_change = index in available_lanes
            color = (48, 150, 124) if open_for_change else (48, 91, 100) if current else (40, 48, 58)
            card = pygame.Rect(24 + index * 112, 101, 104, 29)
            pygame.draw.rect(surface, color, card, border_radius=5)
            gain = lane["speed_gain_m_s"]
            label = f"{index + 1}: {lane['estimated_speed_m_s']:.0f} ({gain:+.0f})"
            surface.blit(small.render(label, True, (245, 248, 250)), (card.x + 7, card.y + 5))
        pygame.display.flip()
    except Exception:
        pass


def run(render_mode: str | None = "human", max_episodes: int | None = None, seed: int = 1) -> None:
    print("Starting five-lane HighwayEnv.")
    print("JEV continuously manages speed and lane changes; available safe overtakes are chosen automatically.")
    episode_seed = seed
    completed_segments = 0
    episode_results: list[dict] = []
    while max_episodes is None or completed_segments < max_episodes:
        driver = ContinuousJevDriver()
        env = gym.make(
            "highway-v0",
            render_mode=render_mode,
            config={
                "lanes_count": 5,
                "vehicles_count": 24,
                "vehicles_density": 1.0,
                "initial_lane_id": 2,
                "duration": 60,
                "simulation_frequency": 15,
                "policy_frequency": 3,
                "action": {
                    "type": "DiscreteMetaAction",
                    "longitudinal": True,
                    "lateral": True,
                    "target_speeds": TARGET_SPEEDS_M_S,
                },
                "observation": {
                    "type": "Kinematics",
                    "vehicles_count": 15,
                    "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                    "absolute": True,
                },
                "screen_width": 1200,
                "screen_height": 520,
                "centering_position": [0.28, 0.5],
                "scaling": 5.5,
                "reward_speed_range": [20, 44],
                "right_lane_reward": 0.0,
                "offroad_terminal": False,
            },
        )
        try:
            env.reset(seed=episode_seed)
            print(f"Traffic scenario seed: {episode_seed}")
            episode_seed += 1
            sim_time = 0.0
            episode_speeds: list[float] = []
            episode_lane_changes = 0
            previous_lane = int(env.unwrapped.vehicle.lane_index[2])
            driver.request(highway_state(env, sim_time))
            while True:
                new_decision = driver.update()
                state = highway_state(env, sim_time)
                driver.sync_lane_target(state["lane_index_from_left"])
                state["committed_target_lane"] = driver.target_lane
                validation_state = dict(state)
                valid_now = action_options(validation_state)
                action_name = driver.action
                if new_decision:
                    valid_lane_request = False
                    existing_target_lane = driver.target_lane
                    if action_name.startswith(LANE_ACTION_PREFIX):
                        try:
                            requested_lane = int(action_name.removeprefix(LANE_ACTION_PREFIX))
                            if (
                                0 <= requested_lane < state["lanes_count"]
                                and (action_name in valid_now or existing_target_lane is None or requested_lane == existing_target_lane)
                            ):
                                driver.target_lane = requested_lane
                                valid_lane_request = True
                        except ValueError:
                            pass
                    if action_name not in valid_now and not valid_lane_request:
                        action_name = (
                            "slow_down"
                            if "slow_down" in valid_now
                            else "hold_lane"
                            if "hold_lane" in valid_now
                            else next(iter(valid_now))
                        )
                    driver.action = action_name
                if new_decision and driver.decisions <= 15:
                    print(
                        f"State #{driver.decisions}: {driver.action}, "
                        f"speed={state['ego_speed_m_s']:.1f}/{state['current_target_speed_m_s']:.0f}m/s, "
                        f"lane={state['lane_index_from_left'] + 1}/5, "
                        f"lead={state['lead_gap_m']}m, ttc={state['front_time_to_collision_s']}s, "
                        f"options={list(valid_now)}"
                    )
                if not new_decision:
                    action_name = "hold_lane"
                current_lane = state["lane_index_from_left"]
                if driver.target_lane is not None:
                    target_action = f"{LANE_ACTION_PREFIX}{driver.target_lane}"
                    if driver.target_lane == current_lane:
                        driver.target_lane = None
                    elif target_action not in valid_now:
                        pending_lane = int(env.unwrapped.vehicle.target_lane_index[2])
                        if pending_lane != current_lane:
                            # A lateral transition has begun but the route is
                            # no longer safe; steer back to the current lane.
                            driver.target_lane = None
                            action_name = "overtake_left" if pending_lane > current_lane else "return_right"
                            driver.lane_change_ready_at = time.monotonic() + 1.0
                        else:
                            # Keep the requested destination and wait for its
                            # route to become safe before starting the turn.
                            action_name = "slow_down" if "slow_down" in valid_now else "hold_lane"
                    else:
                        pending_lane = int(env.unwrapped.vehicle.target_lane_index[2])
                        if pending_lane != current_lane:
                            # Let the current lateral transition finish before
                            # commanding the next adjacent lane.
                            action_name = "hold_lane"
                        elif time.monotonic() >= driver.lane_change_ready_at:
                            action_name = "overtake_left" if driver.target_lane < current_lane else "return_right"
                            driver.lane_change_ready_at = time.monotonic() + 1.0
                        else:
                            action_name = "hold_lane"
                if driver.target_lane is None:
                    pending_lane = int(env.unwrapped.vehicle.target_lane_index[2])
                    if pending_lane != current_lane:
                        # IDLE/SLOWER do not cancel HighwayEnv's pending lane
                        # target. Reverse it explicitly if its route was masked.
                        action_name = "overtake_left" if pending_lane > current_lane else "return_right"
                        driver.lane_change_ready_at = time.monotonic() + 1.0
                if action_name in {"accelerate", "slow_down"}:
                    now = time.monotonic()
                    if now < driver.speed_action_ready_at:
                        action_name = "hold_lane"
                    else:
                        # A target-speed level represents a sustained decision.
                        # Rate-limit increments so rapid API responses cannot
                        # make the car alternate or jump through speeds unrealistically.
                        driver.speed_action_ready_at = now + 0.9
                driver.request(state)
                ttc = state["front_time_to_collision_s"]
                emergency_headway = (
                    isinstance(ttc, (int, float)) and ttc < 6.0
                ) or (
                    isinstance(state["lead_gap_m"], (int, float))
                    and state["lead_gap_m"] < 30
                )
                if emergency_headway and action_name not in {"overtake_left", "return_right"}:
                    action_name = (
                        "slow_down"
                        if state["current_target_speed_m_s"] > state["min_target_speed_m_s"]
                        else "hold_lane"
                    )
                    driver.action = action_name
                action = ACTION_INDEX.get(action_name, 1)
                if action_name == "overtake_left":
                    action = 0
                elif action_name == "return_right":
                    action = 2
                _observation, _reward, terminated, truncated, info = env.step(action)
                ego = env.unwrapped.vehicle
                episode_speeds.append(float(ego.speed))
                current_lane = int(ego.lane_index[2])
                episode_lane_changes += int(current_lane != previous_lane)
                previous_lane = current_lane
                sim_time += 1.0 / 3.0
                if render_mode == "human":
                    overlay_hud(state, driver)
                    time.sleep(0.25)
                else:
                    time.sleep(0.1)
                if terminated or truncated:
                    reason = "collision" if info.get("crashed") else "episode complete"
                    average_speed = float(np.mean(episode_speeds)) if episode_speeds else 0.0
                    print(
                        f"Drive segment finished ({reason}); mean speed {average_speed:.1f} m/s, "
                        f"lane changes {episode_lane_changes}."
                    )
                    completed_segments += 1
                    episode_results.append({
                        "seed": episode_seed - 1,
                        "mean_speed_m_s": round(average_speed, 2),
                        "lane_changes": episode_lane_changes,
                        "collision": bool(info.get("crashed", False)),
                        "completed": bool(truncated and not info.get("crashed", False)),
                        "steps": len(episode_speeds),
                        "jev_decisions": driver.decisions,
                    })
                    if max_episodes is not None:
                        artifacts = Path(__file__).resolve().parent / "artifacts"
                        artifacts.mkdir(parents=True, exist_ok=True)
                        evaluation = {
                            "episodes": len(episode_results),
                            "seed_start": seed,
                            "mean_speed_m_s": round(
                                float(np.mean([row["mean_speed_m_s"] for row in episode_results])), 2
                            ),
                            "collision_rate": sum(row["collision"] for row in episode_results)
                            / len(episode_results),
                            "completion_rate": sum(row["completed"] for row in episode_results)
                            / len(episode_results),
                            "episode_results": episode_results,
                        }
                        (artifacts / f"jev_eval_{seed}_{len(episode_results)}.json").write_text(
                            json.dumps(evaluation, indent=2), encoding="utf-8"
                        )
                    break
        except KeyboardInterrupt:
            return
        finally:
            env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="run without opening the simulation window")
    parser.add_argument("--episodes", type=int, default=1, help="number of segments; default run stays open")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    run(render_mode=None if args.headless else "human",
        max_episodes=args.episodes if args.headless else None,
        seed=args.seed)
