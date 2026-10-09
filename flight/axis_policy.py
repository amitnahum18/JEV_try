"""Matched single-JEV and simultaneous two-JEV navigation policies."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "car"))
from jev_client import MODEL, choose_action
from course import VERTICAL_ACTIONS, HORIZONTAL_ACTIONS, JOINT_ACTIONS
from navigation import NAVIGATION_INSTRUCTIONS
from maneuver_context import compact_state, FOCUSED_INSTRUCTIONS, BUFFERED_INSTRUCTIONS

OBJECTIVE = (
    "Pilot a fixed-wing aircraft through the next numbered rectangular gate, then "
    "the remaining gates in order. Avoid the frame and ground. Maximize gates passed. "
    "Use the provided physical state and gate geometry, account for momentum and "
    "attitude response, start turns early and stop or reverse before overshooting. "
    "World +x follows the course, +y is LEFT, +z is UP. Euler pitch positive means "
    "nose DOWN and Euler roll negative means bank LEFT. Decisions last the stated "
    "duration. Pitch and bank commands execute simultaneously through the same "
    "attitude stabilizer. Neither nominal pitch nor wings-level locks altitude or "
    "heading. Banking may affect lift and vertical motion. Use all available relevant "
    "measurements; any constant-velocity crossing forecast is only an estimate, "
    "not a prediction of your next maneuver. The stabilizer has no gate knowledge. "
)
FIELD_REFERENCES = (
    "Base the choice on these specific state fields: `aircraft.position_world_xyz_m`, "
    "`aircraft.velocity_world_xyz_m_s`, `aircraft.roll_pitch_yaw_deg`, "
    "`next_gates[0].center_m`, `next_gates[0].relative_world_xyz_m`, "
    "`next_gates[0].half_width_m`, `next_gates[0].half_height_m`, and `controls`. "
    "The first item of `next_gates` is the immediate priority. Positive second "
    "component of `next_gates[0].relative_world_xyz_m` places that gate to your left, "
    "negative to your right. Positive third component places it above, negative below. "
    "Choose the maneuver from `simultaneous_axes` that best aligns current motion "
    "with this opening, accounting for velocity and attitude rather than position alone. "
)


def policy_state(observation: dict, telemetry: str, history: list[dict], interior_margin_m: float = 0., include_pitch_context: bool = False, include_pitch_center: bool = False) -> dict:
    if telemetry in {"full", "previous_full", "focused"}:
        state = json.loads(json.dumps(observation))
        if telemetry == "previous_full":
            state.pop("navigation", None)
    elif telemetry == "basic":
        state = {key: observation[key] for key in (
            "task", "time_seconds", "gates_passed", "gates_total", "previous_action",
            "controls", "pass_rule", "environment")}
        state["aircraft"] = {key: observation["aircraft"][key] for key in (
            "position_world_xyz_m", "velocity_world_xyz_m_s", "roll_pitch_yaw_deg")}
        state["next_gates"] = [{key: gate[key] for key in (
            "number", "center_m", "half_width_m", "half_height_m",
            "relative_world_xyz_m", "distance_m")} for gate in observation["next_gates"]]
        # Keep command descriptions/conventions identical; remove this derived dynamic estimate.
        state["controls"] = dict(state["controls"])
        state["controls"].pop("ideal_turn_radius_at_25_deg_m", None)
    else:
        raise ValueError(f"Unknown telemetry: {telemetry}")
    # Same history fields in both groups, without leaking rich feedback into basic input.
    state["recent_maneuvers"] = [{"action": row["action"], "time_seconds": row["time_seconds"]}
                                  for row in history[-4:]]
    vertical_actions = dict(VERTICAL_ACTIONS)
    up = observation["controls"].get("pitch_up_nose_up_deg", 6)
    down = observation["controls"].get("pitch_down_nose_up_deg", -6)
    if up != 6 or down != -6:
        vertical_actions["up"] = f"Target nose-up pitch {up:g} degrees to climb."
        vertical_actions["down"] = f"Target nose-down pitch {abs(down):g} degrees to descend."
    if observation["controls"].get("extended_pitch_actions"):
        vertical_actions["up_strong"] = f"Target strong nose-up pitch {observation['controls']['strong_pitch_up_nose_up_deg']:g} degrees."
        vertical_actions["down_strong"] = f"Target strong nose-down pitch {abs(observation['controls']['strong_pitch_down_nose_up_deg']):g} degrees."
    if observation["controls"].get("pitch_targets_nose_up_deg"):
        vertical_actions = {key: f"Fixed nose-up pitch target {angle:g} degrees; negative means nose DOWN."
                            for key, angle in observation["controls"]["pitch_targets_nose_up_deg"].items()}
    state["simultaneous_axes"] = {"vertical": vertical_actions, "horizontal": HORIZONTAL_ACTIONS}
    if "navigation" in state:
        previous_gates = history[-1].get("state", {}).get("next_gates", []) if history else []
        previous_gate = previous_gates[0]["number"] if previous_gates else None
        if previous_gate is None and history:
            previous_gate = (history[-1].get("state", {}).get("target") or {}).get("number")
        target = state["navigation"]["target_update"]
        target.setdefault("active_gate_number", state["next_gates"][0]["number"] if state["next_gates"] else None)
        target["previous_decision_gate_number"] = previous_gate
        target["changed_since_previous_decision"] = previous_gate is not None and previous_gate != target["active_gate_number"]
    return compact_state(state, history, interior_margin_m, include_pitch_context, include_pitch_center) if telemetry == "focused" else state


def requests_for(state: dict, architecture: str, prompt_version: int = 3) -> dict:
    if prompt_version not in (1, 2, 3, 4, 5, 6, 7, 8):
        raise ValueError("Unknown prompt version")
    objective = OBJECTIVE + (FIELD_REFERENCES if prompt_version >= 2 else "")
    if prompt_version >= 3:
        objective += NAVIGATION_INSTRUCTIONS
    if prompt_version >= 4:
        if "target" not in state:
            raise ValueError("Prompt 4/5 requires focused telemetry")
        objective = BUFFERED_INSTRUCTIONS if prompt_version >= 5 else FOCUSED_INSTRUCTIONS
    if prompt_version == 6:
        objective += (
            "Evaluate BOTH axes independently from the opening gaps. Positive "
            "`target.vertical_gap_to_interior_up_deg` supports UP; negative supports DOWN. "
            "A zero vertical gap supports LEVEL. A horizontal turn does not cancel the "
            "need for a simultaneous climb or descent. Use the state conditions in each "
            "choice's criteria, not just the maneuver name. Keep corrections timely even "
            "when their angle is small. A pitch target changes gradually; do not assume "
            "level holds current climb angle or altitude. "
        )
    pitch_targets = state.get("control", {}).get("pitch_up_level_down_nose_up_deg")
    if pitch_targets is None:
        controls = state.get("controls", {})
        pitch_targets = [controls.get("pitch_up_nose_up_deg", 6), 2, controls.get("pitch_down_nose_up_deg", -6)]
    up, _, down = pitch_targets
    vertical_actions = dict(VERTICAL_ACTIONS)
    joint_actions = JOINT_ACTIONS
    if up != 6 or down != -6:
        vertical_actions["up"] = f"Target nose-up pitch {up:g} degrees to climb."
        vertical_actions["down"] = f"Target nose-down pitch {abs(down):g} degrees to descend."
    strong_pitch = state.get("control", {}).get("strong_pitch_up_down_nose_up_deg")
    if strong_pitch is None and state.get("controls", {}).get("extended_pitch_actions"):
        strong_pitch = [state["controls"]["strong_pitch_up_nose_up_deg"], state["controls"]["strong_pitch_down_nose_up_deg"]]
    if strong_pitch:
        vertical_actions["up_strong"] = f"Target strong nose-up pitch {strong_pitch[0]:g} degrees; a steeper climb with greater speed and lift demand."
        vertical_actions["down_strong"] = f"Target strong nose-down pitch {abs(strong_pitch[1]):g} degrees; a faster descent requiring earlier recovery."
    target_table = state.get("control", {}).get("pitch_targets_nose_up_deg") or state.get("controls", {}).get("pitch_targets_nose_up_deg")
    if target_table:
        vertical_actions = {key: f"Target nose-up pitch {angle:g} degrees (negative means nose DOWN)." for key, angle in target_table.items()}
    if up != 6 or down != -6 or strong_pitch or target_table:
        joint_actions = {f"{v}__{h}": f"{vtext} Simultaneously: {htext}"
                         for v, vtext in vertical_actions.items() for h, htext in HORIZONTAL_ACTIONS.items()}
    if architecture == "single":
        specs = {"joint": (joint_actions, f"Choose one joint pitch-and-bank maneuver from the {len(joint_actions)} choices.")}
    elif architecture == "parallel":
        specs = {
            "vertical": (vertical_actions, "You control pitch only. Choose up, down or level. Another JEV simultaneously chooses bank using this same state. Its new choice is unknown to you."),
            "horizontal": (HORIZONTAL_ACTIONS, "You control bank only. Choose straight, left, right, roll_left or roll_right. Another JEV simultaneously chooses pitch using this same state. Its new choice is unknown to you."),
        }
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    if prompt_version in (6, 7, 8) and architecture == "single":
        pitch_conditions = {
            "up": f"`target.vertical_gap_to_interior_up_deg` is positive: current motion crosses below the preferred opening. Climb with nose-up target {up:g} degrees.",
            "down": f"`target.vertical_gap_to_interior_up_deg` is negative: current motion crosses above the preferred opening. Descend with nose-down target {abs(down):g} degrees.",
            "level": "`target.vertical_gap_to_interior_up_deg` is zero or near zero: current vertical direction passes inside the preferred opening. Nominal nose-up trim 2 degrees; keep accounting for response lag.",
        }
        if strong_pitch:
            pitch_conditions["up"] += " Prefer this gentle correction for a small positive gap or near alignment."
            pitch_conditions["down"] += " Prefer this gentle correction for a small negative gap or near alignment."
            pitch_conditions["up_strong"] = f"`target.vertical_gap_to_interior_up_deg` is substantially positive, roughly above 3 degrees, or the next high opening requires a large timely climb. Strong nose-up target {strong_pitch[0]:g} degrees. Account for measured climb response and airspeed; release before overshooting the opening."
            pitch_conditions["down_strong"] = f"`target.vertical_gap_to_interior_up_deg` is substantially negative, roughly below -3 degrees, or the next low opening requires a large timely descent. Strong nose-down target {abs(strong_pitch[1]):g} degrees. Account for measured descent response and altitude; recover before overshooting the opening."
        for key in vertical_actions:
            if key not in pitch_conditions:
                pitch_conditions[key] = vertical_actions[key] + " Use the measured vertical gap and recent response to make an intermediate correction without overshooting."
        if prompt_version >= 7:
            objective += (
                "Pitch sustains vertical motion. A zero vertical gap means the CURRENT climb/descent path is safe; "
                "it does NOT mean choose level. Level targets +2 degrees nose-up and will arrest a descent or climb. "
                "Use `target.approximate_pitch_window_nose_up_deg` and the pitch target table to choose a pitch "
                "that keeps the future flight path inside the opening. The window uses current measured pitch-path "
                "difference and is approximate, particularly during a turn. Correct toward its interior early, "
                "using response history and measured path-angle rate. Maintain a suitable descent/climb pitch "
                "when motion aligns instead of alternating strong pitch and level. Account for inertia and speed. "
                "When an actual safe crossing is imminent prioritize keeping it safe over recentering. "
                "Choose bank independently using the horizontal gaps and leveling estimate. "
            )
            pitch_conditions = {key: f"Fixed nose-up pitch target {angle:g} degrees. Choose when this target best sustains or corrects motion toward the interior of `target.approximate_pitch_window_nose_up_deg`, accounting for actual flight-path rate, bank and measured recent responses. Negative targets sustain descent; positive targets can sustain climb. +2 trim is not a hold of current vertical velocity."
                                for key, angle in target_table.items()}
        if prompt_version == 8:
            objective += (
                "Treat `target.approximate_center_pitch_nose_up_deg` as a signed geometric pitch estimate. "
                "Positive means nose UP; negative means nose DOWN. Select a pitch target close to this estimate "
                "using the stated numerical ranges. It is recalculated from the CURRENT gate and measured "
                "pitch/path difference every decision. Do not keep an old pitch when this estimate changes. "
                "If the vertical gap becomes increasingly negative, select a stronger DOWN correction; if "
                "increasingly positive, select stronger UP. Pitch and bank act simultaneously; a left/right "
                "turn does not justify ignoring the required descent/climb. "
            )
            ordered = sorted(target_table.items(), key=lambda item: item[1])
            pitch_conditions = {}
            for index, (key, angle) in enumerate(ordered):
                low = (ordered[index - 1][1] + angle) / 2 if index else None
                high = (angle + ordered[index + 1][1]) / 2 if index + 1 < len(ordered) else None
                region = (f"greater than or equal to {low:g}" if low is not None else "any negative value") + (f" and less than {high:g}" if high is not None else " with no upper limit")
                pitch_conditions[key] = f"`target.approximate_center_pitch_nose_up_deg` is {region}. Fixed pitch {angle:g} degrees nose-up: {'DESCEND, nose DOWN' if angle < 0 else 'CLIMB, nose UP' if angle > 2 else 'nominal trim'}. Reevaluate this numeric estimate each interval. Account for response lag and keep actual opening safe near crossing."
        bank_conditions = {
            "straight": "`target.horizontal_gap_to_interior_if_leveled_left_deg` is zero or near zero: leveling aligns with the preferred opening. Release bank toward zero degrees.",
            "left": "`target.horizontal_gap_to_interior_left_deg` is positive and modest: a moderate left turn corrects the remaining gap. Target left bank 25 degrees.",
            "right": "`target.horizontal_gap_to_interior_left_deg` is negative and modest: a moderate right turn corrects the remaining gap. Target right bank 25 degrees.",
            "roll_left": "`target.horizontal_gap_to_interior_left_deg` is substantially positive (roughly above 15 degrees): start or sustain a strong left turn, target left bank 40 degrees. Account for measured inertia before release.",
            "roll_right": "`target.horizontal_gap_to_interior_left_deg` is substantially negative (roughly below -15 degrees): start or sustain a strong right turn, target right bank 40 degrees. Account for measured inertia before release.",
        }
        specs = {"joint": ({key: pitch_conditions[key.split("__")[0]] + " Simultaneously: " + bank_conditions[key.split("__")[1]]
                            for key in joint_actions}, specs["joint"][1])}
    return {axis: {"model": MODEL, "state": state, "questions": {"action": {
        "type": "choice", "instructions": objective + instruction, "criteria": actions}}}
        for axis, (actions, instruction) in specs.items()}


def decide(requests: dict) -> dict:
    def call(payload: dict):
        question = payload["questions"]["action"]
        return choose_action(state=payload["state"], actions=question["criteria"],
                             instructions=question["instructions"], timeout_seconds=20.)

    started = time.perf_counter()
    if len(requests) == 1:
        decisions = {"joint": call(requests["joint"])}
        action = decisions["joint"].choice
    else:
        # Actual concurrent API calls on the exact same frozen simulator snapshot.
        # Both must succeed. There is no automatic action fallback or route planner.
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {axis: pool.submit(call, payload) for axis, payload in requests.items()}
            decisions = {axis: future.result() for axis, future in futures.items()}
        action = decisions["vertical"].choice + "__" + decisions["horizontal"].choice
    elapsed = time.perf_counter() - started
    costs = [decision.cost_usd for decision in decisions.values()]
    tokens = [decision.input_tokens for decision in decisions.values()]
    return {"action": action, "confidence": min(d.confidence for d in decisions.values()),
            "confidence_note": "Minimum component confidence; not a calibrated joint probability.",
            "latency_seconds": elapsed, "api_calls": len(decisions),
            "cost_usd": sum(costs) if all(c is not None for c in costs) else None,
            "input_tokens": sum(tokens) if all(t is not None for t in tokens) else None,
            "axis_decisions": {axis: asdict(d) for axis, d in decisions.items()},
            "request_bytes": sum(len(json.dumps(p).encode("utf-8")) for p in requests.values())}
