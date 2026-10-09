"""Compact measured navigation context; no action selection or physics rollout."""
from __future__ import annotations

import math

from navigation import rounded

FOCUSED_SCHEMA = "flight_opening_context_v1"
BUFFERED_SCHEMA = "flight_opening_context_v2"
PITCH_SCHEMA = "flight_opening_context_v3"


def window_gap(angle: float | None, window: list | None) -> float | None:
    """Signed correction to the nearest edge of an unwrapped angular interval."""
    if angle is None or window is None:
        return None
    return rounded(window[0] - angle if angle < window[0] else
                   window[1] - angle if angle > window[1] else 0.)


def compact_state(observation: dict, history: list[dict], interior_margin_m: float = 0., include_pitch_context: bool = False, include_pitch_center: bool = False) -> dict:
    nav = observation["navigation"]
    aircraft = observation["aircraft"]
    immediate = nav["immediate_gate"]
    target = None
    if immediate:
        gate = observation["next_gates"][0]
        target = {
            "number": gate["number"],
            "distance_to_plane_m": immediate["distance_to_plane_forward_m"],
            "seconds_to_plane_if_velocity_unchanged": immediate["time_to_plane_constant_forward_speed_s"],
            "position_error_left_up_m": [gate["current_center_error_left_m"], gate["current_center_error_up_m"]],
            "center_direction_left_up_deg": [gate["center_bearing_world_left_deg"], gate["center_elevation_world_up_deg"]],
            "center_error_from_motion_left_up_deg": [gate["velocity_track_error_left_deg"], gate["flight_path_error_up_deg"]],
            "track_window_left_deg": immediate["constant_direction_track_window_left_deg"],
            "climb_window_up_deg": immediate["constant_direction_climb_window_up_deg"],
            "horizontal_gap_left_deg": window_gap(aircraft["velocity_track_left_deg"], immediate["constant_direction_track_window_left_deg"]),
            "vertical_gap_up_deg": window_gap(aircraft["flight_path_climb_angle_deg"], immediate["constant_direction_climb_window_up_deg"]),
            "horizontal_gap_if_leveled_now_left_deg": window_gap(nav["leveling_estimate"]["estimated_final_track_if_wings_leveled_now_deg"], immediate["constant_direction_track_window_left_deg"]),
            "crossing_error_left_up_m": [immediate["constant_velocity_crossing_error_left_m"], immediate["constant_velocity_crossing_error_up_m"]],
            "crossing_clearance_left_up_m": [immediate["constant_velocity_crossing_clearance_left_m"], immediate["constant_velocity_crossing_clearance_up_m"]],
            "opening_half_width_height_with_aircraft_margin_m": [gate["half_width_m"] - 1.25, gate["half_height_m"] - .5],
            "geometry_note": "Window gaps measure correction to ANY safe direction through the opening, not its center. Zero means current constant motion passes inside that axis. Windows move as the plane moves. No camera visibility filter.",
        }
        if interior_margin_m > 0:
            dx, dy, dz = gate["relative_world_xyz_m"]
            width = max(0., gate["half_width_m"] - 1.25 - interior_margin_m)
            height = max(0., gate["half_height_m"] - .5 - interior_margin_m)
            track = math.radians(aircraft["velocity_track_left_deg"])
            preferred_track = [rounded(math.degrees(math.atan2(dy + side * width, dx))) for side in (-1, 1)] if dx > .1 else None
            preferred_climb = [rounded(math.degrees(math.atan2((dz + side * height) * math.cos(track), dx))) for side in (-1, 1)] if dx > .1 and math.cos(track) > .01 else None
            target.update({
                "preferred_interior_margin_m": interior_margin_m,
                "preferred_track_window_left_deg": preferred_track,
                "preferred_climb_window_up_deg": preferred_climb,
                "horizontal_gap_to_interior_left_deg": window_gap(aircraft["velocity_track_left_deg"], preferred_track),
                "vertical_gap_to_interior_up_deg": window_gap(aircraft["flight_path_climb_angle_deg"], preferred_climb),
                "horizontal_gap_to_interior_if_leveled_left_deg": window_gap(nav["leveling_estimate"]["estimated_final_track_if_wings_leveled_now_deg"], preferred_track),
                "interior_note": "A preferred 1m extra clearance buffer, not a changed gate size or scoring rule. The original opening windows and crossing clearances remain the actual pass condition.",
            })
    responses = []
    for row in history[-4:]:
        before = row.get("state", {})
        after = row.get("observed_result", {})
        first = before.get("aircraft") or before.get("motion")
        last = after.get("aircraft")
        if not first or not last:
            continue
        # Both full and compact request states carry these explicit measurements.
        duration = row.get("applied_steps", 0) / 120.
        if duration <= 0:
            continue
        delta = last["velocity_track_left_deg"] - first["velocity_track_left_deg"]
        delta = math.degrees(math.atan2(math.sin(math.radians(delta)), math.cos(math.radians(delta))))
        previous_target = (before.get("target") or {}).get("number")
        if previous_target is None and before.get("next_gates"):
            previous_target = before["next_gates"][0]["number"]
        responses.append({
            "action": row["action"], "target_number": previous_target,
            "duration_seconds": rounded(duration),
            "bank_left_start_end_deg": [first["bank_left_deg"], last["bank_left_deg"]],
            "track_change_left_deg": rounded(delta),
            "mean_track_turn_left_deg_s": rounded(delta / duration),
            "climb_angle_change_up_deg": rounded(last["flight_path_climb_angle_deg"] - first["flight_path_climb_angle_deg"]),
        })
    y = aircraft["position_world_xyz_m"][1]
    vy = aircraft["velocity_world_xyz_m_s"][1]
    boundary_time = (60. - y) / vy if vy > .01 else (-60. - y) / vy if vy < -.01 else None
    result = {
        "schema": BUFFERED_SCHEMA if interior_margin_m > 0 else FOCUSED_SCHEMA,
        "task": observation["task"],
        "time_seconds": observation["time_seconds"],
        "gates_passed": observation["gates_passed"], "gates_total": observation["gates_total"],
        "conventions": "Positive angles and rates mean LEFT or UP; negative mean RIGHT or DOWN. All directions are world motion directions, not rolled body axes. The display camera does not limit gate data.",
        "motion": {key: aircraft[key] for key in (
            "position_world_xyz_m", "velocity_world_xyz_m_s", "speed_m_s", "bank_left_deg",
            "nose_up_pitch_deg", "velocity_track_left_deg", "flight_path_climb_angle_deg", "climb_rate_m_s")},
        "measured_rates": {key: nav["measurements"][key] for key in (
            "bank_left_rate_deg_s", "track_turn_rate_left_deg_s", "flight_path_angle_rate_up_deg_s")},
        "target": target,
        "target_update": dict(nav["target_update"]),
        "upcoming_gates": [{key: g[key] for key in (
            "number", "center_m", "half_width_m", "half_height_m")} for g in observation["next_gates"]],
        "following_transition": nav["next_gate_transition"],
        "measured_recent_responses": responses,
        "boundary": {"lateral_limits_world_m": [-60, 60],
                     "remaining_left_right_m": [rounded(60 - y), rounded(y + 60)],
                     "seconds_to_side_boundary_if_velocity_unchanged": rounded(boundary_time)},
        "control": {"action_duration_seconds": observation["controls"]["action_duration_seconds"],
                    "bank_moderate_strong_deg": [25, 40],
                    "pitch_up_level_down_nose_up_deg": [
                        observation["controls"].get("pitch_up_nose_up_deg", observation["controls"]["up_down_pitch_deg"]),
                        observation["controls"]["nominal_pitch_nose_up_deg"],
                        observation["controls"].get("pitch_down_nose_up_deg", -observation["controls"]["up_down_pitch_deg"])],
                    "strong_pitch_up_down_nose_up_deg": [
                        observation["controls"]["strong_pitch_up_nose_up_deg"],
                        observation["controls"]["strong_pitch_down_nose_up_deg"]]
                        if observation["controls"].get("extended_pitch_actions") else None,
                    "joint_choice_count": observation["controls"].get("joint_choice_count", 15),
                    "estimated_extra_turn_if_leveled_left_deg": nav["leveling_estimate"]["estimated_additional_track_turn_left_deg"],
                    "coasting_estimate_note": "Approximate empirical zero-bank turn decay; no guarantee, no new action simulation. Turning and pitch affect one another."},
    }
    targets = observation["controls"].get("pitch_targets_nose_up_deg")
    if targets:
        result["control"]["pitch_targets_nose_up_deg"] = targets
    if include_pitch_context:
        result["schema"] = PITCH_SCHEMA
        controls = observation["controls"]
        if not targets:
            targets = {"up": controls["pitch_up_nose_up_deg"], "level": 2., "down": controls["pitch_down_nose_up_deg"]}
            if controls.get("extended_pitch_actions"):
                targets.update(up_strong=controls["strong_pitch_up_nose_up_deg"], down_strong=controls["strong_pitch_down_nose_up_deg"])
        result["control"]["pitch_targets_nose_up_deg"] = targets
        offset = aircraft["nose_up_pitch_deg"] - aircraft["flight_path_climb_angle_deg"]
        result["motion"]["nose_pitch_minus_path_angle_deg"] = rounded(offset)
        if target:
            window = target.get("preferred_climb_window_up_deg") or target["climb_window_up_deg"]
            target["approximate_pitch_window_nose_up_deg"] = [rounded(value + offset) for value in window] if window else None
            target["pitch_estimate_note"] = "Opening's climb-angle window plus the currently measured nose-pitch minus flight-path-angle difference. Approximate steady-motion geometry; bank, lift, speed and lag can change this difference. No action rollout or selection."
            if include_pitch_center and window:
                target["approximate_center_pitch_nose_up_deg"] = rounded(sum(window) / 2. + offset)
    return result


FOCUSED_INSTRUCTIONS = (
    "Fly through every numbered gate in order. Choose pitch and bank for the next interval. "
    "Read `target.horizontal_gap_left_deg` and `target.vertical_gap_up_deg`. "
    "These signed gaps are corrections to the nearest safe edge of the gate opening. "
    "Negative horizontal gap requires RIGHT correction; positive requires LEFT. "
    "Negative vertical gap requires descent; positive requires climb. "
    "Zero gap means this motion direction already passes inside the opening. "
    "Prioritize the opening windows over the center bearing, especially immediately before crossing: "
    "center angles can grow dramatically even while the crossing is safe. "
    "When absolute horizontal gap exceeds 15 degrees, a strong bank toward the needed correction "
    "is usually appropriate. Sustain it while the gap remains large; do not reverse simply because "
    "`motion.bank_left_deg` or `measured_rates.track_turn_rate_left_deg_s` already has the desired sign. "
    "Those are evidence your turn is working, not evidence the needed correction is complete. "
    "Use `target.horizontal_gap_if_leveled_now_left_deg`, `control.estimated_extra_turn_if_leveled_left_deg` "
    "and `measured_recent_responses` to reduce bank near alignment and avoid overshoot. "
    "Keep vertical motion within `target.climb_window_up_deg`; nominal level is trim, not altitude hold. "
    "Account for measured response lag rather than reversing pitch every interval. "
    "Use `following_transition` to prepare an exit toward the following gate while still clearing "
    "the current opening. Reassess immediately when `target_update.changed_since_previous_decision` "
    "is true. Upcoming gates remain valid outside the display camera. No image is supplied. "
    "Stay within `boundary.lateral_limits_world_m`. Estimates assume unchanged motion and are approximate. "
)

BUFFERED_INSTRUCTIONS = FOCUSED_INSTRUCTIONS + (
    "Prefer the 1m interior buffer: use `target.horizontal_gap_to_interior_left_deg`, "
    "`target.vertical_gap_to_interior_up_deg` and "
    "`target.horizontal_gap_to_interior_if_leveled_left_deg` instead of correcting only "
    "after the actual opening is missed. A small gap outside the interior still needs a timely "
    "moderate correction; do not coast for several intervals until it becomes large. "
    "Aim well inside the frame, release a strong bank early enough for its remaining turn. "
    "Keep the actual opening safe when it is too late to improve the buffer. "
)
