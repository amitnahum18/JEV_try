"""Navigation measurements and explicitly limited geometric/response estimates.

This module supplies features to JEV. It never selects or overrides a maneuver,
queries a model, advances physics, or looks up a precomputed route.
"""
from __future__ import annotations

from functools import lru_cache
import json
import math
from pathlib import Path

NAVIGATION_SCHEMA = "flight_navigation_v3"
RESPONSE_FIT = Path(__file__).resolve().parent / "artifacts" / "bank_level_response_fit.json"


def rounded(value: float | None, places: int = 3) -> float | None:
    return round(value, places) if value is not None and math.isfinite(value) else None


def angle_deg(value: float) -> float:
    return rounded(math.degrees(math.atan2(math.sin(value), math.cos(value))))


@lru_cache(maxsize=1)
def response_fit() -> dict | None:
    try:
        fit = json.loads(RESPONSE_FIT.read_text(encoding="utf-8"))
        if not 0 < fit["tau_seconds"] < 2:
            return None
        return {key: fit[key] for key in (
            "tau_seconds", "tau_seconds_p10", "tau_seconds_p90",
            "sample_count", "episode_count", "source_control_profile")}
    except (OSError, ValueError, KeyError, TypeError):
        return None


def navigation_features(observation: dict, control_profile: str = "banked_v2") -> dict:
    aircraft = observation["aircraft"]
    position = aircraft["position_world_xyz_m"]
    vx, vy, vz = aircraft["velocity_world_xyz_m_s"]
    ax, ay, az = aircraft["acceleration_world_xyz_m_s2"]
    roll, pitch, _ = map(math.radians, aircraft["roll_pitch_yaw_deg"])
    p, q, r = map(math.radians, aircraft["body_roll_pitch_yaw_rates_deg_s"])
    horizontal_speed = math.hypot(vx, vy)
    speed_squared = horizontal_speed ** 2 + vz ** 2
    acceleration_available = observation["time_seconds"] > 0
    attitude_rates_available = abs(math.cos(pitch)) > .01
    roll_rate = p + math.tan(pitch) * (q * math.sin(roll) + r * math.cos(roll)) if attitude_rates_available else None
    pitch_rate = q * math.cos(roll) - r * math.sin(roll)
    heading_rate = (q * math.sin(roll) + r * math.cos(roll)) / math.cos(pitch) if attitude_rates_available else None
    track_rate = (vx * ay - vy * ax) / horizontal_speed ** 2 if horizontal_speed > 1 and acceleration_available else None
    horizontal_acceleration = (vx * ax + vy * ay) / horizontal_speed if horizontal_speed > 1 and acceleration_available else None
    climb_angle_rate = ((horizontal_speed * az - vz * horizontal_acceleration) / speed_squared
                        if horizontal_acceleration is not None and speed_squared > 1 else None)
    bank = aircraft["bank_left_deg"]
    measured = {
        "altitude_world_m": position[2],
        "height_above_flat_ground_m": position[2],
        "ground_reference": "Flat simulator ground at world z=0; height is body-center clearance, not terrain sensing.",
        "bank_left_deg": bank, "nose_up_pitch_deg": aircraft["nose_up_pitch_deg"],
        "heading_left_deg": aircraft["heading_left_deg"],
        "velocity_track_left_deg": aircraft["velocity_track_left_deg"],
        "flight_path_climb_angle_deg": aircraft["flight_path_climb_angle_deg"],
        "climb_rate_up_m_s": vz,
        "bank_left_rate_deg_s": rounded(-math.degrees(roll_rate)) if roll_rate is not None else None,
        "nose_up_pitch_rate_deg_s": rounded(-math.degrees(pitch_rate)),
        "heading_turn_rate_left_deg_s": rounded(math.degrees(heading_rate)) if heading_rate is not None else None,
        "track_turn_rate_left_deg_s": rounded(math.degrees(track_rate)) if track_rate is not None else None,
        "flight_path_angle_rate_up_deg_s": rounded(math.degrees(climb_angle_rate)) if climb_angle_rate is not None else None,
        "vertical_acceleration_up_m_s2": az if acceleration_available else None,
        "horizontal_acceleration_m_s2": rounded(horizontal_acceleration),
        "acceleration_sample_available": acceleration_available,
        "observed_horizontal_turn_radius_m": rounded(horizontal_speed / abs(track_rate)) if track_rate is not None and abs(track_rate) > math.radians(.5) else None,
        "rate_method": "Euler attitude rates from body angular velocity; track/climb rates from world velocity and last 1/120s finite-difference acceleration. No averaging; transient rates can be noisy.",
    }
    fit = response_fit() if control_profile == "banked_v2" else None
    leveling = {
        "available": fit is not None and track_rate is not None,
        "assumptions": "Future zero-bank command. First-order bank decay, with measured track turn rate assumed to decay with the same time constant. No roll inertia, yaw lag, sideslip or pitch/lift coupling. These are approximate estimates, not a PyFlyt rollout or an instruction.",
        "validity": "Fit from correlated existing zero-bank-command samples in banked_v2. Not validated on independent flights; p10/p90 describe fitted sample variation, not prediction confidence.",
        "bank_tolerance_deg": 3,
        "response_fit": fit,
        "estimated_seconds_to_bank_within_3_deg": rounded(fit["tau_seconds"] * math.log(abs(bank) / 3)) if fit and abs(bank) > 3 else (0. if fit else None),
        "estimated_additional_track_turn_left_deg": rounded(math.degrees(track_rate) * fit["tau_seconds"]) if fit and track_rate is not None else None,
        "estimated_final_track_if_wings_leveled_now_deg": angle_deg(math.radians(aircraft["velocity_track_left_deg"]) + track_rate * fit["tau_seconds"]) if fit and track_rate is not None else None,
        "additional_track_turn_p10_p90_tau_deg": [rounded(math.degrees(track_rate) * tau) for tau in sorted((fit["tau_seconds_p10"], fit["tau_seconds_p90"]), reverse=track_rate < 0)] if fit and track_rate is not None else None,
        "current_bank_and_track_turn_have_same_sign": bank * track_rate >= 0 if track_rate is not None else None,
    }
    navigation = {"schema": NAVIGATION_SCHEMA,
                  "sign_convention": "Positive LEFT/UP. Angles in degrees, angular rates in degrees/second. Position error and motion-direction error are different quantities. Null means unavailable.",
                  "measurements": measured, "leveling_estimate": leveling,
                  "immediate_gate": None, "next_gate_transition": None, "target_update": {}}
    gates = observation["next_gates"]
    if not gates:
        return navigation
    gate = gates[0]
    dx, dy, dz = gate["relative_world_xyz_m"]
    horizontal_distance = math.hypot(dx, dy)
    plane_time = gate["estimated_seconds_to_gate_plane"]
    usable_width = max(0., gate["half_width_m"] - 1.25)
    usable_height = max(0., gate["half_height_m"] - .5)
    bearing_rate = (dy * vx - dx * vy) / horizontal_distance ** 2 if horizontal_distance > 1 else None
    radial_horizontal_rate = -(dx * vx + dy * vy) / horizontal_distance if horizontal_distance > 1 else None
    elevation_rate = ((-horizontal_distance * vz - dz * radial_horizontal_rate) / (horizontal_distance ** 2 + dz ** 2)
                      if radial_horizontal_rate is not None else None)
    track_error_rate = bearing_rate - track_rate if bearing_rate is not None and track_rate is not None else None
    climb_error_rate = elevation_rate - climb_angle_rate if elevation_rate is not None and climb_angle_rate is not None else None
    track = math.radians(aircraft["velocity_track_left_deg"])
    climb = math.radians(aircraft["flight_path_climb_angle_deg"])
    forecast = gate["straight_velocity_crossing_forecast"]
    next_interval = observation["controls"]["action_duration_seconds"]
    alignment = {
        "number": gate["number"],
        "distance_to_plane_forward_m": dx,
        "horizontal_distance_to_center_m": rounded(horizontal_distance),
        "gate_center_altitude_world_m": gate["center_m"][2],
        "height_error_up_m": dz, "lateral_position_error_left_m": dy,
        "usable_altitude_bounds_world_m": [rounded(gate["center_m"][2] - usable_height), rounded(gate["center_m"][2] + usable_height)],
        "usable_lateral_bounds_world_m": [rounded(gate["center_m"][1] - usable_width), rounded(gate["center_m"][1] + usable_width)],
        "height_inside_opening_band_now": abs(dz) <= usable_height,
        "lateral_position_inside_opening_band_now": abs(dy) <= usable_width,
        "heading_error_left_deg": gate["heading_error_left_deg"],
        "track_alignment_error_left_deg": gate["velocity_track_error_left_deg"],
        "climb_alignment_error_up_deg": gate["flight_path_error_up_deg"],
        "gate_center_bearing_rate_left_deg_s": rounded(math.degrees(bearing_rate)) if bearing_rate is not None else None,
        "track_alignment_error_rate_left_deg_s": rounded(math.degrees(track_error_rate)) if track_error_rate is not None else None,
        "climb_alignment_error_rate_up_deg_s": rounded(math.degrees(climb_error_rate)) if climb_error_rate is not None else None,
        "height_error_rate_up_m_s": -vz, "lateral_position_error_rate_left_m_s": -vy,
        "time_to_plane_constant_forward_speed_s": plane_time,
        "constant_velocity_crossing_error_left_m": forecast["center_error_left_m"] if forecast else None,
        "constant_velocity_crossing_error_up_m": forecast["center_error_up_m"] if forecast else None,
        "constant_velocity_crossing_clearance_left_m": forecast["lateral_clearance_margin_m"] if forecast else None,
        "constant_velocity_crossing_clearance_up_m": forecast["vertical_clearance_margin_m"] if forecast else None,
        "constant_forward_speed_center_velocity_left_up_m_s": [rounded(dy / plane_time), rounded(dz / plane_time)] if plane_time is not None and plane_time > .05 else None,
        "geometric_estimate_assumption": "Velocity requirements assume constant current world-x speed until the gate plane. Angular windows assume fixed world-y/world-z travel direction. Opening-band checks do not mean the gate has been passed.",
        "track_error_after_one_interval_constant_rates_deg": angle_deg(math.radians(gate["velocity_track_error_left_deg"]) + track_error_rate * next_interval) if track_error_rate is not None else None,
        "constant_rates_estimate_assumption": "Current motion-direction rate and gate-bearing rate held fixed for one decision interval, without predicting the selected maneuver.",
        "track_error_after_leveling_constant_gate_bearing_deg": angle_deg(math.radians(gate["velocity_track_error_left_deg"]) - track_rate * fit["tau_seconds"]) if fit and track_rate is not None else None,
        "leveling_alignment_assumption": "Combines exponential turn-decay estimate with fixed current gate bearing. Gate bearing actually changes while traveling; this is not a predicted gate crossing.",
    }
    if dx > .1:
        low = math.atan2(dy - usable_width, dx)
        high = math.atan2(dy + usable_width, dx)
        alignment["constant_direction_track_window_left_deg"] = [rounded(math.degrees(low)), rounded(math.degrees(high))]
        alignment["current_track_inside_angular_window"] = low <= track <= high
        # At the x plane, vertical displacement is vz * dx/vx; elevation uses
        # horizontal speed, so its window depends on the measured track angle.
        if math.cos(track) > .01:
            path_low = math.atan2((dz - usable_height) * math.cos(track), dx)
            path_high = math.atan2((dz + usable_height) * math.cos(track), dx)
            alignment["constant_direction_climb_window_up_deg"] = [rounded(math.degrees(path_low)), rounded(math.degrees(path_high))]
            alignment["current_climb_angle_inside_angular_window"] = path_low <= climb <= path_high
        else:
            alignment["constant_direction_climb_window_up_deg"] = None
            alignment["current_climb_angle_inside_angular_window"] = None
    else:
        alignment["constant_direction_track_window_left_deg"] = None
        alignment["current_track_inside_angular_window"] = None
        alignment["constant_direction_climb_window_up_deg"] = None
        alignment["current_climb_angle_inside_angular_window"] = None
    navigation["immediate_gate"] = alignment
    if len(gates) > 1:
        following = gates[1]
        between = [following["center_m"][i] - gate["center_m"][i] for i in range(3)]
        navigation["next_gate_transition"] = {
            "from_gate": gate["number"], "to_gate": following["number"],
            "center_to_center_forward_left_up_m": [rounded(v) for v in between],
            "height_change_up_m": rounded(between[2]),
            "lateral_change_left_m": rounded(between[1]),
            "center_to_center_bearing_left_deg": rounded(math.degrees(math.atan2(between[1], between[0]))),
            "center_to_center_elevation_up_deg": rounded(math.degrees(math.atan2(between[2], math.hypot(*between[:2])))),
            "note": "Static geometry for looking ahead; not a flight path or a required action. Pass the immediate gate first.",
        }
    return navigation


NAVIGATION_INSTRUCTIONS = (
    "When `navigation` is present, prioritize `navigation.immediate_gate.track_alignment_error_left_deg`, "
    "`navigation.immediate_gate.climb_alignment_error_up_deg`, their rates and the crossing clearances. "
    "A gate left of your POSITION can already be right of your MOTION DIRECTION. "
    "A positive bank turns left even if velocity is already aligned; continued bank can overshoot. "
    "Read `navigation.measurements.track_turn_rate_left_deg_s`, "
    "`navigation.measurements.bank_left_rate_deg_s`, `navigation.immediate_gate.height_error_up_m`, "
    "and the track/climb angular windows. Consider `navigation.leveling_estimate` "
    "when timing a bank release; it is an approximate empirical estimate, not a guaranteed result. "
    "The next gate transition describes upcoming geometry. All estimates are optional evidence, "
    "not control commands; make your own choice using measurements and inertia. "
    "When `navigation.target_update.changed_since_previous_decision` is true, the target has "
    "changed: reassess the current attitude and velocity for the new gate. The previous "
    "decision was for `navigation.target_update.previous_decision_gate_number`. "
)
