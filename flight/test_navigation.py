"""Local navigation/driver regressions. Model responses in driver tests are mocked."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from axis_policy import policy_state, requests_for
from maneuver_context import window_gap
from course import Config, Course
from navigation import navigation_features
from run_course import episode, ROOT


class NavigationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        course = Course(Config())
        try:
            cls.initial = course.observation(None)
        finally:
            course.close()

    def state(self):
        return copy.deepcopy(self.initial)

    def test_initial_acceleration_is_unavailable(self):
        nav = self.initial["navigation"]
        self.assertIsNone(nav["measurements"]["track_turn_rate_left_deg_s"])
        self.assertFalse(nav["leveling_estimate"]["available"])

    def test_dense_pitch_choices_keep_course_and_scoring(self):
        from course import generate_gates
        coarse = Config(seed=44, gates=8, layout="vertical_challenge", extended_pitch_actions=True)
        dense = Config(seed=44, gates=8, layout="vertical_challenge", extended_pitch_actions=True, dense_pitch_actions=True)
        self.assertEqual(generate_gates(coarse), generate_gates(dense))
        self.assertEqual(coarse.joint_choice_count, 25)
        self.assertEqual(dense.joint_choice_count, 45)
        self.assertEqual(dense.pitch_targets()["down_medium"], -10.)
        self.assertEqual(Config().joint_choice_count, 15)

    def test_pitch_context_is_measured_and_all_choices_reach_jev(self):
        course = Course(Config(extended_pitch_actions=True, dense_pitch_actions=True))
        try:
            observation = course.observation(None)
            state = policy_state(observation, "focused", [], 1., include_pitch_context=True)
            offset = state["motion"]["nose_pitch_minus_path_angle_deg"]
            window = state["target"]["preferred_climb_window_up_deg"]
            for expected, actual in zip(window, state["target"]["approximate_pitch_window_nose_up_deg"]):
                self.assertAlmostEqual(expected + offset, actual, places=2)
            requests = requests_for(state, "single", 7)
            self.assertEqual(len(requests), 1)
            self.assertEqual(len(requests["joint"]["questions"]["action"]["criteria"]), 45)
            self.assertEqual(set(requests["joint"]["questions"]["action"]["criteria"]), course.allowed_joint)
            v8 = policy_state(observation, "focused", [], 1., include_pitch_context=True, include_pitch_center=True)
            self.assertIn("approximate_center_pitch_nose_up_deg", v8["target"])
            choices = requests_for(v8, "single", 8)["joint"]["questions"]["action"]["criteria"]
            self.assertEqual(len(choices), 45)
            self.assertIn("less than -13", choices["down_strong__straight"])
            self.assertIn("DESCEND, nose DOWN", choices["down_medium__left"])
            full = policy_state(observation, "full", [])
            self.assertEqual(len(requests_for(full, "single", 3)["joint"]["questions"]["action"]["criteria"]), 45)
            for key in course.pitch_targets:
                course._control(key + "__straight")
        finally:
            course.close()

    def test_opening_gap_uses_nearest_edge(self):
        self.assertEqual(window_gap(44., [-18., -11.]), -55.)
        self.assertEqual(window_gap(44., [-89., 89.]), 0.)
        self.assertEqual(window_gap(-25., [-18., -11.]), 7.)
        self.assertIsNone(window_gap(None, [-18., -11.]))

    def test_focused_state_response_history_and_gate_update(self):
        first = policy_state(self.initial, "focused", [])
        feedback = self.state()
        feedback["aircraft"]["velocity_track_left_deg"] = -5.
        feedback["aircraft"]["bank_left_deg"] = -32.
        history = [{"state": first, "action": "level__roll_right", "time_seconds": 0,
                    "applied_steps": 60, "observed_result": feedback}]
        next_state = self.state()
        next_state["next_gates"] = next_state["next_gates"][1:]
        next_state["navigation"] = navigation_features(next_state)
        next_state["navigation"]["target_update"]["active_gate_number"] = 2
        result = policy_state(next_state, "focused", history)
        self.assertTrue(result["target_update"]["changed_since_previous_decision"])
        self.assertEqual(result["target_update"]["previous_decision_gate_number"], 1)
        self.assertEqual(result["measured_recent_responses"][0]["mean_track_turn_left_deg_s"], -10.)
        self.assertNotIn("target", self.initial)
        requests = requests_for(result, "single", 4)
        self.assertEqual(len(requests["joint"]["questions"]["action"]["criteria"]), 15)
        json.dumps(requests, allow_nan=False)

    def test_preferred_buffer_does_not_change_pass_geometry(self):
        original = self.state()
        result = policy_state(original, "focused", [], interior_margin_m=1.)
        target = result["target"]
        self.assertEqual(target["opening_half_width_height_with_aircraft_margin_m"], [6.75, 5.5])
        self.assertGreater(target["preferred_track_window_left_deg"][0], target["track_window_left_deg"][0])
        self.assertLess(target["preferred_track_window_left_deg"][1], target["track_window_left_deg"][1])
        self.assertEqual(original, self.initial)

    def test_track_turn_rate_sign_and_units(self):
        state = self.state()
        state["time_seconds"] = 1
        state["aircraft"]["velocity_world_xyz_m_s"] = [20., 0., 0.]
        for ay in (-4., 4.):
            state["aircraft"]["acceleration_world_xyz_m_s2"] = [0., ay, 0.]
            rate = navigation_features(state)["measurements"]["track_turn_rate_left_deg_s"]
            self.assertAlmostEqual(rate, math.degrees(ay / 20.), places=2)

    def test_body_yaw_is_transformed_to_heading_rate(self):
        state = self.state()
        state["aircraft"]["roll_pitch_yaw_deg"] = [30., 0., 0.]
        state["aircraft"]["body_roll_pitch_yaw_rates_deg_s"] = [0., 0., 10.]
        measurement = navigation_features(state)["measurements"]
        self.assertAlmostEqual(measurement["heading_turn_rate_left_deg_s"], 10 * math.cos(math.pi / 6), places=2)
        self.assertAlmostEqual(measurement["nose_up_pitch_rate_deg_s"], 5., places=2)

    def test_angular_windows_match_constant_velocity_geometry(self):
        for lateral in (-20., 0., 20.):
            for height in (-10., 0., 10.):
                for track in (-.4, 0., .4):
                    for climb in (-.15, 0., .15):
                        state = self.state()
                        gate = state["next_gates"][0]
                        gate["relative_world_xyz_m"] = [100., lateral, height]
                        state["aircraft"]["velocity_track_left_deg"] = math.degrees(track)
                        state["aircraft"]["flight_path_climb_angle_deg"] = math.degrees(climb)
                        nav = navigation_features(state)["immediate_gate"]
                        lateral_at_plane = 100 * math.tan(track)
                        vertical_at_plane = 100 * math.tan(climb) / math.cos(track)
                        self.assertEqual(nav["current_track_inside_angular_window"], abs(lateral_at_plane - lateral) <= 6.75)
                        self.assertEqual(nav["current_climb_angle_inside_angular_window"], abs(vertical_at_plane - height) <= 5.5)

    def test_zero_speed_and_no_gates_produce_valid_json(self):
        state = self.state()
        state["time_seconds"] = 1
        state["aircraft"]["velocity_world_xyz_m_s"] = [0., 0., 0.]
        state["next_gates"] = []
        nav = navigation_features(state)
        self.assertIsNone(nav["immediate_gate"])
        self.assertIsNone(nav["measurements"]["track_turn_rate_left_deg_s"])
        json.dumps(nav, allow_nan=False)

    def test_previous_and_basic_inputs_do_not_leak_new_fields(self):
        for version in ("previous_full", "basic"):
            state = policy_state(self.initial, version, [])
            self.assertNotIn("navigation", state)
        self.assertIn("navigation", policy_state(self.initial, "full", []))
        self.assertIn("navigation", self.initial)

    def test_target_change_context(self):
        state = self.state()
        state["next_gates"] = state["next_gates"][1:]
        state["navigation"]["target_update"]["active_gate_number"] = 2
        history = [{"state": self.initial, "action": "up__right", "time_seconds": 4.5}]
        result = policy_state(state, "full", history)
        target = result["navigation"]["target_update"]
        self.assertEqual(target["previous_decision_gate_number"], 1)
        self.assertTrue(target["changed_since_previous_decision"])
        self.assertNotIn("changed_since_previous_decision", state["navigation"]["target_update"])

    def test_legacy_response_estimate_is_disabled(self):
        state = self.state()
        state["time_seconds"] = 1
        self.assertIsNone(navigation_features(state, "legacy")["leveling_estimate"]["response_fit"])

    def test_live_driver_keeps_full_input_across_decisions(self):
        seen = []

        def mocked_decision(requests):
            seen.append(requests)
            return {"action": "level__straight", "confidence": .5, "cost_usd": 0.,
                    "latency_seconds": 0., "api_calls": 0, "axis_decisions": {}}

        with tempfile.TemporaryDirectory(prefix="navigation_unit_", dir=ROOT / "flight" / "artifacts") as directory:
            self.assertTrue(Path(directory).resolve().is_relative_to((ROOT / "flight" / "artifacts").resolve()))
            with patch("run_course.decide", mocked_decision):
                record, _ = episode(Config(max_seconds=.55), "parallel", Path(directory), telemetry="full")
            self.assertEqual(record["status"], "time_limit")
            self.assertEqual(len(seen), 2)
            for requests in seen:
                self.assertIn("navigation", requests["vertical"]["state"])
                self.assertEqual(requests["vertical"]["state"], requests["horizontal"]["state"])

    def test_live_driver_focused_buffered_input(self):
        seen = []

        def mocked_decision(requests):
            seen.append(requests)
            return {"action": "level__straight", "confidence": .5, "cost_usd": 0.,
                    "latency_seconds": 0., "api_calls": 0, "axis_decisions": {}}

        with tempfile.TemporaryDirectory(prefix="focused_unit_", dir=ROOT / "flight" / "artifacts") as directory:
            with patch("run_course.decide", mocked_decision):
                record, _ = episode(Config(max_seconds=.55), "single", Path(directory),
                                    telemetry="focused", prompt_version=5)
            self.assertEqual(record["feature_schema"], "flight_opening_context_v2")
            self.assertEqual(len(seen), 2)
            for request in seen:
                self.assertEqual(request["joint"]["state"]["target"]["preferred_interior_margin_m"], 1.)
            self.assertEqual(len(seen[1]["joint"]["state"]["measured_recent_responses"]), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
