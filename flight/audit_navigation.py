"""Replay scored flights and verify every request against current simulator state."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np

from axis_policy import policy_state
from course import Config, Course, CONTROL_HZ
from compare_axes import write_json


def audit(path_string: str) -> dict:
    path = Path(path_string)
    record = json.loads(path.read_text(encoding="utf-8"))
    course = Course(Config(**record["config"]))
    checked, transition_checks, payload_checks = 0, 0, 0
    try:
        for index, row in enumerate(record["decisions"]):
            np.testing.assert_allclose(course.sim.elapsed_time, row["time_seconds"], atol=1e-8)
            observation = course.observation(record["decisions"][index - 1]["action"] if index else None)
            expected = policy_state(observation, record["telemetry"], record["decisions"][:index],
                                    interior_margin_m=1. if record.get("prompt_version", 3) >= 5 else 0.,
                                    include_pitch_context=record.get("prompt_version", 3) >= 7, include_pitch_center=record.get("prompt_version", 3) >= 8)
            # Diagnostic metadata after terminal failure is not part of any API
            # request. Request-state fields must match the reconstructed state.
            assert expected == row["state"], f"Request state mismatch: {path} decision {index + 1}"
            assert len(row["requests"]) == (2 if record["architecture"] == "parallel" else 1)
            for request in row["requests"].values():
                assert request["state"] == row["state"]
                payload_checks += 1
            for _ in range(row.get("applied_steps", round(course.config.decision_seconds * CONTROL_HZ))):
                previous_gate_count = course.passed
                course.step(row["action"])
                current = course.observation(row["action"])
                if course.passed < len(course.gates):
                    gate = course.gates[course.passed]
                    first, nav = current["next_gates"][0], current["navigation"]["immediate_gate"]
                    assert first["number"] == gate["number"] == nav["number"]
                    assert first["center_m"] == gate["center_m"]
                    delta = np.array(gate["center_m"]) - course.position()
                    np.testing.assert_allclose(first["relative_world_xyz_m"], delta, atol=.00051)
                    assert abs(nav["height_error_up_m"] - delta[2]) < .00051
                    assert abs(nav["lateral_position_error_left_m"] - delta[1]) < .00051
                else:
                    assert current["next_gates"] == []
                    assert current["navigation"]["immediate_gate"] is None
                if course.passed != previous_gate_count:
                    transition_checks += 1
                    assert current["navigation"]["target_update"]["last_passed_gate"] == course.passed
                    if course.status == "flying" and course.config.refresh_on_gate_pass:
                        assert index + 1 < len(record["decisions"])
                        following = record["decisions"][index + 1]
                        np.testing.assert_allclose(following["time_seconds"], course.sim.elapsed_time, atol=1e-8)
                        state = following["state"]
                        target_number = state["target"]["number"] if record["telemetry"] == "focused" else state["next_gates"][0]["number"]
                        target_update = state["target_update"] if record["telemetry"] == "focused" else state["navigation"]["target_update"]
                        assert target_number == course.passed + 1
                        assert target_update["changed_since_previous_decision"]
                checked += 1
            assert course.passed == row["observed_result"]["gates_passed"]
        assert course.status == record["status"]
        assert course.passed == record["gates_passed"]
        np.testing.assert_allclose(course.position(), record["final_position_m"], atol=1e-8)
        return {"recording": str(path), "physics_steps_checked": checked, "gate_transitions_checked": transition_checks,
                "exact_request_payloads_checked": payload_checks, "status": "passed"}
    finally:
        course.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    paths = sorted(args.directory.glob("*/*/episode.json"))
    if not paths:
        parser.error("No evaluation episodes found")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(audit, map(str, paths)))
    summary = {"episodes_checked": len(results), "all_passed": all(r["status"] == "passed" for r in results),
               "physics_steps_checked": sum(r["physics_steps_checked"] for r in results),
               "gate_transitions_checked": sum(r["gate_transitions_checked"] for r in results),
               "exact_request_payloads_checked": sum(r["exact_request_payloads_checked"] for r in results),
               "episodes": results}
    write_json(args.directory / "gate_update_audit.json", summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "episodes"}, indent=2))


if __name__ == "__main__":
    main()
