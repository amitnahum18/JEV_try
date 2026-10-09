"""Check whether choices respond to mirrored one-gate geometry, without flying."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import json
from pathlib import Path

from axis_policy import policy_state, requests_for, decide
from compare_axes import CONDITIONS, write_json
from course import Course, Config
from navigation import navigation_features


def mirror_gate_geometry(observation: dict) -> dict:
    # This probe starts at y=0 with zero lateral velocity/attitude. Only the static
    # gate is reflected; every gate-derived signed lateral feature must also flip.
    result = copy.deepcopy(observation)
    assert result["aircraft"]["position_world_xyz_m"][1] == 0
    assert result["aircraft"]["velocity_world_xyz_m_s"][1] == 0
    assert result["aircraft"]["roll_pitch_yaw_deg"] == [0., 0., 0.]
    for gate in result["next_gates"]:
        for field in ("center_m", "relative_world_xyz_m", "relative_body_forward_left_up_m"):
            gate[field][1] *= -1
        for field in ("bearing_left_deg", "current_center_error_left_m", "normalized_lateral_center_error",
                      "heading_error_left_deg", "center_bearing_world_left_deg", "velocity_track_error_left_deg"):
            gate[field] *= -1
        forecast = gate["straight_velocity_crossing_forecast"]
        if forecast:
            forecast["position_world_xyz_m"][1] *= -1
            forecast["center_error_left_m"] *= -1
    # Recompute all new navigation geometry from the reflected gate. Leaving the
    # original navigation block here would leak contradictory left/right features.
    result["navigation"] = navigation_features(result)
    return result


def probe(job: tuple) -> dict:
    repeat, architecture, telemetry, observation, prompt_version = job
    state = policy_state(observation, telemetry, [])
    requests = requests_for(state, architecture, prompt_version)
    target_y = state["next_gates"][0]["center_m"][1]
    row = {"repeat": repeat, "architecture": architecture, "telemetry": telemetry,
           "target_side": "left" if target_y > 0 else "right", "target_y_m": target_y,
           "requests": requests}
    try:
        row.update(decide(requests))
        horizontal = row["action"].split("__")[1]
        side = "left" if horizontal in {"left", "roll_left"} else "right" if horizontal in {"right", "roll_right"} else "straight"
        row.update({"chosen_side": side, "toward_gate": side == row["target_side"]})
    except Exception as exc:
        row["error"] = str(exc)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--prompt-version", type=int, choices=(1, 2, 3), default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("Use positive repeats")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "mirror_probe.json").exists():
        parser.error("Existing probe output; choose a fresh directory")
    course = Course(Config(seed=42, gates=1))
    try:
        original = course.observation(None)
    finally:
        course.close()
    reflected = mirror_gate_geometry(original)
    jobs = [(repeat, architecture, telemetry, observation, args.prompt_version)
            for repeat in range(1, args.repeats + 1)
            for architecture, telemetry in CONDITIONS
            for observation in (original, reflected)]
    rows = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(probe, job) for job in jobs]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print({key: row[key] for key in ("repeat", "architecture", "telemetry", "target_side", "action", "chosen_side", "error") if key in row}, flush=True)
    summary = []
    for architecture, telemetry in CONDITIONS:
        group = [r for r in rows if (r["architecture"], r["telemetry"]) == (architecture, telemetry)]
        valid = [r for r in group if "error" not in r]
        summary.append({"architecture": architecture, "telemetry": telemetry, "valid": len(valid),
                        "toward_gate": sum(r["toward_gate"] for r in valid),
                        "left_choices": sum(r["chosen_side"] == "left" for r in valid),
                        "right_choices": sum(r["chosen_side"] == "right" for r in valid),
                        "straight_choices": sum(r["chosen_side"] == "straight" for r in valid)})
    note = ("Synthetic reflection of one static gate in an actual initial simulator snapshot. "
            "Two repeats per side; no flight is executed and no claim of optimal trajectory is made. "
            "Toward-gate measures initial steering sign only. This is separate from the matched flight benchmark.")
    write_json(args.output_dir / "mirror_probe.json", {"prompt_version": args.prompt_version, "limitations": note, "summary": summary, "decisions": rows})
    lines = ["# Initial steering response to mirrored gate", "", note, "",
             "| Architecture | Input | Valid calls | Toward gate | Left choices | Right choices | Straight |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for row in summary:
        lines.append(f"| {row['architecture']} | {row['telemetry']} | {row['valid']} | {row['toward_gate']} | {row['left_choices']} | {row['right_choices']} | {row['straight_choices']} |")
    (args.output_dir / "mirror_probe.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
