"""Run a matched 2x2 experiment: one/two JEVs with basic/full telemetry."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random
from statistics import mean
import time

from dotenv import load_dotenv
from axis_policy import ROOT, MODEL, policy_state, requests_for, decide
from course import Config, Course, CONTROL_HZ
from navigation import NAVIGATION_SCHEMA
from maneuver_context import FOCUSED_SCHEMA, BUFFERED_SCHEMA, PITCH_SCHEMA

load_dotenv(ROOT / ".env")
CONDITIONS = [(architecture, telemetry) for architecture in ("single", "parallel")
              for telemetry in ("basic", "full")]


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def run_episode(job: tuple) -> dict:
    settings, architecture, telemetry, directory, prompt_version = job
    config = Config(**settings)
    course = Course(config)
    output = Path(directory) / f"{architecture}_{telemetry}_seed{config.seed}"
    output.mkdir(parents=True, exist_ok=True)
    path = output / "episode.json"
    rows, trajectory = [], []
    record = {"schema_version": 5, "feature_schema": NAVIGATION_SCHEMA if telemetry != "previous_full" else "flight_full_v2",
              "policy": f"{architecture}_{telemetry}", "model": MODEL,
              "architecture": architecture, "telemetry": telemetry,
              "prompt_version": prompt_version,
              "config": asdict(config), "decisions": rows, "trajectory": trajectory,
              "course_gates": course.gates,
              "course_hash": hashlib.sha256(json.dumps(course.gates, sort_keys=True).encode()).hexdigest(),
              "timing": "Physics pauses during API calls; decision quality, not real-time delay tolerance.",
              "navigation": f"{config.joint_choice_count} simultaneous pitch-and-bank combinations.",
              "recording": str(path.resolve())}
    if telemetry == "focused":
        record["feature_schema"] = PITCH_SCHEMA if prompt_version >= 7 else BUFFERED_SCHEMA if prompt_version >= 5 else FOCUSED_SCHEMA
    action = None
    api_attempts = 0
    started = time.perf_counter()
    print(f"START {record['policy']} seed={config.seed}", flush=True)
    try:
        while course.status == "flying":
            observation = course.observation(action)
            state = policy_state(observation, telemetry, rows, interior_margin_m=1. if prompt_version >= 5 else 0., include_pitch_context=prompt_version >= 7, include_pitch_center=prompt_version >= 8)
            requests = requests_for(state, architecture, prompt_version)
            write_json(output / "latest_jev_input.json", {"requests": requests})
            api_attempts += len(requests)
            result = decide(requests)
            action = result["action"]
            row = {"number": len(rows) + 1, "time_seconds": course.sim.elapsed_time,
                   "state": state, "requests": requests, **result}
            rows.append(row)
            gate_at_start = course.passed
            applied_steps = 0
            for _ in range(round(config.decision_seconds * CONTROL_HZ)):
                course.step(action)
                applied_steps += 1
                if course.sim.aviary_steps % 6 == 0 or course.status != "flying":
                    physics = course.sim.state(0)
                    trajectory.append({"time_seconds": course.sim.elapsed_time,
                                       "position_m": physics[3].tolist(), "rpy_rad": physics[1].tolist()})
                if course.status != "flying" or (config.refresh_on_gate_pass and course.passed != gate_at_start):
                    break
            row["applied_steps"] = applied_steps
            feedback = course.observation(action)
            row["observed_result"] = {"aircraft": feedback["aircraft"],
                                      "next_gates": feedback["next_gates"],
                                      "navigation": feedback["navigation"],
                                      "gates_passed": course.passed, "status": course.status}
            record.update(course.summary())
            write_json(path, record)
    except Exception as exc:
        course.status = "api_error" if len(rows) == 0 or api_attempts > sum(r["api_calls"] for r in rows) else "error"
        record["error"] = str(exc)
        print(f"ERROR {record['policy']} seed={config.seed}: {exc}", flush=True)
    finally:
        record.update(course.summary())
        known_costs = [r["cost_usd"] for r in rows if r["cost_usd"] is not None]
        record.update({"decision_count": len(rows), "api_calls": sum(r["api_calls"] for r in rows),
                       "api_attempts": api_attempts, "known_cost_usd": sum(known_costs),
                       "cost_complete": len(known_costs) == len(rows) and course.status != "api_error",
                       "wall_seconds": time.perf_counter() - started,
                       "mean_decision_latency_seconds": mean(r["latency_seconds"] for r in rows) if rows else None,
                       "mean_request_bytes_per_decision": mean(r["request_bytes"] for r in rows) if rows else None,
                       "finished_at": datetime.now().astimezone().isoformat()})
        write_json(path, record)
        course.close()
    print(f"RESULT {record['policy']} seed={config.seed}: {record['status']} {record['gates_passed']}/{config.gates}", flush=True)
    return {key: value for key, value in record.items() if key not in {"decisions", "trajectory", "course_gates"}}


def report(output: Path, records: list[dict], settings: dict) -> None:
    aggregate = []
    for architecture, telemetry in CONDITIONS:
        group = [r for r in records if (r["architecture"], r["telemetry"]) == (architecture, telemetry)]
        valid = [r for r in group if r["status"] not in {"api_error", "error", "physics_error"}]
        if not group:
            continue
        aggregate.append({"architecture": architecture, "telemetry": telemetry,
                          "episodes": len(group), "valid_episodes": len(valid),
                          "mean_gates_passed": mean(r["gates_passed"] for r in valid) if valid else None,
                          "total_gates_passed": sum(r["gates_passed"] for r in valid),
                          "completed": sum(r["status"] == "completed" for r in valid),
                          "status_counts": dict(Counter(r["status"] for r in group)),
                          "mean_decision_latency_seconds": mean(r["mean_decision_latency_seconds"] for r in valid if r["decision_count"]) if valid else None,
                          "mean_cost_usd_per_valid_episode": mean(r["known_cost_usd"] for r in valid) if valid else None,
                          "mean_request_bytes_per_decision": mean(r["mean_request_bytes_per_decision"] for r in valid if r["decision_count"]) if valid else None})
    by_key = {(r["config"]["seed"], r["architecture"], r["telemetry"]): r for r in records}
    seeds = sorted({r["config"]["seed"] for r in records})
    effects = []
    for label, first, second in (
        ("parallel_minus_single_basic", ("single", "basic"), ("parallel", "basic")),
        ("parallel_minus_single_full", ("single", "full"), ("parallel", "full")),
        ("full_minus_basic_single", ("single", "basic"), ("single", "full")),
        ("full_minus_basic_parallel", ("parallel", "basic"), ("parallel", "full")),
    ):
        differences = []
        for seed in seeds:
            a, b = by_key.get((seed, *first)), by_key.get((seed, *second))
            if a and b and all(r["status"] not in {"api_error", "error", "physics_error"} for r in (a, b)):
                differences.append({"seed": seed, "gate_difference": b["gates_passed"] - a["gates_passed"]})
        effects.append({"effect": label, "matched_pairs": len(differences), "differences": differences,
                        "mean_gate_difference": mean(d["gate_difference"] for d in differences) if differences else None})
    for seed in seeds:
        assert len({r["course_hash"] for r in records if r["config"]["seed"] == seed}) == 1, "Course mismatch"
    note = ("Small matched-seed exploratory sample; no statistical significance or mechanistic understanding claim. "
            "Full telemetry tests a bundle, not individual features. API responses are not deterministic. "
            "Two axes see the same state but do not negotiate; aerodynamic coupling remains. "
            "API/physics errors excluded from navigation averages and shown separately. Failed-request costs may be unavailable.")
    write_json(output / "comparison.json", {"settings": settings, "aggregate": aggregate,
               "paired_effects": effects, "episodes": records, "limitations": note})
    lines = ["# JEV flight: axes and telemetry comparison", "", note, "",
             "All groups: identical seeded slalom courses, aircraft physics, 15 joint actions, stabilizer and decision interval.", "",
             "| Architecture | Input | Valid runs | Mean gates | Finished | Decision latency (s) | Cost/run ($) | Request bytes/decision |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in aggregate:
        def fmt(key, places=3):
            value = row[key]
            return f"{value:.{places}f}" if value is not None else "n/a"
        lines.append(f"| {row['architecture']} | {row['telemetry']} | {row['valid_episodes']}/{row['episodes']} | {fmt('mean_gates_passed', 2)} | {row['completed']} | {fmt('mean_decision_latency_seconds')} | {fmt('mean_cost_usd_per_valid_episode', 5)} | {fmt('mean_request_bytes_per_decision', 0)} |")
    lines += ["", "## Matched effects", "", "| Comparison | Pairs | Mean additional gates |", "|---|---:|---:|"]
    for effect in effects:
        value = effect["mean_gate_difference"]
        lines.append(f"| {effect['effect']} | {effect['matched_pairs']} | {value:+.2f} |" if value is not None else f"| {effect['effect']} | 0 | n/a |")
    lines += ["", "## Individual flights", "", "| Seed | Architecture | Input | Gates | Status | API calls |", "|---|---|---|---:|---|---:|"]
    for row in sorted(records, key=lambda r: (r["config"]["seed"], r["architecture"], r["telemetry"])):
        lines.append(f"| {row['config']['seed']} | {row['architecture']} | {row['telemetry']} | {row['gates_passed']}/{row['gates_total']} | {row['status']} | {row['api_calls']} |")
    (output / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--level", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--decision-seconds", type=float, default=.5)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--prompt-version", type=int, choices=(1, 2, 3), default=3)
    args = parser.parse_args()
    if args.seeds < 1 or not 1 <= args.workers <= 4 or not 1 / CONTROL_HZ <= args.decision_seconds <= 3:
        parser.error("Use positive seeds, 1-4 workers, and decision-seconds between 1/120 and 3.")
    output = args.output_dir or ROOT / "flight" / "recordings" / datetime.now().strftime("axes_%Y%m%d_%H%M%S")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "comparison.json").exists():
        parser.error("Output directory already contains a comparison. Choose a fresh directory.")
    jobs = []
    rng = random.Random(20260926)
    for seed in range(args.seed, args.seed + args.seeds):
        config = Config(seed=seed, level=args.level,
                        decision_seconds=round(args.decision_seconds * CONTROL_HZ) / CONTROL_HZ)
        conditions = list(CONDITIONS)
        rng.shuffle(conditions)
        jobs.extend((asdict(config), architecture, telemetry, str(output), args.prompt_version) for architecture, telemetry in conditions)
    settings = {"seeds": list(range(args.seed, args.seed + args.seeds)), "level": args.level,
                "decision_seconds": jobs[0][0]["decision_seconds"], "workers": args.workers, "prompt_version": args.prompt_version, "feature_schema": NAVIGATION_SCHEMA,
                "job_order": [{"seed": j[0]["seed"], "architecture": j[1], "telemetry": j[2]} for j in jobs]}
    write_json(output / "plan.json", settings)
    records = []
    # Separate processes isolate PyBullet clients. Threads are only used for JEV axis requests.
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_episode, job) for job in jobs]
        for future in as_completed(futures):
            records.append(future.result())
            report(output, records, settings)
    print(f"REPORT {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
