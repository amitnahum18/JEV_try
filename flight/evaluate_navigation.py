"""Compare previous input, added fields, added prompt, and immediate gate refresh."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import random
from statistics import mean

from compare_axes import run_episode, write_json, ROOT
from course import Config

ARMS = {
    "previous": {"telemetry": "previous_full", "prompt": 2, "refresh": False},
    "fields_only": {"telemetry": "full", "prompt": 2, "refresh": False},
    "expanded": {"telemetry": "full", "prompt": 3, "refresh": False},
    "expanded_refresh": {"telemetry": "full", "prompt": 3, "refresh": True},
}


def run_job(job: tuple) -> dict:
    arm, episode_job = job
    record = run_episode(episode_job)
    record["arm"] = arm
    path = Path(record["recording"])
    detailed = json.loads(path.read_text(encoding="utf-8"))
    detailed["arm"] = arm
    switches = []
    for event in detailed["gate_events"]:
        if event["gate"] > record["gates_passed"]:
            continue
        following = [row for row in detailed["decisions"] if row["time_seconds"] >= event["time_seconds"] - .00051]
        first = following[0] if following else None
        switches.append({"passed_gate": event["gate"], "crossing_time_seconds": event["time_seconds"],
                         "first_next_request_time_seconds": first["time_seconds"] if first else None,
                         "request_delay_seconds": max(0., first["time_seconds"] - event["time_seconds"]) if first else None,
                         "first_next_request_gate": first["state"]["next_gates"][0]["number"] if first and first["state"]["next_gates"] else None})
    detailed["gate_request_transitions"] = record["gate_request_transitions"] = switches
    write_json(path, detailed)
    return record


def report(output: Path, records: list[dict], plan: dict) -> None:
    aggregate = []
    for architecture in ("single", "parallel"):
        for arm in ARMS:
            group = [r for r in records if r["architecture"] == architecture and r["arm"] == arm]
            if not group:
                continue
            valid = [r for r in group if r["status"] not in {"api_error", "error", "physics_error"}]
            aggregate.append({"architecture": architecture, "arm": arm, "runs": len(group), "valid_runs": len(valid),
                              "mean_gates": mean(r["gates_passed"] for r in valid) if valid else None,
                              "total_gates": sum(r["gates_passed"] for r in valid),
                              "completed": sum(r["status"] == "completed" for r in valid),
                              "at_least_one_gate": sum(r["gates_passed"] > 0 for r in valid),
                              "gates_per_seed": {str(r["config"]["seed"]): r["gates_passed"] for r in valid},
                              "known_cost_usd": sum(r["known_cost_usd"] for r in group)})
    pairs = []
    indexed = {(r["architecture"], r["arm"], r["config"]["seed"]): r for r in records}
    for architecture in ("single", "parallel"):
        for a, b in (("previous", "fields_only"), ("previous", "expanded"), ("fields_only", "expanded"), ("expanded", "expanded_refresh")):
            differences = []
            for seed in plan["seeds"]:
                first, second = indexed.get((architecture, a, seed)), indexed.get((architecture, b, seed))
                if first and second and all(r["status"] not in {"api_error", "error", "physics_error"} for r in (first, second)):
                    differences.append({"seed": seed, "difference": second["gates_passed"] - first["gates_passed"]})
            pairs.append({"architecture": architecture, "comparison": f"{b}_minus_{a}", "matched_pairs": len(differences),
                          "mean_gate_difference": mean(r["difference"] for r in differences) if differences else None,
                          "differences": differences})
    for seed in plan["seeds"]:
        assert len({r["course_hash"] for r in records if r["config"]["seed"] == seed}) <= 1
    limitations = ("Exploratory matched-seed sample. Prior and current input are rerun concurrently to reduce historical/API-time confounding. "
                   "Fields-only arm uses the previous prompt to isolate the added fields; expanded also changes prompt and transition context. "
                   "Expanded-refresh additionally interrupts a maneuver when a gate is passed. Physics pauses for API calls. "
                   "No claim of statistical significance or internal model understanding. Input additions are a bundle, not per-feature tests.")
    write_json(output / "comparison.json", {"plan": plan, "aggregate": aggregate, "paired_effects": pairs,
               "episodes": records, "limitations": limitations})
    lines = ["# Navigation input and gate-refresh evaluation", "", limitations, "",
             "| Architecture | Condition | Valid runs | Gates per seed | Total gates | Mean gates | Completed |",
             "|---|---|---:|---|---:|---:|---:|"]
    for r in aggregate:
        value = f"{r['mean_gates']:.2f}" if r["mean_gates"] is not None else "n/a"
        seed_values = ", ".join(f"{seed}:{gates}" for seed, gates in sorted(r["gates_per_seed"].items()))
        lines.append(f"| {r['architecture']} | {r['arm']} | {r['valid_runs']}/{r['runs']} | {seed_values} | {r['total_gates']} | {value} | {r['completed']} |")
    lines += ["", "## Paired effects", "", "| Architecture | Difference | Pairs | Mean extra gates |", "|---|---|---:|---:|"]
    for r in pairs:
        value = f"{r['mean_gate_difference']:+.2f}" if r["mean_gate_difference"] is not None else "n/a"
        lines.append(f"| {r['architecture']} | {r['comparison']} | {r['matched_pairs']} | {value} |")
    lines += ["", "## Flights", "", "| Seed | Architecture | Condition | Gates | Status |", "|---|---|---|---:|---|"]
    for r in sorted(records, key=lambda r: (r["config"]["seed"], r["architecture"], r["arm"])):
        lines.append(f"| {r['config']['seed']} | {r['architecture']} | {r['arm']} | {r['gates_passed']}/{r['gates_total']} | {r['status']} |")
    (output / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.seeds < 1 or not 1 <= args.workers <= 4:
        parser.error("Use positive seeds and 1-4 workers")
    output = args.output_dir or ROOT / "flight" / "recordings" / datetime.now().strftime("navigation_eval_%Y%m%d_%H%M%S")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "plan.json").exists():
        parser.error("Use a fresh output directory")
    jobs = []
    rng = random.Random(20260927)
    for seed in range(args.seed, args.seed + args.seeds):
        conditions = [(architecture, arm) for architecture in ("single", "parallel") for arm in ARMS]
        rng.shuffle(conditions)
        for architecture, arm in conditions:
            spec = ARMS[arm]
            config = Config(seed=seed, refresh_on_gate_pass=spec["refresh"])
            jobs.append((arm, (asdict(config), architecture, spec["telemetry"], str(output / arm), spec["prompt"])))
    plan = {"seeds": list(range(args.seed, args.seed + args.seeds)), "arms": ARMS, "workers": args.workers,
            "job_order": [{"arm": arm, "architecture": j[1], "seed": j[0]["seed"]} for arm, j in jobs]}
    write_json(output / "plan.json", plan)
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_job, job) for job in jobs]
        for future in as_completed(futures):
            records.append(future.result())
            report(output, records, plan)
    print(f"REPORT {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
