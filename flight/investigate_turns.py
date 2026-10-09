"""Recorded sharp-turn input probes and matched complete JEV flights."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
from pathlib import Path
from statistics import mean

from axis_policy import policy_state, requests_for, decide
from compare_axes import ROOT, run_episode, write_json
from course import Config
from maneuver_context import compact_state

SOURCE = ROOT / "flight/recordings/navigation_evaluation_20260926/expanded_refresh/single_full_seed44/episode.json"


def probe_one(job):
    label, repeat, index, source, telemetry, prompt = job
    old = source["decisions"][index]
    state = policy_state(old["state"], telemetry, source["decisions"][:index])
    requests = requests_for(state, "single", prompt)
    context = compact_state(old["state"], source["decisions"][:index])
    result = {"label": label, "repeat": repeat, "source_decision": old["number"],
              "source_time_seconds": old["time_seconds"], "telemetry": telemetry,
              "prompt_version": prompt, "requests": requests, "context": context}
    try:
        result.update(decide(requests))
    except Exception as exc:
        result["error"] = str(exc)
    print({k: result[k] for k in ("label", "repeat", "telemetry", "action", "error") if k in result}, flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("probe", "flights"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--versions", type=int, nargs="+", choices=(3, 4, 5), default=[3, 4])
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "results.json").exists():
        parser.error("Choose a fresh output directory")
    records = []
    if args.mode == "probe":
        source = json.loads(SOURCE.read_text(encoding="utf-8"))
        labels = {17.283333333333335: "near_safe_gate3", 17.3: "new_gate4", 17.8: "right_turn_in_progress", 19.8: "late_large_right_gap"}
        jobs = [(label, repeat, index, source, telemetry, prompt)
                for index, row in enumerate(source["decisions"]) if row["time_seconds"] in labels
                for label in [labels[row["time_seconds"]]] for repeat in (1, 2)
                for telemetry, prompt in (("full", 3), ("focused", 4))]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for future in as_completed([pool.submit(probe_one, job) for job in jobs]):
                records.append(future.result())
                write_json(output / "results.json", {"source": str(SOURCE), "records": records,
                    "note": "Static recorded snapshots; these calls do not execute flight. Camera is absent from both requests."})
    else:
        jobs = [(asdict(Config(seed=seed)), "single", "full" if prompt == 3 else "focused", str(output / f"v{prompt}"), prompt)
                for seed in args.seeds for prompt in args.versions]
        write_json(output / "plan.json", {"seeds": args.seeds, "jobs": jobs, "note": "Same physics, course, actions, margins, boundaries and interval. Telemetry plus prompt changed together."})
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for future in as_completed([pool.submit(run_episode, job) for job in jobs]):
                records.append(future.result())
                aggregates = []
                for version in args.versions:
                    group = [r for r in records if r["prompt_version"] == version]
                    valid = [r for r in group if r["status"] not in {"error", "api_error", "physics_error"}]
                    aggregates.append({"prompt": version, "runs": len(group), "valid": len(valid),
                                       "mean_gates": mean(r["gates_passed"] for r in valid) if valid else None,
                                       "completed": sum(r["status"] == "completed" for r in valid)})
                write_json(output / "results.json", {"records": records, "aggregate": aggregates,
                    "note": "Exploratory paired seeds. API nondeterminism; no individual-feature or statistical significance claim."})
    print(f"REPORT {output}", flush=True)


if __name__ == "__main__":
    main()
