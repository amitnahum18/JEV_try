"""Run real single-JEV attempts on the open-frame hard-turn course."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path

from compare_axes import ROOT, run_episode, write_json
from course import Config, generate_gates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[44, 45, 46])
    parser.add_argument("--gates", type=int, default=8)
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--level", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--layout", choices=("hard_turns", "vertical_challenge", "maze"), default="vertical_challenge")
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--prompt-version", type=int, choices=(5, 6, 7, 8))
    parser.add_argument("--pitch-options", choices=("coarse", "dense"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.prompt_version is None:
        args.prompt_version = 8 if args.layout == "vertical_challenge" else 6
    if args.pitch_options is None:
        args.pitch_options = "dense" if args.layout == "vertical_challenge" and args.prompt_version >= 8 else "coarse"
    if args.seconds is None:
        args.seconds = 120. if args.layout == "vertical_challenge" else 85.
    output = args.output_dir.resolve()
    if (output / "plan.json").exists() or (output / "results.json").exists():
        parser.error("Choose a fresh output directory")
    output.mkdir(parents=True, exist_ok=True)
    jobs = []
    for seed in args.seeds:
        config = Config(seed=seed, gates=args.gates, max_seconds=args.seconds,
                        level=args.level, layout=args.layout,
                        extended_pitch_actions=args.layout == "vertical_challenge",
                        dense_pitch_actions=args.pitch_options == "dense")
        jobs.append((asdict(config), "single", "focused", str(output), args.prompt_version))
    write_json(output / "plan.json", {
        "created_at": datetime.now().astimezone().isoformat(), "jobs": jobs,
        "actions": Config(**jobs[0][0]).joint_choice_count, "architecture": "single", "prompt_version": args.prompt_version,
        "layout": args.layout, "walls": args.layout == "maze",
        "course_examples": {str(seed): generate_gates(Config(**job[0])) for seed, job in zip(args.seeds, jobs)},
        "pitch_options": args.pitch_options,
        "note": "One JEV with explicit joint pitch/bank choices. Coarse vertical mode has 25 choices, dense vertical mode 45. Prompt v8 adds measured geometric center-pitch estimate and explicit numerical choice criteria. No action is selected or overridden by the input builder. Open frames; historical maze retains walls. Preserve every attempt."})
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(run_episode, job) for job in jobs]):
            records.append(future.result())
            write_json(output / "results.json", {"episodes": records,
                "completed": sum(r["status"] == "completed" for r in records),
                "attempts_finished": len(records), "planned_attempts": len(jobs),
                "note": "Exploratory attempts, not a reliability estimate. Selection for video is identified explicitly."})
    selected = max(records, key=lambda r: (r["status"] == "completed", r["gates_passed"], r["score"]))
    write_json(output / "selected_for_video.json", {"selected": selected,
        "selection": "Completed flight first, then greatest gates passed; all attempts preserved in results.json."})
    print("SELECTED " + json.dumps({k: selected[k] for k in ("recording", "status", "gates_passed", "gates_total", "simulated_seconds")}), flush=True)


if __name__ == "__main__":
    main()
