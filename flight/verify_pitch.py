"""Recorded real-JEV pitch ablation on unchanged vertical course geometry."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
import json
from compare_axes import run_episode, write_json, ROOT
from course import Config


def main():
    output = ROOT / "flight/recordings/pitch_verification_20260927"
    if (output / "plan.json").exists():
        raise RuntimeError("Preserve previous trials: choose a fresh experiment directory.")
    output.mkdir(parents=True, exist_ok=True)
    jobs = []
    for name, dense, version in [("angles_only", True, 6), ("guidance_only", False, 7), ("combined", True, 7)]:
        config = Config(seed=44, gates=8, max_seconds=120., layout="vertical_challenge",
                        extended_pitch_actions=True, dense_pitch_actions=dense)
        jobs.append((name, (asdict(config), "single", "focused", str(output / name), version)))
    write_json(output / "plan.json", {"experiments": jobs,
        "baseline": str(ROOT / "flight/recordings/vertical_25_choices_preview_20260926/single_level1_seed44_232048_671155.json"),
        "purpose": "Ablation, same geometry/scoring/physics/decision interval. Three independent single-JEV flights, not multiple JEVs controlling one aircraft. Preserve failures. Small exploratory sample."})
    results = []
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(run_episode, job): name for name, job in jobs}
        for future in as_completed(futures):
            result = future.result()
            result["experiment"] = futures[future]
            results.append(result)
            write_json(output / "results.json", {"episodes": results})
            print(json.dumps({k: result[k] for k in ("experiment", "status", "gates_passed", "recording")}), flush=True)


if __name__ == "__main__":
    main()
