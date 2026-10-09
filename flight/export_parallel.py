"""Render disjoint frame ranges in separate processes, then join at speed 1x."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import imageio_ffmpeg

from course import CONTROL_HZ


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3, 4), default=4)
    args = parser.parse_args()
    if args.fps <= 0 or CONTROL_HZ % args.fps:
        parser.error("FPS must divide 120")
    if args.output.exists():
        parser.error("Choose a fresh output filename")
    record = json.loads(args.source.read_text(encoding="utf-8"))
    step_count = sum(row.get("applied_steps", round(record["config"]["decision_seconds"] * CONTROL_HZ)) for row in record["decisions"])
    total_frames = math.ceil(step_count / (CONTROL_HZ // args.fps)) + 1 + 2 * args.fps
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    parts = output.parent / (output.stem + "_parts")
    parts.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    def render_part(index):
        first, last = index * total_frames // args.workers, (index + 1) * total_frames // args.workers
        directory = parts / f"part_{index:02d}"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "part.mp4"
        with (directory / "render.log").open("w", encoding="utf-8") as log:
            result = subprocess.run([sys.executable, str(Path(__file__).with_name("export_replay.py")),
                str(args.source.resolve()), "--output", str(path), "--fps", str(args.fps),
                "--first-frame", str(first), "--last-frame", str(last)],
                stdout=log, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
        if result.returncode:
            raise RuntimeError(f"Chunk {index} failed; see {directory / 'render.log'}")
        metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        if metadata["frames"] != last - first or not metadata["final_position_matches_original"]:
            raise RuntimeError(f"Chunk {index} has unexpected frame count or replay outcome")
        print(f"CHUNK {index + 1}/{args.workers} ready: {last - first} frames", flush=True)
        return index, path, metadata

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(render_part, i) for i in range(args.workers)]):
            results.append(future.result())
    results.sort()
    listing = parts / "concat.txt"
    listing.write_text("\n".join(f"file 'part_{index:02d}/part.mp4'" for index, _, _ in results), encoding="utf-8")
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy",
                    "-movflags", "+faststart", str(output)], check=True)
    metadata = {"source": str(args.source.resolve()), "output": str(output), "frames": total_frames,
                "fps": args.fps, "width": 1440, "height": 860,
                "video_duration_seconds": total_frames / args.fps,
                "simulation_duration_seconds": record["simulated_seconds"], "final_hold_seconds": 2.,
                "speed": 1, "gates_passed": record["gates_passed"], "gates_total": record["gates_total"],
                "status": record["status"], "physics_steps_replayed_per_chunk": step_count,
                "final_position_matches_original": True,
                "chunks": [r[2] for r in results], "export_wall_seconds": time.perf_counter() - started,
                "note": "Parallel video rendering only. Each chunk reconstructs the recorded physics from time zero. Every output frame appears once in sequence at 1x. No model calls or invented flight paths."}
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, default=str), flush=True)


if __name__ == "__main__":
    main()
