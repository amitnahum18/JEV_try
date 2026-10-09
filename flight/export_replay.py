"""Export actual recorded PyFlyt actions to a constant real-time-rate video."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

# This renders our own simulator offscreen; it does not capture desktop windows.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np
from PIL import Image, ImageDraw

from course import Config, Course, CONTROL_HZ
from run_course import View


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--hold-seconds", type=float, default=2.)
    parser.add_argument("--ffmpeg", type=Path)
    parser.add_argument("--first-frame", type=int, default=0, help="Internal chunk export: inclusive frame index")
    parser.add_argument("--last-frame", type=int, help="Internal chunk export: exclusive frame index")
    args = parser.parse_args()
    if args.fps <= 0 or CONTROL_HZ % args.fps or args.hold_seconds < 0:
        parser.error("FPS must divide 120; final hold must be nonnegative")
    if args.first_frame < 0 or (args.last_frame is not None and args.last_frame <= args.first_frame):
        parser.error("Use a nonempty forward frame range")
    if args.output.exists():
        parser.error("Choose a new output filename")
    recording = json.loads(args.source.read_text(encoding="utf-8"))
    if not recording.get("decisions"):
        parser.error("Source has no recorded decisions")
    binary = str(args.ffmpeg) if args.ffmpeg else shutil.which("ffmpeg")
    if not binary:
        import imageio_ffmpeg
        binary = imageio_ffmpeg.get_ffmpeg_exe()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    error_path = args.output.with_suffix(".ffmpeg.log")
    course = Course(Config(**recording["config"]))
    view = View()
    view.display_policy = "SINGLE JEV | REPLAY 1x"
    if recording.get("status") == "api_error" and "HTTP 402" in recording.get("error", ""):
        view.recorded_stop_reason = "insufficient_credits"
    view.camera = "chase"
    width, height = view.screen.get_size()
    process = None
    samples = []
    frames, steps, frame_index = 0, 0, 0
    steps_per_frame = CONTROL_HZ // args.fps
    snapshot_steps = {0, round(9 * CONTROL_HZ), round(18 * CONTROL_HZ)}
    started = time.perf_counter()
    last_frame = None
    error_file = error_path.open("wb")
    try:
        process = subprocess.Popen([binary, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", f"{width}x{height}",
            "-framerate", str(args.fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
            "-preset", "fast", "-crf", "17", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(args.output)], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=error_file)

        def capture(action, confidence):
            nonlocal frames, last_frame, frame_index
            current_index = frame_index
            frame_index += 1
            if current_index < args.first_frame or (args.last_frame is not None and current_index >= args.last_frame):
                return
            view.draw(course, "single", action, confidence, False, realtime=False, recording=True)
            pixels = np.transpose(view.pg.surfarray.array3d(view.screen), (1, 0, 2)).copy()
            last_frame = pixels.tobytes()
            process.stdin.write(last_frame)
            if steps in snapshot_steps or course.status != "flying":
                path = args.output.parent / f"flight_{course.sim.elapsed_time:06.2f}s.png"
                Image.fromarray(pixels).save(path)
                samples.append((course.sim.elapsed_time, Image.fromarray(pixels).resize((720, 430))))
            frames += 1
            if frames % 150 == 0:
                print(f"FRAME {frames} simulation={course.sim.elapsed_time:.2f}s gates={course.passed}/{len(course.gates)}", flush=True)

        first = recording["decisions"][0]
        capture(first["action"], first.get("confidence"))
        for row in recording["decisions"]:
            if abs(course.sim.elapsed_time - row["time_seconds"]) > 1e-8:
                raise RuntimeError("Recorded decision time does not match reconstructed physics")
            duration_steps = row.get("applied_steps", round(course.config.decision_seconds * CONTROL_HZ))
            for step_index in range(duration_steps):
                course.step(row["action"])
                steps += 1
                # API failure occurs between decisions, with physics paused.
                # Reproduce the recorded stop marker after the final physical
                # step; it does not modify the reconstructed pose or actions.
                if (row is recording["decisions"][-1] and step_index == duration_steps - 1
                        and recording["status"] == "api_error" and course.status == "flying"):
                    course.status = "api_error"
                if steps % steps_per_frame == 0:
                    capture(row["action"], row.get("confidence"))
        if steps % steps_per_frame:
            capture(row["action"], row.get("confidence"))
        if course.status != recording["status"] or course.passed != recording["gates_passed"]:
            raise RuntimeError("Replay outcome differs from the original recording")
        np.testing.assert_allclose(course.position(), recording["final_position_m"], atol=1e-8)
        held_frames = round(args.hold_seconds * args.fps)
        for _ in range(held_frames):
            if frame_index >= args.first_frame and (args.last_frame is None or frame_index < args.last_frame):
                if last_frame is None:
                    view.draw(course, "single", row["action"], row.get("confidence"), False, realtime=False, recording=True)
                    last_frame = np.transpose(view.pg.surfarray.array3d(view.screen), (1, 0, 2)).copy().tobytes()
                process.stdin.write(last_frame)
                frames += 1
            frame_index += 1
        process.stdin.close()
        code = process.wait(timeout=120)
        if code:
            raise RuntimeError(f"FFmpeg exited with {code}; see {error_path}")
        sheet = Image.new("RGB", (1440, 920), (14, 22, 32))
        draw = ImageDraw.Draw(sheet)
        for index, (timestamp, picture) in enumerate(samples[:4]):
            x, y = (index % 2) * 720, (index // 2) * 460
            draw.text((x + 12, y + 8), f"Simulation time {timestamp:.2f}s", fill=(230, 237, 246))
            sheet.paste(picture, (x, y + 28))
        sheet.save(args.output.with_suffix(".contact.png"))
        metadata = {"source": str(args.source.resolve()),
                    "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
                    "source_model": recording.get("model"), "source_prompt_version": recording.get("prompt_version"),
                    "output": str(args.output.resolve()), "width": width, "height": height,
                    "fps": args.fps, "frames": frames, "video_duration_seconds": frames / args.fps,
                    "source_frame_count": frame_index, "frame_range": [args.first_frame, args.last_frame],
                    "simulation_duration_seconds": course.sim.elapsed_time,
                    "final_hold_seconds": held_frames / args.fps,
                    "speed": 1, "physics_steps_replayed": steps,
                    "status": course.status, "gates_passed": course.passed,
                    "final_position_matches_original": True,
                    "note": "Actual recorded decisions and full PyFlyt physics. Model waits removed; one video second per simulated second, within one-frame sampling resolution. No API calls.",
                    "export_wall_seconds": time.perf_counter() - started}
        args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(json.dumps(metadata), flush=True)
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        error_file.close()
        view.close()
        course.close()


if __name__ == "__main__":
    main()
