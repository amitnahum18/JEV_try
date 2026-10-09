"""Verify delivered MP4 timing against the actual simulation replay manifest."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import imageio_ffmpeg
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--ffprobe", type=Path)
    args = parser.parse_args()
    probe_binary = str(args.ffprobe) if args.ffprobe else shutil.which("ffprobe")
    if not probe_binary:
        parser.error("Provide --ffprobe or add ffprobe to PATH")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    probe = json.loads(subprocess.check_output([probe_binary, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height,r_frame_rate,nb_frames,duration:format=duration,size",
        "-of", "json", str(args.video)]))
    stream = probe["streams"][0]
    assert stream["width"] == manifest["width"] and stream["height"] == manifest["height"]
    assert stream["r_frame_rate"] == f"{manifest['fps']}/1"
    assert int(stream["nb_frames"]) == manifest["frames"]
    assert abs(float(probe["format"]["duration"]) - manifest["video_duration_seconds"]) < 1 / manifest["fps"]
    assert manifest["speed"] == 1 and manifest["final_position_matches_original"]
    duration = float(probe["format"]["duration"])
    sheet = Image.new("RGB", (1440, 912), (14,22,32))
    draw = ImageDraw.Draw(sheet)
    times = [duration*.08, duration*.34, duration*.66, duration-.5]
    for index, timestamp in enumerate(times):
        pixels = subprocess.check_output([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-ss", str(timestamp),
            "-i", str(args.video), "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])
        frame = Image.frombytes("RGB", (manifest["width"], manifest["height"]), pixels)
        frame.save(args.video.parent / f"verified_frame_{index}.png")
        x, y = (index % 2)*720, (index//2)*456
        draw.text((x+8,y+7), f"Actual MP4 at {timestamp:.2f}s", fill=(240,240,240))
        sheet.paste(frame.resize((720,430)), (x,y+26))
    sheet.save(args.video.parent / "verified_contact_sheet.png")
    verification = {"passed": True, "probe": probe, "source_export": manifest,
                    "video_sha256": hashlib.sha256(args.video.read_bytes()).hexdigest(),
                    "frames_inspected_seconds": times, "contact_sheet": "verified_contact_sheet.png"}
    (args.video.parent / "verification.json").write_text(json.dumps(verification,indent=2),encoding="utf-8")
    print(json.dumps({"passed":True,"frames":manifest["frames"],"duration":duration,
                      "gates":manifest["gates_passed"],"status":manifest["status"]}))


if __name__ == "__main__":
    main()
