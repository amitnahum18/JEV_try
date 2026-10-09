"""Run a single JEV driver in dense traffic and display its grid input."""

from __future__ import annotations

import json
import argparse
import time
from pathlib import Path

import numpy as np
import pygame
import imageio_ffmpeg

from grid_jev import GridJevPolicy, describe_lane_options
from grid_world import (
    FIXED_TARGET_SPEED_M_S,
    GRID_ROWS,
    lane_action_choices,
    lane_change_is_safe,
    lane_change_in_progress,
    lane_kinematics_summary,
    lane_change_safety_summary,
    occupancy_grid,
    traffic_forecast,
    target_speed_for_scene,
    urgent_overtake_action,
)
from rl_baseline import make_env


ROOT = Path(__file__).resolve().parent
SIM_STEP_S = 1.0 / 3.0
GRID_COLUMNS = 31
EGO_COLUMN = GRID_COLUMNS // 2
ROAD_WIDTH = 1180
ROAD_HEIGHT = 470
HEADER_HEIGHT = 44
FOOTER_HEIGHT = 126
WINDOW_HEIGHT = HEADER_HEIGHT + ROAD_HEIGHT + FOOTER_HEIGHT


def draw_jev_panel(screen, fonts, frame, env, action_label, extra) -> None:
    font, small_font = fonts
    screen.fill((17, 25, 35))
    pygame.draw.rect(screen, (255, 171, 76), (0, 0, ROAD_WIDTH, HEADER_HEIGHT), border_radius=7)
    screen.blit(font.render("JEV DRIVER - LONGER LOOKAHEAD", True, (248, 250, 252)), (15, 10))
    road = pygame.surfarray.make_surface(np.swapaxes(frame, 0, 1))
    road = pygame.transform.smoothscale(road, (ROAD_WIDTH, ROAD_HEIGHT))
    screen.blit(road, (0, HEADER_HEIGHT))

    footer_y = HEADER_HEIGHT + ROAD_HEIGHT + 10
    pygame.draw.rect(screen, (10, 17, 25), (10, footer_y, ROAD_WIDTH - 20, FOOTER_HEIGHT - 16), border_radius=8)
    pygame.draw.rect(screen, (255, 171, 76), (10, footer_y, ROAD_WIDTH - 20, FOOTER_HEIGHT - 16), 1, border_radius=8)
    vehicle = env.unwrapped.vehicle
    speed = float(vehicle.speed)
    lane = int(vehicle.lane_index[2]) + 1
    target_speed = float(vehicle.target_speed)
    screen.blit(small_font.render(f"SPEED  {speed:4.1f} / {target_speed:.0f} m/s", True, (226, 234, 242)), (24, footer_y + 11))
    screen.blit(small_font.render(f"LANE  {lane}/5", True, (226, 234, 242)), (390, footer_y + 11))
    screen.blit(small_font.render(f"ACTION  {action_label}", True, (255, 171, 76)), (24, footer_y + 45))
    screen.blit(small_font.render(extra, True, (181, 195, 208)), (390, footer_y + 45))


def run(seed: int = 101, vehicles_count: int = 50, record_video: bool = False) -> None:
    snapshot_path = ROOT / "artifacts" / f"jev_only_seed{seed}.png"
    report_path = ROOT / "artifacts" / f"jev_only_seed{seed}.json"
    video_path = ROOT / "artifacts" / f"jev_only_seed{seed}.mp4"
    env = make_env(
        render_mode="rgb_array",
        screen_size=(ROAD_WIDTH, ROAD_HEIGHT),
        vehicles_count=vehicles_count,
        grid_observation=False,
        lane_only=True,
    )
    env.reset(seed=seed)
    env.unwrapped.vehicle.target_speed = FIXED_TARGET_SPEED_M_S
    driver = GridJevPolicy()
    pygame.init()
    screen = pygame.display.set_mode((ROAD_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption(f"JEV only - {vehicles_count} traffic cars - Esc to close")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 27)
    small_font = pygame.font.Font(None, 27)

    speeds: list[float] = []
    targets: list[float] = []
    trajectory: list[dict] = []
    lane_changes = 0
    safety_rejections = 0
    urgent_overtakes = 0
    previous_lane = int(env.unwrapped.vehicle.lane_index[2])
    current_action = "WAITING FOR JEV"
    frame = env.render()
    grid = occupancy_grid(env, columns=GRID_COLUMNS)
    terminal_info: dict = {}
    done = False
    report_saved = False
    final_usage: dict | None = None
    sim_time = 0.0
    last_step = time.perf_counter()
    end_at: float | None = None
    video_writer = None
    if record_video:
        video_fps = round(1 / SIM_STEP_S)
        video_writer = imageio_ffmpeg.write_frames(
            str(video_path),
            (ROAD_WIDTH, WINDOW_HEIGHT),
            fps=video_fps,
            quality=8,
            codec="libx264",
            macro_block_size=1,
            output_params=["-preset", "ultrafast"],
        )
        video_writer.send(None)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                pygame.quit()
                env.close()
                return

        now = time.perf_counter()
        record_step_frame = False
        if not done and now - last_step >= SIM_STEP_S:
            last_step = now
            grid = occupancy_grid(env, columns=GRID_COLUMNS)
            actions = lane_action_choices(grid)
            changing_lanes = lane_change_in_progress(env)
            if changing_lanes:
                actions = {"stay": "Hold position and finish the current lane change before starting another."}
            lane_safety = lane_change_safety_summary(env, actions)
            lane_kinematics = lane_kinematics_summary(env, grid, actions)
            urgent_action = urgent_overtake_action(lane_kinematics, lane_safety)
            desired_speed, speed_limited = target_speed_for_scene(
                float(env.unwrapped.vehicle.speed), FIXED_TARGET_SPEED_M_S,
                lane_kinematics, lane_safety, changing_lanes,
                current_target_speed_m_s=float(env.unwrapped.vehicle.target_speed),
            )
            env.unwrapped.vehicle.target_speed = desired_speed
            state = {
                "occupancy_grid": grid.tolist(),
                "grid_shape": [GRID_ROWS, GRID_COLUMNS],
                "meters_per_cell": 20,
                "ego_cell": [
                    int(np.argwhere(grid == 2)[0][0]) if np.any(grid == 2) else GRID_ROWS // 2,
                    EGO_COLUMN,
                ],
                "fixed_target_speed_m_s": FIXED_TARGET_SPEED_M_S,
                "current_target_speed_m_s": round(desired_speed, 1),
                "speed_mode": "traffic_limited" if speed_limited else "cruise",
                "available_actions": list(actions),
                "lane_change_in_progress": changing_lanes,
                "lane_kinematics": lane_kinematics,
                "lane_change_safety": lane_safety,
                "traffic_forecast": traffic_forecast(env, columns=GRID_COLUMNS),
            }
            new_decision, action_name = driver.update()
            driver.request(state, describe_lane_options(actions, state["lane_kinematics"], lane_safety))
            action_id = {"move_down": 0, "stay": 1, "move_up": 2}.get(action_name, 1) if new_decision else 1
            if changing_lanes:
                action_id = 1
            rejected = action_id != 1 and not lane_change_is_safe(env, action_id)
            if rejected:
                safety_rejections += 1
                action_id = 1
            urgent = False
            if action_id == 1 and urgent_action is not None and not changing_lanes:
                action_id = {"move_down": 0, "move_up": 2}[urgent_action]
                urgent = True
                urgent_overtakes += 1
            current_action = (
                f"URGENT {urgent_action.upper().replace('_', ' ')}" if urgent
                else "SAFETY HOLD" if rejected
                else {0: "MOVE DOWN", 1: "STAY", 2: "MOVE UP"}[action_id]
            )
            before_vehicle = env.unwrapped.vehicle
            before_speed = float(before_vehicle.speed)
            before_lane = int(before_vehicle.lane_index[2])
            current_lane_facts = state["lane_kinematics"].get("stay", {})
            _observation, _reward, terminated, truncated, terminal_info = env.step(action_id)
            ego = env.unwrapped.vehicle
            speeds.append(float(ego.speed))
            targets.append(float(ego.target_speed))
            lane = int(ego.lane_index[2])
            lane_changes += int(lane != previous_lane)
            previous_lane = lane
            done = bool(terminated or truncated)
            record_step_frame = True
            trajectory.append({
                "step": len(speeds),
                "lane_before": before_lane,
                "speed_before_m_s": round(before_speed, 2),
                "requested_action": action_name if new_decision else "no_new_jev_decision",
                "applied_action": current_action,
                "safety_rejected": rejected,
                "urgent_overtake": urgent,
                "target_speed_m_s": round(float(ego.target_speed), 1),
                "speed_limited": speed_limited,
                "current_lane_front_gap_m": current_lane_facts.get("nearest_ahead_m"),
                "current_lane_front_ttc_s": current_lane_facts.get("ahead_ttc_s"),
                "lane_safety": lane_safety,
                "lane_after": lane,
                "speed_after_m_s": round(float(ego.speed), 2),
                "collision": bool(terminal_info.get("crashed", False)),
            })
            frame = env.render()
            grid = occupancy_grid(env, columns=GRID_COLUMNS)
            sim_time += SIM_STEP_S
            if done:
                driver.wait_for_pending(timeout_seconds=7.0)
                final_usage = driver.usage_report()
                result = {
                    "seed": seed,
                    "traffic_vehicles": vehicles_count,
                    "agent": "JEV",
                    "observation": "5x31 occupancy grid: 0 empty, 1 other car, 2 ego",
                    "grid_shape": [GRID_ROWS, GRID_COLUMNS],
                    "lookahead_m": EGO_COLUMN * 20,
                    "lookbehind_m": EGO_COLUMN * 20,
                    "fixed_target_speed_m_s": FIXED_TARGET_SPEED_M_S,
                    "minimum_target_speed_m_s": round(min(targets), 1) if targets else FIXED_TARGET_SPEED_M_S,
                    "speed_limited_steps": sum(1 for item in trajectory if item["speed_limited"]),
                    "steps": len(speeds),
                    "mean_speed_m_s": round(float(np.mean(speeds)), 2) if speeds else 0.0,
                    "lane_changes": lane_changes,
                    "safety_rejections": safety_rejections,
                    "urgent_overtakes": urgent_overtakes,
                    "collision": bool(terminal_info.get("crashed", False)),
                    "jev_usage": final_usage,
                    "trajectory": trajectory,
                }
                report_path.parent.mkdir(parents=True, exist_ok=True)
                report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
                report_saved = True
                end_at = now + 5.0

        status = "COLLISION" if terminal_info.get("crashed") else "COMPLETE" if done else "DRIVING"
        extra = f"Step {len(speeds)}   Lane changes {lane_changes}   {status}"
        draw_jev_panel(screen, (font, small_font), frame, env, current_action, extra)
        pygame.display.flip()
        if video_writer is not None and record_step_frame:
            frame_rgb = np.ascontiguousarray(pygame.surfarray.array3d(screen).transpose(1, 0, 2))
            video_writer.send(frame_rgb)
        if done and end_at is not None and now >= end_at:
            if video_writer is not None:
                for _ in range(round(5 / SIM_STEP_S) - 1):
                    video_writer.send(frame_rgb)
                video_writer.close()
            pygame.image.save(screen, snapshot_path)
            pygame.quit()
            env.close()
            if report_saved:
                print(f"JEV result: {report_path}")
                print("JEV usage: " + json.dumps(final_usage, separators=(",", ":")))
                if record_video:
                    print(f"JEV video: {video_path}")
            return
        clock.tick(30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--vehicles", type=int, default=50)
    parser.add_argument("--record", action="store_true", help="save a real-time MP4 video of the displayed JEV run")
    args = parser.parse_args()
    run(args.seed, args.vehicles, args.record)
