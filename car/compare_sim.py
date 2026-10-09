"""Show JEV and the trained DQN driving the same traffic seed side by side."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pygame
from stable_baselines3 import DQN

from grid_jev import GridJevPolicy, describe_lane_options
from grid_world import (
    EGO_COLUMN,
    FIXED_TARGET_SPEED_M_S,
    GRID_COLUMNS,
    GRID_ROWS,
    grid_action_allowed,
    lane_action_choices,
    lane_change_in_progress,
    lane_change_is_safe,
    lane_change_safety_summary,
    lane_kinematics_summary,
    occupancy_grid,
    traffic_forecast,
    target_speed_for_scene,
    urgent_overtake_action,
)
from rl_baseline import GRID_MODEL_PATH, make_env


ROOT = Path(__file__).resolve().parent
PANEL_W, PANEL_H = 720, 360
HEADER_H, FOOTER_H = 44, 116
FPS, SIM_STEP_S = 30, 1.0 / 3.0
def grid_action_choices(grid: np.ndarray) -> dict[str, str]:
    ego_cells = np.argwhere(grid == 2)
    # A terminal or briefly inconsistent simulator frame must not crash the UI.
    # Keep the policy in its lane until the occupancy view contains the ego again.
    if len(ego_cells) != 1:
        return {"stay": "Keep the current lane; the grid has no reliable ego cell."}
    ego_row = int(ego_cells[0][0])
    actions = {"stay": "Stay in your current lane and keep the fixed high speed."}
    if ego_row > 0 and grid_action_allowed(grid, 2):
        actions["move_up"] = "Move one lane up (toward the physical left/top row)."
    if ego_row < GRID_ROWS - 1 and grid_action_allowed(grid, 0):
        actions["move_down"] = "Move one lane down (toward the physical right/bottom row)."
    return actions


def draw_panel(screen, font, small_font, x: int, title: str, accent: tuple[int, int, int], frame: np.ndarray,
               env, action_label: str, extra: str, grid: np.ndarray, grid_font, cell_font) -> None:
    pygame.draw.rect(screen, (17, 25, 35), (x, 0, PANEL_W, PANEL_H + HEADER_H + FOOTER_H))
    pygame.draw.rect(screen, accent, (x, 0, PANEL_W, HEADER_H), border_radius=7)
    screen.blit(font.render(title, True, (248, 250, 252)), (x + 15, 10))
    if frame is not None:
        image = pygame.surfarray.make_surface(np.swapaxes(frame, 0, 1))
        image = pygame.transform.smoothscale(image, (PANEL_W, PANEL_H))
        screen.blit(image, (x, HEADER_H))
    footer_y = HEADER_H + PANEL_H + 10
    if env is None:
        speed, lane = 0.0, 0
    else:
        vehicle = env.unwrapped.vehicle
        speed = float(vehicle.speed)
        target_speed = float(vehicle.target_speed)
        lane = int(vehicle.lane_index[2]) + 1
    if env is None:
        target_speed = 0.0
    pygame.draw.rect(screen, (10, 17, 25), (x + 10, footer_y, PANEL_W - 20, FOOTER_H - 16), border_radius=8)
    pygame.draw.rect(screen, accent, (x + 10, footer_y, PANEL_W - 20, FOOTER_H - 16), 1, border_radius=8)
    screen.blit(small_font.render(f"Speed  {speed:4.1f}/{target_speed:.0f} target m/s     Lane  {lane}/5", True, (226, 234, 242)), (x + 24, footer_y + 12))
    screen.blit(small_font.render(f"Action  {action_label}", True, accent), (x + 24, footer_y + 40))
    screen.blit(small_font.render(extra, True, (181, 195, 208)), (x + 24, footer_y + 69))
    screen.blit(grid_font.render("INPUT GRID: 0 EMPTY  1 TRAFFIC  2 EGO", True, (181, 195, 208)),
                (x + 410, footer_y + 5))
    cell_w, cell_h = 16, 13
    cell_colors = {0: (25, 35, 47), 1: (104, 190, 235), 2: accent}
    for row in range(GRID_ROWS):
        for col in range(GRID_COLUMNS):
            value = int(grid[row, col])
            rect = pygame.Rect(x + 412 + col * cell_w, footer_y + 25 + row * cell_h, cell_w - 2, cell_h - 2)
            pygame.draw.rect(screen, cell_colors[value], rect, border_radius=2)
            text_color = (142, 158, 174) if value == 0 else (8, 20, 29)
            text = cell_font.render(str(value), True, text_color)
            screen.blit(text, text.get_rect(center=rect.center))


def metrics(name: str, env, state: dict, speeds: list[float], lane_changes: int, done: bool, info: dict) -> dict:
    return {
        "model": name,
        "mean_speed_m_s": round(float(np.mean(speeds)), 2) if speeds else 0.0,
        "lane_changes": lane_changes,
        "collision": bool(info.get("crashed", False)),
        "completed": bool(done and not info.get("crashed", False)),
        "steps": len(speeds),
    }


def run(seed: int = 101, vehicles_count: int = 50, model_path: Path = GRID_MODEL_PATH.with_suffix(".zip")) -> None:
    model = DQN.load(str(model_path))
    env_dqn = make_env(render_mode="rgb_array", screen_size=(PANEL_W, PANEL_H), vehicles_count=vehicles_count,
                       grid_observation=True, lane_only=True)
    env_jev = make_env(render_mode="rgb_array", screen_size=(PANEL_W, PANEL_H), vehicles_count=vehicles_count,
                       grid_observation=False, lane_only=True)
    driver = GridJevPolicy()
    pygame.init()
    screen = pygame.display.set_mode((PANEL_W * 2, HEADER_H + PANEL_H + FOOTER_H))
    pygame.display.set_caption(f"{vehicles_count} traffic cars • Same seed • DQN vs JEV • Esc to close")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 27)
    small_font = pygame.font.Font(None, 23)
    grid_font = pygame.font.Font(None, 17)
    cell_font = pygame.font.Font(None, 15)

    observation, _ = env_dqn.reset(seed=seed)
    env_jev.reset(seed=seed)
    env_jev.unwrapped.vehicle.target_speed = FIXED_TARGET_SPEED_M_S
    grid_dqn = observation.reshape(GRID_ROWS, GRID_COLUMNS).astype(np.int8)
    grid_jev = occupancy_grid(env_jev)
    speeds = {"DQN": [], "JEV": []}
    lanes = {"DQN": 0, "JEV": 0}
    previous_lanes = {
        "DQN": int(env_dqn.unwrapped.vehicle.lane_index[2]),
        "JEV": int(env_jev.unwrapped.vehicle.lane_index[2]),
    }
    done = {"DQN": False, "JEV": False}
    terminal_info = {"DQN": {}, "JEV": {}}
    safety_rejections = 0
    urgent_overtakes = 0
    labels = {"DQN": "STARTING", "JEV": "WAITING FOR JEV"}
    grids = {"DQN": grid_dqn, "JEV": grid_jev}
    frames = {"DQN": env_dqn.render(), "JEV": env_jev.render()}
    sim_time = 0.0
    last_step = time.perf_counter()
    running = True
    report_saved = False
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False

        now = time.perf_counter()
        if now - last_step >= SIM_STEP_S:
            last_step = now
            if not done["DQN"]:
                predicted, _ = model.predict(observation, deterministic=True)
                action_id = int(np.asarray(predicted).item())
                grid_dqn = observation.reshape(GRID_ROWS, GRID_COLUMNS).astype(np.int8)
                if not grid_action_allowed(grid_dqn, action_id):
                    action_id = 1
                observation, _reward, terminated, truncated, info = env_dqn.step(action_id)
                ego = env_dqn.unwrapped.vehicle
                speeds["DQN"].append(float(ego.speed))
                current_lane = int(ego.lane_index[2])
                lanes["DQN"] += int(current_lane != previous_lanes["DQN"])
                previous_lanes["DQN"] = current_lane
                labels["DQN"] = {0: "MOVE DOWN", 1: "STAY", 2: "MOVE UP"}.get(action_id, "UNKNOWN")
                done["DQN"] = bool(terminated or truncated)
                terminal_info["DQN"] = info
                frames["DQN"] = env_dqn.render()
                grids["DQN"] = occupancy_grid(env_dqn)
            if not done["JEV"]:
                grid_jev = occupancy_grid(env_jev)
                actions = lane_action_choices(grid_jev)
                changing_lanes = lane_change_in_progress(env_jev)
                if changing_lanes:
                    actions = {"stay": "Finish the current lane change before starting another."}
                lane_safety = lane_change_safety_summary(env_jev, actions)
                lane_kinematics = lane_kinematics_summary(env_jev, grid_jev, actions)
                urgent_action = urgent_overtake_action(lane_kinematics, lane_safety)
                desired_speed, speed_limited = target_speed_for_scene(
                    float(env_jev.unwrapped.vehicle.speed), FIXED_TARGET_SPEED_M_S,
                    lane_kinematics, lane_safety, changing_lanes,
                    current_target_speed_m_s=float(env_jev.unwrapped.vehicle.target_speed),
                )
                env_jev.unwrapped.vehicle.target_speed = desired_speed
                state = {
                    "occupancy_grid": grid_jev.tolist(),
                    "grid_shape": [GRID_ROWS, GRID_COLUMNS],
                    "meters_per_cell": 20,
                    "ego_cell": [int(np.argwhere(grid_jev == 2)[0][0]) if np.any(grid_jev == 2) else GRID_ROWS // 2, EGO_COLUMN],
                    "fixed_target_speed_m_s": FIXED_TARGET_SPEED_M_S,
                    "current_target_speed_m_s": round(desired_speed, 1),
                    "speed_mode": "traffic_limited" if speed_limited else "cruise",
                    "available_actions": list(actions),
                    "lane_change_in_progress": changing_lanes,
                    "lane_kinematics": lane_kinematics,
                    "lane_change_safety": lane_safety,
                    "traffic_forecast": traffic_forecast(env_jev, columns=GRID_COLUMNS),
                }
                new_decision, action_name = driver.update()
                driver.request(state, describe_lane_options(actions, state["lane_kinematics"], lane_safety))
                action_id = {"move_down": 0, "stay": 1, "move_up": 2}.get(action_name, 1) if new_decision else 1
                if changing_lanes:
                    action_id = 1
                rejected = action_id != 1 and not lane_change_is_safe(env_jev, action_id)
                if rejected:
                    safety_rejections += 1
                    action_id = 1
                urgent = action_id == 1 and urgent_action is not None and not changing_lanes
                if urgent:
                    action_id = {"move_down": 0, "move_up": 2}[urgent_action]
                    urgent_overtakes += 1
                _observation, _reward, terminated, truncated, info = env_jev.step(action_id)
                ego = env_jev.unwrapped.vehicle
                speeds["JEV"].append(float(ego.speed))
                current_lane = int(ego.lane_index[2])
                lanes["JEV"] += int(current_lane != previous_lanes["JEV"])
                previous_lanes["JEV"] = current_lane
                labels["JEV"] = (
                    f"URGENT {urgent_action.upper().replace('_', ' ')}" if urgent
                    else "SAFETY HOLD" if rejected
                    else {0: "MOVE DOWN", 1: "STAY", 2: "MOVE UP"}.get(action_id, "STAY")
                )
                done["JEV"] = bool(terminated or truncated)
                terminal_info["JEV"] = info
                frames["JEV"] = env_jev.render()
                grids["JEV"] = occupancy_grid(env_jev)
            sim_time += SIM_STEP_S

        extra_dqn = f"Steps {len(speeds['DQN']):3d}   Lane changes {lanes['DQN']}   " + (
            "COLLISION" if terminal_info["DQN"].get("crashed") else "COMPLETE" if done["DQN"] else "DRIVING"
        )
        extra_jev = f"Steps {len(speeds['JEV']):3d}   Lane changes {lanes['JEV']}   " + (
            f"JEV calls {driver.decisions}   Safety holds {safety_rejections}   Urgent {urgent_overtakes}   " + ("COLLISION" if terminal_info["JEV"].get("crashed") else "COMPLETE" if done["JEV"] else "DRIVING")
        )
        screen.fill((8, 13, 20))
        draw_panel(screen, font, small_font, 0, f"DQN  •  Stable-Baselines3  •  {len(env_dqn.unwrapped.road.vehicles) - 1} traffic cars", (78, 163, 255), frames["DQN"],
                   env_dqn, labels["DQN"], extra_dqn, grids["DQN"], grid_font, cell_font)
        draw_panel(screen, font, small_font, PANEL_W, f"JEV  •  typesafe/jev-1.13  •  {len(env_jev.unwrapped.road.vehicles) - 1} traffic cars", (46, 213, 157), frames["JEV"],
                   env_jev, labels["JEV"], extra_jev, grids["JEV"], grid_font, cell_font)
        pygame.display.flip()
        clock.tick(FPS)
        if done["DQN"] and done["JEV"] and not report_saved:
            dqn_result = metrics("DQN", env_dqn, {}, speeds["DQN"], lanes["DQN"], done["DQN"], terminal_info["DQN"])
            jev_result = metrics("JEV", env_jev, {}, speeds["JEV"], lanes["JEV"], done["JEV"], terminal_info["JEV"])
            report = {
                "seed": seed,
                "traffic_vehicles": vehicles_count,
                "dqn_training_traffic_vehicles": vehicles_count,
                "observation": "5x15 occupancy grid: 0 empty, 1 other car, 2 ego",
                "fixed_target_speed_m_s": FIXED_TARGET_SPEED_M_S,
                "action_set": ["move_up", "stay", "move_down"],
                "distribution_shift_stress_test": False,
                "jev_safety_rejections": safety_rejections,
                "jev_urgent_overtakes": urgent_overtakes,
                "paired_results": [dqn_result, jev_result],
            }
            (ROOT / "artifacts").mkdir(parents=True, exist_ok=True)
            (ROOT / "artifacts" / f"parallel_eval_{seed}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            print("PARALLEL RESULT " + json.dumps(report))
            report_saved = True

    env_dqn.close()
    env_jev.close()
    pygame.quit()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--vehicles", type=int, default=50, help="traffic vehicles in both parallel simulations")
    parser.add_argument("--model", type=Path, default=GRID_MODEL_PATH.with_suffix(".zip"))
    args = parser.parse_args()
    run(args.seed, args.vehicles, args.model)


if __name__ == "__main__":
    main()
