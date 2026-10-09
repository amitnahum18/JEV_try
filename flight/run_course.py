"""Run, compare, or interactively replay JEV's PyFlyt obstacle-course decisions."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import time
import textwrap

import numpy as np
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "car"))
from jev_client import MODEL, JevError, choose_action
from course import ACTIONS, CONTROL_HZ, Config, Course
from axis_policy import policy_state, requests_for, decide
from navigation import NAVIGATION_INSTRUCTIONS, NAVIGATION_SCHEMA
from maneuver_context import FOCUSED_SCHEMA, BUFFERED_SCHEMA, PITCH_SCHEMA

load_dotenv(ROOT / ".env")
PROMPT = (
    "You pilot a fixed-wing aircraft through a static obstacle course in a simulator. "
    "Choose exactly one of the six navigation maneuvers. Pass the next numbered gate "
    "inside its rectangular opening, then continue in order. Avoid solid gate frames "
    "and the ground. Use aircraft velocity, current attitude, gate-relative coordinates, "
    "bearing, elevation, center errors and time to the gate. The straight-velocity "
    "crossing forecast estimates where current velocity would cross the opening "
    "if no further turn or acceleration occurred. Negative clearance means missing "
    "the opening. Compare heading with velocity track: the nose and actual motion "
    "can point differently. Account for inertia: begin correcting "
    "early and reverse a maneuver when continuing would overshoot. Prefer small "
    "corrections; a roll maneuver banks the aircraft and changes its heading. "
    "There is no hold action. Keep altitude and heading near the opening and avoid "
    "overcorrecting when already aligned. Positive world y is LEFT; positive z is UP. "
    "Left/right use a moderate bank, while roll_left/roll_right use a stronger bank. "
    "Up/down return the wings toward level. Use the next gates to anticipate the "
    "slalom; straighten the velocity before reaching the frame instead of endlessly "
    "turning toward its center. Read the controls in the state for exact magnitudes. "
    "The basic stabilizer translates your action but does not select a route."
    " Use `aircraft.position_world_xyz_m`, `aircraft.velocity_world_xyz_m_s`, "
    "`aircraft.roll_pitch_yaw_deg`, and `next_gates[0].relative_world_xyz_m` "
    "for the immediate target. The second relative coordinate is signed LEFT, "
    "the third is signed UP. The first gate in `next_gates` is the priority."
) + NAVIGATION_INSTRUCTIONS


def write_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    for attempt in range(8):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if attempt == 7:
                raise
            # A Windows reader or virus scanner can briefly lock the destination.
            time.sleep(.02 * (attempt + 1))


class View:
    def __init__(self, preview_scale: float = 1.):
        import pygame
        self.pg = pygame
        self.preview_scale = preview_scale
        pygame.init()
        self.screen = pygame.display.set_mode((1440, 860))
        pygame.display.set_caption("JEV Flight Lab - PyFlyt obstacle course")
        self.font = pygame.font.SysFont("segoeui", 20)
        self.small = pygame.font.SysFont("segoeui", 16)
        self.big = pygame.font.SysFont("segoeui", 30, bold=True)
        self.clock = pygame.time.Clock()
        self.paused = False
        self.camera = "chase"
        self.show_input = False
        self.input_state = None
        self.input_scroll = 0
        self.was_paused_before_input = False
        self.mono = pygame.font.SysFont("consolas", 14)

    def events(self) -> str | None:
        for event in self.pg.event.get():
            if event.type == self.pg.QUIT:
                return "close"
            if event.type == self.pg.MOUSEWHEEL and self.show_input:
                self.input_scroll = max(0, self.input_scroll - event.y * 3)
            if event.type == self.pg.KEYDOWN:
                if event.key == self.pg.K_ESCAPE:
                    return "close"
                if event.key == self.pg.K_SPACE and not self.show_input:
                    self.paused = not self.paused
                if event.key == self.pg.K_c:
                    self.camera = "overview" if self.camera == "chase" else "chase"
                if event.key == self.pg.K_i:
                    self.show_input = not self.show_input
                    if self.show_input:
                        self.was_paused_before_input = self.paused
                        self.paused = True
                    else:
                        self.paused = self.was_paused_before_input
                if self.show_input and event.key in (self.pg.K_DOWN, self.pg.K_PAGEDOWN):
                    self.input_scroll += 15 if event.key == self.pg.K_PAGEDOWN else 1
                if self.show_input and event.key in (self.pg.K_UP, self.pg.K_PAGEUP):
                    self.input_scroll = max(0, self.input_scroll - (15 if event.key == self.pg.K_PAGEUP else 1))
                if event.key == self.pg.K_r:
                    return "retry"
                if event.key == self.pg.K_n:
                    return "new"
                if event.key == self.pg.K_b:
                    return "random"
                if event.key == self.pg.K_j:
                    return "jev"
                if event.key in (self.pg.K_1, self.pg.K_2, self.pg.K_3):
                    return f"level{event.unicode}"
        return None

    def draw(self, course: Course, policy: str, action: str | None, confidence, thinking: bool,
             *, realtime: bool = True, recording: bool = False) -> None:
        pg = self.pg
        self.screen.fill((14, 22, 32))
        self.screen.blit(self.big.render("JEV FLIGHT LAB", True, (110, 226, 194)), (18, 12))
        title = f"{getattr(self, 'display_policy', policy).upper()} | LEVEL {course.config.level} | SEED {course.config.seed}"
        self.screen.blit(self.font.render(title, True, (224, 232, 240)), (760, 18))
        image = course.image(round(1060 * self.preview_scale), round(596 * self.preview_scale), self.camera)
        surface = pg.surfarray.make_surface(np.transpose(image, (1, 0, 2)))
        if surface.get_size() != (1060, 596):
            surface = pg.transform.smoothscale(surface, (1060, 596))
        self.screen.blit(surface, (0, 58))
        obs = course.observation(action)
        aircraft = obs["aircraft"]
        next_gate = obs["next_gates"][0] if obs["next_gates"] else None
        angles = aircraft["roll_pitch_yaw_deg"]
        action_label = (action or "WAITING").upper()
        if action and "__" in action:
            vertical, horizontal = action.split("__")
            action_label = f"PITCH {vertical.upper()} / {horizontal.upper()}"
        lines = [
            f"GATES {course.passed}/{len(course.gates)}  |  SCORE {course.score}  |  {course.sim.elapsed_time:.1f}s",
            f"SPEED {aircraft['speed_m_s']:.1f} m/s   ALT {aircraft['position_world_xyz_m'][2]:.1f} m",
            f"BANK {-angles[0]:+.1f} deg   NOSE UP {-angles[1]:+.1f} deg",
            f"HEADING {aircraft['heading_left_deg']:+.1f} deg   TRACK {aircraft['velocity_track_left_deg']:+.1f} deg",
            f"CLIMB {aircraft['climb_rate_m_s']:+.1f} m/s   LOAD {aircraft['specific_force_load_g']:.2f}g",
            action_label if action and "__" in action else (f"ACTION {action_label}   CONF {confidence:.0%}" if confidence is not None else f"ACTION {action_label}"),
        ]
        if next_gate:
            forecast = next_gate["straight_velocity_crossing_forecast"]
            lines.extend([
                f"GATE {next_gate['number']}   DIST {next_gate['distance_m']:.1f} m",
                f"CENTER LEFT {next_gate['current_center_error_left_m']:+.1f} m",
                f"CENTER UP {next_gate['current_center_error_up_m']:+.1f} m",
                f"TRACK ERROR {next_gate['velocity_track_error_left_deg']:+.1f} deg",
                f"CLIMB ERROR {next_gate['flight_path_error_up_deg']:+.1f} deg",
            ])
            if forecast:
                lines.append(f"PREDICTED ERROR L {forecast['center_error_left_m']:+.1f} / U {forecast['center_error_up_m']:+.1f} m")
        for index, line in enumerate(lines):
            self.screen.blit(self.small.render(line, True, (228, 237, 246)), (1077, 78 + index * 25))
        phase = "PAUSED" if self.paused else "JEV THINKING" if thinking else course.status.upper().replace("_", " ")
        color = (255, 189, 100) if thinking else (110, 226, 194)
        self.screen.blit(self.font.render(phase, True, color), (1077, 390))
        self.screen.blit(self.small.render("ACTUAL CONTROL SURFACE DEFLECTION", True, (110, 226, 194)), (1077, 431))
        actuators = aircraft["actuators"]
        measured = actuators["measured_surface_deflection_deg"]
        for index, (name, limit) in enumerate((("left_aileron", 30), ("right_aileron", 30), ("elevator", 20), ("rudder", 20), ("throttle", 1))):
            y = 467 + index * 29
            value = actuators.get("measured_throttle_normalized", 0.) if name == "throttle" else measured[name]
            label = name.upper().replace("_", " ")
            self.screen.blit(self.small.render(label, True, (193, 209, 225)), (1077, y - 9))
            pg.draw.line(self.screen, (62, 80, 98), (1240, y), (1330, y), 7)
            pg.draw.line(self.screen, (170, 184, 197), (1285, y - 7), (1285, y + 7))
            normalized = max(-1., min(1., value / limit))
            pg.draw.line(self.screen, (110, 226, 194), (1285, y), (1285 + round(normalized * 43), y), 5)
            amount = f"{value:.0%}" if name == "throttle" else f"{value:+.1f} deg"
            self.screen.blit(self.small.render(amount, True, (236, 242, 248)), (1343, y - 9))
        timing_label = "Recorded flight | simulation speed 1x" if recording else "Physics pauses during model calls."
        self.screen.blit(self.small.render(timing_label, True, (160, 176, 194)), (1077, 620))
        self._attitude(angles)
        if course.status != "flying":
            pg.draw.rect(self.screen, (14, 25, 39), (200, 290, 680, 110), border_radius=10)
            message = "COURSE COMPLETED" if course.status == "completed" else course.status.upper().replace("_", " ")
            if recording and getattr(self, "recorded_stop_reason", None) == "insufficient_credits":
                message = "STOPPED: API CREDITS EXHAUSTED"
            self.screen.blit(self.big.render(message, True, (255, 194, 99)), (228, 308))
            end_label = f"{course.passed} GATES PASSED | ACTUAL RECORDED JEV DECISIONS" if recording and course.status == "completed" else (f"{course.passed}/{len(course.gates)} GATES | ACTUAL RECORDED ATTEMPT" if recording else "Attempt ended. R: retry   J: new JEV attempt   I: inspect input")
            self.screen.blit(self.font.render(end_label, True, (232, 240, 248)), (228, 353))
        elif thinking:
            pg.draw.rect(self.screen, (14, 25, 39), (280, 602, 580, 36), border_radius=5)
            self.screen.blit(self.small.render("JEV is choosing the next maneuver â€” physics paused", True, (255, 204, 119)), (304, 610))
        pg.draw.rect(self.screen, (23, 37, 52), (0, 654, 1440, 206))
        hint = "R: retry  N: new seed  J: JEV  B: random  1/2/3: level  C: camera  I: exact input  SPACE: pause  ESC: exit"
        if recording:
            hint = "RECORDED JEV FLIGHT | 1x SIMULATION SPEED | PITCH AND BANK COMMANDS APPLIED TOGETHER"
        self.screen.blit(self.font.render(hint, True, (220, 231, 240)), (20, 665))
        self._map(course)
        self.screen.blit(self.small.render("Orange: actual flight path | Gray: gate centers | Green: openings", True, (184, 205, 221)), (650, 719))
        if next_gate:
            self.screen.blit(self.font.render(f"GATE {next_gate['number']} NAVIGATION", True, (110, 226, 194)), (650, 750))
            errors = f"LEFT {next_gate['current_center_error_left_m']:+.2f} m   UP {next_gate['current_center_error_up_m']:+.2f} m"
            self.screen.blit(self.font.render(errors, True, (239, 244, 249)), (650, 775))
            navigation = obs["navigation"]
            turn_rate = navigation["measurements"]["track_turn_rate_left_deg_s"]
            rate_label = f"{turn_rate:+.1f} deg/s" if turn_rate is not None else "unavailable"
            alignment = f"TRACK ERROR {next_gate['velocity_track_error_left_deg']:+.1f} deg   TURN RATE LEFT {rate_label}"
            self.screen.blit(self.small.render(alignment, True, (239, 244, 249)), (650, 803))
            leveling = navigation["leveling_estimate"]
            extra = leveling["estimated_additional_track_turn_left_deg"]
            duration = leveling["estimated_seconds_to_bank_within_3_deg"]
            estimate = f"LEVELING EST. {duration:.2f}s   EXTRA TURN EST. {extra:+.1f} deg   I: inputs" if extra is not None and duration is not None else "Leveling estimate unavailable   I: exact inputs"
            self.screen.blit(self.small.render(estimate, True, (255, 194, 99)), (650, 826))
        if self.show_input:
            self._input(course, action)
        if not recording:
            pg.display.flip()
        if realtime:
            self.clock.tick(20)

    def _attitude(self, angles) -> None:
        pg = self.pg
        cx, cy = 974, 149
        pg.draw.circle(self.screen, (22, 41, 59), (cx, cy), 62)
        pg.draw.line(self.screen, (88, 115, 137), (cx - 55, cy), (cx + 55, cy), 2)
        bank = math.radians(float(angles[0]))
        dx, dy = 44 * math.cos(bank), 44 * math.sin(bank)
        pg.draw.line(self.screen, (255, 193, 88), (cx - dx, cy - dy), (cx + dx, cy + dy), 5)
        pg.draw.line(self.screen, (255, 193, 88), (cx, cy - 13), (cx, cy + 15), 3)
        self.screen.blit(self.small.render(f"BANK {-angles[0]:+.1f}", True, (236, 244, 249)), (cx - 42, cy + 33))

    def _map(self, course: Course) -> None:
        pg = self.pg
        box = pg.Rect(20, 708, 600, 133)
        pg.draw.rect(self.screen, (12, 22, 33), box, border_radius=5)
        x_max = course.gates[-1]["center_m"][0] + 35
        def point(position):
            return (box.x + 15 + (position[0] + 15) / (x_max + 15) * (box.width - 30),
                    box.centery - position[1] / 40 * (box.height / 2 - 6))
        centers = [point(gate["center_m"]) for gate in course.gates]
        if len(centers) > 1:
            pg.draw.lines(self.screen, (70, 88, 105), False, centers, 1)
        for index, gate in enumerate(course.gates):
            x, y, z = gate["center_m"]
            color = (84, 150, 239) if index < course.passed else (80, 215, 169)
            pg.draw.line(self.screen, color, point([x, y - gate["half_width_m"], z]), point([x, y + gate["half_width_m"], z]), 3)
        points = [point(position) for position in course.trail]
        if len(points) > 1:
            pg.draw.lines(self.screen, (255, 179, 73), False, points, 2)
        p = point(course.position())
        pg.draw.circle(self.screen, (255, 230, 165), (round(p[0]), round(p[1])), 4)
        yaw = course.sim.state(0)[1, 2]
        pg.draw.line(self.screen, (255, 230, 165), p, (p[0] + 15 * math.cos(yaw), p[1] - 15 * math.sin(yaw)), 2)

    def _input(self, course: Course, action) -> None:
        pg = self.pg
        pg.draw.rect(self.screen, (9, 17, 26), (10, 70, 1040, 574))
        title = "CONTROLLER OBSERVATION"
        if self.input_state and ("questions" in self.input_state or "requests" in self.input_state):
            title = "EXACT INPUT SENT TO JEV"
        elif self.input_state and "recorded_requests" in self.input_state:
            title = "RECORDED INPUT AND CURRENT REPLAY STATE"
        self.screen.blit(self.font.render(title + "  |  I: close  |  scroll to inspect", True, (110, 226, 194)), (28, 82))
        raw = json.dumps(self.input_state or course.observation(action), ensure_ascii=False, indent=2)
        lines = []
        for line in raw.splitlines():
            lines.extend(textwrap.wrap(line, width=118, replace_whitespace=False, drop_whitespace=False) or [""])
        self.input_scroll = min(self.input_scroll, max(0, len(lines) - 26))
        for index, line in enumerate(lines[self.input_scroll:self.input_scroll + 26]):
            self.screen.blit(self.mono.render(line, True, (216, 229, 242)), (28, 117 + index * 19))

    def snapshot(self, path: Path) -> None:
        self.pg.image.save(self.screen, str(path))

    def close(self) -> None:
        self.pg.quit()


def episode(config: Config, policy: str, output: Path, view: View | None = None,
            replay: dict | None = None, telemetry: str = "full", prompt_version: int = 3) -> tuple[dict, str | None]:
    if policy in {"jev", "random"}:
        telemetry, prompt_version = "full", 3
    output.mkdir(parents=True, exist_ok=True)
    suffix = datetime.now().strftime("%H%M%S_%f")
    path = output / f"{policy}_level{config.level}_seed{config.seed}_{suffix}.json"
    course = Course(config)
    actions = dict(ACTIONS)
    instructions = PROMPT
    if config.control_profile == "legacy":
        actions.update({"left": "Left rudder turn with wings returning toward level.",
                        "right": "Right rudder turn with wings returning toward level.",
                        "roll_left": "Left coordinated turn with an 18-degree bank.",
                        "roll_right": "Right coordinated turn with an 18-degree bank."})
        instructions += " Legacy control exception: left/right use level-wing rudder turns; roll actions use 18-degree banks, as documented in this state's controls."
    if view:
        view.display_policy = f"replay {replay['policy']}" if replay else policy
        view.pg.display.set_caption(f"JEV Flight Lab | {view.display_policy} | seed {config.seed}")
        view.input_state = None
        view.show_input = False
        view.input_scroll = 0
    rng = np.random.default_rng(config.seed + 10000)
    action, confidence = None, None
    rows, trajectory = [], []
    record = {"schema_version": 5, "feature_schema": NAVIGATION_SCHEMA,
              "prompt_version": prompt_version, "policy": policy, "model": MODEL if policy in {"jev", "single", "parallel"} else None,
              "telemetry": telemetry if policy in {"single", "parallel"} else "full",
              "architecture": policy if policy in {"single", "parallel"} else None,
              "physics_library": f"PyFlyt {importlib.metadata.version('PyFlyt')}",
              "started_at": datetime.now().astimezone().isoformat(),
              "timing": "Physics pauses for JEV decisions; actions last at most the decision interval and refresh immediately after a gate passage.",
              "navigation": replay["navigation"] if replay else (f"{config.joint_choice_count} joint pitch/bank choices." if policy in {"single", "parallel"} else "Six discrete choices; basic attitude/rate and speed stabilization only."),
              "decisions": rows, "trajectory": trajectory,
              "course_gates": course.gates, "replay_source": replay.get("recording") if replay else None}
    if telemetry == "focused":
        record["feature_schema"] = PITCH_SCHEMA if prompt_version >= 7 else BUFFERED_SCHEMA if prompt_version >= 5 else FOCUSED_SCHEMA
    if replay:
        record["feature_schema"] = replay.get("feature_schema", NAVIGATION_SCHEMA)
        record["prompt_version"] = replay.get("prompt_version")
    pool = ThreadPoolExecutor(max_workers=1)
    future = None
    remaining = 0
    request_state = None
    command = None
    total_cost = 0.
    print(f"START {policy} level={config.level} seed={config.seed}", flush=True)
    try:
        while course.status == "flying":
            if view:
                command = view.events()
                if command:
                    course.status = "user_stopped"
                    break
                if view.paused:
                    view.draw(course, policy, action, confidence, future is not None)
                    continue
            if remaining == 0 and future is None:
                request_state = course.observation(action)
                request_state["recent_maneuvers"] = [{"action": row["action"], "time_seconds": row["time_seconds"],
                                                    "observed_result": row.get("observed_result")} for row in rows[-4:]]
                if policy == "jev":
                    payload = {"model": MODEL, "state": request_state,
                               "questions": {"action": {"type": "choice", "instructions": instructions, "criteria": actions}}}
                    write_json(output / "latest_jev_input.json", payload)
                    if view:
                        view.input_state = payload
                    future = pool.submit(choose_action, state=request_state, actions=actions,
                                         instructions=instructions, timeout_seconds=12.)
                elif policy in {"single", "parallel"}:
                    request_state = policy_state(request_state, telemetry, rows, interior_margin_m=1. if prompt_version >= 5 else 0., include_pitch_context=prompt_version >= 7, include_pitch_center=prompt_version >= 8)
                    axis_requests = requests_for(request_state, policy, prompt_version)
                    write_json(output / "latest_jev_input.json", {"requests": axis_requests})
                    if view:
                        view.input_state = {"requests": axis_requests}
                    future = pool.submit(decide, axis_requests)
                elif replay:
                    if len(rows) >= len(replay["decisions"]):
                        course.status = "replay_exhausted"
                        break
                    original = replay["decisions"][len(rows)]
                    if view:
                        view.input_state = {"recorded_requests": original.get("requests"), "recorded_state": original["state"], "replay_physics_state": request_state}
                    action = original["action"]
                    confidence = original.get("confidence")
                    rows.append({"number": len(rows) + 1, "time_seconds": course.sim.elapsed_time,
                                 "state": request_state, "action": action, "confidence": confidence})
                    remaining = original.get("applied_steps", round(config.decision_seconds * CONTROL_HZ))
                else:
                    if view:
                        view.input_state = request_state
                    action = str(rng.choice(list(ACTIONS)))
                    confidence = None
                    rows.append({"number": len(rows) + 1, "time_seconds": course.sim.elapsed_time,
                                 "state": request_state, "action": action})
                    remaining = round(config.decision_seconds * CONTROL_HZ)
            if future is not None:
                if future.done():
                    decision = future.result()
                    future = None
                    if isinstance(decision, dict):
                        action, confidence = decision["action"], decision["confidence"]
                        total_cost += decision["cost_usd"] or 0.
                        rows.append({"number": len(rows) + 1, "time_seconds": course.sim.elapsed_time,
                                     "state": request_state, "requests": axis_requests, **decision})
                    else:
                        action, confidence = decision.choice, decision.confidence
                        total_cost += decision.cost_usd or 0.
                        rows.append({"number": len(rows) + 1, "time_seconds": course.sim.elapsed_time,
                                     "state": request_state, "action": action, "confidence": confidence,
                                     "probabilities": decision.probabilities,
                                     "latency_seconds": decision.latency_seconds, "cost_usd": decision.cost_usd})
                    remaining = round(config.decision_seconds * CONTROL_HZ)
                else:
                    if view:
                        view.draw(course, policy, action, confidence, True)
                    else:
                        time.sleep(.01)
                    continue
            before_gates = course.passed
            steps = min(remaining, 6 if view else remaining)
            applied_steps = 0
            for _ in range(steps):
                course.step(action)
                applied_steps += 1
                if course.sim.aviary_steps % 6 == 0 or course.status != "flying":
                    physics_state = course.sim.state(0)
                    trajectory.append({"time_seconds": course.sim.elapsed_time,
                                       "position_m": physics_state[3].tolist(),
                                       "rpy_rad": physics_state[1].tolist()})
                if course.status != "flying" or (not replay and config.refresh_on_gate_pass and course.passed != before_gates):
                    break
            remaining -= applied_steps
            rows[-1]["applied_steps"] = rows[-1].get("applied_steps", 0) + applied_steps
            if not replay and config.refresh_on_gate_pass and course.passed != before_gates:
                remaining = 0
            if rows and (remaining == 0 or course.status != "flying"):
                feedback = course.observation(action)
                after = feedback["aircraft"]
                gate_after = feedback["next_gates"][0] if feedback["next_gates"] else None
                rows[-1]["observed_result"] = {"time_seconds": feedback["time_seconds"],
                                               "aircraft": after,
                                               "next_gates": feedback["next_gates"],
                                               "navigation": feedback["navigation"],
                                               "status": course.status,
                                               "bank_left_deg": after["bank_left_deg"],
                                               "nose_up_pitch_deg": after["nose_up_pitch_deg"],
                                               "heading_left_deg": after["heading_left_deg"],
                                               "velocity_track_left_deg": after["velocity_track_left_deg"],
                                               "climb_rate_m_s": after["climb_rate_m_s"],
                                               "gates_passed": course.passed,
                                               "next_gate": gate_after["number"] if gate_after else None,
                                               "center_error_left_m": gate_after["current_center_error_left_m"] if gate_after else None,
                                               "center_error_up_m": gate_after["current_center_error_up_m"] if gate_after else None}
            if before_gates != course.passed:
                print(f"GATE {policy} {course.passed}/{len(course.gates)} at {course.sim.elapsed_time:.2f}s", flush=True)
            record.update(course.summary())
            record["jev_cost_usd"] = total_cost
            write_json(path, record)
            if view:
                view.draw(course, policy, action, confidence, False)
    except Exception as exc:
        course.status = "api_error" if isinstance(exc, JevError) else "error"
        record["error"] = str(exc)
        print(f"ERROR {policy}: {exc}", flush=True)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        record.update(course.summary())
        record.update({"jev_cost_usd": total_cost, "finished_at": datetime.now().astimezone().isoformat(),
                       "recording": str(path.resolve()), "decision_count": len(rows)})
        write_json(path, record)
        print(f"RESULT {policy}: {record['status']} gates={record['gates_passed']}/{record['gates_total']} score={record['score']}", flush=True)
        if view:
            view.draw(course, policy, action, confidence, False)
            view.snapshot(output / "latest_view.png")
            if command is None:
                while command is None:
                    command = view.events()
                    view.draw(course, policy, action, confidence, False)
        course.close()
    return record, command


def report(output: Path, records: list[dict]) -> None:
    compact = [{key: value for key, value in row.items()
                if key not in {"decisions", "trajectory", "course_gates"}} for row in records]
    write_json(output / "comparison.json", {"episodes": compact,
               "note": "Controls are recorded per episode. Compare only matched seeds, action spaces and course settings. Preliminary sample, not a general intelligence score."})
    lines = ["# PyFlyt obstacle-course results", "", "| Policy | Level | Seed | Gates | Score | Status |", "|---|---|---|---|---|---|"]
    for row in compact:
        config = row["config"]
        lines.append(f"| {row['policy']} | {config['level']} | {config['seed']} | {row['gates_passed']}/{row['gates_total']} | {row['score']} | {row['status']} |")
    (output / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--policy", choices=("jev", "random", "single", "parallel"), default="single")
    parser.add_argument("--telemetry", choices=("basic", "full", "focused"))
    parser.add_argument("--prompt-version", type=int, choices=(1, 2, 3, 4, 5, 6, 7, 8))
    parser.add_argument("--pitch-options", choices=("coarse", "dense"))
    parser.add_argument("--compare-random", action="store_true")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--level", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--layout", choices=("slalom", "legacy", "maze", "hard_turns", "vertical_challenge"), default="vertical_challenge")
    parser.add_argument("--gates", type=int)
    parser.add_argument("--decision-seconds", type=float, default=.5)
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--preview-quality", choices=("full", "fast"), default="full",
                        help="Fast lowers only the live 3D render resolution; physics and HUD stay unchanged")
    args = parser.parse_args()
    if args.seed is None:
        args.seed = 44 if args.layout in {"hard_turns", "vertical_challenge"} else 42
    if args.gates is None:
        args.gates = 8 if args.layout in {"hard_turns", "vertical_challenge"} else 5
    if args.seconds is None:
        args.seconds = 120. if args.layout == "vertical_challenge" else 85. if args.layout == "hard_turns" else 35.
    if args.telemetry is None:
        args.telemetry = "focused" if args.policy == "single" else "full"
    if args.prompt_version is None:
        args.prompt_version = (8 if args.layout == "vertical_challenge" else 6 if args.layout == "hard_turns" else 5) if args.telemetry == "focused" else 3
    if args.pitch_options is None:
        args.pitch_options = "dense" if args.layout == "vertical_challenge" and args.prompt_version >= 8 else "coarse"
    if not args.replay and ((args.telemetry == "focused") != (args.prompt_version in (4, 5, 6, 7, 8))):
        parser.error("Focused telemetry and prompt version 4/5/6/7/8 must be used together.")
    if not args.replay and args.telemetry == "focused" and args.policy not in {"single", "parallel"}:
        parser.error("Focused telemetry uses a joint-axis single or parallel policy.")
    if args.episodes < 1 or args.gates < 1 or not 1 / CONTROL_HZ <= args.decision_seconds <= 3 or args.seconds <= 0:
        parser.error("Use positive episodes/gates/seconds and decision-seconds between 1/120 and 3.")
    if not args.headless and (args.compare_random or args.episodes != 1):
        parser.error("Multi-episode comparisons use --headless. Open recorded episodes with --replay.")
    if args.replay and (args.compare_random or args.episodes != 1):
        parser.error("--replay accepts exactly one episode")
    output = args.output_dir or ROOT / "flight" / "recordings" / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    output.mkdir(parents=True, exist_ok=True)
    replay = json.loads(args.replay.read_text(encoding="utf-8")) if args.replay else None
    if replay and replay.get("schema_version", 1) < 2:
        replay["config"].setdefault("layout", "legacy")
        replay["config"].setdefault("control_profile", "legacy")
    config = Config(**replay["config"]) if replay else Config(
        seed=args.seed, level=args.level, gates=args.gates,
        decision_seconds=round(args.decision_seconds * CONTROL_HZ) / CONTROL_HZ, max_seconds=args.seconds, layout=args.layout,
        extended_pitch_actions=args.layout == "vertical_challenge", dense_pitch_actions=args.pitch_options == "dense")
    if not replay and args.policy in {"single", "parallel"} and args.layout == "legacy":
        parser.error("Joint-axis policies use the slalom layout and banked_v2 controls.")
    if args.compare_random and args.policy in {"single", "parallel"}:
        parser.error("Use compare_axes.py for matched joint-axis comparisons.")
    view = None if args.headless else View(preview_scale=.5 if args.preview_quality == "fast" else 1.)
    records = []
    try:
        if view:
            policy = "replay" if replay else args.policy
            while True:
                record, command = episode(config, policy, output, view, replay, args.telemetry, args.prompt_version)
                records.append(record)
                report(output, records)
                if command == "close":
                    break
                if not (policy == "replay" and command == "retry"):
                    replay = None
                    if policy == "replay":
                        policy = args.policy
                if command == "new":
                    config.seed += 1
                elif command in {"random", "jev"}:
                    policy = command
                elif command and command.startswith("level"):
                    config.level = int(command[-1])
                view.paused = False
        else:
            policies = ("jev", "random") if args.compare_random else (args.policy,)
            for episode_index in range(args.episodes):
                for policy in policies:
                    run_config = Config(**vars(config))
                    run_config.seed += episode_index
                    record, _ = episode(run_config, "replay" if replay else policy, output, replay=replay, telemetry=args.telemetry, prompt_version=args.prompt_version)
                    records.append(record)
                    report(output, records)
    finally:
        if view:
            view.close()
    print(f"REPORT {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
