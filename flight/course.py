"""A fixed-wing obstacle course with actual PyFlyt/PyBullet flight physics."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np
from PyFlyt.core import Aviary
from navigation import navigation_features

CONTROL_HZ = 120
ACTIONS = {
    "up": "Pitch the nose up to climb, with wings returning toward level.",
    "down": "Pitch the nose down to descend, with wings returning toward level.",
    "left": "Turn left with a 25-degree left bank and coordinated rudder; nominal nose-up pitch.",
    "right": "Turn right with a 25-degree right bank and coordinated rudder; nominal nose-up pitch.",
    "roll_left": "Bank left more strongly, 40 degrees, for a tighter coordinated turn; nominal pitch.",
    "roll_right": "Bank right more strongly, 40 degrees, for a tighter coordinated turn; nominal pitch.",
}
VERTICAL_ACTIONS = {
    "up": "Target nose-up pitch 6 degrees to climb.",
    "down": "Target nose-down pitch 6 degrees to descend.",
    "level": "Target nominal nose-up trim of 2 degrees; not an altitude hold.",
}
HORIZONTAL_ACTIONS = {
    "straight": "Target wings level, zero coordinated yaw rate; not a heading hold.",
    "left": "Target left bank 25 degrees, coordinated turn.",
    "right": "Target right bank 25 degrees, coordinated turn.",
    "roll_left": "Target left bank 40 degrees, stronger coordinated turn.",
    "roll_right": "Target right bank 40 degrees, stronger coordinated turn.",
}
JOINT_ACTIONS = {f"{vertical}__{horizontal}": f"{vtext} Simultaneously: {htext}"
                 for vertical, vtext in VERTICAL_ACTIONS.items()
                 for horizontal, htext in HORIZONTAL_ACTIONS.items()}
ALL_JOINT_KEYS = set(JOINT_ACTIONS) | {
    f"{vertical}__{horizontal}" for vertical in ("up_strong", "down_strong", "up_soft", "up_medium", "down_soft", "down_medium")
    for horizontal in HORIZONTAL_ACTIONS}


@dataclass
class Config:
    seed: int = 42
    level: int = 1
    gates: int = 5
    decision_seconds: float = 0.5
    max_seconds: float = 35.0
    layout: str = "slalom"
    control_profile: str = "banked_v2"
    refresh_on_gate_pass: bool = True
    pitch_up_deg: float = 6.
    pitch_down_deg: float = 6.
    extended_pitch_actions: bool = False
    strong_pitch_up_deg: float = 20.
    strong_pitch_down_deg: float = 16.
    dense_pitch_actions: bool = False

    def pitch_targets(self) -> dict[str, float]:
        targets = {"up": self.pitch_up_deg, "down": -self.pitch_down_deg, "level": 2.}
        if self.extended_pitch_actions:
            targets.update(up_strong=self.strong_pitch_up_deg, down_strong=-self.strong_pitch_down_deg)
        if self.dense_pitch_actions:
            targets.update(up_soft=4., up_medium=12., down_soft=-3., down_medium=-10.)
        return targets

    @property
    def joint_choice_count(self) -> int:
        return len(self.pitch_targets()) * len(HORIZONTAL_ACTIONS)

    @property
    def speed(self) -> float:
        return (20.0, 25.0, 30.0)[self.level - 1]

    @property
    def half_width(self) -> float:
        return (8.0, 6.5, 5.0)[self.level - 1]

    @property
    def half_height(self) -> float:
        return (6.0, 5.0, 4.0)[self.level - 1]


def generate_gates(config: Config) -> list[dict]:
    """Seeded static geometry. This function does not select flight actions."""
    if config.layout not in {"legacy", "slalom", "maze", "hard_turns", "vertical_challenge"}:
        raise ValueError(f"Unknown layout: {config.layout}")
    rng = np.random.default_rng(config.seed)
    lateral, altitude = 0., 22.
    offset = (3., 4.5, 6.)[config.level - 1]
    first_side = int(rng.choice([-1, 1])) if config.layout in {"slalom", "maze", "hard_turns", "vertical_challenge"} else 1
    # Variable spacing, two consecutive turns to the same side, then reversals.
    # Height changes require climb/descent while banking, not a straight corridor.
    maze_lateral = [13., -18., 25., -24., 7., 27., -20., 13.]
    maze_altitude = [24., 31., 20., 34., 17., 28., 36., 22.]
    maze_spacing = [95., 155., 130., 160., 145., 160., 155., 160.]
    # Open frames only. Wider lateral reversals than the wall course, with
    # sufficient distance for the existing 40-degree bank at level-one speed.
    hard_lateral = [18., -32., 35., -34., 34., 8., -33., 31.]
    hard_altitude = [24., 30., 20., 33., 19., 29., 35., 22.]
    hard_spacing = [95., 155., 155., 170., 170., 140., 165., 175.]
    # Larger climbs/descents with forward room for an actual fixed-wing maneuver.
    vertical_altitude = [26., 48., 17., 58., 20., 54., 16., 42.]
    vertical_spacing = [95., 220., 210., 280., 240., 240., 250., 220.]
    x = 0.
    gates = []
    for number in range(config.gates):
        if config.layout == "slalom":
            amplitude = (14., 18., 22.)[config.level - 1]
            lateral = first_side * (-1) ** number * amplitude + float(rng.uniform(-1.5, 1.5))
            altitude = 22. + math.sin(number * 1.4) * (3.5 + config.level) + float(rng.uniform(-1, 1))
            center = [95. + 110. * number, lateral, altitude]
        elif config.layout in {"maze", "hard_turns", "vertical_challenge"}:
            lateral_shape, altitude_shape, spacing = (
                (hard_lateral, vertical_altitude, vertical_spacing) if config.layout == "vertical_challenge"
                else (hard_lateral, hard_altitude, hard_spacing) if config.layout == "hard_turns"
                else (maze_lateral, maze_altitude, maze_spacing))
            index = number % len(lateral_shape)
            x += spacing[index] if number == 0 else max(130., spacing[index])
            amplitude_scale = (1., 1.08, 1.16)[config.level - 1]
            lateral = first_side * lateral_shape[index] * amplitude_scale + float(rng.uniform(-.8, .8))
            altitude = altitude_shape[index] + float(rng.uniform(-.6, .6))
            center = [x, lateral, altitude]
        else:
            lateral = float(np.clip(lateral + rng.uniform(-offset, offset), -15, 15))
            altitude = float(np.clip(altitude + rng.uniform(-offset / 2, offset / 2), 14, 34))
            center = [70. + 75. * number, lateral, altitude]
        gates.append({"number": number + 1, "center_m": center,
                      "half_width_m": config.half_width, "half_height_m": config.half_height})
    return gates


class Course:
    def __init__(self, config: Config):
        self.config = config
        self.pitch_targets = config.pitch_targets()
        self.allowed_joint = {f"{v}__{h}" for v in self.pitch_targets for h in HORIZONTAL_ACTIONS}
        self.sim = Aviary(
            start_pos=np.array([[0., 0., 22.]]), start_orn=np.zeros((1, 3)),
            drone_type="fixedwing", render=False, physics_hz=240, world_scale=50.,
            drone_options={"control_hz": CONTROL_HZ,
                           "starting_velocity": np.array([config.speed, 0., 0.])},
            seed=config.seed,
        )
        # Direct surface mode keeps elevator and rudder separate on PyFlyt 0.29.
        self.sim.set_mode(-1)
        self.aircraft_id = self.sim.drones[0].Id
        self.rng = np.random.default_rng(config.seed)
        self.gates = generate_gates(config)
        self.gate_bodies = []
        for gate in self.gates:
            self._make_gate(gate)
        self.sim.register_all_new_bodies()
        self.passed = 0
        self.status = "flying"
        self.score = 0
        self.events = []
        self.pitch_integral = 0.
        self.minimum_clearance = None
        self.previous_position = self.position()
        self.world_acceleration = np.zeros(3)
        self.previous_velocity = np.array(self.sim.getBaseVelocity(self.aircraft_id)[0])
        self.trail = [self.previous_position.tolist()]
        self.last_surface_commands = np.zeros(6)
        self.control_targets = {}

    def _box(self, position: list, half: list, color: list) -> int:
        collision = self.sim.createCollisionShape(self.sim.GEOM_BOX, halfExtents=half)
        visual = self.sim.createVisualShape(self.sim.GEOM_BOX, halfExtents=half, rgbaColor=color)
        return self.sim.createMultiBody(baseMass=0, baseCollisionShapeIndex=collision,
                                        baseVisualShapeIndex=visual, basePosition=position)

    def _make_gate(self, gate: dict) -> None:
        x, y, z = gate["center_m"]
        w, h = gate["half_width_m"], gate["half_height_m"]
        thickness = .35
        color = [0.1, .8, .65, 1]
        bodies = [
            self._box([x, y - w - thickness, z], [.4, thickness, h + 2 * thickness], color),
            self._box([x, y + w + thickness, z], [.4, thickness, h + 2 * thickness], color),
            self._box([x, y, z - h - thickness], [.4, w, thickness], color),
            self._box([x, y, z + h + thickness], [.4, w, thickness], color),
        ]
        if self.config.layout == "maze":
            # Four actual collision panels around each opening. All navigation
            # still comes from JEV; the wall cannot be passed around inside y±60.
            wall_bottom, wall_top, wall_left, wall_right = 0., 65., -62., 62.
            left_edge, right_edge = y - w - 2 * thickness, y + w + 2 * thickness
            bottom_edge, top_edge = z - h - 2 * thickness, z + h + 2 * thickness
            wall_color = [0.38, .47, .58, 1]
            if gate["number"] % 2 == 0:
                wall_color = [.46, .39, .56, 1]
            for start, end in ((wall_left, left_edge), (right_edge, wall_right)):
                bodies.append(self._box([x, (start + end) / 2, (wall_bottom + wall_top) / 2],
                                        [.4, (end - start) / 2, (wall_top - wall_bottom) / 2], wall_color))
            for start, end in ((wall_bottom, bottom_edge), (top_edge, wall_top)):
                bodies.append(self._box([x, y, (start + end) / 2],
                                        [.4, w + 2 * thickness, (end - start) / 2], wall_color))
        self.gate_bodies.append(bodies)
        # A white, non-colliding center cross makes the aim point visible.
        marker = self.sim.createVisualShape(self.sim.GEOM_BOX, halfExtents=[.025, .35, .025], rgbaColor=[1., .85, .15, 1.])
        self.sim.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=marker, basePosition=[x, y, z])
        marker = self.sim.createVisualShape(self.sim.GEOM_BOX, halfExtents=[.025, .025, .35], rgbaColor=[1., .85, .15, 1.])
        self.sim.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=marker, basePosition=[x, y, z])
        # Non-colliding runway stripes provide distance cues.
        self.sim.addUserDebugText(str(gate["number"]), [x, y, z + h + 1], [1, 1, 1], 2.)

    def position(self) -> np.ndarray:
        return np.array(self.sim.getBasePositionAndOrientation(self.aircraft_id)[0])

    def observation(self, previous_action: str | None) -> dict:
        state = self.sim.state(0)
        position, quaternion = self.sim.getBasePositionAndOrientation(self.aircraft_id)
        velocity, _ = self.sim.getBaseVelocity(self.aircraft_id)
        rotation = np.array(self.sim.getMatrixFromQuaternion(quaternion)).reshape(3, 3)
        speed = float(np.linalg.norm(velocity))
        horizontal_speed = float(np.linalg.norm(velocity[:2]))
        heading = math.atan2(rotation[1, 0], rotation[0, 0])
        track = math.atan2(velocity[1], velocity[0])
        climb_angle = math.atan2(velocity[2], horizontal_speed)
        degrees = lambda value: round(math.degrees(value), 3)
        wrap = lambda value: math.atan2(math.sin(value), math.cos(value))
        surface_state = self.sim.drones[0].lifting_surfaces.get_states()
        surfaces = ["left_aileron", "right_aileron", "elevator", "rudder", "main_wing"]
        limits = [30., 30., 20., 20., 0.]
        next_gates = []
        for gate in self.gates[self.passed:self.passed + 3]:
            delta = np.array(gate["center_m"]) - position
            body_delta = rotation.T @ delta
            forward_speed = float(velocity[0])
            time_to_plane = delta[0] / forward_speed if forward_speed > .1 and delta[0] >= 0 else None
            target_bearing = math.atan2(delta[1], delta[0])
            target_elevation = math.atan2(delta[2], np.linalg.norm(delta[:2]))
            forecast = None
            if time_to_plane is not None:
                crossing = np.array(position) + np.array(velocity) * time_to_plane
                lateral_error = gate["center_m"][1] - crossing[1]
                vertical_error = gate["center_m"][2] - crossing[2]
                forecast = {"assumption": "Straight constant WORLD velocity; no turn or acceleration. This is a geometric estimate, not a planned route.",
                            "position_world_xyz_m": np.round(crossing, 3).tolist(),
                            "center_error_left_m": round(lateral_error, 3),
                            "center_error_up_m": round(vertical_error, 3),
                            "lateral_clearance_margin_m": round(gate["half_width_m"] - 1.25 - abs(lateral_error), 3),
                            "vertical_clearance_margin_m": round(gate["half_height_m"] - .5 - abs(vertical_error), 3)}
            next_gates.append({
                **gate, "relative_world_xyz_m": np.round(delta, 3).tolist(),
                "relative_body_forward_left_up_m": np.round(body_delta, 3).tolist(),
                "distance_m": round(float(np.linalg.norm(delta)), 3),
                "bearing_left_deg": round(math.degrees(math.atan2(body_delta[1], body_delta[0])), 2),
                "elevation_up_deg": round(math.degrees(math.atan2(body_delta[2], np.linalg.norm(body_delta[:2]))), 2),
                "estimated_seconds_to_gate_plane": round(time_to_plane, 3) if time_to_plane is not None else None,
                "distance_to_gate_plane_m": round(float(delta[0]), 3),
                "current_center_error_left_m": round(float(delta[1]), 3),
                "current_center_error_up_m": round(float(delta[2]), 3),
                "normalized_lateral_center_error": round(float(delta[1] / (gate["half_width_m"] - 1.25)), 3),
                "normalized_vertical_center_error": round(float(delta[2] / (gate["half_height_m"] - .5)), 3),
                "heading_error_left_deg": degrees(wrap(target_bearing - heading)),
                "center_bearing_world_left_deg": degrees(target_bearing),
                "center_elevation_world_up_deg": degrees(target_elevation),
                "velocity_track_error_left_deg": degrees(wrap(target_bearing - track)),
                "flight_path_error_up_deg": degrees(wrap(target_elevation - climb_angle)),
                "closing_speed_to_center_m_s": round(float(np.dot(velocity, delta) / max(np.linalg.norm(delta), .001)), 3),
                "straight_velocity_crossing_forecast": forecast,
            })
        observation = {
            "task": "Fly through the next static gate opening in order, avoid the solid frame and ground.",
            "time_seconds": round(self.sim.elapsed_time, 3), "gates_passed": self.passed,
            "gates_total": len(self.gates), "previous_action": previous_action,
            "aircraft": {
                "position_world_xyz_m": np.round(position, 3).tolist(),
                "velocity_world_xyz_m_s": np.round(velocity, 3).tolist(),
                "velocity_body_forward_left_up_m_s": np.round(state[2], 3).tolist(),
                "speed_m_s": round(speed, 3),
                "roll_pitch_yaw_deg": np.round(np.rad2deg(state[1]), 3).tolist(),
                "bank_left_deg": degrees(-state[1, 0]),
                "nose_up_pitch_deg": degrees(-state[1, 1]),
                "body_roll_pitch_yaw_rates_deg_s": np.round(np.rad2deg(state[0]), 3).tolist(),
                "heading_left_deg": degrees(heading), "velocity_track_left_deg": degrees(track),
                "flight_path_climb_angle_deg": degrees(climb_angle),
                "climb_rate_m_s": round(float(velocity[2]), 3),
                "horizontal_speed_m_s": round(horizontal_speed, 3),
                "acceleration_world_xyz_m_s2": np.round(self.world_acceleration, 3).tolist(),
                "acceleration_body_forward_left_up_m_s2": np.round(rotation.T @ self.world_acceleration, 3).tolist(),
                "specific_force_load_g": round(float(np.linalg.norm(self.world_acceleration - np.array([0., 0., -9.81])) / 9.81), 3),
                "body_reference_angle_of_attack_deg": degrees(math.atan2(-state[2, 2], state[2, 0])),
                "body_sideslip_left_deg": degrees(math.atan2(state[2, 1], np.linalg.norm(state[2, [0, 2]]))),
                "actuators": {"command_normalized": dict(zip(surfaces + ["throttle"], np.round(self.last_surface_commands, 4).tolist())),
                              "measured_surface_deflection_deg": {name: round(float(value * limit), 3) for name, value, limit in zip(surfaces, surface_state, limits)},
                              "measured_throttle_normalized": round(float(self.sim.drones[0].motors.get_states()[0]), 4),
                              "attitude_targets": self.control_targets},
            },
            "next_gates": next_gates,
            "controls": {"action_duration_seconds": self.config.decision_seconds,
                         "target_speed_m_s": self.config.speed,
                         "up_down_pitch_deg": self.config.pitch_up_deg,
                         "pitch_up_nose_up_deg": self.config.pitch_up_deg,
                         "pitch_down_nose_up_deg": -self.config.pitch_down_deg,
                         "extended_pitch_actions": self.config.extended_pitch_actions,
                         "strong_pitch_up_nose_up_deg": self.config.strong_pitch_up_deg,
                         "strong_pitch_down_nose_up_deg": -self.config.strong_pitch_down_deg,
                         "joint_choice_count": self.config.joint_choice_count,
                         **({"pitch_targets_nose_up_deg": self.pitch_targets} if self.config.dense_pitch_actions else {}),
                         "left_right_yaw_rate_deg_s": 12 if self.config.control_profile == "legacy" else degrees(9.81 * math.tan(math.radians(25)) / max(speed, 10)),
                         "roll_left_right_bank_deg": 18 if self.config.control_profile == "legacy" else 40,
                         "nominal_pitch_nose_up_deg": 2,
                         "turn_bank_deg": 25 if self.config.control_profile != "legacy" else 0,
                         "strong_roll_bank_deg": 40 if self.config.control_profile != "legacy" else 18,
                         "ideal_turn_radius_at_25_deg_m": round(speed * speed / (9.81 * math.tan(math.radians(25))), 2),
                         "frame_convention": "World +x forward along course, +y left, +z up. Positive Euler pitch means nose DOWN; negative Euler roll means bank LEFT.",
                         "stabilizer": "Basic attitude/rate and speed controller; does not know gate locations or choose navigation actions."},
            "pass_rule": "Cross each gate's x plane inside its opening with a 1.25m lateral and 0.5m vertical center margin.",
            "environment": {"wind_world_xyz_m_s": [0., 0., 0.], "gravity_m_s2": 9.81,
                            "physics_hz": 240, "control_hz": CONTROL_HZ,
                            "gate_geometry": "Static rectangular openings in world x planes; positive center error means move LEFT or UP."},
        }
        observation["navigation"] = navigation_features(observation, self.config.control_profile)
        last_pass = next((event for event in reversed(self.events) if event["gate"] <= self.passed), None)
        activated_at = last_pass["time_seconds"] if last_pass else 0.
        observation["navigation"]["target_update"] = {
            "active_gate_number": next_gates[0]["number"] if next_gates else None,
            "last_passed_gate": last_pass["gate"] if last_pass else None,
            "last_pass_time_seconds": round(activated_at, 6) if last_pass else None,
            "active_gate_since_seconds": round(activated_at, 6) if next_gates else None,
            "seconds_since_target_became_active": round(self.sim.elapsed_time - activated_at, 6) if next_gates else None,
            "decision_refresh_on_gate_pass": self.config.refresh_on_gate_pass,
        }
        return observation

    def _control(self, action: str | None) -> None:
        # This low-level controller has no access to course coordinates.
        vertical, horizontal = action, action
        if action in ALL_JOINT_KEYS:
            if self.config.control_profile == "legacy":
                raise ValueError("Joint controls require banked_v2 physics control profile")
            vertical, horizontal = action.split("__")
        rates, angles, velocity, _ = self.sim.state(0)
        roll, pitch, _ = angles
        desired_roll = math.radians(-18 if action == "roll_left" else 18 if action == "roll_right" else 0)
        desired_pitch = math.radians(-self.pitch_targets.get(vertical, 2.))
        speed = float(np.linalg.norm(velocity))
        desired_yaw_rate = math.radians(12 if action == "left" else -12 if action == "right" else 0)
        if action in {"roll_left", "roll_right"}:
            desired_yaw_rate = -9.81 * math.tan(desired_roll) / max(speed, 10)
        if self.config.control_profile != "legacy":
            banks = {"left": -25, "right": 25, "roll_left": -40, "roll_right": 40}
            desired_roll = math.radians(banks.get(horizontal, 0))
            desired_yaw_rate = -9.81 * math.tan(desired_roll) / max(speed, 10)
        self.control_targets = {"bank_left_deg": round(-math.degrees(desired_roll), 2),
                                "nose_up_pitch_deg": round(-math.degrees(desired_pitch), 2),
                                "yaw_rate_left_deg_s": round(math.degrees(desired_yaw_rate), 2)}
        self.pitch_integral = float(np.clip(self.pitch_integral + (desired_pitch - pitch) / CONTROL_HZ, -.6, .6))
        aileron = np.clip(2.0 * (desired_roll - roll) - .4 * rates[0], -.6, .6)
        elevator = np.clip(3.0 * (desired_pitch - pitch) - .55 * rates[1] + .35 * self.pitch_integral, -.6, .6)
        rudder = np.clip(-1.5 * (desired_yaw_rate - rates[2]), -.5, .5)
        throttle = np.clip(.4 + .035 * (self.config.speed - speed), .1, .85)
        self.last_surface_commands = np.array([aileron, -aileron, elevator, rudder, 0., throttle])
        self.sim.set_setpoint(0, self.last_surface_commands)

    def step(self, action: str | None) -> None:
        if self.status != "flying":
            return
        allowed_joint = self.allowed_joint
        if action is not None and action not in ACTIONS and action not in allowed_joint:
            raise ValueError(f"Unknown action: {action}")
        self._control(action)
        self.sim.step()
        position = self.position()
        new_velocity = np.array(self.sim.getBaseVelocity(self.aircraft_id)[0])
        self.world_acceleration = (new_velocity - self.previous_velocity) * CONTROL_HZ
        self.previous_velocity = new_velocity
        if self.sim.aviary_steps % 12 == 0:
            self.trail.append(position.tolist())
        if self.sim.getContactPoints(bodyA=self.aircraft_id):
            self.status = "collision"
            self.score -= 100
        elif not np.all(np.isfinite(self.sim.state(0))):
            self.status = "physics_error"
        elif abs(position[1]) > 60 or position[2] < 1 or position[2] > 80 or position[0] < -20:
            self.status = "out_of_bounds"
        elif abs(self.sim.state(0)[1, 0]) > math.radians(75):
            self.status = "loss_of_control"
        elif self.passed < len(self.gates):
            gate = self.gates[self.passed]
            center = np.array(gate["center_m"])
            if self.previous_position[0] < center[0] <= position[0]:
                fraction = (center[0] - self.previous_position[0]) / (position[0] - self.previous_position[0])
                crossing = self.previous_position + fraction * (position - self.previous_position)
                lateral_margin = gate["half_width_m"] - 1.25 - abs(crossing[1] - center[1])
                vertical_margin = gate["half_height_m"] - .5 - abs(crossing[2] - center[2])
                margin = float(min(lateral_margin, vertical_margin))
                self.minimum_clearance = margin if self.minimum_clearance is None else min(self.minimum_clearance, margin)
                event = {"gate": self.passed + 1, "time_seconds": self.sim.elapsed_time,
                         "crossing_xyz_m": crossing.tolist(), "clearance_margin_m": margin}
                self.events.append(event)
                if margin >= 0:
                    self.passed += 1
                    self.score += 100
                    for body in self.gate_bodies[self.passed - 1]:
                        self.sim.changeVisualShape(body, -1, rgbaColor=[.2, .45, .95, 1])
                    if self.passed == len(self.gates):
                        self.status = "completed"
                else:
                    self.status = "gate_missed"
                    self.score -= 50
        if self.status == "flying" and self.sim.elapsed_time >= self.config.max_seconds:
            self.status = "time_limit"
        self.previous_position = position

    def image(self, width: int = 960, height: int = 540, camera: str = "chase") -> np.ndarray:
        position = self.position()
        if camera == "overview":
            eye = position + np.array([-22., -30., 20.])
            target = position + np.array([22., 0., 0.])
        else:
            yaw = self.sim.state(0)[1, 2]
            forward = np.array([math.cos(yaw), math.sin(yaw), 0.])
            eye = position - 6.5 * forward + np.array([0., 0., 2.5])
            target = position + 28. * forward
        view = self.sim.computeViewMatrix(eye.tolist(), target.tolist(), [0, 0, 1])
        projection = self.sim.computeProjectionMatrixFOV(65, width / height, .1, 700)
        pixels = self.sim.getCameraImage(width, height, viewMatrix=view, projectionMatrix=projection,
                                        renderer=self.sim.ER_TINY_RENDERER, flags=self.sim.ER_NO_SEGMENTATION_MASK)[2]
        return np.array(pixels, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]

    def summary(self) -> dict:
        return {"config": asdict(self.config), "status": self.status, "score": self.score,
                "gates_passed": self.passed, "gates_total": len(self.gates),
                "simulated_seconds": self.sim.elapsed_time,
                "minimum_crossing_clearance_margin_m": self.minimum_clearance,
                "final_position_m": self.position().tolist(), "gate_events": self.events}

    def close(self) -> None:
        self.sim.disconnect()
