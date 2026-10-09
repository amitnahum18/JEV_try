"""Two-drone Sky Relay demo controlled by Jev's typed decisions."""

from __future__ import annotations

import argparse
import math

import numpy as np
from PyFlyt.core import Aviary

from jev_client import choose_action


CONTROL_HZ = 120
DECISION_PERIOD_SECONDS = 2.0
ALTITUDE = 2.0

ASSIGNMENTS = {
    "drone_1_west_drone_2_east": "Drone 1 takes the west checkpoint; Drone 2 takes the east checkpoint.",
    "drone_1_east_drone_2_west": "Drone 1 takes the east checkpoint; Drone 2 takes the west checkpoint.",
}


def checkpoint_positions(sim_time: float) -> tuple[np.ndarray, np.ndarray]:
    """Return two checkpoints moving gently on separate, visible lanes."""
    west = np.array([-2.5, 1.2 * math.sin(sim_time * 0.45), ALTITUDE])
    east = np.array([2.5, 1.2 * math.sin(sim_time * 0.45 + math.pi), ALTITUDE])
    return west, east


def positions(aviary: Aviary) -> list[np.ndarray]:
    return [aviary.state(index)[3, :].copy() for index in range(2)]


def update_setpoints(
    aviary: Aviary,
    assignment: str,
    sim_time: float,
    checkpoint_ids: tuple[int, int],
) -> None:
    west, east = checkpoint_positions(sim_time)
    for checkpoint_id, position in zip(checkpoint_ids, (west, east)):
        aviary.resetBasePositionAndOrientation(checkpoint_id, position.tolist(), [0, 0, 0, 1])
    targets = (
        (west, east)
        if assignment == "drone_1_west_drone_2_east"
        else (east, west)
    )
    for drone_index, target in enumerate(targets):
        aviary.set_setpoint(
            drone_index,
            np.array([target[0], target[1], 0.0, target[2]], dtype=np.float64),
        )


def make_state(
    aviary: Aviary,
    current_assignment: str,
    sim_time: float,
) -> dict:
    drone_positions = positions(aviary)
    drone_velocities = [aviary.state(index)[2, :].copy() for index in range(2)]
    west, east = checkpoint_positions(sim_time)
    return {
        "time_seconds": round(sim_time, 2),
        "drones": {
            f"drone_{index + 1}": {
                "position_m": np.round(position, 2).tolist(),
                "velocity_m_s": np.round(drone_velocities[index], 2).tolist(),
            }
            for index, position in enumerate(drone_positions)
        },
        "checkpoints": {
            "west": {"position_m": np.round(west, 2).tolist(), "motion": "sweeping side to side"},
            "east": {"position_m": np.round(east, 2).tolist(), "motion": "sweeping side to side"},
        },
        "current_assignment": current_assignment,
        "note": "Assign exactly one drone to each checkpoint. The simulator computes positions; choose only the team assignment.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the two-drone Jev Sky Relay demo.")
    parser.add_argument("--seconds", type=float, default=40.0, help="simulated flight duration")
    parser.add_argument("--headless", action="store_true", help="run without the PyBullet window")
    parser.add_argument("--decision-period", type=float, default=DECISION_PERIOD_SECONDS)
    parser.add_argument("--linger", action="store_true", help="keep the visible simulation open until its window is closed")
    args = parser.parse_args()

    if args.seconds <= 0 or args.decision_period <= 0:
        parser.error("--seconds and --decision-period must be positive")
    if args.linger and args.headless:
        parser.error("--linger requires the visible PyBullet window")

    start_pos = np.array([[-2.5, 0.0, ALTITUDE], [2.5, 0.0, ALTITUDE]])
    start_orn = np.zeros_like(start_pos)
    options = {"control_hz": CONTROL_HZ, "drone_model": "primitive_drone"}
    aviary = Aviary(
        start_pos=start_pos,
        start_orn=start_orn,
        drone_type="quadx",
        drone_options=options,
        render=not args.headless,
        physics_hz=240,
        world_scale=12.0,
    )
    aviary.set_mode(7)
    checkpoint_ids = []
    for color in ([0.15, 0.75, 1.0, 1.0], [1.0, 0.55, 0.12, 1.0]):
        visual_id = aviary.createVisualShape(
            aviary.GEOM_SPHERE,
            radius=0.22,
            rgbaColor=color,
        )
        checkpoint_ids.append(
            aviary.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=visual_id,
                basePosition=[0.0, 0.0, ALTITUDE],
            )
        )
    aviary.register_all_new_bodies()

    assignment = "drone_1_west_drone_2_east"
    sim_time = 0.0
    steps_per_decision = max(1, round(args.decision_period * CONTROL_HZ))
    remaining_steps = math.ceil(args.seconds * CONTROL_HZ)
    decision_number = 0

    try:
        while remaining_steps > 0:
            state = make_state(aviary, assignment, sim_time)
            decision = choose_action(
                state=state,
                actions=ASSIGNMENTS,
                instructions=(
                    "Choose which drone should take each moving checkpoint. Keep one drone per checkpoint. "
                    "Use the supplied positions, velocities and current assignment. Prefer the assignment "
                    "that avoids crossing the drones' paths; preserve the current assignment when there is no clear benefit to switching."
                ),
            )
            assignment = decision.choice
            decision_number += 1
            print(
                f"Jev decision {decision_number}: {assignment} "
                f"(confidence={decision.confidence:.2f}, latency={decision.latency_seconds:.3f}s)"
            )

            chunk = min(steps_per_decision, remaining_steps)
            for _ in range(chunk):
                update_setpoints(aviary, assignment, sim_time, tuple(checkpoint_ids))
                aviary.step()
                sim_time += 1.0 / CONTROL_HZ
            remaining_steps -= chunk
        if args.linger:
            print("Simulation is live. Close the PyBullet window to stop.", flush=True)
            while aviary.isConnected():
                update_setpoints(aviary, assignment, sim_time, tuple(checkpoint_ids))
                aviary.step()
                sim_time += 1.0 / CONTROL_HZ
    finally:
        aviary.disconnect()


if __name__ == "__main__":
    main()
