"""Asynchronous JEV policy that chooses only grid-based lane movements."""

from __future__ import annotations

import queue
import threading
import math

from jev_client import MODEL, choose_action


GRID_INSTRUCTIONS = (
    "Drive a five-lane one-way highway. Your goal is safe, fast forward progress. "
    "occupancy_grid uses 0=empty, 1=other car, 2=you. Rows go from physical left/top to right/bottom. "
    "ego_cell gives your row and column. Columns before your column are behind; columns after it are ahead; "
    "each column is 20 meters. Use the full visible distance to plan early. "
    "traffic_forecast contains projected occupancy grids up to 4.5 seconds ahead, assuming current speeds and lanes. "
    "move_up is one lane left/top, move_down is one lane right/bottom, stay keeps your lane. Choose only an "
    "action listed in available_actions. Cruise speed is high; when no safe exit exists the speed supervisor "
    "may reduce speed to match a slower lead car, then restore cruise speed when an overtake is safe. "
    "Each action option includes front/rear gaps, TTC, and a live safety-gate prediction for the next 1.5 s. "
    "The safety gate is authoritative: request lane changes only for options marked SAFE; the simulator will "
    "reject any option marked UNSAFE. Your objective is to overtake traffic decisively while avoiding contact. "
    "URGENT OVERTAKE RULE: inspect lane_kinematics.stay. If the current-lane front TTC is 7 seconds or less "
    "and at least one adjacent lane is marked SAFE, you MUST choose the SAFE adjacent lane with the longest "
    "predicted front gap on this decision. Staying is allowed only when no adjacent lane is SAFE or a lane "
    "change is already in progress. Do not wait for TTC to become critical. "
    "When a slower car is ahead in your current lane, take a safe adjacent lane instead of following it. Among "
    "safe lanes, prefer the one with the longest open route and the best front gap. Avoid a lane marked without "
    "a viable adjacent escape route in 3 seconds; enter such a lane only if the current TTC is under 1.5 seconds "
    "and no route-viable lane is available. Do not add extra arbitrary "
    "gap or TTC thresholds beyond the safety gate. Do not request an unsafe lane change to preserve speed. "
    "If no adjacent lane is marked SAFE, stay in your lane for this step and reassess on the next observation. "
    "If lane_change_in_progress is true, choose stay and finish that maneuver before requesting another. "
    "After passing, continue through the clearest available lane with decisive safe overtakes; avoid reversing "
    "direction immediately after a completed change. The simulator safety check is authoritative: never request "
    "a lane marked unsafe, and choose a different safe lane or stay if needed."
)


def describe_lane_options(
    actions: dict[str, str],
    lane_kinematics: dict[str, dict[str, float | None]],
    lane_safety: dict[str, dict[str, float | bool | None]] | None = None,
) -> dict[str, str]:
    """Put the computed safety facts directly beside each JEV action choice."""
    described: dict[str, str] = {}
    for action, description in actions.items():
        facts = lane_kinematics.get(action, {})
        front_gap = facts.get("nearest_ahead_m")
        front_ttc = facts.get("ahead_ttc_s")
        front_close = facts.get("ahead_closing_speed_m_s")
        rear_gap = facts.get("nearest_behind_m")
        rear_ttc = facts.get("rear_ttc_s")
        rear_close = facts.get("rear_closing_speed_m_s")
        safety = (lane_safety or {}).get(action, {})
        front = (
            "no car visible ahead"
            if front_gap is None
            else f"front gap {front_gap:.0f} m, closing {front_close:.1f} m/s, TTC {front_ttc if front_ttc is not None else 'not closing'}"
        )
        rear = (
            "no car visible behind"
            if rear_gap is None
            else f"rear gap {rear_gap:.0f} m, closing {rear_close:.1f} m/s, TTC {rear_ttc if rear_ttc is not None else 'not closing'}"
        )
        verdict = ""
        if action != "stay":
            front_pred = safety.get("front_predicted_gap_m")
            rear_pred = safety.get("rear_predicted_gap_m")
            origin_pred = safety.get("origin_front_predicted_gap_m")
            front_text = "clear" if front_pred is None else f"front predicted gap {front_pred:.0f} m"
            rear_text = "clear" if rear_pred is None else f"rear predicted gap {rear_pred:.0f} m"
            origin_text = "clear" if origin_pred is None else f"origin-lane front gap during transition {origin_pred:.0f} m"
            route_text = (
                f"route viable for 3 s ({safety.get('future_open_lane_count', 0)} open lanes)"
                if safety.get("route_viable") else "dead end predicted within 3 s"
            )
            verdict = (
                f" Safety gate: {'SAFE' if safety.get('safe') else 'UNSAFE'} over "
                f"{safety.get('horizon_s', 1.5):.1f} s ({front_text}; {rear_text}; {origin_text}; "
                f"{route_text}; destination minimum 12 m, transition-origin minimum 8 m)."
            )
        described[action] = f"{description} {front}; {rear}.{verdict}"
    return described


class GridJevPolicy:
    def __init__(self):
        self.results: queue.Queue[tuple | str] = queue.Queue()
        self.pending = False
        self.lock = threading.Lock()
        self.action = "stay"
        self.confidence = 0.0
        self.probabilities: dict[str, float] = {}
        self.decisions = 0
        self.generation = 0
        self.calls_started = 0
        self.successful_responses = 0
        self.failed_calls = 0
        self.stale_responses_discarded = 0
        self.latencies: list[float] = []
        self.input_tokens: list[int] = []
        self.costs_usd: list[float] = []
        self.confidences: list[float] = []
        self.response_choices: dict[str, int] = {}
        self.finished = threading.Event()
        self.finished.set()

    def request(self, state: dict, actions: dict[str, str]) -> None:
        if len(actions) <= 1:
            with self.lock:
                self.generation += 1
                self.action = "stay"
                while not self.results.empty():
                    try:
                        discarded = self.results.get_nowait()
                        if isinstance(discarded, tuple):
                            self.stale_responses_discarded += 1
                    except queue.Empty:
                        break
            return
        with self.lock:
            if self.pending or not self.results.empty():
                return
            self.pending = True
            self.calls_started += 1
            self.finished.clear()
            generation = self.generation

        def ask() -> None:
            try:
                result = choose_action(
                    state=state,
                    actions=actions,
                    instructions=GRID_INSTRUCTIONS,
                    timeout_seconds=6.0,
                )
                with self.lock:
                    self.successful_responses += 1
                    self.latencies.append(float(result.latency_seconds))
                    self.confidences.append(float(result.confidence))
                    self.response_choices[result.choice] = self.response_choices.get(result.choice, 0) + 1
                    if result.input_tokens is not None:
                        self.input_tokens.append(int(result.input_tokens))
                    if result.cost_usd is not None:
                        self.costs_usd.append(float(result.cost_usd))
                    if generation == self.generation:
                        self.results.put((result.choice, result.confidence, result.probabilities, result.latency_seconds))
                    else:
                        self.stale_responses_discarded += 1
            except Exception as exc:
                with self.lock:
                    self.failed_calls += 1
                    if generation == self.generation:
                        self.results.put(str(exc))
            finally:
                with self.lock:
                    self.pending = False
                    self.finished.set()

        threading.Thread(target=ask, name="jev-grid-policy", daemon=True).start()

    def update(self) -> tuple[bool, str]:
        changed = False
        choice = "stay"
        while True:
            try:
                result = self.results.get_nowait()
            except queue.Empty:
                break
            changed = True
            if isinstance(result, tuple):
                self.action, self.confidence, self.probabilities, latency = result
                self.decisions += 1
                choice = self.action
                print(
                    f"JEV grid #{self.decisions}: {self.action} "
                    f"({self.confidence:.0%}; {latency:.2f}s)"
                )
            else:
                self.action = "stay"
                choice = "stay"
                print(f"JEV grid unavailable; staying in lane: {result}")
        return changed, choice

    def wait_for_pending(self, timeout_seconds: float = 7.0) -> bool:
        """Wait for the final outstanding provider call before writing usage stats."""
        return self.finished.wait(timeout_seconds)

    def usage_report(self) -> dict:
        """Return non-secret provider usage and decision statistics for the run report."""
        with self.lock:
            sorted_latencies = sorted(self.latencies)
            p95_index = max(0, math.ceil(0.95 * len(sorted_latencies)) - 1)
            return {
                "provider": "OpenRouter",
                "model": MODEL,
                "api_calls_started": self.calls_started,
                "successful_responses": self.successful_responses,
                "failed_calls": self.failed_calls,
                "stale_responses_discarded": self.stale_responses_discarded,
                "decisions_consumed_by_simulator": self.decisions,
                "response_choices": dict(self.response_choices),
                "mean_latency_seconds": round(sum(self.latencies) / len(self.latencies), 3)
                if self.latencies else None,
                "p95_latency_seconds": round(sorted_latencies[p95_index], 3)
                if sorted_latencies else None,
                "input_tokens_total": sum(self.input_tokens) if self.input_tokens else None,
                "responses_with_token_usage": len(self.input_tokens),
                "reported_cost_usd_total": round(sum(self.costs_usd), 6) if self.costs_usd else None,
                "responses_with_reported_cost": len(self.costs_usd),
                "mean_confidence": round(sum(self.confidences) / len(self.confidences), 4)
                if self.confidences else None,
            }
