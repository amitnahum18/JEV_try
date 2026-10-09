"""Small client for Jev's typed choice API via OpenRouter.

Jev chooses among explicit actions; this module intentionally does not generate
free-form text or continuous flight controls.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv


load_dotenv()


ENDPOINT = "https://openrouter.ai/api/alpha/decisions"
MODEL = "typesafe/jev-1.13"


class JevError(RuntimeError):
    """Raised when Jev cannot return a valid decision."""


@dataclass(frozen=True)
class Decision:
    choice: str
    confidence: float
    probabilities: dict[str, float]
    latency_seconds: float
    input_tokens: int | None = None
    cost_usd: float | None = None
    role_moves: dict[str, str] = field(default_factory=dict)
    role_agreement: float | None = None
    tactical_review: str | None = None


def choose_action(
    *,
    state: dict[str, Any],
    actions: dict[str, str],
    instructions: str,
    timeout_seconds: float = 8.0,
) -> Decision:
    """Ask Jev to select one named action from ``actions``.

    Pass compact, computed simulator facts in ``state``. Include a safe action
    such as ``hold_position`` in ``actions`` so Jev can choose it when needed.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise JevError("Set OPENROUTER_API_KEY in the environment first.")
    if not actions:
        raise ValueError("actions must contain at least one choice")

    payload = {
        "model": MODEL,
        "state": state,
        "questions": {
            "action": {
                "type": "choice",
                "instructions": instructions,
                "criteria": actions,
            }
        },
    }
    request = Request(
        ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    started = time.perf_counter()
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise JevError(f"OpenRouter returned HTTP {exc.code}: {detail}") from exc
    except (URLError, TimeoutError) as exc:
        raise JevError(f"Could not reach Jev: {exc}") from exc
    latency = time.perf_counter() - started

    try:
        answer = result["answers"]["action"]
        choice = answer["choice"]
        confidence = float(answer["confidence"])
        probabilities = {
            str(name): float(value)
            for name, value in answer["probabilities"].items()
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise JevError("Jev response did not contain a valid choice answer.") from exc

    if choice not in actions:
        raise JevError(f"Jev returned an unknown action: {choice!r}")

    usage = result.get("usage", {})
    return Decision(
        choice=choice,
        confidence=confidence,
        probabilities=probabilities,
        latency_seconds=latency,
        input_tokens=usage.get("input_tokens"),
        cost_usd=usage.get("cost"),
    )
