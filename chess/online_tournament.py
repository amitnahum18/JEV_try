"""Run JEV up a cost-aware ladder of online chess-capable LLMs."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import chess
import chess.pgn
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "car"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from chess_app import jev_decision, jev_single_decision  # noqa: E402

load_dotenv(ROOT / ".env")

# Sorted by the ChessBench LLM-only chess ratings checked on 2026-09-26.
# We use low reasoning effort for every tier to keep games moving; this is a
# speed-oriented comparison and won't reproduce ChessBench's own settings.
LEVELS: list[dict[str, Any]] = [
    {"name": "Qwen3.7 Plus", "model": "qwen/qwen3.7-plus", "rating": 1168},
    {"name": "Claude Sonnet 5", "model": "anthropic/claude-sonnet-5", "rating": 1198},
    {"name": "MiniMax M3", "model": "minimax/minimax-m3", "rating": 1225},
    {"name": "GLM 5.3 Flash", "model": "z-ai/glm-5.3-flash", "rating": 1330},
    {"name": "GPT-6 Luna", "model": "openai/gpt-6-luna", "rating": 1433},
    {"name": "DeepSeek V4.1 Flash", "model": "deepseek/deepseek-v4.1-flash", "rating": 1615},
    {"name": "GPT-6 Sol", "model": "openai/gpt-6-sol", "rating": 1627},
    {"name": "GPT-6 Astra", "model": "openai/gpt-6-astra", "rating": 2020},
    {"name": "Gemini 3.8 Flash", "model": "google/gemini-3.8-flash", "rating": 2025},
    {"name": "Claude Opus 5.5", "model": "anthropic/claude-opus-5.5", "rating": 2067},
]
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
USER_AGENT = "JEV-Chess-Ladder/1.0"


def call_opponent(board: chess.Board, tier: dict[str, Any], *, timeout: float) -> dict[str, Any]:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is missing.")

    legal = list(board.legal_moves)
    legal_entries = [{"uci": move.uci(), "san": board.san(move)} for move in legal]
    schema = {
        "name": "chess_move",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"uci": {"type": "string", "enum": [m.uci() for m in legal]}},
            "required": ["uci"],
            "additionalProperties": False,
        },
    }
    recent_moves = list(board.move_stack[-12:])
    history_san: list[str] = []
    replay = chess.Board()
    for index, move in enumerate(board.move_stack):
        san = replay.san(move)
        replay.push(move)
        if index >= len(board.move_stack) - len(recent_moves):
            history_san.append(san)
    repetition_count = int(board.can_claim_threefold_repetition()) + 1

    payload = {
        "model": tier["model"],
        "temperature": 0,
        # Some providers account hidden reasoning against this budget even
        # when it is excluded from the returned message. Keep low effort for
        # speed, but leave enough room for the actual JSON move.
        "max_tokens": 8192,
        "reasoning": {"effort": "low", "exclude": True},
        "response_format": {"type": "json_schema", "json_schema": schema},
        "messages": [
            {
                "role": "system",
                "content": (
                    "Play legal tournament chess. Choose your strongest move, calculate the opponent's "
                    "best reply, avoid hanging material, and avoid repeating a position without need. "
                    "Return only the required JSON move."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "fen": board.fen(),
                        "you_are": "White" if board.turn else "Black",
                        "recent_moves_san": history_san,
                        "legal_moves": legal_entries,
                        "repetition_count": repetition_count,
                    },
                    separators=(",", ":"),
                ),
            },
        ],
    }
    started = time.perf_counter()
    retried_without_reasoning = False
    for attempt in range(2):
        attempt_payload = payload
        if attempt:
            # A few providers can exhaust even a large low-effort budget on a
            # complex position. Retry that same move request with reasoning off
            # so an output-length limit does not end the whole ladder.
            attempt_payload = {
                **payload,
                "max_tokens": 2048,
                "reasoning": {"effort": "none", "exclude": True},
            }
            retried_without_reasoning = True
        req = Request(
            ENDPOINT,
            data=json.dumps(attempt_payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "User-Agent": USER_AGENT,
            },
            method="POST",
        )
        try:
            with urlopen(req, timeout=timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenRouter HTTP {exc.code} for {tier['model']}: {detail[:800]}") from exc
        except (URLError, TimeoutError) as exc:
            raise RuntimeError(f"OpenRouter request failed for {tier['model']}: {exc}") from exc
        choice = (result.get("choices") or [{}])[0]
        if choice.get("finish_reason") != "length" or attempt:
            break

    try:
        content = result["choices"][0]["message"]["content"]
        answer = json.loads(content) if isinstance(content, str) else content
        move_uci = answer["uci"]
        move = chess.Move.from_uci(move_uci)
        if move not in board.legal_moves:
            raise ValueError("model selected a move that is not legal in this position")
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
        # Keep enough response metadata to diagnose provider/schema mismatches,
        # but never include request headers or credentials in the tournament log.
        choice = (result.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        debug = {
            "finish_reason": choice.get("finish_reason"),
            "refusal": message.get("refusal"),
            "content": message.get("content"),
            "provider": result.get("provider"),
            "usage": result.get("usage"),
        }
        raise RuntimeError(
            f"{tier['model']} did not return a valid legal UCI move; "
            f"response={json.dumps(debug, ensure_ascii=False)[:1000]}"
        ) from exc

    usage = result.get("usage") or {}
    return {
        "move": move,
        "latency_seconds": round(time.perf_counter() - started, 3),
        "input_tokens": usage.get("prompt_tokens"),
        "output_tokens": usage.get("completion_tokens"),
        "cost_usd": usage.get("cost"),
        "provider": result.get("provider"),
        "retried_without_reasoning": retried_without_reasoning,
    }


def save_pgn(game: chess.pgn.Game, path: Path, result: str) -> None:
    game.headers["Result"] = result
    path.write_text(game.accept(chess.pgn.StringExporter(headers=True, variations=False, comments=True)) + "\n", encoding="utf-8")


def run_ladder(
    output: Path, *, system: str, max_plies: int, timeout: float, seed: int
) -> dict[str, Any]:
    random.seed(seed)
    output.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "system": system,
        "jev_color": "white",
        "promotion_rule": "Advance only after a JEV win; stop on draw, loss, API error, or move cap.",
        "ranking_source": "ChessBench LLM leaderboard; relative benchmark ratings only.",
        "reasoning_effort": "low (speed setting; not ChessBench-equivalent)",
        "max_plies_per_game": max_plies,
        "games": [],
        "completed": False,
    }
    summary_path = output / "tournament.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for index, tier in enumerate(LEVELS, 1):
        slug = tier["model"].split("/", 1)[-1].replace(".", "_").replace(":", "_")
        pgn_path = output / f"game_{index:02d}_{slug}.pgn"
        records_path = output / f"game_{index:02d}_{slug}.json"
        board = chess.Board()
        game = chess.pgn.Game()
        game.headers.update({
            "Event": f"JEV Online Model Ladder - {system}",
            "Site": "OpenRouter",
            "Date": datetime.now().strftime("%Y.%m.%d"),
            "Round": str(index),
            "White": "JEV",
            "Black": tier["name"],
            "Result": "*",
            "Annotator": "JEV Chess Lab",
        })
        node = game
        record: dict[str, Any] = {
            "round": index,
            "opponent": tier["name"],
            "jev_system": system,
            "model_id": tier["model"],
            "chessbench_rating": tier["rating"],
            "reasoning_effort": "low",
            "pgn": pgn_path.name,
            "decisions": records_path.name,
            "status": "playing",
            "moves": [],
        }
        summary["games"].append(record)
        save_pgn(game, pgn_path, "*")
        records_path.write_text("[]", encoding="utf-8")
        print(f"\nSTAGE {index}/{len(LEVELS)}: JEV vs {tier['name']} ({tier['rating']} ChessBench)", flush=True)

        try:
            while not board.is_game_over(claim_draw=True) and len(board.move_stack) < max_plies:
                fen_before = board.fen()
                if board.turn == chess.WHITE:
                    move, decision = (
                        jev_single_decision(board) if system == "single" else jev_decision(board)
                    )
                    actor = "JEV"
                    entry = {
                        "decision_mode": system,
                        "confidence": decision.confidence,
                        "council_role_moves": decision.role_moves,
                        "council_agreement": decision.role_agreement,
                        "tactical_review": decision.tactical_review,
                        "latency_seconds": round(decision.latency_seconds, 3),
                        "input_tokens": decision.input_tokens,
                        "cost_usd": decision.cost_usd,
                    }
                else:
                    response = call_opponent(board, tier, timeout=timeout)
                    move = response.pop("move")
                    actor = tier["name"]
                    entry = response
                san = board.san(move)
                node = node.add_variation(move)
                board.push(move)
                move_record = {
                    "ply": len(board.move_stack),
                    "player": actor,
                    "fen_before": fen_before,
                    "uci": move.uci(),
                    "san": san,
                    **entry,
                }
                record["moves"].append(move_record)
                print(f"{actor}: {san}", flush=True)
                result_so_far = board.result(claim_draw=True) if board.is_game_over(claim_draw=True) else "*"
                save_pgn(game, pgn_path, result_so_far)
                records_path.write_text(json.dumps(record["moves"], indent=2), encoding="utf-8")

            if board.is_game_over(claim_draw=True):
                outcome = board.outcome(claim_draw=True)
                result_value = board.result(claim_draw=True)
                record.update({
                    "status": "jev_win" if result_value == "1-0" else "draw" if result_value == "1/2-1/2" else "jev_loss",
                    "result": result_value,
                    "termination": outcome.termination.name.lower() if outcome else "unknown",
                    "winner": "JEV" if result_value == "1-0" else tier["name"] if result_value == "0-1" else None,
                })
            else:
                record.update({"status": "move_cap", "result": "*", "termination": "move_cap", "winner": None})
            save_pgn(game, pgn_path, record["result"])
        except Exception as exc:
            record.update({"status": "error", "result": "*", "termination": "api_or_engine_error", "error": str(exc)})
            save_pgn(game, pgn_path, "*")
            records_path.write_text(json.dumps(record["moves"], indent=2), encoding="utf-8")
            print(f"STOP: {exc}", flush=True)

        summary["last_completed_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        if record["status"] != "jev_win":
            summary["stopped_after"] = tier["name"]
            summary["stop_reason"] = record["status"]
        summary["completed"] = record["status"] in {"jev_loss", "draw"} or (
            index == len(LEVELS) and record["status"] == "jev_win"
        )
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"RESULT: {record['status']} ({record['result']})", flush=True)
        if record["status"] != "jev_win":
            break
        if index < len(LEVELS):
            print("JEV won; moving up to the next model.\n", flush=True)

    summary["finished_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _track_score(summary: dict[str, Any]) -> dict[str, Any]:
    games = summary.get("games", [])
    wins = sum(game.get("status") == "jev_win" for game in games)
    jev_moves = [
        move
        for game in games
        for move in game.get("moves", [])
        if move.get("player") == "JEV"
    ]
    costs = [move.get("cost_usd") for move in jev_moves if move.get("cost_usd") is not None]
    latencies = [move.get("latency_seconds") for move in jev_moves if move.get("latency_seconds") is not None]
    last_game = games[-1] if games else None
    return {
        "system": summary.get("system"),
        "stages_won": wins,
        "last_opponent": last_game.get("opponent") if last_game else None,
        "last_result": last_game.get("status") if last_game else "not_started",
        "games_recorded": len(games),
        "jev_moves": len(jev_moves),
        "jev_cost_usd": round(sum(costs), 6),
        "average_jev_move_seconds": round(sum(latencies) / len(latencies), 3) if latencies else None,
        "tournament_completed": summary.get("completed", False),
        "folder": summary.get("folder"),
    }


def _competition_winner(scores: list[dict[str, Any]]) -> str | None:
    if len(scores) != 2:
        return None
    first, second = scores
    # A technical error or capped game is not evidence that one system is stronger.
    if first["last_result"] in {"error", "move_cap", "not_started"} or second["last_result"] in {
        "error", "move_cap", "not_started"
    }:
        return None
    if first["stages_won"] != second["stages_won"]:
        return first["system"] if first["stages_won"] > second["stages_won"] else second["system"]
    quality = {"jev_win": 2, "draw": 1, "jev_loss": 0}
    first_quality = quality.get(first["last_result"], -1)
    second_quality = quality.get(second["last_result"], -1)
    if first_quality != second_quality:
        return first["system"] if first_quality > second_quality else second["system"]
    return "tie"


def run_competition(
    output: Path, *, max_plies: int, timeout: float, seed: int, systems: list[str]
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    root_summary: dict[str, Any] = {
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "format": "Independent ladders against the same opponent sequence and settings.",
        "promotion_rule": "Each system advances only after its own win; draw, loss, API error, or move cap stops that system.",
        "systems": {},
        "comparison": [],
        "completed": False,
    }
    summary_path = output / "competition.json"
    summary_path.write_text(json.dumps(root_summary, indent=2), encoding="utf-8")
    for system in systems:
        folder = output / system
        print(f"\n{'=' * 15} {system.upper()} JEV {'=' * 15}", flush=True)
        summary = run_ladder(
            folder, system=system, max_plies=max_plies, timeout=timeout, seed=seed
        )
        summary["folder"] = str(folder)
        # Persist the folder in both the track report and root scoreboard.
        (folder / "tournament.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        root_summary["systems"][system] = summary
        root_summary["comparison"] = [
            _track_score(track) for track in root_summary["systems"].values()
        ]
        root_summary["winner"] = _competition_winner(root_summary["comparison"])
        root_summary["completed"] = len(root_summary["systems"]) == len(systems)
        summary_path.write_text(json.dumps(root_summary, indent=2), encoding="utf-8")
    root_summary["finished_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    summary_path.write_text(json.dumps(root_summary, indent=2), encoding="utf-8")
    return root_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-plies", type=int, default=160)
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--system", choices=("all", "single", "team"), default="all")
    args = parser.parse_args()
    if args.max_plies < 2 or args.timeout < 1:
        parser.error("--max-plies must be >=2 and --timeout >=1")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output_dir or (Path(__file__).resolve().parent / "recordings" / f"jev_systems_competition_{stamp}")
    systems = ["single", "team"] if args.system == "all" else [args.system]
    result = run_competition(
        output, max_plies=args.max_plies, timeout=args.timeout, seed=args.seed, systems=systems
    )
    print(f"\nRecorded JEV systems competition: {output}", flush=True)
    print(json.dumps(result["comparison"], indent=2))


if __name__ == "__main__":
    main()
