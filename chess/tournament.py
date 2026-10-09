"""Run JEV up an opponent ladder and persist every game for video replay."""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import chess
import chess.engine
import chess.pgn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "car"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from chess_app import OPPONENTS, find_stockfish, jev_decision, weak_opponent_turn


def save_pgn(game: chess.pgn.Game, path: Path, result: str) -> None:
    game.headers["Result"] = result
    exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=True)
    content = game.accept(exporter)
    temp_path = path.with_suffix(".pgn.tmp")
    temp_path.write_text(content + "\n", encoding="utf-8")
    temp_path.replace(path)


def material_label(level: dict[str, Any]) -> str:
    return level["name"]


def opponent_move(board: chess.Board, level: dict[str, Any], engine_path: str | None) -> chess.Move:
    kind = level["kind"]
    if kind in {"random", "greedy"}:
        return weak_opponent_turn(board, material_only=kind == "greedy")
    if not engine_path:
        raise RuntimeError("Stockfish was not found for the selected opponent tier.")
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    try:
        if "Skill Level" in engine.options:
            engine.configure({"Skill Level": level["skill"]})
        result = engine.play(board, chess.engine.Limit(time=level["time"]))
        if result.move is None:
            raise RuntimeError(f"{level['name']} did not return a move.")
        return result.move
    finally:
        engine.quit()


def run_tournament(output_dir: Path, *, max_plies: int, seed: int) -> dict[str, Any]:
    random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    engine_path = find_stockfish()
    ladder: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "jev_color": "white",
        "promotion_rule": "Advance only after a JEV win; stop on draw, loss, API error, or move cap.",
        "max_plies_per_game": max_plies,
        "games": ladder,
        "completed": False,
    }
    summary_path = output_dir / "tournament.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for index, level in enumerate(OPPONENTS, start=1):
        slug = level["name"].split(" - ", 1)[0].lower()
        pgn_path = output_dir / f"game_{index:02d}_{slug}.pgn"
        decision_path = output_dir / f"game_{index:02d}_{slug}.json"
        board = chess.Board()
        game = chess.pgn.Game()
        game.headers["Event"] = "JEV Chess Ladder"
        game.headers["Site"] = "Local simulation"
        game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
        game.headers["Round"] = str(index)
        game.headers["White"] = "JEV"
        game.headers["Black"] = material_label(level)
        game.headers["Result"] = "*"
        game.headers["Annotator"] = "JEV Chess Lab"
        node = game
        records: list[dict[str, Any]] = []
        game_record: dict[str, Any] = {
            "round": index,
            "opponent": level["name"],
            "pgn": pgn_path.name,
            "decisions": decision_path.name,
            "status": "playing",
            "moves": records,
        }
        ladder.append(game_record)
        print(f"\nGAME {index}: JEV vs {level['name']}", flush=True)
        save_pgn(game, pgn_path, "*")

        try:
            while not board.is_game_over(claim_draw=True) and len(board.move_stack) < max_plies:
                before_fen = board.fen()
                mover = "JEV" if board.turn == chess.WHITE else level["name"]
                if board.turn == chess.WHITE:
                    move, decision = jev_decision(board)
                    record = {
                        "ply": len(board.move_stack) + 1,
                        "player": "JEV",
                        "fen_before": before_fen,
                        "uci": move.uci(),
                        "san": board.san(move),
                        "confidence": decision.confidence,
                        "probabilities": decision.probabilities,
                        "council_role_moves": decision.role_moves,
                        "council_agreement": decision.role_agreement,
                        "tactical_review": decision.tactical_review,
                        "latency_seconds": decision.latency_seconds,
                        "input_tokens": decision.input_tokens,
                        "cost_usd": decision.cost_usd,
                    }
                    node = node.add_variation(move)
                    node.comment = f"JEV confidence: {decision.confidence:.2f}"
                    records.append(record)
                    print(f"JEV: {record['san']}  confidence {decision.confidence:.2f}", flush=True)
                else:
                    move = opponent_move(board, level, engine_path)
                    san = board.san(move)
                    node = node.add_variation(move)
                    records.append({
                        "ply": len(board.move_stack) + 1,
                        "player": level["name"],
                        "fen_before": before_fen,
                        "uci": move.uci(),
                        "san": san,
                    })
                    print(f"{mover}: {san}", flush=True)
                board.push(move)
                result_so_far = board.result(claim_draw=True) if board.is_game_over(claim_draw=True) else "*"
                save_pgn(game, pgn_path, result_so_far)
                decision_path.write_text(json.dumps(records, indent=2), encoding="utf-8")

            if board.is_game_over(claim_draw=True):
                outcome = board.outcome(claim_draw=True)
                result = board.result(claim_draw=True)
                won = result == "1-0"
                game_record.update({
                    "status": "jev_win" if won else "draw" if result == "1/2-1/2" else "jev_loss",
                    "result": result,
                    "termination": outcome.termination.name.lower() if outcome else "unknown",
                    "winner": "JEV" if won else level["name"] if result == "0-1" else None,
                })
            else:
                game_record.update({"status": "move_cap", "result": "*", "termination": "move_cap"})
            save_pgn(game, pgn_path, game_record["result"])
        except Exception as exc:
            game_record.update({"status": "error", "result": "*", "error": str(exc)})
            save_pgn(game, pgn_path, "*")
            decision_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
            print(f"STOP: {exc}", flush=True)

        summary["last_completed_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        summary["completed"] = (
            game_record["status"] in {"jev_loss", "draw"}
            or (index == len(OPPONENTS) and game_record["status"] == "jev_win")
        )
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"RESULT: {game_record['status']} ({game_record['result']})", flush=True)
        if game_record["status"] != "jev_win":
            summary["stopped_after"] = level["name"]
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            break
        if index < len(OPPONENTS):
            print("JEV won; promoting to the next opponent.\n", flush=True)

    summary["finished_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-plies", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.max_plies < 2:
        parser.error("--max-plies must be at least 2")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output_dir or (Path(__file__).resolve().parent / "recordings" / f"ladder_{timestamp}")
    result = run_tournament(output, max_plies=args.max_plies, seed=args.seed)
    print(f"\nRecorded tournament: {output}", flush=True)
    print(json.dumps({"games": len(result["games"]), "stopped_after": result.get("stopped_after")}, indent=2))


if __name__ == "__main__":
    main()
