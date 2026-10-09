"""Compare the unchanged single/council JEV policies against four chess opponents.

Maia uses three separate published neural checkpoints, each with go nodes 1.
Stockfish uses maximum skill with a fixed 200,000-node budget. Neither engine's
search, evaluation, or suggested moves are supplied to JEV.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import chess
import chess.engine
import chess.pgn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from chess_app import find_stockfish, jev_decision, jev_single_decision
from tournament import save_pgn

LEVELS = [
    {"id": "maia_1100", "name": "Maia 1100", "kind": "maia", "training_rating": 1100},
    {"id": "maia_1500", "name": "Maia 1500", "kind": "maia", "training_rating": 1500},
    {"id": "maia_1900", "name": "Maia 1900", "kind": "maia", "training_rating": 1900},
    {"id": "stockfish", "name": "Stockfish (maximum skill)", "kind": "stockfish"},
]
SYSTEMS = {"single": jev_single_decision, "team": jev_decision}


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def write_json(path: Path, data: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def open_opponent(level: dict) -> tuple[chess.engine.SimpleEngine, chess.engine.Limit, dict]:
    if level["kind"] == "maia":
        executable = HERE / "engines" / "lc0" / "lc0.exe"
        weights = HERE / "models" / "maia" / f"maia-{level['training_rating']}.pb.gz"
        if not executable.is_file() or not weights.is_file():
            raise FileNotFoundError(f"Missing Lc0 executable or {weights.name}")
        engine = chess.engine.SimpleEngine.popen_uci(
            [str(executable), f"--weights={weights}", "--backend=blas"], timeout=90
        )
        settings = {"Threads": 1, "MinibatchSize": 1, "NNCacheSize": 200000}
        limit = chess.engine.Limit(nodes=1)
        metadata = {"weights": str(weights.relative_to(HERE)),
                    "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
                    "nodes_per_move": 1, "backend": "blas"}
    else:
        executable = find_stockfish()
        if not executable:
            raise FileNotFoundError("Stockfish executable is missing")
        engine = chess.engine.SimpleEngine.popen_uci(executable, timeout=90)
        settings = {"Threads": 1, "Hash": 64, "Skill Level": 20, "UCI_LimitStrength": False}
        limit = chess.engine.Limit(nodes=200000)
        metadata = {"nodes_per_move": 200000, "skill_level": 20}
    try:
        engine.configure(settings)
    except Exception:
        engine.quit()
        raise
    metadata.update({"engine": engine.id, "settings": settings,
                     "executable_sha256": hashlib.sha256(Path(executable).read_bytes()).hexdigest()})
    return engine, limit, metadata


def run_match(output: Path, level: dict, system: str, max_plies: int) -> dict:
    folder = output / system
    folder.mkdir(parents=True, exist_ok=True)
    stem = level["id"]
    record_path = folder / f"{stem}.json"
    pgn_path = folder / f"{stem}.pgn"
    # Resume an interrupted/API-error match from its saved legal move history.
    if record_path.exists():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record["status"] in {"win", "loss", "draw", "move_cap"}:
            return record
        record["status"] = "playing"
        record.pop("error", None)
        board = chess.Board()
        for item in record["moves"]:
            board.push_uci(item["uci"])
        game = chess.pgn.Game.from_board(board)
    else:
        board = chess.Board()
        game = chess.pgn.Game()
        record = {"system": system, "opponent": level["name"], "level": level,
                  "started_at": now(), "status": "playing", "result": "*",
                  "pgn": f"{system}/{stem}.pgn", "moves": [], "api_retries": []}
    game.headers.update({"Event": "JEV Four Non-LLM Opponents", "Site": "Local",
                         "Date": datetime.now().strftime("%Y.%m.%d"),
                         "White": f"JEV {system}", "Black": level["name"], "Result": "*"})
    node = game.end()
    engine = None
    print(f"START {system} vs {level['name']} (ply {len(board.move_stack)})", flush=True)
    try:
        engine, limit, metadata = open_opponent(level)
        record["opponent_runtime"] = metadata
        write_json(record_path, record)
        save_pgn(game, pgn_path, "*")
        while not board.is_game_over(claim_draw=True) and len(board.move_stack) < max_plies:
            started = time.perf_counter()
            item = {"ply": len(board.move_stack) + 1, "fen_before": board.fen()}
            if board.turn == chess.WHITE:
                for attempt in range(2):
                    try:
                        move, decision = SYSTEMS[system](board)
                        break
                    except Exception as exc:
                        record["api_retries"].append({"ply": item["ply"], "attempt": attempt + 1,
                                                       "error": str(exc)})
                        write_json(record_path, record)
                        if attempt == 1:
                            raise
                item.update({"player": "JEV", "confidence": decision.confidence,
                             "probabilities": decision.probabilities,
                             "role_moves": decision.role_moves, "role_agreement": decision.role_agreement,
                             "tactical_review": decision.tactical_review,
                             "cost_usd": decision.cost_usd, "input_tokens": decision.input_tokens})
            else:
                played = engine.play(board, limit)
                if played.move is None:
                    raise RuntimeError("Chess opponent returned no move")
                move = played.move
                item["player"] = level["name"]
            if move not in board.legal_moves:
                raise RuntimeError(f"Illegal move {move.uci()}")
            item.update({"wall_seconds": time.perf_counter() - started,
                         "uci": move.uci(), "san": board.san(move)})
            node = node.add_variation(move)
            board.push(move)
            item["fen_after"] = board.fen()
            record["moves"].append(item)
            record["updated_at"] = now()
            write_json(record_path, record)
            save_pgn(game, pgn_path, board.result(claim_draw=True))
            if len(board.move_stack) % 10 == 0:
                print(f"PLAY {system} {level['name']} ply={len(board.move_stack)} last={item['san']}", flush=True)
        outcome = board.outcome(claim_draw=True)
        result = board.result(claim_draw=True)
        record.update({"result": result,
                       "status": {"1-0": "win", "0-1": "loss", "1/2-1/2": "draw"}.get(result, "move_cap"),
                       "termination": outcome.termination.name.lower() if outcome else "move_cap"})
    except Exception as exc:
        record.update({"status": "error", "result": "*", "error": str(exc)})
    finally:
        if engine:
            try:
                engine.quit()
            except Exception:
                engine.close()
    jev_moves = [item for item in record["moves"] if item["player"] == "JEV"]
    record.update({"finished_at": now(), "plies": len(board.move_stack),
                   "jev_turns": len(jev_moves),
                   "jev_cost_usd": sum(item.get("cost_usd") or 0 for item in jev_moves),
                   "average_jev_wall_seconds": sum(item["wall_seconds"] for item in jev_moves) / max(1, len(jev_moves))})
    game.headers["Termination"] = record.get("termination", record["status"])
    save_pgn(game, pgn_path, record["result"])
    write_json(record_path, record)
    print(f"RESULT {system} {level['name']}: {record['status']} ({record['result']}) {record['plies']} plies", flush=True)
    return record


def save_summary(output: Path, games: list[dict], max_plies: int) -> None:
    compact = [{key: value for key, value in game.items() if key != "moves"} for game in games]
    compact.sort(key=lambda game: (list(SYSTEMS).index(game["system"]),
                                   next(i for i, level in enumerate(LEVELS) if level["id"] == game["level"]["id"])))
    scores = {}
    for system in SYSTEMS:
        played = [game for game in compact if game["system"] == system]
        wins_in_sequence = 0
        for level in LEVELS:
            match = next((game for game in played if game["level"]["id"] == level["id"]), None)
            if not match or match["status"] != "win":
                break
            wins_in_sequence += 1
        scores[system] = {status: sum(game["status"] == status for game in played)
                          for status in ("win", "draw", "loss", "error", "move_cap")}
        scores[system]["consecutive_levels_passed"] = wins_in_sequence
    summary = {"updated_at": now(), "jev_color": "white", "max_plies": max_plies,
               "protocol": "All four opponents tested regardless of earlier losses; only wins advance the ladder.",
               "rating_note": "Maia names are training rating targets, not measured playing Elo or guaranteed ordering.",
               "sample_note": "One game per system/opponent; preliminary comparison, not an Elo estimate.",
               "completed": len(games) == 8 and all(game["status"] in {"win", "draw", "loss"} for game in games),
               "scores": scores, "games": compact}
    write_json(output / "benchmark.json", summary)
    with (output / "results.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        fields = ["system", "opponent", "status", "result", "plies", "jev_cost_usd", "average_jev_wall_seconds", "pgn"]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(compact)
    lines = ["# JEV vs four non-LLM chess opponents", "", "| Opponent | Single JEV | JEV council |", "|---|---|---|"]
    for level in LEVELS:
        cells = []
        for system in SYSTEMS:
            match = next((game for game in compact if game["system"] == system and game["level"]["id"] == level["id"]), None)
            cells.append(f"{match['status']} ({match['result']})" if match else "pending")
        lines.append(f"| {level['name']} | {' | '.join(cells)} |")
    lines.extend(["", summary["rating_note"], "", summary["sample_note"], "",
                  "Maia: nodes=1, CPU BLAS. Stockfish: Skill Level=20, Threads=1, Hash=64 MiB, 200,000 nodes/move.",
                  "JEV receives the board and legal choices; no opponent search/evaluation is supplied."])
    (output / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-plies", type=int, default=200)
    args = parser.parse_args()
    if args.max_plies < 2:
        parser.error("--max-plies must be >= 2")
    output = args.output_dir or HERE / "recordings" / f"non_llm_{datetime.now():%Y%m%d_%H%M%S}"
    output.mkdir(parents=True, exist_ok=True)
    print(f"RECORDINGS {output.resolve()}", flush=True)
    games = []
    # Two simultaneous matches keep the network calls moving, without artificial delay.
    with ThreadPoolExecutor(max_workers=2) as pool:
        for level in LEVELS:
            futures = [pool.submit(run_match, output, level, system, args.max_plies) for system in SYSTEMS]
            for future in as_completed(futures):
                games.append(future.result())
                save_summary(output, games, args.max_plies)
    print(json.dumps({"output": str(output), "games": len(games)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
