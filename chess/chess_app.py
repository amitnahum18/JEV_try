"""Play a model-vs-model chess game: JEV against Stockfish."""

from __future__ import annotations

import os
import random
import shutil
import sys
import threading
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import chess
import chess.engine


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "car"))
from jev_client import Decision, choose_action  # noqa: E402


BG = "#10151d"
PANEL = "#19222d"
FG = "#e8edf3"
MUTED = "#a3afbd"
ACCENT = "#7bc8a4"
LIGHT = "#e8dcc4"
DARK = "#76906f"
SELECTED = "#e7c85a"

PIECES = {
    "P": chr(0x2659), "N": chr(0x2658), "B": chr(0x2657),
    "R": chr(0x2656), "Q": chr(0x2655), "K": chr(0x2654),
    "p": chr(0x265F), "n": chr(0x265E), "b": chr(0x265D),
    "r": chr(0x265C), "q": chr(0x265B), "k": chr(0x265A),
}

OPPONENTS = [
    {"name": "Beginner - Random", "kind": "random"},
    {"name": "Novice - Material hunter", "kind": "greedy"},
    {"name": "Club - Stockfish 3", "kind": "stockfish", "skill": 3, "time": 0.35},
    {"name": "Strong - Stockfish 10", "kind": "stockfish", "skill": 10, "time": 0.75},
    {"name": "Pro - Stockfish 20", "kind": "stockfish", "skill": 20, "time": 1.2},
]
PIECE_VALUES = {
    chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 335,
    chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 20_000,
}


def find_stockfish() -> str | None:
    configured = os.environ.get("STOCKFISH_PATH")
    if configured and Path(configured).is_file():
        return configured
    installed = shutil.which("stockfish")
    if installed:
        return installed
    bundled = Path(__file__).resolve().parent / "stockfish"
    return next((str(path) for path in bundled.rglob("*.exe")
                 if "stockfish" in path.name.lower()), None)


def move_options(board: chess.Board) -> tuple[dict[str, str], dict[str, chess.Move]]:
    """Give JEV a stable action id for every currently legal move."""
    actions: dict[str, str] = {}
    moves: dict[str, chess.Move] = {}
    for index, move in enumerate(board.legal_moves):
        action_id = f"move_{index}"
        san = board.san(move)
        actions[action_id] = f"Play {san} (UCI: {move.uci()})"
        moves[action_id] = move
    return actions, moves


def chess_state(board: chess.Board) -> dict[str, object]:
    return {
        "fen": board.fen(),
        "board_ascii": str(board),
        "side_to_move": "White" if board.turn == chess.WHITE else "Black",
        "in_check": board.is_check(),
        "legal_moves_san_and_uci": [
            {"san": board.san(move), "uci": move.uci()}
            for move in board.legal_moves
        ],
    }


SINGLE_JEV_PROMPT = (
    "You are JEV playing tournament chess. Select the strongest legal move, not just "
    "an active-looking one. Compare candidate lines at least 10 plies ahead (five full "
    "moves), considering the opponent's strongest reply. Check for forced mates, hanging "
    "pieces or pawns, and repeated positions. When clearly ahead, preserve passed pawns "
    "and convert the advantage; when behind, seek active chances. Prefer a forced win "
    "over a draw, and don't shuffle the king when a pawn plan can make progress. Only "
    "choose one of the provided legal actions. The board is in FEN and ASCII; moves "
    "include SAN/UCI."
)


def jev_single_decision(board: chess.Board) -> tuple[chess.Move, Decision]:
    """Baseline: one JEV call selects a move without a council."""
    actions, move_by_id = move_options(board)
    decision = choose_action(
        state=chess_state(board), actions=actions, instructions=SINGLE_JEV_PROMPT
    )
    return move_by_id[decision.choice], decision


def jev_decision(board: chess.Board) -> tuple[chess.Move, Decision]:
    actions, move_by_id = move_options(board)
    state = chess_state(board)
    shared = (
        "You are one member of JEV's chess decision council. "
        + SINGLE_JEV_PROMPT.removeprefix("You are JEV playing tournament chess. ")
        + " "
    )
    roles = {
        "tactics": shared + (
            "ROLE: TACTICIAN. Examine checks, captures, forks, promotions, and forced "
            "sequences first. Reject any move that loses to the opponent's best reply."
        ),
        "strategy": shared + (
            "ROLE: STRATEGIST. Find a concrete multi-move plan, improve the worst "
            "placed piece, protect the king, and convert passed pawns toward promotion."
        ),
        "risk": shared + (
            "ROLE: RISK REVIEWER. Judge whether aggression is justified by the actual "
            "position: press for a win when ahead or when a forcing line exists; reduce "
            "risk when ahead without need; take calculated risks when behind. Avoid "
            "repetition unless it is the best available result."
        ),
    }

    results: dict[str, Decision] = {}
    errors: list[Exception] = []
    with ThreadPoolExecutor(max_workers=len(roles), thread_name_prefix="jev-council") as pool:
        futures = {
            pool.submit(choose_action, state=state, actions=actions, instructions=prompt): role
            for role, prompt in roles.items()
        }
        for future in as_completed(futures):
            role = futures[future]
            try:
                results[role] = future.result()
            except Exception as exc:
                errors.append(exc)
    if not results:
        raise RuntimeError(f"All JEV council members failed: {errors[0] if errors else 'unknown error'}")

    votes: dict[str, int] = {}
    for result in results.values():
        votes[result.choice] = votes.get(result.choice, 0) + 1
    probability_totals = {action_id: 0.0 for action_id in actions}
    for result in results.values():
        for action_id in actions:
            probability_totals[action_id] += result.probabilities.get(action_id, 0.0)
    average_probabilities = {
        action_id: score / len(results) for action_id, score in probability_totals.items()
    }
    selected = max(
        actions,
        key=lambda action_id: (votes.get(action_id, 0), average_probabilities[action_id]),
    )

    # If the panel proposes a move that lets the opponent legally capture the
    # moved piece, ask JEV for a focused adversarial review before committing.
    # This is local move legality only; no chess engine/tool is involved.
    probe = board.copy()
    probe.push(move_by_id[selected])
    proposed_piece = probe.piece_at(move_by_id[selected].to_square)
    immediate_captures = []
    if proposed_piece and PIECE_VALUES[proposed_piece.piece_type] >= PIECE_VALUES[chess.KNIGHT]:
        immediate_captures = [
            move for move in probe.legal_moves
            if probe.is_capture(move) and move.to_square == move_by_id[selected].to_square
        ]
    review_note = None
    if immediate_captures:
        capture_sans = [probe.san(move) for move in immediate_captures]
        review_state = {
            **state,
            "panel_proposal": {
                "san": board.san(move_by_id[selected]),
                "uci": move_by_id[selected].uci(),
                "votes": votes.get(selected, 0),
                "panel_size": len(roles),
            },
            "position_after_proposal_fen": probe.fen(),
            "opponent_legal_captures_of_moved_piece": capture_sans,
        }
        review_prompt = shared + (
            "ADVERSARIAL AUDITOR: The panel's proposed move leaves the moved piece "
            "legally capturable immediately. Examine the resulting position and the "
            "listed legal captures. Determine whether the sacrifice has concrete "
            "compensation or is a blunder. Choose the strongest move from the original "
            "position; keep the proposal only if it survives this tactical audit."
        )
        try:
            review = choose_action(
                state=review_state, actions=actions, instructions=review_prompt
            )
            results["auditor"] = review
            selected = review.choice
            review_note = (
                "audited; opponent could capture the proposed piece with "
                + ", ".join(capture_sans)
            )
            votes[review.choice] = votes.get(review.choice, 0) + 1
            for action_id in actions:
                average_probabilities[action_id] = (
                    average_probabilities[action_id] * (len(results) - 1)
                    + review.probabilities.get(action_id, 0.0)
                ) / len(results)
        except Exception:
            # Keep the panel's move if the optional review call fails.
            review_note = "review unavailable; kept panel decision"

    supporting = [result.confidence for result in results.values() if result.choice == selected]
    role_moves = {role: move_by_id[result.choice].uci() for role, result in results.items()}
    combined = Decision(
        choice=selected,
        confidence=sum(supporting) / len(supporting) if supporting else 0.0,
        probabilities=average_probabilities,
        latency_seconds=max(result.latency_seconds for result in results.values()),
        input_tokens=(
            sum(result.input_tokens for result in results.values() if result.input_tokens is not None)
            if any(result.input_tokens is not None for result in results.values()) else None
        ),
        cost_usd=(
            sum(result.cost_usd for result in results.values() if result.cost_usd is not None)
            if any(result.cost_usd is not None for result in results.values()) else None
        ),
        role_moves=role_moves,
        role_agreement=votes.get(selected, 0) / (len(roles) + (1 if "auditor" in results else 0)),
        tactical_review=review_note,
    )
    return move_by_id[selected], combined


def jev_turn(board: chess.Board) -> chess.Move:
    return jev_decision(board)[0]


def weak_opponent_turn(board: chess.Board, *, material_only: bool) -> chess.Move:
    legal_moves = list(board.legal_moves)
    if not material_only:
        return random.choice(legal_moves)

    side = board.turn
    best_score = -float("inf")
    best_moves: list[chess.Move] = []
    for move in legal_moves:
        board.push(move)
        score = sum(
            PIECE_VALUES[piece.piece_type] * (1 if piece.color == side else -1)
            for piece in board.piece_map().values()
        )
        if board.is_checkmate():
            score += 100_000
        elif board.is_check():
            score += 20
        board.pop()
        if score > best_score:
            best_score, best_moves = score, [move]
        elif score == best_score:
            best_moves.append(move)
    return random.choice(best_moves)


class ChessMatch:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.opponent_index = 0
        self.opponent = OPPONENTS[self.opponent_index]
        self.root.title(f"JEV vs {self.opponent['name']} | Chess Lab")
        self.root.configure(bg=BG)
        self.root.minsize(790, 620)
        self.board = chess.Board()
        self.jev_color = chess.WHITE
        self.engine_path = find_stockfish()
        self.busy = False
        self.selected: chess.Square | None = None
        self.last_move: chess.Move | None = None
        self.square_buttons: dict[chess.Square, tk.Button] = {}

        self._build_ui()
        self._render()
        self.root.after(350, self._play_next)

    def _build_ui(self) -> None:
        outer = tk.Frame(self.root, bg=BG, padx=24, pady=20)
        outer.pack(fill="both", expand=True)
        header = tk.Frame(outer, bg=BG)
        header.pack(fill="x", pady=(0, 16))
        tk.Label(header, text="CHESS LAB", bg=BG, fg=ACCENT,
                 font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.title_label = tk.Label(
            header, text=f"JEV  vs  {self.opponent['name']}", bg=BG, fg=FG,
            font=("Segoe UI", 21, "bold"),
        )
        self.title_label.pack(anchor="w", pady=(2, 0))

        body = tk.Frame(outer, bg=BG)
        body.pack(fill="both", expand=True)
        board_wrap = tk.Frame(body, bg=PANEL, padx=12, pady=12)
        board_wrap.pack(side="left", anchor="n")
        board_frame = tk.Frame(board_wrap, bg=PANEL)
        board_frame.pack()
        for rank in range(7, -1, -1):
            for file in range(8):
                square = chess.square(file, rank)
                color = LIGHT if (file + rank) % 2 == 1 else DARK
                button = tk.Button(
                    board_frame, text="", width=3, height=1, relief="flat",
                    bd=0, bg=color, activebackground=color, fg="#17202a",
                    font=("Segoe UI Symbol", 31),
                    command=lambda sq=square: self._click_square(sq),
                )
                button.grid(row=7 - rank, column=file, sticky="nsew")
                self.square_buttons[square] = button

        panel = tk.Frame(body, bg=PANEL, width=265, padx=18, pady=16)
        panel.pack(side="left", fill="y", padx=(16, 0))
        panel.pack_propagate(False)
        tk.Label(panel, text="MATCH", bg=PANEL, fg=MUTED,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w")
        self.status = tk.Label(panel, text="Starting...", bg=PANEL, fg=FG,
                               justify="left", wraplength=225,
                               font=("Segoe UI", 12, "bold"))
        self.status.pack(anchor="w", pady=(7, 12))
        self.detail = tk.Label(panel, text="", bg=PANEL, fg=MUTED,
                               justify="left", wraplength=225,
                               font=("Segoe UI", 9))
        self.detail.pack(anchor="w", pady=(0, 14))

        tk.Label(panel, text="MOVE HISTORY", bg=PANEL, fg=MUTED,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w")
        history_frame = tk.Frame(panel, bg=PANEL)
        history_frame.pack(fill="both", expand=True, pady=(6, 14))
        self.history = tk.Text(history_frame, bg="#121a23", fg=FG,
                               insertbackground=FG, relief="flat", wrap="word",
                               font=("Consolas", 10), padx=10, pady=8,
                               state="disabled")
        self.history.pack(fill="both", expand=True)
        self.new_button = tk.Button(
            panel, text="New game", command=self._new_game,
            bg=ACCENT, fg="#102019", activebackground="#91d9b5",
            relief="flat", padx=12, pady=8, font=("Segoe UI", 10, "bold"),
        )
        self.new_button.pack(fill="x")
        self.opponent_button = tk.Button(
            panel, text=self._next_opponent_label(), command=self._next_opponent,
            bg="#263442", fg=FG, activebackground="#344657",
            relief="flat", padx=12, pady=8, font=("Segoe UI", 9),
        )
        self.opponent_button.pack(fill="x", pady=(8, 0))
        self.color_button = tk.Button(
            panel, text="JEV plays White", command=self._toggle_color,
            bg="#263442", fg=FG, activebackground="#344657",
            relief="flat", padx=12, pady=6, font=("Segoe UI", 9),
        )
        self.color_button.pack(fill="x", pady=(8, 0))
        self.retry_button = tk.Button(
            panel, text="Retry move", command=self._play_next,
            bg="#263442", fg=FG, activebackground="#344657",
            relief="flat", padx=12, pady=8, font=("Segoe UI", 9),
        )

    def _render(self) -> None:
        legal_targets: set[chess.Square] = set()
        if self.selected is not None:
            legal_targets = {m.to_square for m in self.board.legal_moves
                             if m.from_square == self.selected}
        for square, button in self.square_buttons.items():
            file, rank = chess.square_file(square), chess.square_rank(square)
            base = LIGHT if (file + rank) % 2 == 1 else DARK
            if square == self.selected or square in legal_targets:
                base = SELECTED if square == self.selected else "#a9bb73"
            piece = self.board.piece_at(square)
            button.configure(text=PIECES[piece.symbol()] if piece else "", bg=base,
                             activebackground=base, state="disabled")
        self._render_history()
        self._update_status()

    def _render_history(self) -> None:
        temp = chess.Board()
        pairs: list[str] = []
        for move in self.board.move_stack:
            if temp.turn == chess.WHITE:
                pairs.append(f"{temp.fullmove_number}. {temp.san(move)}")
            else:
                pairs[-1] += f"   {temp.san(move)}"
            temp.push(move)
        self.history.configure(state="normal")
        self.history.delete("1.0", "end")
        self.history.insert("1.0", "\n".join(pairs) if pairs else "Game has not started")
        self.history.configure(state="disabled")

    def _next_opponent_label(self) -> str:
        index = (self.opponent_index + 1) % len(OPPONENTS)
        tier = OPPONENTS[index]["name"].split(" - ", 1)[0]
        return f"Next opponent: {index + 1}/5 - {tier}"

    def _update_status(self) -> None:
        if self.board.is_game_over():
            result = self.board.result()
            reason = "Checkmate" if self.board.is_checkmate() else "Draw"
            self.status.configure(text=f"{reason} - {result}", fg=ACCENT)
            self.detail.configure(text=self.board.outcome().termination.name.replace("_", " ").title())
            return
        player = "JEV" if self.board.turn == self.jev_color else self.opponent["name"]
        color = "White" if self.board.turn else "Black"
        self.status.configure(text=f"{player} to move", fg=FG)
        self.detail.configure(text=f"{color} to move\n"
                              f"{'Check' if self.board.is_check() else 'Game in progress'}")

    def _play_next(self) -> None:
        if self.busy or self.board.is_game_over():
            return
        if self.opponent["kind"] == "stockfish" and not self.engine_path:
            self.status.configure(text="Stockfish is not installed", fg="#f0b36b")
            self.detail.configure(text="Install Stockfish and set STOCKFISH_PATH, or add it to PATH.")
            return
        self.busy = True
        self.new_button.configure(state="disabled")
        self.opponent_button.configure(state="disabled")
        self.color_button.configure(state="disabled")
        snapshot = self.board.copy()
        fen = snapshot.fen()
        is_jev = snapshot.turn == self.jev_color
        player = "JEV" if is_jev else self.opponent["name"]
        opponent = self.opponent
        self.status.configure(text=f"{player} is thinking...", fg=ACCENT)
        self.detail.configure(text="Evaluating the current position")

        def worker() -> None:
            try:
                if is_jev:
                    move = jev_turn(snapshot)
                elif opponent["kind"] in {"random", "greedy"}:
                    move = weak_opponent_turn(
                        snapshot, material_only=opponent["kind"] == "greedy"
                    )
                else:
                    engine = chess.engine.SimpleEngine.popen_uci(self.engine_path)
                    try:
                        if "Skill Level" in engine.options:
                            engine.configure({"Skill Level": opponent["skill"]})
                        result = engine.play(
                            snapshot, chess.engine.Limit(time=opponent["time"])
                        )
                        move = result.move
                    finally:
                        engine.quit()
                san = snapshot.san(move)
                self.root.after(0, lambda: self._apply_result(fen, move, san, None))
            except Exception as exc:  # Surface API and engine errors in the match panel.
                self.root.after(0, lambda err=str(exc): self._apply_result(fen, None, "", err))

        threading.Thread(target=worker, daemon=True).start()

    def _apply_result(self, fen: str, move: chess.Move | None, san: str, error: str | None) -> None:
        self.busy = False
        self.new_button.configure(state="normal")
        self.opponent_button.configure(state="normal")
        self.color_button.configure(state="normal")
        if self.board.fen() != fen:
            return
        if error:
            self.status.configure(text="Move could not be completed", fg="#f0b36b")
            self.detail.configure(text=error[:260])
            self.retry_button.pack(fill="x", pady=(8, 0))
            return
        assert move is not None
        self.retry_button.pack_forget()
        mover = "JEV" if self.board.turn == self.jev_color else self.opponent["name"]
        self.board.push(move)
        self._render()
        self.detail.configure(text=f"{mover} played {san}\n"
                              f"{'Check' if self.board.is_check() else 'Next turn'}")
        if not self.board.is_game_over():
            self.root.after(200, self._play_next)

    def _new_game(self) -> None:
        self.board.reset()
        self.last_move = None
        self.selected = None
        self.retry_button.pack_forget()
        self._render()
        self._play_next()

    def _next_opponent(self) -> None:
        if self.busy:
            return
        self.opponent_index = (self.opponent_index + 1) % len(OPPONENTS)
        self.opponent = OPPONENTS[self.opponent_index]
        self.root.title(f"JEV vs {self.opponent['name']} | Chess Lab")
        self.title_label.configure(text=f"JEV  vs  {self.opponent['name']}")
        self.opponent_button.configure(text=self._next_opponent_label())
        self._new_game()

    def _toggle_color(self) -> None:
        if self.busy or self.board.move_stack:
            return
        self.jev_color = not self.jev_color
        self.color_button.configure(
            text=f"JEV plays {'White' if self.jev_color else 'Black'}"
        )
        self._update_status()

    def _click_square(self, square: chess.Square) -> None:
        # The board is intentionally read-only: this is an autonomous model match.
        return


def main() -> None:
    root = tk.Tk()
    ChessMatch(root)
    root.mainloop()


if __name__ == "__main__":
    main()
