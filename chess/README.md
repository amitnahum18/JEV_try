# JEV vs Stockfish

## Folder layout

The chess experiment is complete. The active folder keeps the reusable code,
installed opponents, and the final eight-game comparison:

- `chess_app.py` — interactive board and shared single/council JEV policies.
- `non_llm_benchmark.py` — four-opponent comparison runner.
- `tournament.py`, `online_tournament.py` — earlier reusable runners.
- `engines/`, `models/`, `stockfish/` — chess opponents and licenses.
- `recordings/non_llm_four_20260926/` — final results, PGNs, and decision logs.
- `non_llm_opponents.json` — versions, configurations, and model source URLs.
- `archive/recordings/` — earlier experiments and interrupted runs.
- `archive/videos/` — historical video projects and MP4 files.

Video production is archived. The retained chess scripts use their original
paths so their imports continue to work.

## Four non-language chess opponents

The new comparison uses four opponents, in intended difficulty order:

| Opponent | Implementation | Settings |
|---|---|---|
| Maia 1100 | Separate published neural checkpoint | Lc0 0.32.1, CPU BLAS, 1 node/move |
| Maia 1500 | Separate published neural checkpoint | Lc0 0.32.1, CPU BLAS, 1 node/move |
| Maia 1900 | Separate published neural checkpoint | Lc0 0.32.1, CPU BLAS, 1 node/move |
| Stockfish 19 | Chess engine with NNUE evaluation | Skill 20, 1 thread, 64 MiB hash, 200,000 nodes/move |

Maia numbers are human rating bands used for training, not certified playing Elo;
they do not guarantee that every higher-numbered model is harder in every position.
Stockfish uses maximum skill at a fixed computation budget. Neither is an LLM.

Run both JEV policies against all four opponents:

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
.\.venv\Scripts\python.exe chess\non_llm_benchmark.py
```

The single policy makes one JEV call per turn. The council uses the existing
tactics, strategy, risk, and optional audit roles. Engines only play the opposing
side; their evaluation/search is never given to JEV. Two matches run concurrently,
and each opponent process stays open for its entire match.

The comparison tests all opponents even after losses. Ladder advancement counts
consecutive wins from the first opponent. Each system plays White from the standard
starting position once per opponent. This is a preliminary eight-game comparison,
not an Elo measurement. Move caps and API failures are recorded as unfinished.

Outputs: complete PGNs, move-by-move decision JSON, `benchmark.json`, `results.csv`,
and `results.md`. Resume interrupted games with `--output-dir` pointing to the same
directory; completed games are preserved.

Recorded comparison: `recordings/non_llm_four_20260926/`.
Both systems lost all four games by checkmate in this run.

Official sources and licenses:

- Maia: https://github.com/CSSLab/maia-chess; GPL-3.0 license at `models/maia/LICENSE`.
- Lc0: https://github.com/LeelaChessZero/lc0/releases/tag/v0.32.1;
  license at `engines/lc0/COPYING`.
- Stockfish: https://github.com/official-stockfish/Stockfish; bundled in `stockfish/`.
- Each match records engine IDs, settings, and model/executable SHA256 hashes.

## Original interactive board

A local model-vs-model chess match. JEV chooses among the legal moves returned by
the `typesafe/jev-1.13` OpenRouter endpoint; Stockfish replies as the professional
chess engine. The Tkinter board shows the game and move history.

## Setup

1. Install the Python package:

   ```powershell
   .\.venv\Scripts\python.exe -m pip install -r chess\requirements.txt
   ```

2. Install the Stockfish engine for your operating system. Put `stockfish.exe`
   on `PATH`, or set `STOCKFISH_PATH` in the environment to its full path.
3. Set `OPENROUTER_API_KEY` in the project's `.env` file (the existing JEV
   client loads it automatically).
4. Start the match:

   ```powershell
   .\.venv\Scripts\python.exe chess\chess_app.py
   ```

JEV plays White by default. Use **Next opponent** to start a new game at each
successively stronger level: random legal moves, a simple material hunter,
Stockfish skill 3, skill 10, and skill 20. Use **JEV plays White/Black** before
the opening move to switch sides. If a JEV request fails, the game pauses and
**Retry move** retries the same turn.

