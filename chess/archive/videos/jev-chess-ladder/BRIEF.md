workflow: general-video
flow: automation
storyboard: no
message: Compare single JEV and the existing JEV council against four non-language chess opponents.
destination: YouTube
aspect: 16:9
language: Hebrew
audience: chess and AI viewers
length: about 2 minutes
angle: replay all eight real games and finish with a four-row win/loss/draw table.
voice: none

## Intent

A Hebrew, no-narration replay of the real experiment: one JEV call per move versus the existing parallel JEV council. Each system plays White against Maia 1100, Maia 1500, Maia 1900, and Stockfish 19. All opponents are tested even after an earlier loss. Never depict a lost game as passing a level. The recorded run ended in eight losses by checkmate.

## Assets

- `../../recordings/non_llm_four_20260926/benchmark.json` — all eight results and exact opponent settings.
- `../../recordings/non_llm_four_20260926/{single,team}/*.pgn` — complete real game histories.
- Matching per-game JSON files — real JEV choices and confidence values.
- `build_non_llm_video.py` — reproducible generator for this comparison.

## Customizations

- Replay every recorded move on a readable chessboard, labeling single JEV versus team JEV.
- Show move number, confidence, and council agreement.
- Finish with a Hebrew table. Maia numbers describe training rating targets, not measured Elo. Maia uses nodes=1; Stockfish uses maximum skill and 200,000 nodes/move.
- Keep the existing dark board look with muted green accents, no narration, and no added music.
