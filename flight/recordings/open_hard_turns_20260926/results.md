# Open frames with harder turns

Eight physical rectangular frames with no surrounding walls. One JEV, existing 15 joint pitch/bank choices, unchanged aircraft physics and prompt v6. Heights approximately 19-35m. Seed-44 centerline direction changes reach 45.7 degrees.

| Batch | Seed | Frames passed | Status | Flight seconds | Minimum passing margin (m) |
|---|---:|---:|---|---:|---:|
| initial | 44 | 0/8 | api_error | 3.500 | n/a |
| initial | 46 | 8/8 | completed | 67.825 | 0.438 |
| initial | 45 | 8/8 | completed | 65.650 | 0.363 |
| retry | 44 | 8/8 | completed | 66.392 | 0.211 |

All three seeded courses were completed without a crash. There were four attempts: three completions and one transport timeout at 3.5s, followed by a successful retry. No credit-exhaustion error occurred in these new runs.

The selected source is `v6/single_focused_seed46/episode.json`, identified in the initial batch selection file. The completed source is replayed at 1x with native physics and a two-second completion hold. No new model calls are made during video export.

MP4: `../../videos/open_turns_source/jev_8of8_open_frames_1x.mp4`.

Earlier wall recordings remain intact for reproducibility; the new default in `run_maze.py` is the open `hard_turns` layout. The live viewer accepts `--layout hard_turns --gates 8 --seconds 85 --policy single --telemetry focused --prompt-version 6`.

These are three exploratory seeded successes, not a guarantee that JEV can never miss a frame or crash. The frame safety margins and original termination rules were not relaxed.
