# Eight-wall JEV flight maze

## Course and controls

The new `maze` layout uses actual static collision walls around eight
rectangular openings. The openings alternate laterally and vertically:
seed 44 uses heights approximately 24, 31, 20, 34, 17, 29, 36 and 22 metres.
The course requires climbs, descents and turns in both directions.

Each aircraft is controlled by one JEV. It chooses one of the existing 15
joint pitch-and-bank actions. Physics, action amplitudes and the stabilizer
were retained. The scene is a winding obstacle course with forward-facing
openings, not a branching maze or a course with U-turns.

## Recorded attempts

| Prompt | Seed | Openings passed | Result |
|---|---:|---:|---|
| v5 | 44 | 1/8 | Missed opening 2 vertically |
| v5 | 45 | 1/8 | Missed opening 2 |
| v5 | 46 | 1/8 | Missed opening 2 |
| v6 | 44 | 4/8 | API stopped: HTTP 402, insufficient credits |
| v6 | 45 | 4/8 | API stopped: HTTP 402, insufficient credits |
| v6 | 46 | 4/8 | API stopped: HTTP 402, insufficient credits |

Prompt v6 describes each candidate in terms of the signed pitch and bank
gaps provided to JEV. It explicitly asks JEV to assess both axes: turning
does not cancel a required climb or descent. All 15 choices remain
available, and no Python rule selects, filters or overrides JEV's choice.

The initial seed-44 miss was approximately 6.4 centimetres outside the
vertical passing margin. The updated attempts demonstrate greater progress
on these three seeds, but do not establish reliable completion. No attempt
has passed all eight openings; the latter three ended for a technical
reason, not a recorded navigation failure.

## Video selection

`criteria_v6/single_focused_seed44/episode.json` was selected from the tied
four-opening attempts. Every attempted flight is preserved in the two
`results.json` files. The MP4 replays the actual selected actions through
PyFlyt at 1x, removing only API waiting time and retaining the gate counter
and final technical stop. Rendering uses separate offline processes and
makes no new model requests.

Final MP4: `../../videos/jev-maze-normal-speed/jev_maze_partial_normal_speed.mp4`.

Completing the eight-opening challenge requires renewed OpenRouter credits
and additional actual JEV flight time. Do not treat this partial result as
a successful completed maze.
