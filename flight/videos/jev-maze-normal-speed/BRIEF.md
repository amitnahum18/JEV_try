---
workflow: general-video
flow: automation
storyboard: no
message: "Show the actual single-JEV flight through the harder three-dimensional wall maze at normal simulation speed."
destination: local-mp4
aspect: native-simulator
length: "Actual recorded flight plus a two-second hold on its technical stop"
---

## Intent

The user requested a harder maze with altitude changes and turns, followed by
an actual flight recording. Their authorization for automatic production
without review pauses remains in effect. Deliver a local MP4 at one video
second per simulation second.

## Actual result

Eight physical walls with rectangular openings form a winding course. The
selected seed-44 attempt with prompt v6 passed four of eight openings before
OpenRouter returned HTTP 402 for insufficient credits. This is a partial
recording. It does not demonstrate completion of the new course.

## Assets

- `../../recordings/maze_20260926/criteria_v6/single_focused_seed44/episode.json`:
  actual JEV choices, physics-step durations and the API error.
- `assets/flight_1x.mp4`: deterministic replay of those recorded choices using
  the same native aircraft physics and the simulator HUD, at 30 fps.

## Presentation

One continuous, silent scene with the native chase camera and trajectory map.
Preserve altitude, action labels, gate counts and final technical stop banner.
Remove time spent waiting for API responses; do not accelerate simulation
time, invent choices, or append a successful continuation.
