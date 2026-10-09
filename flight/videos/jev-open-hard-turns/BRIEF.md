---
workflow: general-video
flow: automation
storyboard: no
message: "Show the completed single-JEV flight through eight open frames, with sharper turns and altitude changes."
destination: local-mp4
aspect: native-simulator
length: "67.825 seconds of recorded flight plus a two-second completion hold"
---

## Intent

Status: native footage is exported and the composition passed checks. The
later user instruction requests a live preview with larger height changes
and reserves the video-recording decision. Stop production here; do not
export a video of the new vertical challenge until the user requests it.

The user corrected the previous wall maze: remove the surrounding walls and
use open frames with difficult turns. Credits have been replenished. Preserve
their authorization for automatic production and deliver an actual flight
recording at normal simulation speed.

## Actual source

- `../../recordings/open_hard_turns_20260926/v6/single_focused_seed46/episode.json`:
  the selected actual completed 8/8 flight.
- `assets/flight_1x.mp4`: native PyFlyt replay of the recorded JEV choices,
  including the measured flight HUD, controls and trajectory.

The source uses one JEV with the same 15 joint choices and prompt v6. The
layout has eight open rectangular frames, lateral reversals and altitude
changes. There are no surrounding wall panels.

## Presentation

One continuous silent shot at playback rate 1, with the native chase camera.
Remove API waiting time only. Keep the flight data, gate counts and actual
completion banner. Expected length is 2096 frames at 30 fps, approximately
69.866667 seconds including the completion hold and frame quantization.
All attempts, including a transport timeout, are retained in the reports.
