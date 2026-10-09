---
workflow: general-video
flow: automation
storyboard: no
message: "Show the successful single-JEV flight through all five gates at normal simulation speed."
destination: local-mp4
aspect: native-simulator
length: "29.6167 seconds of flight plus 2 seconds holding the completion screen"
---

## Intent

User: "אני רוצה את הסרטון במהירות רגילה של המסלול שהוא עשה".
Replay the actual successful seed-44 episode at one video second per simulation
second and deliver MP4. The user's prior authorization for automatic production
without review pauses remains in effect.

## Assets

- `../../recordings/sharp_turn_20260926/buffered_v5/v5/single_focused_seed44/episode.json`: actual JEV decisions and physics-step durations.
- `assets/flight_1x.mp4`: native simulator HUD and real physics replay, 30 fps.

## Notes

One continuous scene with the existing chase camera, simulator colors and
rasterized fonts. Keep actual gate counts, flight data and trail. Silent output
without model-waiting screens. Expected duration: 950 frames / 30 fps = 31.666667
seconds, including a 2-second completion hold and frame timing quantization.
