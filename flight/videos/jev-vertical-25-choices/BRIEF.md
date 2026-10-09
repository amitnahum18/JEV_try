---
workflow: general-video
flow: automation
storyboard: no
message: "Deliver the actual latest JEV flight with extreme frame heights and 25 joint choices at normal simulation speed."
destination: local-mp4
aspect: native-simulator
length: "27.875 seconds of flight plus a two-second final hold"
---

## Authorization and source

The user first requested a live preview before deciding whether to record it.
Their latest request, "תביא את הסרטון", now authorizes the video export.
Use the saved latest flight, not a different successful earlier course.

Source:
`../../recordings/vertical_25_choices_preview_20260926/single_level1_seed44_232048_671155.json`.

One JEV chooses among 25 pitch/bank combinations: gentle pitch +6/-6 degrees,
nominal trim +2 degrees, strong pitch +20/-16 degrees, and five bank choices.
The course has eight open frames with heights approximately 16-58 metres.

## Actual result

The flight passed two frames and missed the third after 27.875 simulated
seconds. Its minimum passing margin at the third frame was -0.08228m.
This is a real failed attempt, not a completed eight-frame flight. Retain
the actual gate count and terminal "GATE MISSED" banner.

## Presentation

One continuous, silent native simulator scene at playback rate 1. Keep the
aircraft, chase camera, numerical HUD, actual action labels, actuator readings
and trajectory map. Remove API waiting time only. No invented continuation,
new model requests, narration, music, cropping or speed change.

Expected video: 898 frames at 30 fps, approximately 29.933333 seconds,
including the final hold and frame-time quantization.
