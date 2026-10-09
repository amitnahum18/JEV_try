# Latest vertical flight video

The user requested export after viewing the live simulation. The saved
source is `single_level1_seed44_232048_671155.json`; no additional flight or
model requests were made for this export.

| Item | Recorded value |
|---|---|
| Controller | One JEV, 25 joint pitch/bank choices |
| Pitch options | +6, +20, +2, -6, -16 degrees of nose-up attitude |
| Course | Eight open frames, heights approximately 16-58m |
| Outcome | 2/8 passed; third frame missed |
| Flight time | 27.875 simulated seconds |
| Observed altitude range | Approximately 21.93-43.50m |
| Third-frame passing margin | -0.08228m |
| Export | 1440x860, 30 fps, playback speed 1x, 898 frames |
| Video length | Approximately 29.933s, including a two-second final hold |

The third-frame miss is evaluated using the existing passing margins:
1.25m horizontally and 0.5m vertically. The aircraft center crossed the
frame's x plane at z=22.17793m, approximately 8.2cm above the highest center
position allowed by the vertical passing margin. This is recorded as
`gate_missed`, not as a completed course or an API-credit stop.

The video replays all 3345 recorded physical steps, retains the native HUD
and true terminal banner, and removes time waiting for API responses.
Each rendering chunk checks the final position and outcome against the
source. The original recording is retained unchanged.

MP4:
`../../videos/jev-vertical-25-choices/jev_vertical_25_choices_normal_speed.mp4`.

The HyperFrames project pin was updated from 0.8.78 to 0.8.79 before its
composition validation and final render. Video verification is saved in the
project's `verification.json`.
