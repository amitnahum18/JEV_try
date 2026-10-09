# Design

Preserve the simulator's actual visual identity and measured HUD. The airplane
and gate opening are the focal content; the action and flight readings occupy
the right edge, while the actual trajectory map remains below. Use the native
1440 by 860 canvas without cropping or stretching. Existing rasterized Segoe UI
and Consolas text is contained in the video asset; no new DOM typography.
The background is the native navy #0e1620, with existing turquoise gate geometry
and orange trajectory. One uninterrupted flight at 1x; no added camera animation.
