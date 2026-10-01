# Device matrix

Which tier runs on which iPhone, and what accuracy each tier delivers.

| iPhone | Photos | Video | LiDAR |
|---|---|---|---|
| 15, 15 Plus | yes | yes | no (no LiDAR sensor) |
| 15 Pro, 15 Pro Max | yes | yes | yes |
| 16, 16 Plus, 16e | yes | yes | no |
| 16 Pro, 16 Pro Max | yes | yes | yes |
| 17, Air | yes | yes | no |
| 17 Pro, 17 Pro Max | yes | yes | yes |

LiDAR needs the sensor next to the camera lenses, which only the Pro models have. The photo and
video tiers use only the ordinary camera, so every iPhone 15 or newer runs them. They also work on
any other phone, but we only test iPhones.

## Accuracy per tier

| Tier | What the pipeline gets | Gate | Interval half-width on a 4 m wall (90%) | Measured on real rooms |
|---|---|---|---|---|
| LiDAR | depth, pose and intrinsics every frame | walls within 2 cm | about 2 cm | pending, see benchmark/ |
| Video | 30 fps frames, no depth, no poses | walls within 3% | about 40 cm (7 cm with a marker) | pending |
| Photos | 2 to 8 stills per room, no depth, no poses | walls within 8% | about 66 cm (10 cm with a marker) | pending |

The interval column comes from the error model in `src/floorplan/uncertainty.py`. Those are priors.
`floorplan bench` measures how often the tape measurement lands inside the interval, and
`floorplan calibrate --write` scales the intervals until that is 90%. The last column fills in
from that run.

Why the tiers differ:

- **LiDAR** measures depth directly, to a few millimetres at room distances. Its error is mostly
  drift in the phone's pose over a long walk, which `drift.py` corrects.
- **Video** and **photos** get depth and camera positions from a pretrained model (MapAnything).
  The room's shape comes out well. Its absolute size depends on the model's sense of scale, which
  can be well off on a room it has never seen. More views (video) average that down. Fewer views (photos) don't, so
  the photo intervals are the widest.
- A printed marker of known size replaces the model's scale with a measurement. That's why it
  tightens the photo and video intervals so much.

## Processing machine

Any Mac with Apple silicon or a Linux machine with an NVIDIA GPU. CPU-only works but is slow.

Memory: 32 GB recommended for the photo and video tiers. They were developed on an M5 with 16 GB,
where the model (about 5 GB of weights) runs all of a capture's views in one pass: up to 60
keyframes for a video. Under that load the 16 GB machine restarted several times, so on 16 GB run
one capture at a time with nothing else heavy open. Every video is capped at 60 keyframes;
`--fps 1` uses fewer for a clip under a minute. The LiDAR tier does not use the model and runs
comfortably in 16 GB.

The first run downloads about 5 GB of model weights (`uv run floorplan setup`). Nothing is sent to any server after that.
