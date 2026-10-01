# floorplan

Phone captures in, a dimensioned, stitched floor plan out. Every number comes with a 90% interval.
The plan also lists damage, what that damage may be hiding, and a scope of work.

Three input tiers, one output format:

| Tier | Input | Where the 3D comes from |
|---|---|---|
| Photos | a folder of 2 to 8 photos per room | MapAnything, a pretrained model: depth, camera poses and scale from images alone |
| Video | one walkthrough clip | the same model on the sharpest frame of every half second |
| LiDAR | a Stray Scanner recording (depth, poses, intrinsics) | the phone's LiDAR depth, with drift corrected |

How to capture is in [CAPTURE.md](CAPTURE.md). Which iPhone runs which tier is in [DEVICES.md](DEVICES.md).

## Install (clean machine, about 10 minutes plus the download)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh       # uv, the Python package manager
git clone https://github.com/amantibrewal310/floor-plan-generator.git
cd floor-plan-generator
uv sync                                               # Python 3.11 and every dependency
uv run floorplan setup                                # model weights, about 5 GB, once
```

Needs a Mac with Apple silicon or a Linux machine with an NVIDIA GPU. CPU-only works, slowly.

## Run (one command per capture)

```bash
uv run floorplan photos kitchen hall bedroom -o out    # one folder per room
uv run floorplan video IMG_1234.MOV -o out
uv run floorplan lidar 71de12f9 -o out                 # the Stray Scanner folder
```

Each writes `out/plan.svg` (the drawing) and `out/plan.json` (everything, to
[the schema](schema/plan.schema.json)) and prints a summary. This one is two synthetic LiDAR scans
of a two-room flat (`floorplan synth apartment lidar`), stitched at the shared doorway:

```
R1 Room 1: 14.40 m², ceiling 260.1 [258.6, 261.7] cm
  walls (cm, 90% interval): 400.0 [397.2, 402.9], 360.0 [357.1, 363.0], 400.0 [397.1, 403.0], 360.0 [357.1, 363.0]
  door on R1.W2: 89.0 [87.1, 90.9] cm
  door on R1.W4: 85.9 [84.1, 87.8] cm
R2 Room 2: 11.81 m², ceiling 260.1 [258.6, 261.7] cm
  walls (cm, 90% interval): 328.0 [325.2, 330.8], 359.9 [357.1, 362.8], 328.0 [325.2, 330.9], 359.9 [357.0, 362.9]
  door on R2.W4: 91.6 [89.7, 93.5] cm
adjacent: R1 <-> R2
stitch: scan_1.ply: attached by matching doorway
```

Damage, concealed-damage flags and scope lines print after the rooms when there are any.

Useful flags: `--marker-size 0.18` if a printed marker is in view (tightens the scale),
`--no-drift-correction` for the ablation, `--no-damage` to skip damage detection.

## Scoring against a tape measure

```bash
uv run floorplan score out/plan.json truth.json        # gates, errors, interval coverage
uv run floorplan repeat run1/plan.json run2/plan.json  # repeatability gate
uv run floorplan bench benchmark/manifest.json         # the whole benchmark, writes report.md
uv run floorplan calibrate benchmark/out --write       # fit the 90% intervals to the benchmark's truth
uv run floorplan walkcheck <stray folders>             # no tape: share of the walk inside a room
```

The ground-truth format and the benchmark layout are in [benchmark/README.md](benchmark/README.md).
Model outputs are cached by input content, so reruns are exact. Delete `~/.cache/floorplan` to
force the live path.

## How it works

```
photos / video frames ─ MapAnything ─┐
                                     ├─ gravity + floor ─ drift correction ─ plan.extract_rooms ─ stitch ─ plan.json / plan.svg
LiDAR depth + poses ─────────────────┘                                              │
images ─ OWLv2 damage boxes ─ back-projected through each view's points ────────────┴─ damage, flags, scope
```

1. **3D points.** Photos and video go through MapAnything (Meta, Apache-2.0 weights). It returns a
   metric depth map, a camera pose and intrinsics for every image. LiDAR frames are back-projected
   with their own poses. Low-confidence LiDAR depth is dropped, which removes most glass and mirrors.
2. **Gravity.** The average camera up gives a first guess. Floor and ceiling normals refine it.
3. **Drift** (`drift.py`). Phone odometry drifts in heading as you walk. The walls are vertical
   planes on two perpendicular directions, the same across the whole home. The trajectory is cut
   into 1 m chunks, and each chunk's heading snaps to the first chunk's wall directions. Its floor is
   levelled and it slides up to 5 cm onto walls already mapped. On a synthetic walk with 0.8°/m
   drift this takes corner error from 7.5 cm to 2.1 cm.
4. **Rooms** (`plan.py`). Points between 1 m and the ceiling are rasterised at 2 cm. Doorways and
   unseen corners are closed, and flood fill leaves one region per room. Camera positions mark which
   regions are rooms. Each edge is refit to the raw points, which gets mm-level walls out of a 2 cm
   grid. Openings are gaps in the wall band: empty to the floor is a door, wall below and above is a
   window. Ceiling height comes from the points above each room.
5. **Stitching.** A walkthrough is already in one frame. Separate captures (photo folders) join at
   matching doorways, which means equal widths on opposite faces of one wall, with no overlap.
6. **Intervals** (`uncertainty.py`). A per-tier error model, widened for walls that were only partly
   seen. `floorplan bench` checks that about 90% of tape measurements land inside.
7. **Damage** (`damage.py`). OWLv2 finds water stains, mould, cracks, peeling paint and holes. Every
   3D point remembers its pixel, so a box maps onto a wall, floor or ceiling with a metric size. Five
   explicit rules raise concealed-damage flags, and a table turns each region into line items.

## Tests

```bash
uv run pytest -m "not slow"   # about 1 minute
uv run pytest                 # adds the full photo-tier run through MapAnything
```

## Limitations

- Walls are assumed vertical and mostly at right angles. Other angles work, but short off-axis
  walls under 35 cm are dropped.
- Openings wider than about 1.4 m with no wall above them (archways, open plan) merge the rooms on
  either side.
- Photo and video scale comes from the model, so without a marker those intervals are wide on
  purpose. See DEVICES.md.
- A room photographed from one side comes out as the rectangle its seen walls span, flagged
  `unclosed`, with wide intervals on the unseen walls. Photograph every wall.
- Video must be one continuous take. An edited video with cuts between rooms is posed as one walk
  and comes out in fragments.
- Damage detection is open vocabulary and untuned. Expect misses on faint stains and false hits on
  dark patterns.
