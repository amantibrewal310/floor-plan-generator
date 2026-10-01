# Technical report

No number in this report comes from anywhere but a run you can repeat. Numbers on the real
benchmark are in `benchmark/out/report.md`, written by `floorplan bench`.

## 1. Architecture

Every tier ends as the same thing: per-frame 3D points in a z-up metric frame, in capture order,
each point tagged with the pixel it came from. Everything after that is shared.

| Stage | Module | Photos | Video | LiDAR |
|---|---|---|---|---|
| 3D points | `recon.py`, `lidar.py` | MapAnything, all photos of a room in one pass | MapAnything on keyframes (sharpest per 0.5 s, up to 60) | LiDAR depth (confidence 2 only) with ARKit poses |
| Gravity | `recon.gravity_rotation`, `lidar.level` | camera up, refined by floor/ceiling normals | same | ARKit gravity, then a floor-plane fit |
| Drift | `drift.py` | none (stills have no trajectory) | on | on |
| Rooms, walls, openings | `plan.py` | shared | shared | shared |
| Stitching | `stitch.py` | doorway matching across folders | one frame already | one frame already |
| Intervals | `uncertainty.py` | widest | middle | tightest |
| Damage | `damage.py` | model's own images | keyframes | about 40 RGB frames from rgb.mp4 |

The geometry core predates the ML tiers. It rasterises the wall band (1 m up to just under the
ceiling), closes doorway and unseen-corner gaps with line kernels, flood-fills, then refits each
edge to the raw points with a robust median. That refit is where cm accuracy comes from. The grid
only has to be right to a cell.

**Why MapAnything.** The spec's photo tier (2 to 8 stills, no poses, "any picture in") rules out
structure from motion. COLMAP needs dozens of overlapping views and texture on the walls. The
original version of this repo used it with printed markers and could not meet the tier.
MapAnything predicts metric depth, poses and intrinsics jointly from unposed images, runs locally
(8 s for 6 photos on an M5), and has Apache-2.0 weights. Depth Anything 3 was the alternative. Its
metric multi-view model is CC BY-NC, and it pins `numpy<2` and `xformers`, which does not install
cleanly on macOS.

**Why Stray Scanner.** The LiDAR tier asks for depth, poses and intrinsics. Stray Scanner is free,
exports exactly that as plain files, and documents the format. RoomPlan-based apps export fitted
boxes instead, which hides the raw data the drift ablation needs. The parser for RoomPlan JSON and
PLY/OBJ meshes stays in as an extra input.

## 2. Tiers and device matrix

See `DEVICES.md`. LiDAR runs on the Pro models of iPhone 15, 16 and 17. Photos and video run on
any iPhone 15 or newer. The intervals widen from LiDAR to video to photos because the scale source
gets weaker. LiDAR measures depth. Video averages the model's scale estimate over many frames.
Photos average it over a handful.

## 3. Drift handling

ARKit's visual-inertial odometry sees gravity, so roll and pitch don't drift. Heading, height and
position do, a little per metre walked. `drift.py` re-integrates the trajectory in 1 m chunks:

1. Heading. Each chunk's wall direction (mod 90°) is measured from its own points and snapped to
   the first chunk's. The correction is tracked incrementally, prediction plus a residual under
   10°, so total drift can pass the 45° ambiguity of a right-angled frame.
2. Height. The chunk's floor is shifted to the global floor.
3. Position. The chunk slides onto walls already mapped, found with 1-D histograms along the two
   wall directions. The slide is capped at 5 cm. The first version allowed 25 cm and snapped a
   whole room onto the far face of the 12 cm interior wall. The cap is below half a wall's thickness.

Each chunk's motion is then hung off the corrected end of the previous one, so every later chunk
inherits the fix.

Ablation on a synthetic two-room walk whose logged poses drift 0.8°/m (about 20° by the end),
from `test_stray_lidar_drift_correction`:

| | wall edges found | max wall error | corner RMSE | doors |
|---|---|---|---|---|
| poses as-is | 6 + 7 (walls smeared) | 11.1 cm | 7.5 cm | 2/3 |
| corrected | 4 + 4 | 2.2 cm | 2.1 cm | 3/3 |
| no drift, corrected | 4 + 4 | 2.2 cm | 2.0 cm | 3/3 |

The last row shows the correction costs nothing on a clean capture.

On the three real sample walkthroughs (no tape, so the truth-free walk check and the stitched
footprint), drift correction on and off:

| capture | walk | footprint on / off (m²) | walk inside a room on / off | components on / off |
|---|---|---|---|---|
| c00a170fe1 | 14 m | 17.6 / 17.7 | 92.6 / 93.9% | 1 / 1 |
| 1a8384c3f6 | 53 m | 53.4 / 53.7 | 96.6 / 96.0% | 3 / 4 |
| c7d28f72c6 | 98 m | 42.4 / 43.2 | 86.2 / 82.1% | 7 / 6 |

ARKit drifts little over walks this short, so the correction moves the footprint by 0.5 to 1.8%.
It helps most on the longest walk and is slightly worse on the shortest, where there is almost no
drift to remove. Regenerate with `floorplan lidar <folder> [--no-drift-correction]` and
`floorplan walkcheck [--no-drift-correction]`. The ablation with tape truth is in
`benchmark/out/report.md` once the benchmark is captured.

Known limit: heading snapping assumes the rooms share one right-angled frame. A room at 30° to the
rest of the house would have its walls pulled by up to 10°, until the snap window rejects it.

## 4. Error budget

1-sigma per tier, from `uncertainty.py`. These are priors; `floorplan calibrate` scales them to the benchmark (section 5).

| Source | LiDAR | Video | Photos |
|---|---|---|---|
| Scale (fraction of length) | 0.2% | 6% | 10% |
| Each wall face (fit, grid, noise) | 5 mm | 25 mm | 40 mm |
| Each jamb of an opening | 8 mm | 35 mm | 50 mm |
| Ceiling (fraction of height) | 0.3% | 4% | 6% |
| Unseen part of a wall | +25% of the unseen length, per end | same | same |
| With a printed marker | n/a | scale 0.6% | scale 0.6% |

For a 4 m wall the 90% half-widths come out as 1.8 cm, 40 cm and 66 cm (7 cm and 10 cm with a
marker). The photo and video priors started at 4% and 2.5% scale. On real furnished rooms outside
the benchmark those intervals held the truth well under 90% of the time, so they were widened
2.5x and about 2x. The photo and video scale terms dominate everything else, so the useful work on those
tiers is on scale.

## 5. Calibration analysis

A 90% interval has to contain the truth about 90% of the time. Too narrow is the expensive
failure: a confident wrong number on thin input.

**Method.** `floorplan calibrate benchmark/out` takes every measurement that has ground truth
(wall, room area, opening width, ceiling height) and expresses its error in units of the
interval's half-width under the prior. Per tier and quantity, the factor is the 90% quantile of
those ratios with the split-conformal finite-sample correction (the ceil((n+1)·0.9)-th smallest of
n). Scaling the prior's sigma by that factor puts at least 90% of the benchmark's truth inside.
`--write` saves the factors to `src/floorplan/calibration.json`, which every later run applies, and
each `plan.json` records the factors it was made with (`interval_calibration`), so a refit divides
them out instead of compounding. A tier is only narrowed with 20 or more measurements; with fewer
it can only widen.

**Status.** Built and tested (`test_calibrate_widens_intervals_until_90pct_of_truth_is_inside`).
The factors need tape-measured captures, so until the benchmark is in `benchmark/raw/` every tier
runs at its prior (factor 1, no `calibration.json`). On real photo sets held out from development,
a factor fitted on one set of properties carried over to another set, raising its coverage
substantially without reaching 90%: the photo tier's errors have long tails (rooms not
photographed all round), which a single factor can only cover by widening every interval.

## 6. Fix loop

The full declaration and result are in `FIX_LOOP.md`. In short:

- **Worst gate:** the stitched plan from the sample walkthroughs was not connected. Only 59.5 to
  80.8% of the walk fell inside any room, no two rooms were adjacent, and the hallway was missing
  from every plan.
- **Hypothesis, declared before the fix:** the doorway-closing kernels bridge across a hallway
  (two parallel walls about a door's width apart) and fill it as wall.
- **Attempt 1** (`6c8f79a`) shipped the declared change. The prediction was wrong (59.5 to 59.8%):
  the closing was indeed what filled the hallway, but through clutter blobs rather than clean walls.
- **Attempt 2** (`595d4da`): free, walked space inside the home that no room covers becomes a room,
  and a door links to the room whose outline is within 0.5 m. Inside share 72.9/80.8/59.5% to
  92.6/96.6/86.2%, adjacent pairs 0 to 2/5/1. Meaningful movement, short of the gate (95% and
  one component); the report says why (a hallway split by clutter, ragged walked-space outlines).
- Nothing that was right got worse: all rooms in the three sample plans and all 14 synthetic cases
  are unchanged.

## 7. Known failure modes

| Condition | What happens | What we do |
|---|---|---|
| Mirrors | the model or LiDAR sees a room behind the wall | protocol: don't aim at mirrors. LiDAR: confidence filter. Detection of mirror rooms: not built |
| Glass (windows, shower screens) | LiDAR passes through, the model guesses | a gap with wall above and below becomes a window, which is correct |
| Wet-look or glossy floors | reflections give phantom depth below the floor | the floor comes from the densest height bin, so reflections don't move it. Damage detector may flag them |
| Low light | blur, noisy LiDAR confidence | protocol: all lights on. Keyframes pick the sharpest frame per window |
| Plain white walls | fine for the model and LiDAR (the old SfM path failed here) | none needed |
| Open plan or archways over 1.4 m | two rooms merge into one | limitation, reported as one room |
| Mixed portrait and landscape photos | the model crops all photos to one shape | warning in the output, protocol says landscape |
| A corner nobody saw | the room doesn't close | diagonal gap closing as a fallback, the edge refit restores the corner |
| A room photographed from one side (two or three walls in view) | nothing encloses | the rectangle the seen walls span, flagged `unclosed` in the output, unseen walls get low coverage and wide intervals |
| One short shot of a small room (a bathroom) | the seen walls lie along one line | an outline smaller than 1.5 m² or narrower than 0.5 m fails with a reason instead of a 0 m² room |
| An edited video (cuts between rooms) | keyframes from different rooms are posed as one walk; rooms come out as fragments | protocol: one continuous take. Not detected yet: cutting at scene changes and treating each shot as a separate capture is the next step |
| A hallway narrowed by furniture | the walked hallway comes back in pieces, its outline not snapped to the walls | adjacency still links rooms through doors; merging pieces is the next fix (FIX_LOOP.md) |
| Stray Scanner's `rgb.mp4` run through the video tier | frames are stored in sensor orientation (sideways when the phone is upright) and the camera points at the floor; the model's trajectory does not hold together and no room is found | use the LiDAR tier for Stray recordings; the video tier expects the Camera app's video, which carries its rotation |
| A bed or sofa hiding most of the floor | the largest horizontal layer is the furniture top, not the floor | floor = the lowest layer at least two views agree on |
