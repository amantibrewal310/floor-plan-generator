# Technical report (draft)

Sections 1 to 4 and 7 describe the system as built. Sections 5 and 6 need the real benchmark and
say so. No number in this report comes from anywhere but a run you can repeat.

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

The last row shows the correction costs nothing on a clean capture. The real ablation will be in
`benchmark/out/report.md`.

Known limit: heading snapping assumes the rooms share one right-angled frame. A room at 30° to the
rest of the house would have its walls pulled by up to 10°, until the snap window rejects it.

## 4. Error budget

1-sigma per tier, from `uncertainty.py`. These are priors, and section 6 replaces them.

| Source | LiDAR | Video | Photos |
|---|---|---|---|
| Scale (fraction of length) | 0.2% | 2.5% | 4% |
| Each wall face (fit, grid, noise) | 5 mm | 15 mm | 25 mm |
| Each jamb of an opening | 8 mm | 25 mm | 35 mm |
| Ceiling (fraction of height) | 0.3% | 2.5% | 4% |
| Unseen part of a wall | +25% of the unseen length, per end | same | same |
| With a printed marker | n/a | scale 0.6% | scale 0.6% |

For a 4 m wall the 90% half-widths come out as 1.8 cm, 17 cm and 27 cm (5 cm and 7 cm with a
marker). The photo and video scale terms dominate everything else, so the useful work on those
tiers is on scale.

## 5. Calibration analysis

Pending the real benchmark. `floorplan score` reports interval coverage per capture, the share of
tape measurements inside their 90% interval. The target is 85 to 95%. Below that, the tier's
constants go up. Above it, they come down. On synthetic LiDAR the coverage is 93 to 100%, but
synthetic noise is the noise we chose, so that proves nothing about real rooms.

## 6. Fix loop

Pending: the fix loop has to start from the worst gate on the real benchmark (`FIX_LOOP.md`).
Candidates already visible on synthetic data:

- LiDAR walls come out about 1.2 cm short on the Stray-format synthetic capture while the
  PLY-format capture is unbiased. Openings measure within 2 cm on only 67% there. Both point at
  the depth back-projection or edge refit on sparse 256x192 depth, not at the pose.
- Photo-tier scale. On 6 synthetic renders (random confetti textures, nothing like a real room)
  the model's scale was 39% too large without a marker. With the marker the walls came out 3 to 7%
  long. Real photos should be much closer, but that is exactly what the benchmark has to show.

Synthetic video (`floorplan synth apartment video`, 61 keyframes, printed marker in view, 3 min 45 s
on an M5, 58 s on a cached rerun): both rooms found, doors and adjacency right, drift correction
applied 3.3° of heading. But room 1 came out 4.19 x 3.92 m against 4.00 x 3.60 m (+5% and +9%, so
not a single scale error) and room 2 gained a spurious jog. The ceiling was never in view. The
renders are random confetti textures that look nothing like a room, so this says little about
real captures. It is recorded here because it is the only video number we have so far. The same
run exposed the damage detector firing on 14 texture patches. A region seen in one view now needs
a score of 0.6, and 0.4 otherwise, which cut that to one.

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
