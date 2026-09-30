# floorplan

Turn phone captures into **dimensioned, stitched floor plans** with centimetre-level accuracy.
There are three input tiers:

| tier | input | how scale / gravity is recovered |
|---|---|---|
| **LiDAR** | PLY/OBJ/GLB scans (Polycam, 3D Scanner App, …) or Apple **RoomPlan** JSON | already metric; the up axis is auto-detected, then levelled with a floor-plane fit |
| **Video** | walkthrough `.mp4`/`.mov` | sharpest keyframe per 1/3 s → COLMAP SfM (sequential matching) → ArUco markers on the floor |
| **Photos** | a folder of photos per room | COLMAP SfM (exhaustive matching, EXIF focal prior) → ArUco markers on the floor |

Output: `plan.svg` (a drawing with every wall length in cm, doors, areas and ceiling heights) and
`plan.json` (vertices, wall lengths, door positions and widths in metres, plus capture statistics).

## Install and run

```bash
uv sync                                    # Python 3.11+, numpy, opencv, pycolmap, trimesh
uv run floorplan lidar scan.ply -o out/    # or: room.json exported from RoomPlan
uv run floorplan video walk.mp4 --marker-size 0.18 -o out/
uv run floorplan photos living/ kitchen/ --marker-size 0.18 -o out/
```

Pass several inputs to stitch captures together (one scan, video or photo folder per capture).

### Try it without a phone

Synthetic scenes come with ground truth:

```bash
uv run floorplan synth apartment lidar -o data          # two separate room scans
uv run floorplan lidar data/apartment_lidar/scan_*.ply -o out/
uv run floorplan synth lshape video -o data             # renders a walkthrough (~4 min)
uv run floorplan video data/lshape_video/walkthrough.mp4 --marker-size 0.2 -o out/
uv run floorplan benchmark                              # every tier vs ground truth (~20 min)
uv run floorplan benchmark --quick                      # LiDAR/RoomPlan only (seconds)
```

## Capturing (photo and video tiers)

1. `uv run floorplan markers -o markers/`, then print at 100 % scale. Measure the black square
   and pass it as `--marker-size` in metres; this measurement sets the scale of the whole plan.
2. Lay 2–3 markers flat on the floor, about 1.5 m from where you will stand. Put one on the threshold
   between rooms that you capture separately. It then stitches them exactly.
3. Walk a loop about 1 m from the walls, pointing the camera across the room and holding it
   roughly level. Keep the upper part of the walls in view, including the wall above each door.
   Every few steps, tilt down over a marker from about 1 m away; each marker should fill 60+ px
   of the frame in several views, because this sets the scale. Video: walk slowly. Photos: take
   one every half step, about 40 per room, and include the markers in a third of them.

## How it works

```
LiDAR/RoomPlan ─┐
Video ─ keyframes ─┐                                          ┌─ stitch (shared marker │ doorway)
Photos ────────────┴─ COLMAP SfM ─ ArUco metric+gravity frame ─┤
                                                              └─► plan.extract_rooms ─► SVG / JSON
```

Each tier produces a metric, gravity-aligned point cloud. `plan.extract_rooms` is shared by all of them:

1. **Floor/ceiling** come from the dense horizontal bins in the height histogram.
2. **Room segmentation.** Points between 1.0 m and the ceiling are rasterised at 2 cm in a grid
   aligned with the dominant wall direction. Furniture sits below this band, and the wall above
   each door is inside it, so rooms close off at doorways. Cells whose points span less than
   20 cm of height are dropped as floating clutter (mis-triangulated SfM points, shelf tops).
   Gaps are closed with small isotropic kernels plus straight line kernels, which bridge
   doorways without rounding corners. A flood fill from outside leaves one component per room.
   The closing setting is whichever encloses the most floor area.
3. **Walls.** Each room contour is simplified, and its edges are snapped to the Manhattan frame.
   The frame's angle is re-estimated from wall points, because a 0.3° error costs 1–2 cm at the
   ends of a 5 m wall. Each edge is then **refit to the raw points**: it locks onto the first
   dense surface outward from the room (the interior face, never the far face of a thin wall),
   takes a robust median, and neighbouring edges are re-intersected. That last step turns a 2 cm grid into mm-level walls.
4. **Doors** are runs of wall with no points between 1 m and 1.9 m, above the furniture, and
   empty lower down too. Jamb positions use a density-based estimator, which is unbiased under
   sensor noise; the outermost point is always pushed into the gap.
5. **Stitching.** A capture that spans several rooms (one walkthrough, one multi-room scan) is
   already in one frame. Separate captures are joined through a **shared ArUco marker** (an
   exact rigid transform) or else through a **matching doorway**: equal widths, opposite faces
   of one wall, offset by `--wall-thickness`, with no room overlap.

Scale comes from the markers. The sub-pixel ArUco corners are biased about 0.4 % inward on
blurred images, which would add about 2 cm over 4 m. So each marker's corners are re-derived by
fitting lines to its four outer edges and intersecting them (`sfm.refine_corners`). Scale is the
known side length divided by the triangulated one. The floor plane and heading come from the
same corners.

## Accuracy (synthetic benchmark)

`uv run floorplan benchmark` renders each scene and runs the full pipeline for each tier. It
then compares the result with ground truth after a rigid alignment:

| case | rooms | walls | wall MAE cm | wall max cm | corner RMSE cm | area err % | doors | door MAE cm |
|---|---|---|---|---|---|---|---|---|
| lidar/rect | 1/1 | 4/4 | 0.03 | 0.04 | 0.02 | 0.00 | 1/1 | 0.92 |
| lidar/apartment (2 scans, door stitch) | 2/2 | 8/8 | 0.04 | 0.05 | 0.23 | 0.01 | 3/3 | 1.19 |
| roomplan/lshape | 1/1 | 6/6 | 0.00 | 0.01 | 0.00 | 0.00 | 1/1 | 0.56 |
| photos/rect | 1/1 | 4/4 | 0.01 | 0.02 | 0.01 | 0.01 | 1/1 | 3.73 |
| photos/apartment (2 captures) | 2/2 | 8/8 | 0.14 | 0.20 | 0.19 | 0.08 | 3/3 | 2.38 |
| video/lshape | 1/1 | 6/6 | 0.10 | 0.16 | 0.09 | 0.06 | 1/1 | 0.94 |
| video/apartment (1 walkthrough) | 2/2 | 8/8 | 0.21 | 0.40 | 0.23 | 0.11 | 3/3 | 1.71 |

The LiDAR cloud has 1 cm Gaussian noise, 0.2 % outliers, furniture and a partial ceiling.
Photos and video are ray-traced textured rooms with JPEG/codec noise. Walls come out
sub-centimetre in every tier. Door widths from photos and video are good to about ±4 cm,
because SfM loses features right at occlusion edges. Real captures will be worse, mainly on
plain, textureless walls (see limitations).

## Tests

```bash
uv run pytest -m "not slow"   # LiDAR, RoomPlan, stitching, marker refinement (~2 s)
uv run pytest                 # + full photo SfM case (~1 min)
```

## Limitations

- **Plain, textureless walls** give sparse SfM points. The photo and video tiers need visible
  texture on the upper walls (pictures, shelves, trim) or they fail with "no enclosed room". A
  learned depth model aligned to the SfM points would fix that; it is not included, to keep
  dependencies light. For blank rooms, use the LiDAR tier.
- Rooms are assumed to have vertical walls and to be mostly Manhattan. Other angles are
  supported, but walls shorter than about 35 cm off the main axes are dropped.
- Openings wider than about 1.4 m with no wall above them (archways, open-plan) merge the
  rooms on either side.
- Doorway-based stitching relies on `--wall-thickness` (default 12 cm); markers make it exact.
- Windows are not detected.
- The RoomPlan parser follows the `CapturedRoom`/`CapturedStructure` Codable JSON layout
  (column-major 4×4 transforms, `dimensions` = width, height, depth).
