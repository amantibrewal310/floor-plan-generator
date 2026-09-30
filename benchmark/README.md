# Benchmark

The case study provides no captures, so we build the benchmark ourselves. Its composition is fixed
by the spec so it can't be flattered.

## What to capture

| # | Capture | Tiers | Why |
|---|---|---|---|
| 1 | One multi-room walk: 3+ rooms plus the hallway that connects them | photos, video, LiDAR | stitching, adjacency, drift |
| 2 | One furnished room with staged damage of two classes (e.g. a water stain and a crack) | photos, video, LiDAR | damage, flags, scope |
| 3 | One room of #1 captured a second time at the same tier | at least one tier | repeatability gate |
| 4 | Two rooms of #1, also scanned with a consumer app (free tier) | LiDAR | head-to-head |

Follow `CAPTURE.md` literally for every capture, the same way an evaluator would.

## Ground truth

Measure every room with a laser measurer (or tape) and write `truth.json` next to the captures.
Measure each wall at about 1 m height, wall face to wall face, going round the room in either
direction from any corner.

```json
{"rooms": [
  {"name": "kitchen",
   "walls_m": [3.624, 2.911, 3.618, 2.905],
   "ceiling_m": 2.583,
   "area_m2": 10.52,
   "openings": [{"type": "door", "width_m": 0.823}, {"type": "window", "width_m": 1.214}]}
]}
```

`name` must match the photo folder name. `area_m2` is optional for rectangular rooms. Door width is
the clear opening between the frame's inside faces. For a window, measure the glass opening in the
wall, reveal to reveal.

For the head-to-head, write the app's numbers from its export in the same format, as
`<app>.json`, and keep the export file itself next to it.

## Sample walkthroughs

Three LiDAR walkthroughs recorded with Stray Scanner, with no tape measurements:

| folder | frames | size | walk |
|---|---|---|---|
| `c00a170fe1` | 1715 | 93 MB | 14 m, 2 to 3 rooms |
| `1a8384c3f6` | 5251 | 290 MB | 53 m |
| `c7d28f72c6` | 9745 | 490 MB | 98 m |

They live in `raw/samples/`. Without ground truth they can't score wall lengths, but
`floorplan walkcheck` scores them on what needs no tape: how much of the walk falls inside a room,
and whether the rooms link into one plan. The fix loop (`FIX_LOOP.md`) runs on them.

```
uv run floorplan walkcheck benchmark/raw/samples/c00a170fe1 benchmark/raw/samples/1a8384c3f6 benchmark/raw/samples/c7d28f72c6
```

## Layout

```
benchmark/
  manifest.json          # the list of captures (committed)
  raw/                   # captures, truth.json and app exports (downloaded, not in git)
    samples/c00a170fe1/   # the three sample walkthroughs
    flat/truth.json
    flat/photos/kitchen/IMG_0001.HEIC ...
    flat/video/IMG_1234.MOV
    flat/lidar/71de12f9/ ...
  out/                   # written by the run (not in git)
```

## Running it

```
uv run floorplan bench benchmark/manifest.json
```

This runs every capture, scores it, and writes `benchmark/out/report.md` with the gate table per
tier, repeatability, the drift ablation (footprint with correction on and off), the head-to-head
table and timing. See `src/floorplan/bench.py` for the manifest format.

Model outputs are cached per input set in `~/.cache/floorplan`, keyed by the file contents. A rerun
replays them exactly. Delete the cache to force the live path.
