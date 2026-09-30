# Compliance matrix

Status key: **done** means built and covered by a test. **Built, needs real data** means the code
runs but the number the spec asks for only exists once real captures are in `benchmark/raw/`.
**Not started** is not started.

| Requirement | File | Artifact | Status |
|---|---|---|---|
| Capture route (Route 2: stock apps) | `CAPTURE.md` | one-page protocol, all three tiers | done |
| Device matrix | `DEVICES.md` | tier per iPhone, gates, interval widths | done; measured column pending |
| Photo tier: 2 to 8 stills per room, no depth, no poses | `src/floorplan/recon.py`, `pipeline.py` | `floorplan photos <folders>` | done (`test_photos_six_stills_with_marker`) |
| Photo tier: per-room folders stitch into one plan | `stitch.py` | doorway matching, overlap penalty | done on synthetic data; real stitch pending |
| Video tier: handheld walkthrough | `media.py`, `recon.py` | `floorplan video <clip>` | done; HEVC .MOV decode checked |
| LiDAR tier: depth, poses, intrinsics | `lidar.py` (`load_stray`) | `floorplan lidar <stray folder>` | done (`test_stray_lidar_drift_correction`) |
| Per-room plan: walls, ceiling height, floor area, openings | `plan.py`, `export.py` | `plan.json` rooms[] | done |
| Windows as openings | `plan.py` (`_find_openings`) | openings[].type = window | done; no synthetic windows yet |
| Stitched multi-room plan with adjacency | `stitch.py`, `export.adjacency` | `plan.json` adjacency[], `plan.svg` | done |
| Damage regions per surface, class and metric extent | `damage.py` | damage[] | built, needs real data (`test_damage_lands_on_surface...`) |
| Concealed-damage flags with the rule that fired | `damage.py` (`RULES`) | concealed_flags[] | done |
| Scope line items keyed to surfaces | `damage.py` (`SCOPE`) | scope[] | done |
| Confidence interval on every measurement | `uncertainty.py` | every number is {value, lo, hi} | done; calibration needs real data |
| One command per capture | `cli.py` | `floorplan <tier> ... -o out` | done |
| JSON to a published schema | `schema/plan.schema.json` | validated in `test_output_matches_published_schema` | done |
| Rendered plan | `export.write_svg` | `plan.svg` with ± on dimensions, windows, damage marks | done |
| Drift handling, with ablation | `drift.py`, `--no-drift-correction` | bench report "Drift ablation" | done on synthetic drift; real ablation pending |
| Opening width gate (2 cm on 85%, missed/phantom count) | `evaluate.py` | `floorplan score` | built, needs real data |
| Ceiling gate and bias vs spread | `evaluate.py` | ceiling_mae_cm, ceiling_bias_cm | built, needs real data |
| Repeatability gate | `evaluate.py` (`repeatability`) | `floorplan repeat`, bench table | built, needs real data |
| Photo-tier whole-property stitch, footprint within 8% | `evaluate.py` | footprint in bench report | built, needs real data |
| Benchmark set (multi-room, damage room, all tiers, repeat) | `benchmark/README.md`, `manifest.json` | capture list | needs captures |
| Ground truth for everything | `benchmark/raw/*/truth.json` | tape/laser measurements | needs captures |
| Head-to-head vs a consumer app, 70% beat-or-tie | `bench.py` (`head_to_head`) | bench report table | built, needs captures |
| Benchmark report with timing | `bench.py` | `benchmark/out/report.md` | built, needs captures |
| Reproduction bundle, deterministic cache plus live path | `recon.predict` cache, `floorplan bench` | `~/.cache/floorplan/*.npz` | done |
| Weights fetched by script | `floorplan setup` | about 5 GB into ~/.cache | done |
| Fix loop: declaration, before/after, diff | `FIX_LOOP.md` | | not started (needs the real benchmark's worst gate) |
| Technical report, max 6 pages | `REPORT.md` | | draft; results sections need real data |
| Raw benchmark data | `benchmark/raw/` download | | needs captures |
| Mirrors, glass, wet-look surfaces, low light | `CAPTURE.md`, `lidar.load_stray` confidence filter, `REPORT.md` | | protocol and LiDAR filter done; measured behaviour pending |
| Process evidence | git history | | ongoing |
