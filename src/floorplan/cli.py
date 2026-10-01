"""Command line interface: `floorplan <tier> INPUT... -o OUT`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser(prog="floorplan", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_common(p):
        p.add_argument("inputs", nargs="+", type=Path,
                       help="one entry per capture; several captures are stitched together")
        p.add_argument("-o", "--out", type=Path, default=Path("out"))
        p.add_argument("--wall-thickness", type=float, default=0.12,
                       help="assumed interior wall thickness (m) when stitching by doorway")
        p.add_argument("--no-drift-correction", action="store_true",
                       help="use logged/predicted poses as-is (ablation)")
        p.add_argument("--no-damage", action="store_true", help="skip damage detection (faster)")

    p = sub.add_parser("lidar", help="Stray Scanner export folders, PLY/OBJ/GLB scans or RoomPlan JSON")
    add_common(p)
    p.add_argument("--up", default="auto", help="up axis of the scan: auto, +y, +z, ...")

    for name, hlp in (("photos", "photo folders, one folder per room (2-8 photos each)"),
                      ("video", "walkthrough videos (one file per capture)")):
        p = sub.add_parser(name, help=hlp)
        add_common(p)
        p.add_argument("--marker-size", type=float, default=None,
                       help="optional: black-square size (m) of printed ArUco markers in view; "
                            "overrides the model's scale")
        if name == "video":
            p.add_argument("--fps", type=float, default=2.0, help="keyframes per second")
            p.add_argument("--work", type=Path, help="keep extracted keyframes here")

    p = sub.add_parser("score", help="score a plan.json against tape/laser ground truth")
    p.add_argument("plan", type=Path)
    p.add_argument("truth", type=Path)
    p.add_argument("-o", "--out", type=Path, help="write the full scoring JSON here")

    p = sub.add_parser("repeat", help="repeatability: compare two plan.json of the same room(s)")
    p.add_argument("a", type=Path)
    p.add_argument("b", type=Path)

    p = sub.add_parser("walkcheck", help="truth-free check of walkthroughs: share of the walk inside a "
                                         "room, and whether the rooms link into one plan")
    p.add_argument("inputs", nargs="+", type=Path, help="Stray Scanner folders (lidar) or videos (--tier video)")
    p.add_argument("--tier", choices=["lidar", "video"], default="lidar")
    p.add_argument("--no-drift-correction", action="store_true")

    p = sub.add_parser("bench", help="run a real-capture benchmark manifest and write the report")
    p.add_argument("manifest", type=Path)
    p.add_argument("-o", "--out", type=Path)
    p.add_argument("--no-damage", action="store_true")

    p = sub.add_parser("calibrate", help="fit the 90%% intervals to a benchmark's ground truth")
    p.add_argument("results", nargs="+", type=Path, help="`floorplan bench` output folders or results.json")
    p.add_argument("--write", action="store_true", help="save the factors to calibration.json (used by every run)")

    sub.add_parser("setup", help="download the model weights (about 5 GB, once)")

    p = sub.add_parser("markers", help="write printable ArUco markers (A4, 300 dpi)")
    p.add_argument("-o", "--out", type=Path, default=Path("markers"))
    p.add_argument("--size-mm", type=int, default=180)
    p.add_argument("--count", type=int, default=6)

    p = sub.add_parser("synth", help="generate a synthetic capture with ground truth")
    p.add_argument("scene", choices=["rect", "lshape", "apartment", "hall"])
    p.add_argument("tier", choices=["photos", "video", "lidar", "roomplan", "stray"])
    p.add_argument("-o", "--out", type=Path, default=Path("data"))

    p = sub.add_parser("benchmark", help="run every tier on synthetic scenes and report errors")
    p.add_argument("--quick", action="store_true", help="LiDAR/RoomPlan cases only")
    p.add_argument("--case", help="only cases whose name contains this, e.g. video/")
    p.add_argument("-o", "--out", type=Path, default=Path("benchmark"))

    a = ap.parse_args(argv)

    if a.cmd in ("lidar", "photos", "video"):
        from .pipeline import run
        missing = [str(p) for p in a.inputs if not p.exists()]
        if missing:
            ap.error(f"input not found: {', '.join(missing)} "
                     f"(no sample data ships with the project; try `floorplan synth` to make some)")
        kw = {"wall_thickness": a.wall_thickness, "drift_correction": not a.no_drift_correction,
              "find_damage": not a.no_damage}
        if a.cmd == "lidar":
            kw["up"] = a.up
        else:
            kw["marker_size"] = a.marker_size
            if a.cmd == "video":
                kw.update(fps=a.fps, work=a.work)
        try:
            result = run(a.cmd, a.inputs, a.out, **kw)
        except (ValueError, RuntimeError) as e:  # capture problems: explain, don't dump a traceback
            print(f"error: {e}", file=sys.stderr)
            return 1
        _summary(result)
        print(f"\nwrote {a.out / 'plan.svg'} and {a.out / 'plan.json'}")
    elif a.cmd == "score":
        from .evaluate import load, score, to_json
        res = score(load(a.plan), load(a.truth))
        if a.out:
            a.out.write_text(to_json(res))
        print(to_json(res["summary"]))
    elif a.cmd == "repeat":
        from .evaluate import load, repeatability
        res = repeatability(load(a.a), load(a.b))
        for r in res["rows"]:
            print(f"{r['room']:>12} {r['wall']:>8}: {r['a'] * 100:7.1f} vs {r['b'] * 100:7.1f} cm  "
                  f"diff {r['diff_cm']:.2f} cm  {'pass' if r['pass'] else 'FAIL'}")
        print("repeatability gate:", "PASS" if res["pass"] else "FAIL")
    elif a.cmd == "walkcheck":
        from .evaluate import walk_check
        from .pipeline import lidar_capture, video_capture
        print("capture                 rooms  walk m  inside %  adjacent pairs  components")
        for src in a.inputs:
            cap = (lidar_capture(src, drift_correction=not a.no_drift_correction, find_damage=False)
                   if a.tier == "lidar" else
                   video_capture(src, drift_correction=not a.no_drift_correction, find_damage=False))
            w = walk_check(cap)
            print(f"{w['capture']:22s}  {w['rooms']:5d}  {w['walk_m']:6.1f}  {w['walk_inside_pct']:8.1f}  "
                  f"{w['adjacent_pairs']:14d}  {w['components']:10d}")
    elif a.cmd == "bench":
        from .bench import run_bench
        run_bench(a.manifest, a.out, damage=not a.no_damage)
    elif a.cmd == "calibrate":
        from .calibrate import calibrate
        calibrate(a.results, write=a.write)
    elif a.cmd == "setup":
        from . import damage, recon
        print(f"fetching {recon.MODEL_ID} (3D reconstruction) ...", flush=True)
        recon.load_model()
        print(f"fetching {damage.DETECTOR_ID} (damage detection) ...", flush=True)
        damage.load_detector()
        print("done: weights are cached under ~/.cache (huggingface, torch hub)")
    elif a.cmd == "markers":
        from .media import write_marker_sheet
        write_marker_sheet(a.out, range(a.count), a.size_mm)
        print(f"wrote {a.count} markers to {a.out}/ (use --marker-size {a.size_mm / 1000})")
    elif a.cmd == "synth":
        from . import synth
        sc = synth.make_scene(a.scene)
        d = a.out / f"{a.scene}_{a.tier}"
        d.mkdir(parents=True, exist_ok=True)
        if a.tier == "photos":
            paths = [synth.write_photos(sc, d / f"capture_{i}", i) for i in range(len(sc.walk))]
        elif a.tier == "video":
            paths = [synth.write_video(sc, d / "walkthrough.mp4")]
        elif a.tier == "lidar":
            paths = [synth.write_lidar_ply(sc, d / f"scan_{i}.ply", rooms=[i], seed=i)
                     for i in range(len(sc.rooms))]
        elif a.tier == "stray":
            paths = [synth.write_stray(sc, d / "stray")]
        else:
            paths = [synth.write_roomplan_json(sc, d / "room.json")]
        (d / "ground_truth.json").write_text(json.dumps(synth.tape_truth(sc), indent=2))
        print("\n".join(str(p) for p in paths))
        if a.tier in ("photos", "video"):
            print(f"marker size: {sc.marker_size} m")
    elif a.cmd == "benchmark":
        from .benchmark import run_benchmark
        run_benchmark(a.out, quick=a.quick, case=a.case)


def _summary(result):
    def iv(m, unit="m", k=100):
        return f"{m['value'] * k:.1f} [{m['lo'] * k:.1f}, {m['hi'] * k:.1f}]"
    for r in result["rooms"]:
        h = r["ceiling_height_m"]
        print(f"{r['id']} {r['name']}: {r['floor_area_m2']['value']:.2f} m², "
              f"ceiling {'-' if h is None else iv(h) + ' cm'}")
        print("  walls (cm, 90% interval): " + ", ".join(iv(w["length_m"]) for w in r["walls"]))
        for o in r["openings"]:
            print(f"  {o['type']} on {o['wall']}: {iv(o['width_m'])} cm")
    for d in result["damage"]:
        print(f"damage {d['id']}: {d['class']} on {d['surface']}, "
              f"{d['width_m']['value'] * 100:.0f} x {d['height_m']['value'] * 100:.0f} cm (score {d['score']:.2f})")
    for f in result["concealed_flags"]:
        print(f"concealed {f['id']} [{f['rule']}] on {f['surface']}: {f['reason']}")
    for sc in result["scope"]:
        q = sc["quantity"]
        print(f"scope {sc['id']} {sc['surface']}: {sc['item']} - {q['value']} {sc['unit']} [{q['lo']}, {q['hi']}]")
    for a in result["adjacency"]:
        print(f"adjacent: {' <-> '.join(a['rooms'])}")
    for line in result["stitching"]:
        print("stitch:", line)
    for c in result["captures"]:
        if "warning" in c:
            print("warning:", c["warning"])


if __name__ == "__main__":
    sys.exit(main())
