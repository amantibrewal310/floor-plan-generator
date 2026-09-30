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

    p = sub.add_parser("markers", help="write printable ArUco markers (A4, 300 dpi)")
    p.add_argument("-o", "--out", type=Path, default=Path("markers"))
    p.add_argument("--size-mm", type=int, default=180)
    p.add_argument("--count", type=int, default=6)

    p = sub.add_parser("synth", help="generate a synthetic capture with ground truth")
    p.add_argument("scene", choices=["rect", "lshape", "apartment"])
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
        kw = {"wall_thickness": a.wall_thickness, "drift_correction": not a.no_drift_correction}
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
        (d / "ground_truth.json").write_text(json.dumps(sc.ground_truth(), indent=2))
        print("\n".join(str(p) for p in paths))
        if a.tier in ("photos", "video"):
            print(f"marker size: {sc.marker_size} m")
    elif a.cmd == "benchmark":
        from .benchmark import run_benchmark
        run_benchmark(a.out, quick=a.quick, case=a.case)


def _summary(result):
    for r in result["rooms"]:
        h = f", height {r['height_m']:.2f} m" if r["height_m"] else ""
        print(f"{r['name']}: {r['area_m2']:.2f} m²{h}")
        print("  walls (cm): " + ", ".join(f"{w['length_m'] * 100:.1f}" for w in r["walls"]))
        for d in r["doors"]:
            print(f"  door on wall {d['wall']}: {d['width_m'] * 100:.1f} cm")
    for line in result["stitching"]:
        print("stitch:", line)
    for c in result["captures"]:
        if "warning" in c:
            print("warning:", c["warning"])


if __name__ == "__main__":
    sys.exit(main())
