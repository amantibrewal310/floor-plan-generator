"""End-to-end: captures (any tier) -> stitched, dimensioned floor plan on disk."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from . import export, lidar, plan, sfm
from .stitch import Capture, canonicalize, stitch


def lidar_capture(path: Path, up="auto") -> Capture:
    path = Path(path)
    if path.suffix.lower() == ".json":
        pts = lidar.roomplan_points(path)
    else:
        pts = lidar.load_points(path, up)
    return Capture(path.name, plan.extract_rooms(pts))


def _sfm_capture(name, images: Path, work: Path, marker_size, sequential, focal=None) -> Capture:
    rec = sfm.reconstruct(images, work, sequential=sequential, focal_px=focal)
    mc = sfm.metric_cloud(rec, images, marker_size)
    rooms = plan.extract_rooms(mc.points, floor_z=0.0)
    cap = Capture(name, rooms, mc.markers)
    cap.stats = {"images": mc.n_images, "registered": mc.n_registered, "points": len(mc.points),
                 "markers": sorted(mc.markers), "marker_scale_spread": round(mc.scale_spread, 5)}
    if len(mc.markers) < 2:
        cap.stats["warning"] = ("scale rests on a single marker; for cm accuracy capture 2+ markers "
                                "up close (marker >= 60 px wide in frame)")
    return cap


def photo_capture(folder: Path, marker_size: float, work: Path) -> Capture:
    images = work / "images"
    focal = sfm.prepare_photos(Path(folder), images)
    return _sfm_capture(Path(folder).name, images, work, marker_size, sequential=False, focal=focal)


def video_capture(video: Path, marker_size: float, work: Path, fps=3.0) -> Capture:
    images = work / "images"
    sfm.extract_keyframes(Path(video), images, fps=fps)
    return _sfm_capture(Path(video).name, images, work, marker_size, sequential=True)


def run(tier: str, inputs: list[Path], out: Path, marker_size=0.18, wall_thickness=0.12,
        up="auto", fps=3.0, work: Path | None = None) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    captures = []
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(work) if work else Path(tmp)
        for i, src in enumerate(inputs):
            w = base / f"capture_{i}"
            if tier == "lidar":
                captures.append(lidar_capture(src, up))
            elif tier == "photos":
                captures.append(photo_capture(src, marker_size, w))
            elif tier == "video":
                captures.append(video_capture(src, marker_size, w, fps))
            else:
                raise ValueError(f"unknown tier {tier!r}")
    rooms, log = stitch(captures, wall_thickness)
    rooms = canonicalize(rooms)
    for k, r in enumerate(sorted(rooms, key=lambda r: -r.area)):
        r.name = f"Room {k + 1}"
    meta = {"tier": tier, "inputs": [str(p) for p in inputs], "stitching": log,
            "captures": [getattr(c, "stats", {"source": c.source}) for c in captures]}
    export.write_json(rooms, out / "plan.json", meta)
    export.write_svg(rooms, out / "plan.svg", title=f"Floor plan ({tier})")
    return export.to_dict(rooms, meta)
