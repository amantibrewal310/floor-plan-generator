"""End-to-end: captures (any tier) -> stitched, dimensioned floor plan on disk."""

from __future__ import annotations

import tempfile
from pathlib import Path

from . import export, lidar, media, plan, recon
from .stitch import Capture, canonicalize, stitch


def lidar_capture(path: Path, up="auto") -> Capture:
    path = Path(path)
    if path.suffix.lower() == ".json":
        pts = lidar.roomplan_points(path)
    else:
        pts = lidar.load_points(path, up)
    return Capture(path.name, plan.extract_rooms(pts))


def image_capture(name: str, images: list[Path], marker_size: float | None, one_room: bool) -> Capture:
    """Unposed images -> rooms. `one_room`: the images are one room's photo folder."""
    pred = recon.predict(images)
    R = recon.gravity_rotation(pred)
    scale, how = 1.0, "model (metric depth)"
    if marker_size:
        m = recon.marker_scale(pred, marker_size)
        if m:
            scale, how = m[0], f"ArUco marker ({m[1]} sightings)"
    P = recon.metric_points(pred, R) * scale
    cams = pred.poses[:, :3, 3] @ R.T * scale
    rooms = plan.extract_rooms(P, seeds=cams)
    if one_room:
        rooms = [max(rooms, key=lambda r: r.area)]
        rooms[0].name = name
    cap = Capture(name, rooms)
    cap.stats = {"source": name, "images": len(images), "scale_from": how, "scale": round(scale, 4)}
    return cap


def photo_capture(folder: Path, marker_size=None) -> Capture:
    return image_capture(Path(folder).name, recon.list_images(folder), marker_size, one_room=True)


def video_capture(video: Path, marker_size=None, fps=2.0, work: Path | None = None) -> Capture:
    with tempfile.TemporaryDirectory() as tmp:
        frames = Path(work or tmp) / "frames"
        media.extract_keyframes(Path(video), frames, fps=fps)
        return image_capture(Path(video).name, recon.list_images(frames), marker_size, one_room=False)


def run(tier: str, inputs: list[Path], out: Path, marker_size=None, wall_thickness=0.12,
        up="auto", fps=2.0, work: Path | None = None) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    captures = []
    for i, src in enumerate(inputs):
        if tier == "lidar":
            captures.append(lidar_capture(src, up))
        elif tier == "photos":
            captures.append(photo_capture(src, marker_size))
        elif tier == "video":
            captures.append(video_capture(src, marker_size, fps, work and Path(work) / f"capture_{i}"))
        else:
            raise ValueError(f"unknown tier {tier!r}")
    rooms, log = stitch(captures, wall_thickness)
    rooms = canonicalize(rooms)
    unnamed = [r for r in sorted(rooms, key=lambda r: -r.area) if not r.name or r.name.startswith("Room ")]
    for k, r in enumerate(unnamed):
        r.name = f"Room {k + 1}"
    meta = {"tier": tier, "inputs": [str(p) for p in inputs], "stitching": log,
            "captures": [getattr(c, "stats", {"source": c.source}) for c in captures]}
    export.write_json(rooms, out / "plan.json", meta)
    export.write_svg(rooms, out / "plan.svg", title=f"Floor plan ({tier})")
    return export.to_dict(rooms, meta)
