"""End-to-end: captures (any tier) -> stitched, dimensioned floor plan on disk."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from . import damage, drift, export, lidar, media, plan, recon
from .stitch import Capture, canonicalize, stitch


def lidar_capture(path: Path, up="auto", drift_correction=True, find_damage=True) -> Capture:
    path = Path(path)
    if path.is_dir():  # Stray Scanner export: raw depth + poses + intrinsics
        frames = lidar.load_stray(path)
        # damage: every n-th used frame, about 40 images
        pick = list(range(0, len(frames), max(1, len(frames) // 40)))
        images = (lambda: lidar.stray_rgb(path, [frames[i][3] for i in pick])) if find_damage else None
        return _frames_capture(path.name, [f[:3] for f in frames], drift_correction, level=True,
                               images=images, image_views=pick, view_names=[f"frame {frames[i][3]}" for i in pick])
    if path.suffix.lower() == ".json":
        pts = lidar.roomplan_points(path)
    else:
        pts = lidar.load_points(path, up)
    return Capture(path.name, plan.extract_rooms(pts))


def _frames_capture(name, frames, drift_correction, level=False, one_room=False,
                    images=None, image_views=(), view_names=()) -> Capture:
    """Per-frame (points z up, camera, pixel uv) in capture order -> rooms: correct accumulated
    drift, extract rooms, then place damage found in `images()` (RGB of `image_views`)."""
    # photo/video frames carry which way each point's surface faces (recon.view_points): the floor
    # is the lowest big layer of surfaces seen from above. A learned point cloud of a furnished
    # room is too spread out for the height histogram alone, which picks the bed tops.
    facing = len(frames[0]) > 3
    floor = None
    if facing:
        up = np.concatenate([f[0][f[3] == 1, 2] for f in frames])
        floor = plan.layer(up, lowest=True)
    frames, log = drift.correct(frames, enabled=drift_correction, floor=floor)
    P = np.concatenate([f[0] for f in frames])
    cams = np.array([f[1] for f in frames])
    R = np.eye(3)
    if level:
        P, R = lidar.level(P)
        cams = cams @ R.T
    ceiling_pts = None
    if facing:
        lab = np.concatenate([f[3] for f in frames])
        ceiling_pts = P[lab == 2]
    if floor is None:
        floor, _ = plan.estimate_floor_ceiling(P[:, 2])
    rooms = plan.extract_rooms(P, floor_z=floor, seeds=cams, open_fallback=one_room, walked=not one_room,
                               ceiling_pts=ceiling_pts)
    if one_room:
        rooms = [max(rooms, key=lambda r: r.area)]
        rooms[0].name = name
    cap = Capture(name, rooms, path=cams)
    cap.stats = {"source": name, "frames": len(frames), "drift_correction": drift_correction}
    if any(r.extra.get("unclosed") for r in rooms):
        cap.stats["warning"] = (f"{name}: the photos do not show every wall, so the room is the rectangle "
                                "the seen walls span; unseen sides are estimated (wide intervals)")
    if log:
        cap.stats["drift"] = {"chunks": len(log),
                              "final_yaw_correction_deg": log[-1]["yaw_deg"],
                              "max_chunk_shift_m": max(float(np.abs(c["shift_m"]).max()) for c in log)}
    if images is not None:
        shift = np.array([0, 0, floor])
        views = [(frames[i][0] @ R.T - shift, None, frames[i][2]) for i in image_views]
        dets = damage.detect(images())
        damage.locate(dets, views, rooms, list(view_names))
        cap.stats["damage_views"] = len(dets)
    return cap


def image_capture(name: str, images: list[Path], marker_size: float | None, one_room: bool,
                  drift_correction=True, find_damage=True) -> Capture:
    """Unposed images -> rooms. `one_room`: the images are one room's photo folder."""
    pred = recon.predict(images)
    R = recon.gravity_rotation(pred)
    scale, how = 1.0, "model (metric depth)"
    if marker_size:
        m = recon.marker_scale(pred, marker_size)
        if m:
            scale, how = m[0], f"ArUco marker ({m[1]} sightings)"
    frames, closeups = [], []
    for v in range(len(images)):
        X, uv, facing = recon.view_points(pred, R, v)
        cam = pred.poses[v, :3, 3] @ R.T * scale
        X = X * scale
        if len(X) and np.median(np.linalg.norm(X - cam, axis=1)) < CLOSEUP_M:
            # a view filled by one surface a few tens of cm away gives the model little to place
            # it by, and its walls land in the wrong spot; it adds nothing to the layout anyway
            closeups.append(Path(images[v]).name)
            X, uv, facing = X[:0], uv[:0], facing[:0]
        frames.append((X, cam, uv, facing))
    # a photo folder is a handful of unordered stills: there is no trajectory to correct
    cap = _frames_capture(name, frames, drift_correction and not one_room, one_room=one_room,
                          # the model's own (cropped, resized) images: their pixels are the points' uv
                          images=(lambda: list(pred.img)) if find_damage else None,
                          image_views=range(len(images)), view_names=[Path(p).name for p in images])
    cap.stats.update({"images": len(images), "scale_from": how, "scale": round(scale, 4),
                      "closeups_ignored": len(closeups)})
    advice = capture_advice(cap, len(images), len(closeups))
    if advice:
        cap.stats["warning"] = "; ".join(filter(None, [cap.stats.get("warning"), advice]))
    if len({recon.is_portrait(p) for p in images}) > 1:
        mixed = (f"{name}: portrait and landscape photos mixed; the model crops every image "
                 "to one shape, so shoot a room all in landscape")
        cap.stats["warning"] = "; ".join(filter(None, [cap.stats.get("warning"), mixed]))
    return cap


CLOSEUP_M = 1.0  # median depth of a view below this: a close-up


def capture_advice(cap, n_views, n_closeups) -> str:
    """What to do differently next time, from what the capture shows."""
    tips = []
    if n_closeups > 0.25 * n_views:
        tips.append(f"{n_closeups} of {n_views} views were close-ups (under {CLOSEUP_M:.0f} m), so they were "
                    "ignored: aim across the room at the far wall")
    if cap.rooms and all(r.height is None for r in cap.rooms):
        tips.append("the ceiling was never in view: keep the line where wall meets ceiling in the frame")
    weak = [r for r in cap.rooms if r.wall_support and
            np.mean([c for c, _ in r.wall_support]) < 0.5]
    if weak:
        tips.append(f"{len(weak)} room(s) had less than half their walls filmed: walk all the way round, "
                    "1 m from the walls")
    return "; ".join(tips)


def photo_capture(folder: Path, marker_size=None, find_damage=True) -> Capture:
    return image_capture(Path(folder).name, recon.list_images(folder), marker_size, one_room=True,
                         find_damage=find_damage)


def video_capture(video: Path, marker_size=None, fps=2.0, work: Path | None = None,
                  drift_correction=True, find_damage=True) -> Capture:
    with tempfile.TemporaryDirectory() as tmp:
        frames = Path(work or tmp) / "frames"
        media.extract_keyframes(Path(video), frames, fps=fps)
        return image_capture(Path(video).name, recon.list_images(frames), marker_size, one_room=False,
                             drift_correction=drift_correction, find_damage=find_damage)


def run(tier: str, inputs: list[Path], out: Path, marker_size=None, wall_thickness=0.12,
        up="auto", fps=2.0, work: Path | None = None, drift_correction=True, find_damage=True) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    captures = []
    for i, src in enumerate(inputs):
        if tier == "lidar":
            captures.append(lidar_capture(src, up, drift_correction, find_damage))
        elif tier == "photos":
            captures.append(photo_capture(src, marker_size, find_damage))
        elif tier == "video":
            captures.append(video_capture(src, marker_size, fps, work and Path(work) / f"capture_{i}",
                                          drift_correction, find_damage))
        else:
            raise ValueError(f"unknown tier {tier!r}")
    rooms, log = stitch(captures, wall_thickness)
    rooms = canonicalize(rooms)
    unnamed = [r for r in sorted(rooms, key=lambda r: -r.area) if not r.name or r.name.startswith("Room ")]
    for k, r in enumerate(unnamed):
        r.name = f"Room {k + 1}"
    meta = {"tier": tier, "inputs": [str(p) for p in inputs], "stitching": log,
            "captures": [getattr(c, "stats", {"source": c.source}) for c in captures]}
    marker = any("ArUco" in str(c.get("scale_from", "")) for c in meta["captures"])
    data = export.to_dict(rooms, meta, tier=tier, marker_scale=marker)
    damage.regions(rooms, data, tier)
    export.write_json(data, out / "plan.json")
    export.write_svg(rooms, out / "plan.svg", title=f"Floor plan ({tier})", data=data)
    return data
