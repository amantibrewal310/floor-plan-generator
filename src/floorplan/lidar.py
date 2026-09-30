"""LiDAR tier: point clouds / meshes exported from iPhone/iPad scanning apps, or RoomPlan JSON.

LiDAR data is already metric, so the only job is gravity alignment (z up) before handing the
points to plan.extract_rooms.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

AXES = {"x": 0, "y": 1, "z": 2}


def load_points(path: Path, up: str = "auto", max_points=3_000_000) -> np.ndarray:
    import trimesh

    obj = trimesh.load(str(path))
    if isinstance(obj, trimesh.Scene):
        obj = obj.to_geometry()
    if isinstance(obj, trimesh.Trimesh) and len(obj.faces):
        n = int(min(max_points, max(200_000, obj.area * 3000)))
        pts = np.asarray(trimesh.sample.sample_surface(obj, n, seed=0)[0])
    else:
        pts = np.asarray(obj.vertices)
    pts = pts[np.isfinite(pts).all(1)]
    if len(pts) > max_points:
        pts = pts[np.random.default_rng(0).choice(len(pts), max_points, replace=False)]
    return gravity_align(pts, up)


def gravity_align(pts: np.ndarray, up: str = "auto") -> np.ndarray:
    """Rotate so that `up` becomes +z, then level the floor exactly with a plane fit."""
    if up == "auto":
        up = "+" + "xyz"[_guess_up_axis(pts)]
    sign = -1.0 if up.startswith("-") else 1.0
    ax = AXES[up.lstrip("+-")]
    z = sign * pts[:, ax]
    # right-handed frame with the chosen axis as z
    x = pts[:, (ax + 1) % 3]
    y = sign * pts[:, (ax + 2) % 3]
    P = np.column_stack([x, y, z])
    return _level(P)


def _guess_up_axis(pts) -> int:
    """Floors and ceilings are big horizontal planes: along the up axis the coordinate
    histogram has a very sharp spike."""
    scores = []
    for a in range(3):
        h, _ = np.histogram(pts[:, a], bins=np.arange(pts[:, a].min(), pts[:, a].max() + 0.02, 0.02))
        scores.append(h.max() / max(1.0, np.mean(h[h > 0])))
    return int(np.argmax(scores))


def _level(P: np.ndarray) -> np.ndarray:
    return level(P)[0]


def level(P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotate so the floor plane is exactly horizontal: (levelled points, rotation)."""
    from .plan import estimate_floor_ceiling

    floor, _ = estimate_floor_ceiling(P[:, 2])
    near = P[np.abs(P[:, 2] - floor) < 0.04]
    if len(near) < 100:
        return P, np.eye(3)
    m = near.mean(0)
    n = np.linalg.svd(near - m, full_matrices=False)[2][2]
    n = n if n[2] > 0 else -n
    if n[2] < np.cos(np.radians(10)):  # not a floor: leave it alone
        return P, np.eye(3)
    v = np.cross(n, [0, 0, 1.0])
    s, c = np.linalg.norm(v), n[2]
    if s < 1e-9:
        return P, np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + K + K @ K * ((1 - c) / s**2)
    return P @ R.T, R


def roomplan_points(path: Path, spacing=0.02) -> np.ndarray:
    """Turn a RoomPlan CapturedRoom/CapturedStructure JSON into dense wall/floor samples.

    RoomPlan already gives parametric walls; sampling them lets the rest of the pipeline
    (room segmentation, doors, stitching) stay identical for every tier.
    """
    data = json.loads(Path(path).read_text())
    rooms = data.get("rooms") or [data]  # CapturedStructure wraps several CapturedRooms
    walls, holes = [], []
    for room in rooms:
        for w in room.get("walls", []):
            walls.append(w)
        for key in ("doors", "openings", "windows"):
            for d in room.get(key, []):
                holes.append(d)

    def unpack(item):
        M = np.array(item["transform"], float).reshape(4, 4).T  # column-major
        return M[:3, 3], M[:3, 0] / np.linalg.norm(M[:3, 0]), np.asarray(item["dimensions"], float)

    hole_boxes = [unpack(h) for h in holes]
    rng = np.random.default_rng(0)
    pts = []
    y_floor = min(unpack(w)[0][1] - unpack(w)[2][1] / 2 for w in walls)
    for w in walls:
        c, u, (L, H, _) = unpack(w)
        n = int(L * H / spacing**2)  # random rather than gridded: no quantisation of door jambs
        s, h = rng.uniform(-L / 2, L / 2, n), rng.uniform(-H / 2, H / 2, n)
        P = c + np.outer(s, u) + np.outer(h, [0, 1.0, 0])
        keep = np.ones(len(P), bool)
        for hc, hu, (hw, hh, _) in hole_boxes:
            rel = P - hc
            if abs(np.dot(c - hc, np.cross(u, [0, 1.0, 0]))) > 0.3:  # not on this wall
                continue
            keep &= ~((np.abs(rel @ hu) < hw / 2) & (np.abs(rel[:, 1]) < hh / 2))
        pts.append(P[keep])
    P = np.concatenate(pts)
    # floor and ceiling layers so the heights are unambiguous
    lo, hi = P[:, [0, 2]].min(0), P[:, [0, 2]].max(0)
    gx, gz = np.meshgrid(np.arange(lo[0], hi[0], 0.1), np.arange(lo[1], hi[1], 0.1))
    y_ceil = max(unpack(w)[0][1] + unpack(w)[2][1] / 2 for w in walls)
    floor = np.column_stack([gx.ravel(), np.full(gx.size, y_floor), gz.ravel()])
    ceil = np.column_stack([gx.ravel(), np.full(gx.size, y_ceil), gz.ravel()])
    return gravity_align(np.concatenate([P, floor, ceil]), "+y")


# ---------------------------------------------------------------- raw depth + poses (Stray Scanner)

def _quat_to_mat(x, y, z, w):
    n = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


# camera axes of the logged pose -> OpenCV camera axes (x right, y down, z forward)
CONVENTIONS = {"arkit": np.diag([1.0, -1.0, -1.0]), "opencv": np.eye(3)}


def load_stray(folder: Path, max_frames=400, stride_px=2, conf_min=2, max_depth=5.0):
    """Stray Scanner export -> per-frame (points, camera centre) in a z-up world, in capture order.

    Only confident depth (LiDAR confidence 2 by default) is kept: glass, mirrors and dark or
    shiny surfaces come back with low confidence, so they drop out here instead of making
    phantom walls. Depth further than `max_depth` is noisy and dropped too."""
    folder = Path(folder)
    rows = np.genfromtxt(folder / "odometry.csv", delimiter=",", skip_header=1)
    rows = np.atleast_2d(rows)
    step = max(1, int(math.ceil(len(rows) / max_frames)))
    rows = rows[::step]
    cap = cv2.VideoCapture(str(folder / "rgb.mp4"))
    rgb_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920.0
    cap.release()
    frames = []
    for r in rows:
        k = int(r[1])
        depth = cv2.imread(str(folder / "depth" / f"{k:06d}.png"), cv2.IMREAD_UNCHANGED)
        if depth is None:
            continue
        z = depth.astype(np.float32) / 1000.0
        conf_path = folder / "confidence" / f"{k:06d}.png"
        conf = cv2.imread(str(conf_path), cv2.IMREAD_UNCHANGED) if conf_path.exists() else None
        s = z.shape[1] / rgb_w  # intrinsics are logged for the RGB resolution
        fx, fy, cx, cy = r[9] * s, r[10] * s, r[11] * s, r[12] * s
        v, u = np.mgrid[0:z.shape[0]:stride_px, 0:z.shape[1]:stride_px]
        zz = z[v, u]
        ok = (zz > 0.1) & (zz < max_depth)
        if conf is not None:
            ok &= conf[v, u] >= conf_min
        X = np.column_stack([(u[ok] + 0.5 - cx) / fx * zz[ok], (v[ok] + 0.5 - cy) / fy * zz[ok], zz[ok]])
        frames.append((X, _quat_to_mat(*r[5:9]), r[2:5].copy()))
    if not frames:
        raise ValueError(f"no depth frames found in {folder}")
    conv = _pick_convention(frames)
    Y2Z = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0.0]])  # ARKit world (y up) -> z up
    out = []
    for X, R, t in frames:
        Rw = Y2Z @ R @ CONVENTIONS[conv]
        out.append((X @ Rw.T + Y2Z @ t, Y2Z @ t))
    return out


def _pick_convention(frames, voxel=0.05):
    """The docs do not pin down the camera axes of the logged pose; the right choice is the one
    under which frames from across the capture agree (fewest occupied voxels)."""
    pick = frames[:: max(1, len(frames) // 25)]
    best, best_n = None, None
    for name, C in CONVENTIONS.items():
        P = np.concatenate([X[::4] @ (R @ C).T + t for X, R, t in pick])
        n = len(np.unique(np.floor(P / voxel).astype(np.int64), axis=0))
        if best_n is None or n < best_n:
            best, best_n = name, n
    return best
