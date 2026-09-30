"""Synthetic scenes with ground truth, used for demos, tests and accuracy benchmarks.

World frame: metres, z up, floor at z=0. Each scene can be rendered as
  * photos (JPEGs with EXIF focal length) or a video (mp4), with ArUco markers on the floor
  * a LiDAR-style point cloud (PLY, ARKit y-up convention, arbitrary heading)
  * an Apple RoomPlan-style CapturedRoom JSON
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

HEIGHT = 2.6
DOOR_HEIGHT = 2.05
ARUCO_DICT = cv2.aruco.DICT_4X4_50


@dataclass
class Door:
    room: int
    edge: int
    offset: float  # metres from the edge's start vertex
    width: float


@dataclass
class Scene:
    name: str
    rooms: list[np.ndarray]  # CCW polygons of interior wall faces
    doors: list[Door]
    markers: list[tuple[int, float, float, float]]  # (id, x, y, yaw) lying on the floor
    walk: list[list[tuple[float, float]]]  # camera waypoints, one list per capture
    marker_size: float = 0.20
    height: float = HEIGHT

    def door_segments(self):
        """[(room, edge, p0, p1)] in world coordinates."""
        out = []
        for d in self.doors:
            poly = self.rooms[d.room]
            a, b = poly[d.edge], poly[(d.edge + 1) % len(poly)]
            u = (b - a) / np.linalg.norm(b - a)
            out.append((d.room, d.edge, a + u * d.offset, a + u * (d.offset + d.width)))
        return out

    def walls(self):
        """Vertical quads: (p0, p1, openings[(s0, s1, z0, z1)])."""
        quads = []
        for r, poly in enumerate(self.rooms):
            for e in range(len(poly)):
                a, b = poly[e], poly[(e + 1) % len(poly)]
                ops = [(d.offset, d.offset + d.width, 0.0, DOOR_HEIGHT)
                       for d in self.doors if d.room == r and d.edge == e]
                quads.append((a, b, ops))
        return quads

    def ground_truth(self) -> dict:
        return {
            "rooms": [p.tolist() for p in self.rooms],
            "doors": [{"room": r, "edge": e, "width": float(np.linalg.norm(p1 - p0)),
                       "center": ((p0 + p1) / 2).tolist()}
                      for r, e, p0, p1 in self.door_segments()],
            "height": self.height,
        }


def tape_truth(scene: Scene) -> dict:
    """Ground truth in the tape-measure format of evaluate.py (what a person would write down)."""
    rooms = []
    for i, poly in enumerate(scene.rooms):
        n = len(poly)
        x, y = poly[:, 0], poly[:, 1]
        rooms.append({
            "name": f"Room {i + 1}",
            "walls_m": [round(float(np.linalg.norm(poly[(k + 1) % n] - poly[k])), 4) for k in range(n)],
            "ceiling_m": scene.height,
            "area_m2": round(0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))), 4),
            "openings": [{"type": "door", "width_m": d.width} for d in scene.doors if d.room == i],
        })
    return {"rooms": rooms}


def _poly(*pts):
    return np.array(pts, dtype=float)


def make_scene(name: str) -> Scene:
    if name == "rect":
        return Scene(
            name, [_poly((0, 0), (4.2, 0), (4.2, 3.5), (0, 3.5))],
            [Door(0, 0, 1.0, 0.9)],
            [(0, 1.6, 1.5, 0.3), (1, 2.6, 2.0, 1.2), (2, 2.2, 1.0, -0.5)],
            [[(1.0, 0.9), (3.2, 0.9), (3.3, 2.6), (1.0, 2.6)]],
        )
    if name == "lshape":
        return Scene(
            name, [_poly((0, 0), (5.0, 0), (5.0, 2.5), (2.8, 2.5), (2.8, 4.2), (0, 4.2))],
            [Door(0, 5, 1.2, 0.85)],
            [(0, 1.4, 1.2, 0.2), (1, 3.6, 1.2, 1.0), (2, 1.4, 3.0, -0.6)],
            [[(0.9, 0.9), (4.1, 0.9), (4.1, 1.7), (2.0, 1.6), (1.9, 3.4), (0.9, 3.3)]],
        )
    if name == "apartment":
        t = 0.12  # interior wall thickness
        a = _poly((0, 0), (4.0, 0), (4.0, 3.6), (0, 3.6))
        b = _poly((4.0 + t, 0), (7.4, 0), (7.4, 3.6), (4.0 + t, 3.6))
        return Scene(
            name, [a, b],
            # Door between rooms spans y in [1.3, 2.2]; offsets are from each edge's start vertex.
            [Door(0, 1, 1.3, 0.9), Door(1, 3, 3.6 - 2.2, 0.9), Door(0, 3, 0.8, 0.85)],
            [(0, 1.8, 1.4, 0.4), (1, 2.4, 2.3, -0.8), (2, 5.6, 1.5, 0.1), (3, 6.2, 2.4, 2.0),
             (4, 4.06, 1.75, 0.0)],  # marker 4 sits on the threshold, visible from both rooms
            [[(0.9, 0.9), (3.1, 0.9), (3.2, 2.7), (0.9, 2.7)],
             [(5.0, 0.9), (6.6, 0.9), (6.6, 2.7), (5.0, 2.7)]],
            marker_size=0.18,
        )
    raise ValueError(f"unknown scene {name!r}; choose rect, lshape or apartment")


SCENES = ("rect", "lshape", "apartment")


# ---------------------------------------------------------------- rendering

def _texture(seed: int, size=1024) -> np.ndarray:
    """Multi-scale blotchy texture with sharp speckles: gives SIFT something to hold on to."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(90, 200, 3)
    tex = np.zeros((size, size), np.float32)
    for cells, amp in ((4, 40), (16, 30), (64, 22), (256, 14)):
        noise = rng.normal(0, 1, (cells, cells)).astype(np.float32)
        tex += amp * cv2.resize(noise, (size, size), interpolation=cv2.INTER_CUBIC)
    img = np.clip(base[None, None, :] + tex[..., None] * rng.uniform(0.6, 1.0, 3), 0, 255)
    for _ in range(250):  # speckles / posters / scuffs
        c = tuple(int(v) for v in rng.uniform(0, 255, 3))
        p = tuple(int(v) for v in rng.integers(0, size, 2))
        if rng.random() < 0.7:
            cv2.circle(img, p, int(rng.integers(3, 14)), c, -1)
        else:
            q = (p[0] + int(rng.integers(-60, 60)), p[1] + int(rng.integers(-60, 60)))
            cv2.rectangle(img, p, q, c, -1)
    return img.astype(np.float32)


def _marker_tile(marker_id: int) -> np.ndarray:
    d = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    m = cv2.aruco.generateImageMarker(d, marker_id, 240)
    m = cv2.copyMakeBorder(m, 60, 60, 60, 60, cv2.BORDER_CONSTANT, value=255)  # quiet zone
    return cv2.cvtColor(m, cv2.COLOR_GRAY2BGR).astype(np.float32)


class Renderer:
    TEXEL = 0.004  # metres per texel

    def __init__(self, scene: Scene, width=960, height=720, hfov_deg=66.0, supersample=2):
        self.scene, self.w, self.h, self.ss = scene, width, height, supersample
        self.f = width / 2 / math.tan(math.radians(hfov_deg) / 2)
        self.walls = scene.walls()
        self.tex = [_texture(100 + i) for i in range(len(self.walls) + 2)]  # walls, floor, ceiling
        self.markers = [(i, x, y, yaw, _marker_tile(i)) for i, x, y, yaw in scene.markers]

    @property
    def focal_35mm(self) -> float:
        return self.f / self.w * 36.0

    def render(self, cam: np.ndarray, yaw: float, pitch: float, depth=False):
        W, H, ss = self.w * self.ss, self.h * self.ss, self.ss
        f = self.f * ss
        u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
        fwd = np.array([math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), math.sin(pitch)])
        right = np.array([math.sin(yaw), -math.cos(yaw), 0.0])
        down = np.cross(fwd, right)
        d = ((u - W / 2) / f)[..., None] * right + ((v - H / 2) / f)[..., None] * down + fwd
        dx, dy, dz = d[..., 0], d[..., 1], d[..., 2]
        cx, cy, cz = cam
        hs = self.scene.height

        best_t = np.full(dx.shape, np.inf)
        surf = np.full(dx.shape, -1)
        tu = np.zeros(dx.shape)
        tv = np.zeros(dx.shape)
        with np.errstate(divide="ignore", invalid="ignore"):
            for k, (a, b, ops) in enumerate(self.walls):
                L = np.linalg.norm(b - a)
                ud = (b - a) / L
                n = np.array([-ud[1], ud[0]])
                denom = n[0] * dx + n[1] * dy
                t = ((a[0] - cx) * n[0] + (a[1] - cy) * n[1]) / denom
                s = (cx + t * dx - a[0]) * ud[0] + (cy + t * dy - a[1]) * ud[1]
                z = cz + t * dz
                ok = (t > 1e-3) & (s >= 0) & (s <= L) & (z >= 0) & (z <= hs) & (t < best_t)
                for s0, s1, z0, z1 in ops:
                    ok &= ~((s > s0) & (s < s1) & (z > z0) & (z < z1))
                best_t = np.where(ok, t, best_t)
                surf = np.where(ok, k, surf)
                tu = np.where(ok, s, tu)
                tv = np.where(ok, z, tv)
            for k, zp in ((len(self.walls), 0.0), (len(self.walls) + 1, hs)):
                t = (zp - cz) / dz
                ok = (t > 1e-3) & (t < best_t)
                best_t = np.where(ok, t, best_t)
                surf = np.where(ok, k, surf)
                tu = np.where(ok, cx + t * dx, tu)
                tv = np.where(ok, cy + t * dy, tv)

        img = np.zeros((H, W, 3), np.float32)
        mapx = (tu / self.TEXEL).astype(np.float32)
        mapy = (tv / self.TEXEL).astype(np.float32)
        for k, tex in enumerate(self.tex):
            m = surf == k
            if m.any():
                shade = 1.0 if k < len(self.walls) else (0.85 if k == len(self.walls) else 1.1)
                s = cv2.remap(tex, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
                img[m] = np.clip(s[m] * shade, 0, 255)
        floor = surf == len(self.walls)
        for _, mx, my, myaw, tile in self.markers:  # markers painted onto the floor
            half = self.scene.marker_size / 2 * tile.shape[0] / 240.0  # include quiet zone
            lx = (tu - mx) * math.cos(myaw) + (tv - my) * math.sin(myaw)
            ly = -(tu - mx) * math.sin(myaw) + (tv - my) * math.cos(myaw)
            m = floor & (np.abs(lx) < half) & (np.abs(ly) < half)
            if m.any():
                px = ((lx + half) / (2 * half) * tile.shape[1]).astype(np.float32)
                py = ((half - ly) / (2 * half) * tile.shape[0]).astype(np.float32)
                s = cv2.remap(tile, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                img[m] = s[m]
        img = cv2.resize(img, (self.w, self.h), interpolation=cv2.INTER_AREA).astype(np.uint8)
        if depth:  # rays are (x, y, 1) in camera coordinates, so t is already z-depth
            z = np.where(np.isfinite(best_t), best_t, 0.0)[ss // 2::ss, ss // 2::ss]
            return img, z.astype(np.float32)
        return img


def camera_path(waypoints, step: float) -> np.ndarray:
    """Closed loop through waypoints, resampled every `step` metres."""
    pts = np.array(waypoints + [waypoints[0]], float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    s = np.arange(0, cum[-1], step)
    return np.stack([np.interp(s, cum, pts[:, 0]), np.interp(s, cum, pts[:, 1])], 1)


def _look_yaw(p, poly):
    c = poly.mean(0)
    return math.atan2(c[1] - p[1], c[0] - p[0])


def _room_of(p, scene):
    for i, poly in enumerate(scene.rooms):
        if cv2.pointPolygonTest(poly.astype(np.float32), (float(p[0]), float(p[1])), False) >= 0:
            return poly
    return scene.rooms[0]


def write_photos(scene: Scene, out: Path, capture: int, seed=0) -> Path:
    """A capture = photos taken while walking a loop around one room, aimed across the room."""
    rng = np.random.default_rng(seed + capture)
    r = Renderer(scene)
    out.mkdir(parents=True, exist_ok=True)
    poly = _room_of(np.mean(scene.walk[capture], 0), scene)
    n = 0
    for p in camera_path(scene.walk[capture], 0.55):
        base = _look_yaw(p, poly)
        for dyaw, pitch in ((-0.55, -0.2), (0.0, -0.6), (0.55, -0.2)):  # walls, markers, walls
            cam = np.array([p[0], p[1], 1.45 + rng.normal(0, 0.03)])
            img = r.render(cam, base + dyaw + rng.normal(0, 0.05), pitch + rng.normal(0, 0.05))
            img = np.clip(img + rng.normal(0, 2.0, img.shape), 0, 255).astype(np.uint8)
            _save_jpeg_with_focal(img, out / f"IMG_{n:04d}.jpg", r.focal_35mm)
            n += 1
    return out


def _save_jpeg_with_focal(bgr, path, focal_35mm):
    im = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    exif = Image.Exif()
    exif[0x8769] = {0xA405: int(round(focal_35mm))}  # ExifIFD.FocalLengthIn35mmFilm
    im.save(path, quality=92, exif=exif)


def walk_path(scene: Scene, step: float):
    """Every waypoint loop in turn, walking between rooms through the connecting door.
    Returns positions and, per position, a heading to look along (None = look at the room)."""
    loops = [camera_path(w, step) for w in scene.walk]
    path, look = [], []
    for i, loop in enumerate(loops):
        path += list(loop)
        look += [None] * len(loop)
        if i + 1 < len(loops):
            door = next((p0 + p1) / 2 for _, _, p0, p1 in scene.door_segments())
            legs = [loop[-1], door - np.array([0.6, 0]), door + np.array([0.6, 0]), loops[i + 1][0]]
            for a, b in zip(legs[:-1], legs[1:]):
                n = max(1, int(np.linalg.norm(b - a) / step))
                heading = math.atan2(b[1] - a[1], b[0] - a[0])
                path += [a + (b - a) * k / n for k in range(n)]
                look += [heading] * n
    return path, look


def write_video(scene: Scene, out: Path, fps=15, speed=0.35, seed=0) -> Path:
    """One continuous walkthrough of every waypoint loop in the scene (rooms joined through doors)."""
    rng = np.random.default_rng(seed)
    r = Renderer(scene, width=1280, height=720, hfov_deg=70.0, supersample=1)
    path, look = walk_path(scene, speed / fps)
    out.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (r.w, r.h))
    yaw = None
    for k, (p, heading) in enumerate(zip(path, look)):
        target = heading if heading is not None else _look_yaw(p, _room_of(p, scene)) + 0.45 * math.sin(k / fps * 0.8)
        yaw = target if yaw is None else yaw + math.remainder(target - yaw, 2 * math.pi) * 0.08
        cam = np.array([p[0], p[1], 1.45 + 0.02 * math.sin(k / fps * 5)])
        # mostly level, with a glance down at the floor markers every few seconds
        pitch = -0.15 - 0.4 * max(0.0, math.sin(k / fps * 0.9)) ** 3
        img = r.render(cam, yaw, pitch)
        vw.write(np.clip(img + rng.normal(0, 2.0, img.shape), 0, 255).astype(np.uint8))
    vw.release()
    return out


# ---------------------------------------------------------------- lidar / roomplan

def _arkit(points_zup: np.ndarray, heading: float, offset: np.ndarray) -> np.ndarray:
    """z-up metres -> ARKit world (y up) with an arbitrary heading and origin."""
    c, s = math.cos(heading), math.sin(heading)
    x = c * points_zup[:, 0] - s * points_zup[:, 1] + offset[0]
    y = s * points_zup[:, 0] + c * points_zup[:, 1] + offset[1]
    return np.stack([x, points_zup[:, 2] + offset[2], -y], 1)


def lidar_points(scene: Scene, rooms: list[int] | None = None, density=600, noise=0.01, seed=0):
    """Surface samples of the chosen rooms (walls, floor, ceiling, furniture) with sensor noise."""
    rng = np.random.default_rng(seed)
    rooms = list(range(len(scene.rooms))) if rooms is None else rooms
    pts = []
    quads = scene.walls()
    idx = 0
    for r, poly in enumerate(scene.rooms):
        for e in range(len(poly)):
            a, b, ops = quads[idx]
            idx += 1
            if r not in rooms:
                continue
            L = np.linalg.norm(b - a)
            n = int(density * L * scene.height)
            s, z = rng.uniform(0, L, n), rng.uniform(0, scene.height, n)
            keep = np.ones(n, bool)
            for s0, s1, z0, z1 in ops:
                keep &= ~((s > s0) & (s < s1) & (z > z0) & (z < z1))
            u = (b - a) / L
            pts.append(np.column_stack([a + np.outer(s[keep], u), z[keep]]))
        if r not in rooms:
            continue
        lo, hi = poly.min(0), poly.max(0)
        area = np.prod(hi - lo)
        for zp, frac in ((0.0, 1.0), (scene.height, 0.6)):  # ceiling only partially scanned
            n = int(density * area * frac)
            xy = rng.uniform(lo, hi, (n, 2))
            inside = np.array([cv2.pointPolygonTest(poly.astype(np.float32), (float(x), float(y)), False) >= 0
                               for x, y in xy])
            pts.append(np.column_stack([xy[inside], np.full(inside.sum(), zp)]))
        # clutter: a sofa against the first wall and a table in the middle
        a, b = poly[0], poly[1]
        u = (b - a) / np.linalg.norm(b - a)
        nrm = np.array([-u[1], u[0]])
        for c0, size, h in (((a + b) / 2 + nrm * 0.45, (1.8, 0.8), 0.85), (poly.mean(0), (1.2, 0.8), 0.75)):
            n = int(density * 4 * size[0] * size[1])
            loc = rng.uniform(-0.5, 0.5, (n, 3)) * np.array([size[0], size[1], 0]) + [0, 0, 0]
            loc[:, 2] = rng.uniform(0, h, n)
            xy = c0 + np.outer(loc[:, 0], u) + np.outer(loc[:, 1], nrm)
            pts.append(np.column_stack([xy, loc[:, 2]]))
    P = np.concatenate(pts)
    P += rng.normal(0, noise, P.shape)
    outliers = rng.uniform(P.min(0) - 0.5, P.max(0) + 0.5, (len(P) // 500, 3))
    return np.concatenate([P, outliers])


def write_lidar_ply(scene: Scene, out: Path, rooms=None, seed=0) -> Path:
    import trimesh
    rng = np.random.default_rng(seed)
    P = _arkit(lidar_points(scene, rooms, seed=seed), rng.uniform(-math.pi, math.pi), rng.uniform(-3, 3, 3))
    out.parent.mkdir(parents=True, exist_ok=True)
    trimesh.PointCloud(P).export(out)
    return out


def write_roomplan_json(scene: Scene, out: Path, seed=0) -> Path:
    """Mimics the Codable export of RoomPlan's CapturedRoom (walls/doors with 4x4 column-major transforms)."""
    rng = np.random.default_rng(seed)
    heading, off = rng.uniform(-math.pi, math.pi), rng.uniform(-3, 3, 3)

    def transform(center_zup, along_zup):
        c = _arkit(center_zup[None], heading, off)[0]
        x = _arkit(np.array([[*along_zup, 0.0]]), heading, np.zeros(3))[0]
        y = np.array([0.0, 1.0, 0.0])
        z = np.cross(x, y)
        m = np.eye(4)
        m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = x, y, z, c
        return m.T.reshape(-1).tolist()  # column-major

    walls, doors = [], []
    for r, poly in enumerate(scene.rooms):
        for e in range(len(poly)):
            a, b = poly[e], poly[(e + 1) % len(poly)]
            L = float(np.linalg.norm(b - a))
            u = (b - a) / L
            wid = f"W-{r}-{e}"
            walls.append({"identifier": wid, "category": {"wall": {}},
                          "dimensions": [L, scene.height, 0.0],
                          "transform": transform(np.array([*(a + b) / 2, scene.height / 2]), u)})
            for d in scene.doors:
                if d.room == r and d.edge == e:
                    c = a + u * (d.offset + d.width / 2)
                    doors.append({"identifier": f"D-{r}-{e}", "parentIdentifier": wid,
                                  "category": {"door": {"isOpen": True}},
                                  "dimensions": [d.width, DOOR_HEIGHT, 0.0],
                                  "transform": transform(np.array([*c, DOOR_HEIGHT / 2]), u)})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"version": 2, "walls": walls, "doors": doors, "windows": [],
                               "openings": [], "objects": [], "floors": []}, indent=1))
    return out


# ---------------------------------------------------------------- stray scanner (raw LiDAR)

def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def _quat(R):
    """Rotation matrix -> (qx, qy, qz, qw)."""
    w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return x, y, z, w


def write_stray(scene: Scene, out: Path, fps=6, speed=0.35, drift_deg_per_m=0.8, seed=0) -> Path:
    """A Stray Scanner export of one walkthrough: depth/*.png (uint16 mm, 256x192),
    confidence/*.png, odometry.csv (ARKit camera-to-world, y up) and rgb.mp4.

    The depth is rendered from the true trajectory, but the logged poses drift like visual-inertial
    odometry does: a heading error growing with distance walked, plus 1% scale and slight
    vertical creep. That is what the drift correction has to undo."""
    rng = np.random.default_rng(seed)
    hfov = 63.0
    rgb, dep = Renderer(scene, 640, 480, hfov, 1), Renderer(scene, 256, 192, hfov, 1)
    path, look = walk_path(scene, speed / fps)
    (out / "depth").mkdir(parents=True, exist_ok=True)
    (out / "confidence").mkdir(exist_ok=True)
    vw = cv2.VideoWriter(str(out / "rgb.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (rgb.w, rgb.h))
    M = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0.0]])  # z-up world -> ARKit world (y up)
    G, g0 = _rz(rng.uniform(-math.pi, math.pi)), rng.uniform(-2, 2, 3)  # arbitrary session origin
    fx = rgb.w / 2 / math.tan(math.radians(hfov) / 2)
    rows = ["timestamp, frame, x, y, z, qx, qy, qz, qw, fx, fy, cx, cy"]
    yaw, walked, p_drift = None, 0.0, None
    for k, (p, heading) in enumerate(zip(path, look)):
        target = heading if heading is not None else _look_yaw(p, _room_of(p, scene)) + 0.45 * math.sin(k / fps * 0.8)
        yaw = target if yaw is None else yaw + math.remainder(target - yaw, 2 * math.pi) * 0.25
        cam = np.array([p[0], p[1], 1.45 + 0.02 * math.sin(k / fps * 5)])
        pitch = -0.1 + 0.25 * math.sin(k / fps * 0.7)  # looks up at the ceiling and down at the floor
        img = rgb.render(cam, yaw, pitch)
        vw.write(np.clip(img + rng.normal(0, 2.0, img.shape), 0, 255).astype(np.uint8))
        _, z = dep.render(cam, yaw, pitch, depth=True)
        z = z * (1 + rng.normal(0, 0.004, z.shape)) + rng.normal(0, 0.004, z.shape)  # LiDAR noise
        cv2.imwrite(str(out / "depth" / f"{k:06d}.png"), np.clip(z * 1000, 0, 65535).astype(np.uint16))
        cv2.imwrite(str(out / "confidence" / f"{k:06d}.png"), np.where(z > 0, 2, 0).astype(np.uint8))

        # dead-reckoned pose: integrate the true steps through a slowly rotating, stretched frame
        if p_drift is None:
            p_drift, prev = cam.copy(), cam
        else:
            step = cam - prev
            walked += float(np.linalg.norm(step[:2]))
            p_drift = p_drift + _rz(math.radians(drift_deg_per_m) * walked) @ step * 1.01 + [0, 0, 0.002 * np.linalg.norm(step[:2])]
            prev = cam
        err = _rz(math.radians(drift_deg_per_m) * walked)
        fwd = np.array([math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), math.sin(pitch)])
        right = np.array([math.sin(yaw), -math.cos(yaw), 0.0])
        down = np.cross(fwd, right)
        R = M @ G @ err @ np.column_stack([right, -down, -fwd])  # ARKit camera: x right, y up, z back
        t = M @ (G @ p_drift + g0)
        q = _quat(R)
        rows.append(f"{k / fps:.4f}, {k:06d}, {t[0]:.6f}, {t[1]:.6f}, {t[2]:.6f}, "
                    f"{q[0]:.7f}, {q[1]:.7f}, {q[2]:.7f}, {q[3]:.7f}, {fx:.3f}, {fx:.3f}, {rgb.w / 2}, {rgb.h / 2}")
    vw.release()
    (out / "odometry.csv").write_text("\n".join(rows) + "\n")
    K = np.array([[fx, 0, rgb.w / 2], [0, fx, rgb.h / 2], [0, 0, 1]])
    np.savetxt(out / "camera_matrix.csv", K, delimiter=",")
    return out
