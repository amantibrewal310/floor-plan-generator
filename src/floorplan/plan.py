"""Metric, gravity-aligned point cloud -> rooms with wall dimensions and doors.

Every input tier ends up here. Coordinates are metres, z up.

1. Find floor/ceiling heights from the z histogram.
2. Rasterise the upper wall band (above furniture, which also closes door gaps because the
   wall above a door is scanned) into a 2 cm occupancy grid.
3. Flood-fill from outside; the remaining enclosed components are rooms.
4. Simplify each room contour to a polygon and snap its edges to the dominant (Manhattan) axes.
5. Refit every edge to the raw points near it (robust median), then re-intersect neighbouring
   edges. This step is what gets cm accuracy out of a 2 cm grid.
6. Doors are gaps in wall points along an edge, below door height.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Room:
    polygon: np.ndarray  # (N, 2) CCW vertices, metres
    doors: list[tuple[int, float, float]] = field(default_factory=list)  # (edge, start, end) along edge
    height: float | None = None
    name: str = ""
    windows: list[tuple[int, float, float]] = field(default_factory=list)  # same layout as doors
    # per edge: fraction of its length backed by wall points, and the spread of those points (m)
    wall_support: list[tuple[float, float]] = field(default_factory=list)
    extra: dict = field(default_factory=dict)  # damage regions etc., attached later

    @property
    def area(self) -> float:
        x, y = self.polygon[:, 0], self.polygon[:, 1]
        return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    def edges(self):
        n = len(self.polygon)
        return [(self.polygon[i], self.polygon[(i + 1) % n]) for i in range(n)]

    def door_segments(self, which="doors"):
        out = []
        for e, s0, s1 in getattr(self, which):
            a, b = self.edges()[e]
            u = (b - a) / np.linalg.norm(b - a)
            out.append((e, a + u * s0, a + u * s1))
        return out

    def transformed(self, R: np.ndarray, t: np.ndarray) -> "Room":
        extra = dict(self.extra)
        if "damage" in extra:
            extra["damage"] = [{**d, "center": np.r_[R @ d["center"][:2] + t, d["center"][2]]}
                               for d in extra["damage"]]
        return Room(self.polygon @ R.T + t, list(self.doors), self.height, self.name, list(self.windows),
                    list(self.wall_support), extra)


def estimate_floor_ceiling(z: np.ndarray, bin_size=0.01) -> tuple[float, float | None]:
    """Floor/ceiling = the lowest/highest z bins that are dense enough to be horizontal surfaces."""
    lo, hi = np.percentile(z, [0.5, 99.5])
    bins = np.arange(lo - 0.05, hi + 0.05, bin_size)
    hist, edges = np.histogram(z, bins)
    hist = np.convolve(hist, [1, 2, 1], "same")
    thresh = min(max(4 * np.median(hist[hist > 0]), 0.2 * hist.max()), hist.max())
    dense = np.flatnonzero(hist >= thresh)

    def refine(i):
        c = (edges[i] + edges[i + 1]) / 2
        sel = z[np.abs(z - c) < 0.03]
        return float(np.median(sel))

    floor = refine(dense[0])
    ceiling = refine(dense[-1]) if dense[-1] != dense[0] else None
    if ceiling is not None and ceiling - floor < 1.8:
        ceiling = None
    return floor, ceiling


def layer(z: np.ndarray, lowest=True, frac=0.25, bin_size=0.02) -> float | None:
    """Height of the lowest (or highest) horizontal layer holding at least `frac` of the biggest
    one. For points of horizontal surfaces only, where the floor is the lowest big layer (beds and
    tables sit above it) and the ceiling the highest (a bulkhead hangs below it)."""
    z = np.asarray(z)
    if len(z) < 200:
        return None
    lo, hi = np.percentile(z, [0.2, 99.8])
    hist, edges = np.histogram(z, np.arange(lo - 0.05, hi + 0.05, bin_size))
    hist = np.convolve(hist, [1, 2, 1], "same")
    peaks = np.flatnonzero((hist >= frac * hist.max()) & (hist >= np.roll(hist, 1)) & (hist >= np.roll(hist, -1)))
    i = peaks[0] if lowest else peaks[-1]
    c = (edges[i] + edges[i + 1]) / 2
    return float(np.median(z[np.abs(z - c) < 0.03]))


def dominant_angle(segments) -> float:
    """Length-weighted mean edge direction modulo 90 degrees, in (-45, 45] degrees."""
    acc = 0j
    for a, b in segments:
        d = b - a
        acc += np.hypot(*d) * np.exp(4j * math.atan2(d[1], d[0]))
    return float(np.angle(acc) / 4)


def extract_rooms(points: np.ndarray, floor_z: float | None = None, ceiling_z: float | None = None,
                  res=0.02, min_area=1.5, seeds: np.ndarray | None = None, open_fallback=False,
                  walked=False, ceiling_pts: np.ndarray | None = None) -> list[Room]:
    """`seeds`: optional points known to be inside rooms (camera positions); rooms that
    contain none are dropped. `open_fallback`: the points are one room, so if its walls were not
    captured all the way round, return the rectangle they span instead of failing. `walked`: the
    seeds are a walkthrough's camera path, so walked space no room covers (a hallway) is a room too.
    `ceiling_pts`: points of surfaces seen from below (photo and video tiers). Each room's ceiling
    is then the highest layer of them above it, instead of a guess from all points' heights."""
    points = np.asarray(points, float)
    points = points[np.isfinite(points).all(1)]
    if ceiling_z is None and ceiling_pts is not None and len(ceiling_pts):
        # the wall band stops under the lowest ceiling anywhere (a hall bulkhead, say)
        ceiling_z = layer(ceiling_pts[:, 2], lowest=True)
    if floor_z is None or ceiling_z is None:
        f, c = estimate_floor_ceiling(points[:, 2])
        floor_z = f if floor_z is None else floor_z
        ceiling_z = c if ceiling_z is None else ceiling_z
    P = points - [0, 0, floor_z]
    height = (ceiling_z - floor_z) if ceiling_z is not None else None
    top = (height or 2.5) - 0.12
    upper = P[(P[:, 2] > 1.0) & (P[:, 2] < top)]
    if len(upper) < 50:
        raise ValueError("too few wall points above 1.0 m; is the cloud metric and gravity aligned?")

    # Work in a grid aligned with the dominant wall direction so that doorway gaps can be
    # bridged with straight line kernels without rounding off room corners.
    theta0 = _grid_angle(upper[:, :2], res)
    Rm = _rot(-theta0)
    xy = upper[:, :2] @ Rm.T
    origin = xy.min(0) - 1.0
    shape = np.ceil((xy.max(0) + 1.0 - origin) / res).astype(int)[::-1]
    ij = ((xy - origin) / res).astype(int)
    counts = np.zeros(shape, np.int32)
    np.add.at(counts, (ij[:, 1], ij[:, 0]), 1)
    nz = counts[counts > 0]
    occ = (counts >= max(1, int(0.1 * np.percentile(nz, 90)))).astype(np.uint8)
    # Walls are vertical: keep cells whose 10 cm neighbourhood spans a good height range.
    # Drops floating clumps (mis-triangulated SfM points, shelf tops) that would become notches.
    zmin = np.full(shape, np.inf, np.float32)
    zmax = np.full(shape, -np.inf, np.float32)
    np.minimum.at(zmin, (ij[:, 1], ij[:, 0]), upper[:, 2])
    np.maximum.at(zmax, (ij[:, 1], ij[:, 0]), upper[:, 2])
    k5 = np.ones((5, 5), np.uint8)
    spread = cv2.dilate(np.where(np.isfinite(zmax), zmax, -1e3), k5) - \
        -cv2.dilate(np.where(np.isfinite(zmin), -zmin, -1e3), k5)
    occ &= (spread >= 0.2).astype(np.uint8)  # 0.2: the strip above a door head is short

    # Close gaps in the wall mask; keep the gentlest setting that encloses the most floor area
    # (a leaky wall loses its whole room, over-closing only nibbles at corners and corridors).
    # A corner nobody saw (occluded, or next to a door: common with a handful of photos) leaves
    # an L-shaped gap that no row or column crosses. Diagonal kernels bridge it, and the edge
    # refit below restores the corner; they over-close, so they are only a fallback.
    seed_ij = None
    if seeds is not None and len(seeds):
        seed_ij = (((np.asarray(seeds)[:, :2] @ Rm.T) - origin) / res).astype(int)
    best, best_area = None, 0.0
    for diagonal in (False, True):
        for line in (0.0, 1.0, 1.4):
            for radius in (0.06, 0.12, 0.2, 0.3):
                labels, comps = _enclose(occ, res, line, radius, diagonal, min_area)
                if seed_ij is not None:  # the photographer stood inside the room
                    h, w = labels.shape
                    ok = (seed_ij >= 0).all(1) & (seed_ij[:, 0] < w) & (seed_ij[:, 1] < h)
                    hit = set(labels[seed_ij[ok, 1], seed_ij[ok, 0]].tolist())
                    comps = [i for i in comps if i in hit]
                area = sum((labels == i).sum() for i in comps) * res * res
                if area > best_area * 1.02:
                    best, best_area = (labels, comps, (line, radius, diagonal)), area
        if best is not None:
            break
    polys = []
    if best is None:
        if not open_fallback:
            raise ValueError("no enclosed room found; the walls were not captured all the way round")
        polys.append(_open_rect(upper[:, :2], theta0))
        labels, comps = None, []
    else:
        labels, comps, setting = best
        regions = [labels == i for i in comps]
        if walked and seed_ij is not None:
            regions += _walked_spaces(occ, res, setting, np.isin(labels, comps), seed_ij, min_area)
        n_enclosed = len(comps)
        for mask in regions:
            cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            c = max(cnts, key=cv2.contourArea)
            approx = cv2.approxPolyDP(c, 0.08 / res, True)[:, 0, :].astype(float)
            poly = ((approx + 0.5) * res + origin) @ Rm  # back to the input frame
            if _signed_area(poly) < 0:
                poly = poly[::-1]
            polys.append(poly)

    # the wall directions come from the enclosed rooms; walked spaces are ragged and follow them
    theta = _refine_theta(polys[:n_enclosed] if best is not None else polys, upper[:, :2], theta0)
    wall_pts = P[(P[:, 2] > 0.1) & (P[:, 2] < top)]
    rooms = []
    for poly in polys:
        rough = poly
        poly = _refine_polygon(poly, upper[:, :2], theta)
        if (poly is None or len(poly) < 3) and best is None:
            poly = rough  # an open room's unseen sides have nothing to refit to
        if poly is None or len(poly) < 3:
            continue
        room = Room(poly, height=_room_height(poly, P, height, None if ceiling_pts is None
                                              else ceiling_pts - [0, 0, floor_z]))
        if best is None:
            room.extra["unclosed"] = True
        room.doors, room.windows = _find_openings(room, wall_pts)
        room.wall_support = [_support(a, b, upper[:, :2]) for a, b in room.edges()]
        rooms.append(room)
    rooms.sort(key=lambda r: -r.area)
    for k, r in enumerate(rooms):
        r.name = f"Room {k + 1}"
    return rooms


def _enclose(occ, res, line, radius, diagonal, min_area):
    """Close gaps in the wall mask, flood-fill from outside: (labels, enclosed component ids)."""
    flood = _flood(occ, res, line, radius, diagonal)
    n, labels, stats, _ = cv2.connectedComponentsWithStats((flood == 1).astype(np.uint8), connectivity=4)
    return labels, [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] * res * res >= min_area]


def _walked_spaces(occ, res, setting, rooms, seed_ij, min_area):
    """Walked space that no room covers, as extra room masks.

    Closing doorways also bridges across a hallway about a metre wide (its two walls are a
    doorway's width apart), and in a hallway or a kitchen the clutter fills whatever is left, so
    those spaces never enclose. Without closing they are open, but they leak out through the
    doorways they connect. Take the space that is free in the scan itself, inside the home (sealed
    from the outside by the door-closing setting that found the rooms) and not already a room;
    keep the pieces the phone walked through."""
    disk = lambda m: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(m / res) + 1,) * 2)
    home = _flood(occ, res, *setting) != 2
    free = _flood(occ, res, 0.0, 0.06, False) != 0
    near_room = cv2.dilate(rooms.astype(np.uint8), disk(0.15)) > 0  # the rim between a room and its walls
    cand = (home & free & ~near_room).astype(np.uint8)
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, disk(0.15))  # slivers narrower than 0.3 m are not walkable
    n, labels, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=4)
    h, w = labels.shape
    ok = (seed_ij >= 0).all(1) & (seed_ij[:, 0] < w) & (seed_ij[:, 1] < h)
    walked = set(labels[seed_ij[ok, 1], seed_ij[ok, 0]].tolist()) - {0}
    return [labels == i for i in sorted(walked) if stats[i, cv2.CC_STAT_AREA] * res * res >= min_area]


def _flood(occ, res, line, radius, diagonal):
    """Close gaps in the wall mask and flood-fill from outside: 0 wall, 2 outside, 1 enclosed."""
    def close(img, kernel):
        # Beyond the grid is empty. OpenCV's default erosion border counts it as wall, which
        # turns the corner seed pixel of the flood fill into wall under a kernel that reaches
        # only outside the image there (the anti-diagonal one).
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)

    k = 2 * int(radius / res) + 1
    walls = close(occ, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    if line:  # thicken first: the two sides of a doorway may sit a cell or two apart
        n = int(line / res)
        walls = cv2.dilate(walls, np.ones((5, 5), np.uint8))
        walls |= close(walls, np.ones((1, n), np.uint8))
        walls |= close(walls, np.ones((n, 1), np.uint8))
        if diagonal:
            d = np.eye(int(min(line, 0.8) / res * 0.7), dtype=np.uint8)
            walls |= close(walls, d)
            walls |= close(walls, d[::-1].copy())
    walls = cv2.dilate(walls, np.ones((3, 3), np.uint8))
    flood = (1 - walls).astype(np.uint8)
    cv2.floodFill(flood, None, (0, 0), 2)
    return flood


def _open_rect(xy, theta, pct=2):
    """A room seen from a few viewpoints often shows two or three of its walls, which leaves no
    enclosure to flood-fill. The wall points still span the room: take the rectangle they cover
    on its axes (robust percentiles, not the extremes, so a glimpse through a door does not
    count). Sides no wall was seen on get low wall_support, which widens their intervals."""
    R = _rot(-theta)
    q = xy @ R.T
    lo, hi = np.percentile(q, pct, 0), np.percentile(q, 100 - pct, 0)
    return np.array([lo, [hi[0], lo[1]], hi, [lo[0], hi[1]]]) @ R


def _rot(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]])


def _grid_angle(xy, res) -> float:
    """Dominant wall direction (mod 90 deg) from the gradient orientations of the occupancy image."""
    o = xy.min(0)
    ij = ((xy - o) / res).astype(int)
    img = np.zeros((ij[:, 1].max() + 1, ij[:, 0].max() + 1), np.float32)
    img[ij[:, 1], ij[:, 0]] = 1
    img = cv2.GaussianBlur(img, (0, 0), 2)
    gx, gy = cv2.Sobel(img, cv2.CV_32F, 1, 0), cv2.Sobel(img, cv2.CV_32F, 0, 1)
    acc = np.sum((gx**2 + gy**2) * np.exp(4j * np.arctan2(gy, gx)))
    return float(np.angle(acc) / 4)


def _signed_area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _refine_theta(polys, pts2d, theta, iters=2):
    """The contour is only good to a grid cell, so re-estimate the Manhattan angle from the
    principal directions of the wall points along each long edge."""
    for _ in range(iters):
        acc = 0j
        for p in polys:
            for i in range(len(p)):
                a, b = p[i], p[(i + 1) % len(p)]
                L = np.linalg.norm(b - a)
                u, manhattan = _snap(b - a, theta)
                if not manhattan or L < 0.8:
                    continue
                rel = pts2d - (a + b) / 2
                along, perp = rel @ u, rel @ np.array([u[1], -u[0]])
                q = pts2d[(np.abs(along) < 0.4 * L) & (perp > -0.10) & (perp < 0.12)]
                if len(q) < 20:
                    continue
                d = np.linalg.svd(q - q.mean(0), full_matrices=False)[2][0]
                acc += len(q) * np.exp(4j * math.atan2(d[1], d[0]))
        if acc == 0:
            break
        theta = float(np.angle(acc) / 4)
    return theta


def _snap(d, theta, tol=math.radians(12)):
    ang = math.atan2(d[1], d[0])
    k = round((ang - theta) / (math.pi / 2))
    snapped = theta + k * math.pi / 2
    if abs(math.remainder(ang - snapped, 2 * math.pi)) < tol:
        return np.array([math.cos(snapped), math.sin(snapped)]), True
    return d / np.linalg.norm(d), False


def _refine_polygon(poly, pts2d, theta):
    """Snap edges to the Manhattan frame, fit each to nearby wall points, re-intersect."""
    lines = []  # (point_on_line, unit_dir, length)
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        L = np.linalg.norm(b - a)
        if L < 1e-6:
            continue
        u, manhattan = _snap(b - a, theta)
        if L < 0.1 or (not manhattan and L < 0.35):  # corner-rounding artefacts of the closing step
            continue
        c = (a + b) / 2
        nrm = np.array([u[1], -u[0]])  # outward for a CCW polygon
        rel = pts2d - c
        along, perp = rel @ u, rel @ nrm
        # The contour lies some cm inside the wall (grid, closing). Search outward from the room
        # and lock onto the first dense surface: the interior face, never the far face of a
        # thin wall behind it.
        near = (np.abs(along) < 0.4 * L) & (perp > -0.10) & (perp < 0.30)
        if near.sum() >= 8:
            h, e = np.histogram(perp[near], bins=np.arange(-0.10, 0.301, 0.01))
            h = np.convolve(h, [1, 1, 1], "same")
            first = e[np.argmax(h >= 0.4 * h.max())] + 0.005
            c = c + nrm * first
            rel = pts2d - c
            along, perp = rel @ u, rel @ nrm
        sel = (np.abs(along) < 0.4 * L) & (np.abs(perp) < 0.06)
        for win in (None, 0.05, 0.03):  # iteratively tighten around the wall face
            if sel.sum() < 8:
                break
            if manhattan:
                off = float(np.median(perp[sel]))
            else:
                q = pts2d[sel]
                m = q.mean(0)
                _, _, vt = np.linalg.svd(q - m)
                u = vt[0] if np.dot(vt[0], u) > 0 else -vt[0]
                nrm = np.array([u[1], -u[0]])
                rel, c = pts2d - m, m
                along, perp = rel @ u, rel @ nrm
                off = 0.0
            c = c + nrm * off
            rel = pts2d - c
            along, perp = rel @ u, rel @ nrm
            sel = (np.abs(along) < 0.4 * L) & (np.abs(perp) < (win or 0.05))
        if not manhattan:
            # an off-axis edge must be backed by wall points along most of its length;
            # otherwise it is a corner the capture barely saw, and the neighbours should meet
            on = along[(np.abs(perp) < 0.04) & (np.abs(along) < L / 2)]
            bins = np.unique(((on + L / 2) / 0.05).astype(int))
            if len(bins) < 0.6 * L / 0.05:
                continue
        lines.append((c, u, L))

    merged = _merge_collinear(lines)
    if len(merged) < 3:
        return None
    verts = _intersect(merged)
    # Re-intersection can leave sliver edges (e.g. a 1 mm jog between two walls that are really
    # one): drop the sliver's line, merge what became collinear, repeat.
    while len(merged) > 3:
        L = np.linalg.norm(np.roll(verts, -1, 0) - verts, axis=1)
        if L.min() >= 0.05:
            break
        i = int(np.argmin(L))
        merged = _merge_collinear(merged[:i] + merged[i + 1:])
        if len(merged) < 3:
            return None
        verts = _intersect(merged)
    return verts


def _merge_collinear(lines):
    """Merge consecutive (near-)collinear lines, including across the wrap-around."""
    def same(a, b):
        nrm = np.array([a[1][1], -a[1][0]])
        return np.dot(a[1], b[1]) > math.cos(math.radians(8)) and abs(np.dot(b[0] - a[0], nrm)) < 0.05

    merged = []
    for ln in lines:
        if merged and same(merged[-1], ln):
            c0, u0, L0 = merged[-1]
            nrm0 = np.array([u0[1], -u0[0]])
            w = L0 / (L0 + ln[2])
            merged[-1] = (c0 + (1 - w) * nrm0 * np.dot(ln[0] - c0, nrm0), u0, L0 + ln[2])
            continue
        merged.append(ln)
    if len(merged) > 2 and same(merged[-1], merged[0]):
        merged.pop()
    return merged


def _intersect(merged):
    """Vertex i = intersection of line i-1 and line i."""
    verts = []
    m = len(merged)
    for i in range(m):
        (c1, u1, _), (c2, u2, _) = merged[i - 1], merged[i]
        A = np.column_stack([u1, -u2])
        if abs(np.linalg.det(A)) < 1e-3:  # parallel neighbours: bridge at the midpoint
            verts.append((c1 + c2) / 2)
            continue
        s = np.linalg.solve(A, c2 - c1)
        verts.append(c1 + s[0] * u1)
    return np.array(verts)


def _find_openings(room: Room, pts: np.ndarray, min_w=0.6, max_w=2.0, min_win=0.35, max_win=3.0):
    """Openings are stretches of wall with no points between 1.0 m and 1.9 m (above furniture,
    below the head). Empty lower down too -> door. Wall below (a sill) and above (a head) ->
    window: glass returns no LiDAR and no learned depth on the wall plane. Returns (doors, windows)."""
    doors, windows = [], []
    for e, (a, b) in enumerate(room.edges()):
        L = np.linalg.norm(b - a)
        if L < min_win + 0.1:
            continue
        u = (b - a) / L
        nrm = np.array([u[1], -u[0]])
        rel = pts[:, :2] - a
        along, perp = rel @ u, rel @ nrm
        on = (np.abs(perp) < 0.05) & (along > -0.05) & (along < L + 0.05)
        mid = np.sort(along[on & (pts[:, 2] > 1.0) & (pts[:, 2] < 1.9)])
        low = along[on & (pts[:, 2] > 0.2) & (pts[:, 2] < 0.9)]
        rho = len(mid) / L  # points per metre of wall in the band
        if len(mid) < 20 or rho < 20:  # too sparse to tell an opening from missing data
            continue
        jamb = np.sort(along[on & (pts[:, 2] > 0.2) & (pts[:, 2] < 1.95)])
        sigma = 1.4826 * np.median(np.abs(perp[on] - np.median(perp[on])))  # sensor noise
        win = float(np.clip(3.5 * sigma, 0.02, 0.15))
        s = np.concatenate([[0.0], mid, [L]])
        for k in np.flatnonzero((np.diff(s) >= min_win - 0.05) & (np.diff(s) <= max_win)):
            s0, s1 = _edge_pos(jamb, s[k], -1, L, win), _edge_pos(jamb, s[k + 1], +1, L, win)
            w = s1 - s0
            low_rho = np.sum((low > s0 + 0.05) & (low < s1 - 0.05)) / max(w - 0.1, 1e-3)
            if low_rho < 0.3 * rho * 0.7 / 0.9:  # lower band empty too -> door
                if min_w <= w <= max_w:
                    doors.append((e, s0, s1))
            elif min_win <= w <= max_win and s0 > 0.05 and s1 < L - 0.05:
                # a window needs wall on both sides; a gap running into a corner is missing data
                s0w, s1w = _edge_pos(mid, s[k], -1, L, win), _edge_pos(mid, s[k + 1], +1, L, win)
                windows.append((e, s0w, s1w))
    return doors, windows


def _room_height(poly, P, fallback, ceiling_pts=None):
    """Ceiling height from the points above this room only (rooms can differ). With
    `ceiling_pts` (surfaces seen from below, floor at 0), the highest layer of those."""
    import cv2 as _cv2
    lo = poly.min(0)
    res = 0.05
    shape = tuple((np.ceil((poly.max(0) - lo) / res) + 2).astype(int)[::-1])
    m = np.zeros(shape, np.uint8)
    _cv2.fillPoly(m, [np.round((poly - lo) / res).astype(np.int32)], 1)
    m = _cv2.erode(m, np.ones((5, 5), np.uint8))  # stay 10 cm clear of the walls
    Q = P if ceiling_pts is None else ceiling_pts
    ij = np.floor((Q[:, :2] - lo) / res).astype(int)
    ok = (ij >= 0).all(1) & (ij[:, 0] < shape[1]) & (ij[:, 1] < shape[0])
    ok[ok] = m[ij[ok, 1], ij[ok, 0]] > 0
    z = Q[ok, 2]
    if ceiling_pts is not None:
        c = layer(z[z > 1.8], lowest=False)  # nothing under 1.8 m is a ceiling
        return c
    if len(z) < 200:
        return fallback
    f, c = estimate_floor_ceiling(z)
    return (c - f) if c is not None else None


def _support(a, b, pts2d, bin_m=0.05):
    """(fraction of the wall's length with wall points on it, robust spread of those points)."""
    L = float(np.linalg.norm(b - a))
    u = (b - a) / L
    rel = pts2d - a
    along, perp = rel @ u, rel @ np.array([u[1], -u[0]])
    on = (np.abs(perp) < 0.05) & (along >= 0) & (along <= L)
    if on.sum() < 5:
        return 0.0, 0.05
    cov = len(np.unique((along[on] / bin_m).astype(int))) / max(1, int(np.ceil(L / bin_m)))
    spread = 1.4826 * float(np.median(np.abs(perp[on] - np.median(perp[on]))))
    return round(min(1.0, cov), 3), round(spread, 4)


def _edge_pos(s, x, side, L, win):
    """Sub-sample position of the end of a run of wall points next to a gap at `x`.

    Noise pushes the outermost point past the true edge; instead count the points in a window
    straddling the edge and divide by the local density (unbiased for symmetric noise)."""
    if x <= 0.0 or x >= L:
        return float(x)
    if side < 0:  # wall to the left of x
        ref = np.sum((s > x - win - 0.4) & (s <= x - win)) / 0.4
        n = np.sum((s > x - win) & (s < x + win))
        return float(x - win + n / ref) if ref > 0 else float(x)
    ref = np.sum((s >= x + win) & (s < x + win + 0.4)) / 0.4
    n = np.sum((s > x - win) & (s < x + win))
    return float(x + win - n / ref) if ref > 0 else float(x)
