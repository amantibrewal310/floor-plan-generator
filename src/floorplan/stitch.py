"""Stitch rooms from separate captures into one plan.

A capture that saw several rooms (one walkthrough video, one multi-room scan) is already in a
single frame. Separate captures are joined by, in order of preference:
  1. a shared ArUco marker (photo/video tiers): exact rigid transform between the captures;
  2. a matching doorway: two doors of equal width on opposite faces of the same wall.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from .plan import Room, dominant_angle


@dataclass
class Capture:
    source: str
    rooms: list[Room]
    markers: dict[int, tuple[np.ndarray, float]] = field(default_factory=dict)  # id -> (xy, yaw)


def _rot(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]])


def _overlap(rooms_a, rooms_b, res=0.05) -> float:
    """Area (m^2) where rooms of A and B overlap."""
    allp = np.concatenate([r.polygon for r in rooms_a + rooms_b])
    lo = allp.min(0)
    shape = tuple((np.ceil((allp.max(0) - lo) / res) + 2).astype(int)[::-1])

    def raster(rooms):
        m = np.zeros(shape, np.uint8)
        for r in rooms:
            cv2.fillPoly(m, [np.round((r.polygon - lo) / res).astype(np.int32)], 1)
        return m

    return float((raster(rooms_a) & raster(rooms_b)).sum() * res * res)


def _door_frames(rooms):
    """(width, centre, outward normal) for every door."""
    out = []
    for r in rooms:
        for e, p0, p1 in r.door_segments():
            d = p1 - p0
            w = float(np.linalg.norm(d))
            out.append((w, (p0 + p1) / 2, np.array([d[1], -d[0]]) / w))
    return out


def stitch(captures: list[Capture], wall_thickness=0.12) -> tuple[list[Room], list[str]]:
    """Returns rooms in the first capture's frame and a log of how each capture was attached."""
    placed = list(captures[0].rooms)
    markers = dict(captures[0].markers)
    log = [f"{captures[0].source}: reference frame"]
    pending = list(captures[1:])
    while pending:
        progress = False
        for cap in list(pending):
            T = _by_marker(cap, markers)
            how = "shared marker"
            if T is None:
                T = _by_door(cap, placed, wall_thickness)
                how = "matching doorway"
            if T is None:
                continue
            R, t = T
            placed += [r.transformed(R, t) for r in cap.rooms]
            for k, (xy, yaw) in cap.markers.items():
                markers.setdefault(k, (R @ xy + t, yaw + math.atan2(R[1, 0], R[0, 0])))
            log.append(f"{cap.source}: attached by {how}")
            pending.remove(cap)
            progress = True
        if not progress:
            for cap in pending:
                log.append(f"{cap.source}: could not be attached (no shared marker or matching door); "
                           "placed to the right")
                shift = np.array([max(r.polygon[:, 0].max() for r in placed) + 1.0, 0.0])
                placed += [r.transformed(np.eye(2), shift - np.array(
                    [min(r.polygon[:, 0].min() for r in cap.rooms), 0])) for r in cap.rooms]
            break
    return placed, log


def _by_marker(cap, markers):
    common = sorted(set(cap.markers) & set(markers))
    if not common:
        return None
    k = common[0]
    (xa, ya), (xb, yb) = markers[k], cap.markers[k]
    R = _rot(ya - yb)
    return R, xa - R @ xb


def _by_door(cap, placed, thickness, width_tol=0.12):
    best, best_cost = None, np.inf
    for wa, ca, na in _door_frames(placed):
        for wb, cb, nb in _door_frames(cap.rooms):
            if abs(wa - wb) > width_tol:
                continue
            # new room's door normal must point back at the placed room, across the wall
            R = _rot(math.atan2(-na[1], -na[0]) - math.atan2(nb[1], nb[0]))
            t = ca + na * thickness - R @ cb
            moved = [r.transformed(R, t) for r in cap.rooms]
            cost = abs(wa - wb) + 10 * _overlap(placed, moved)
            if cost < best_cost:
                best, best_cost = (R, t), cost
    return best if best_cost < 0.5 else None


def canonicalize(rooms: list[Room]) -> list[Room]:
    """Rotate so walls are axis aligned and translate so the plan starts at (0, 0)."""
    theta = dominant_angle([e for r in rooms for e in r.edges()])
    R = _rot(-theta)
    rooms = [r.transformed(R, np.zeros(2)) for r in rooms]
    lo = np.min([r.polygon.min(0) for r in rooms], axis=0)
    return [r.transformed(np.eye(2), -lo) for r in rooms]
