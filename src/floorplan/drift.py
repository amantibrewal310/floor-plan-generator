"""Drift correction for multi-room captures: plane-anchored re-integration of the trajectory.

Visual-inertial odometry (ARKit) knows gravity, so roll and pitch do not drift, but heading,
height and position do, a little per metre walked. Over a multi-room walk that bends the
later rooms away from the first. The walls themselves are the fix: they are vertical planes
on (mostly) two perpendicular directions, the same everywhere in the home.

The trajectory is cut into chunks of about a metre of walking. For each chunk in order:
  1. heading: rotate so its walls line up with the Manhattan frame of the first chunk;
  2. height: shift so its floor sits at the global floor height;
  3. position: slide (at most 5 cm) so its walls land on walls already mapped, if it overlaps
     them enough to tell. A metre of walking drifts far less than that, and the cap is under
     half an interior wall's thickness, so a chunk can never snap onto the wall's far face.
The chunk's own motion is then re-integrated from the corrected end of the previous chunk, so
every later chunk inherits the fix. `enabled=False` returns the poses as logged (the ablation).
"""

from __future__ import annotations

import math

import numpy as np

from .plan import _grid_angle, estimate_floor_ceiling


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def _chunks(cams, chunk_m, max_frames=40):
    out, start, walked = [], 0, 0.0
    for i in range(1, len(cams)):
        walked += float(np.linalg.norm(cams[i, :2] - cams[i - 1, :2]))
        if walked >= chunk_m or i - start >= max_frames:
            out.append((start, i))
            start, walked = i, 0.0
    out.append((start, len(cams)))
    return out


def _walls(P, floor):
    return P[(P[:, 2] > floor + 0.5) & (P[:, 2] < floor + 2.0), :2]


def correct(frames, enabled=True, chunk_m=1.0, max_shift=0.05, res=0.01):
    """frames: [(points (N,3) z up, camera centre (3,))] in capture order.
    Returns corrected frames and a list of per-chunk corrections (for the report)."""
    if not enabled or len(frames) < 2:
        return frames, []
    cams = np.array([c for _, c in frames])
    allz = np.concatenate([P[::20, 2] for P, _ in frames])
    floor0, _ = estimate_floor_ceiling(allz)
    theta0 = None
    map_pts = []
    out, log = [], []
    yaw = 0.0
    prev_raw, prev_new = None, None  # last camera of the previous chunk, before/after correction
    for a, b in _chunks(cams, chunk_m):
        P = np.concatenate([frames[i][0] for i in range(a, b)])
        # 1. heading, relative to the first chunk that saw enough wall. Drift is gradual, so
        #    predict from the previous chunk and only snap the small residual: total drift can
        #    then exceed the 45 degree ambiguity of a Manhattan frame.
        W = _walls(P, floor0)
        if len(W) > 500:
            th = _grid_angle(W, res)
            if theta0 is None:
                theta0 = th
            d = math.remainder(th + yaw - theta0, math.pi / 2)
            if abs(d) < math.radians(10):
                yaw -= d
        R = _rz(yaw)
        # re-integrate: this chunk's motion, rotated, hung off the corrected previous chunk
        pivot_raw = cams[a] if prev_raw is None else prev_raw
        pivot_new = cams[a] if prev_new is None else prev_new

        def move(X):
            return (X - pivot_raw) @ R.T + pivot_new

        Pm = move(P)
        # 2. height
        near = Pm[np.abs(Pm[:, 2] - floor0) < 0.15, 2]
        dz = floor0 - float(np.median(near)) if len(near) > 500 else 0.0
        # 3. position, against the walls mapped so far
        dxy = np.zeros(2)
        if map_pts and theta0 is not None:
            dxy = _slide(np.concatenate(map_pts), _walls(Pm, floor0), theta0, max_shift, res)
        shift = np.array([dxy[0], dxy[1], dz])
        new = [(move(frames[i][0]) + shift, move(frames[i][1]) + shift) for i in range(a, b)]
        out += new
        map_pts.append(_walls(np.concatenate([p for p, _ in new]), floor0)[::3])
        prev_raw, prev_new = cams[b - 1], new[-1][1]
        log.append({"frames": [a, b], "yaw_deg": round(math.degrees(yaw), 3),
                    "shift_m": np.round(shift, 4).tolist()})
    return out, log


def _slide(M, C, theta, max_shift, res):
    """Translation (x, y) that best lays wall points C onto the map M, found separately along
    the two Manhattan axes with 1-D histograms (walls are sharp peaks there). Zero when C does
    not overlap the map enough to be sure."""
    if len(C) < 300 or len(M) < 300:
        return np.zeros(2)
    Rm = np.array([[math.cos(theta), math.sin(theta)], [-math.sin(theta), math.cos(theta)]])
    Mm, Cm = M @ Rm.T, C @ Rm.T
    out = np.zeros(2)
    for ax in (0, 1):
        lo = min(Mm[:, ax].min(), Cm[:, ax].min()) - max_shift - res
        hi = max(Mm[:, ax].max(), Cm[:, ax].max()) + max_shift + res
        bins = np.arange(lo, hi, res)
        hm = np.histogram(Mm[:, ax], bins)[0].astype(float)
        hc = np.histogram(Cm[:, ax], bins)[0].astype(float)
        hm /= hm.sum()
        hc /= hc.sum()
        n = int(max_shift / res)
        scores = [np.minimum(np.roll(hc, s), hm).sum() for s in range(-n, n + 1)]
        k = int(np.argmax(scores))
        if scores[k] < 0.3 or scores[k] < 1.1 * scores[n]:  # little overlap, or no clear gain
            continue
        # sub-bin peak
        if 0 < k < 2 * n:
            y0, y1, y2 = scores[k - 1], scores[k], scores[k + 1]
            den = y0 - 2 * y1 + y2
            frac = 0.5 * (y0 - y2) / den if abs(den) > 1e-12 else 0.0
        else:
            frac = 0.0
        out[ax] = (k - n + float(np.clip(frac, -0.5, 0.5))) * res
    return out @ Rm  # back to the world frame
