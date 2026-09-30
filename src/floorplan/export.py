"""Write the floor plan as JSON (machine readable) and SVG (dimensioned drawing)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .plan import Room


def to_dict(rooms: list[Room], meta: dict | None = None) -> dict:
    out = {"units": "metres", "rooms": [], **(meta or {})}
    for r in rooms:
        edges = r.edges()
        out["rooms"].append({
            "name": r.name,
            "area_m2": round(r.area, 3),
            "height_m": None if r.height is None else round(r.height, 3),
            "vertices": np.round(r.polygon, 4).tolist(),
            "walls": [{"start": np.round(a, 4).tolist(), "end": np.round(b, 4).tolist(),
                       "length_m": round(float(np.linalg.norm(b - a)), 4)} for a, b in edges],
            "doors": [{"wall": e, "start": np.round(p0, 4).tolist(), "end": np.round(p1, 4).tolist(),
                       "width_m": round(float(np.linalg.norm(p1 - p0)), 4)}
                      for e, p0, p1 in r.door_segments()],
        })
    return out


def write_json(rooms, path: Path, meta=None):
    Path(path).write_text(json.dumps(to_dict(rooms, meta), indent=2))


def write_svg(rooms: list[Room], path: Path, title="Floor plan", px_per_m=110.0):
    pts = np.concatenate([r.polygon for r in rooms])
    lo, hi = pts.min(0), pts.max(0)
    pad = 0.9
    W = (hi[0] - lo[0] + 2 * pad) * px_per_m
    H = (hi[1] - lo[1] + 2 * pad) * px_per_m + 40

    def P(p):  # world (y up) -> svg (y down)
        return ((p[0] - lo[0] + pad) * px_per_m, (hi[1] - p[1] + pad) * px_per_m + 40)

    el = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W:.0f}" height="{H:.0f}" '
          f'viewBox="0 0 {W:.0f} {H:.0f}" font-family="Helvetica, Arial, sans-serif">',
          f'<rect width="100%" height="100%" fill="#fbfaf7"/>',
          f'<text x="16" y="28" font-size="18" font-weight="600" fill="#222">{title}</text>']
    for r in rooms:
        d = " ".join(f"{x:.1f},{y:.1f}" for x, y in map(P, r.polygon))
        el.append(f'<polygon points="{d}" fill="#eef2f6" stroke="#1f2933" stroke-width="5" '
                  f'stroke-linejoin="miter"/>')
    arcs = []
    for r in rooms:
        for e, p0, p1 in r.door_segments():  # door: gap in the wall plus a swing arc
            (x0, y0), (x1, y1) = P(p0), P(p1)
            el.append(f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" '
                      f'stroke="#fbfaf7" stroke-width="7"/>')
            c = (p0 + p1) / 2
            if any(np.linalg.norm(c - q) < 0.4 for q in arcs):  # other face of the same doorway
                continue
            arcs.append(c)
            a, b = r.edges()[e]
            u = (b - a) / np.linalg.norm(b - a)
            inward = np.array([-u[1], u[0]])
            w = np.linalg.norm(p1 - p0)
            tip = P(p0 + inward * w)
            el.append(f'<path d="M{x0:.1f},{y0:.1f} L{tip[0]:.1f},{tip[1]:.1f} '
                      f'A{w * px_per_m:.1f},{w * px_per_m:.1f} 0 0,1 {x1:.1f},{y1:.1f}" '
                      f'fill="none" stroke="#7b8794" stroke-width="1.2"/>')
    for r in rooms:
        for a, b in r.edges():
            el += _dimension(a, b, P, px_per_m)
        c = _label_point(r.polygon)
        cx, cy = P(c)
        h = f" · h {r.height:.2f} m" if r.height else ""
        el.append(f'<text x="{cx:.1f}" y="{cy - 4:.1f}" text-anchor="middle" font-size="15" '
                  f'font-weight="600" fill="#1f2933">{r.name}</text>')
        el.append(f'<text x="{cx:.1f}" y="{cy + 14:.1f}" text-anchor="middle" font-size="12" '
                  f'fill="#52606d">{r.area:.2f} m²{h}</text>')
    x0, y0 = 16, H - 16  # 1 m scale bar
    el.append(f'<line x1="{x0}" y1="{y0}" x2="{x0 + px_per_m:.0f}" y2="{y0}" stroke="#222" stroke-width="3"/>')
    el.append(f'<text x="{x0 + px_per_m + 8:.0f}" y="{y0 + 4}" font-size="12" fill="#222">1 m</text>')
    el.append("</svg>")
    Path(path).write_text("\n".join(el))


def _dimension(a, b, P, px_per_m, offset=0.28):
    """Dimension line drawn just inside the room, parallel to the wall."""
    L = float(np.linalg.norm(b - a))
    if L < 0.25:
        return []
    u = (b - a) / L
    inward = np.array([-u[1], u[0]])
    a2, b2 = a + inward * offset, b + inward * offset
    (x0, y0), (x1, y1) = P(a2), P(b2)
    mx, my = P((a2 + b2) / 2 + inward * 0.11)
    ang = math.degrees(math.atan2(y1 - y0, x1 - x0))
    if ang > 90 or ang < -90:
        ang += 180
    tick = inward * 0.06
    t = []
    for q in (a2, b2):
        (tx0, ty0), (tx1, ty1) = P(q - tick), P(q + tick)
        t.append(f'<line x1="{tx0:.1f}" y1="{ty0:.1f}" x2="{tx1:.1f}" y2="{ty1:.1f}" stroke="#c2410c" stroke-width="1.2"/>')
    return [f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" stroke="#c2410c" stroke-width="1"/>',
            *t,
            f'<text x="{mx:.1f}" y="{my:.1f}" font-size="11.5" fill="#c2410c" text-anchor="middle" '
            f'dominant-baseline="middle" transform="rotate({ang:.1f} {mx:.1f} {my:.1f})">{L * 100:.1f} cm</text>']


def _label_point(poly):
    """A point well inside the polygon (pole of inaccessibility on a coarse grid)."""
    import cv2
    lo = poly.min(0)
    res = 0.05
    shape = tuple((np.ceil((poly.max(0) - lo) / res) + 3).astype(int)[::-1])
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.round((poly - lo) / res + 1).astype(np.int32)], 1)
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    y, x = np.unravel_index(np.argmax(dist), dist.shape)
    return (np.array([x, y]) - 1) * res + lo
