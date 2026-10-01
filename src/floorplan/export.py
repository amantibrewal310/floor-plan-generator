"""Write the floor plan as JSON (machine readable) and SVG (dimensioned drawing)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .plan import Room
from .uncertainty import ErrorModel


SCHEMA_VERSION = "1.0"


def to_dict(rooms: list[Room], meta: dict | None = None, tier="lidar", marker_scale=False) -> dict:
    """The published output (schema/plan.schema.json). Every measurement is a 90% interval."""
    em = ErrorModel(tier, marker_scale)
    out = {"schema_version": SCHEMA_VERSION, "units": "metres", "tier": tier, "interval": "90%",
           "interval_calibration": dict(em.k), "rooms": [], "damage": [], "concealed_flags": [], "scope": [], **(meta or {})}
    for k, r in enumerate(rooms):
        rid = f"R{k + 1}"
        edges = r.edges()
        support = list(r.wall_support or [(1.0, 0.0)] * len(edges))
        for e, s0, s1 in r.doors + r.windows:  # an opening is measured, not missing wall
            L = float(np.linalg.norm(edges[e][1] - edges[e][0]))
            support[e] = (min(1.0, round(support[e][0] + (s1 - s0) / L, 3)), support[e][1])
        walls = []
        for i, ((a, b), (cov, spread)) in enumerate(zip(edges, support)):
            walls.append({"id": f"{rid}.W{i + 1}", "start": _r(a), "end": _r(b),
                          "length_m": em.wall(float(np.linalg.norm(b - a)), cov, spread),
                          "coverage": cov})
        openings = []
        for kind in ("doors", "windows"):
            for e, p0, p1 in r.door_segments(kind):
                openings.append({"id": f"{rid}.O{len(openings) + 1}", "type": kind[:-1],
                                 "wall": f"{rid}.W{e + 1}", "start": _r(p0), "end": _r(p1),
                                 "width_m": em.opening(float(np.linalg.norm(p1 - p0)))})
        perim = float(sum(np.linalg.norm(b - a) for a, b in edges))
        cov = float(np.mean([c for c, _ in support])) if support else 1.0
        out["rooms"].append({
            "id": rid, "name": r.name,
            "floor_area_m2": em.area(r.area, perim, cov),
            "perimeter_m": em.wall(perim, cov),
            "ceiling_height_m": em.height(r.height),
            "vertices": np.round(r.polygon, 4).tolist(),
            "walls": walls, "openings": openings,
        })
    out["adjacency"] = adjacency(out["rooms"])
    total = sum(r["floor_area_m2"]["value"] for r in out["rooms"])
    half = np.sqrt(sum(((r["floor_area_m2"]["hi"] - r["floor_area_m2"]["lo"]) / 2) ** 2 for r in out["rooms"]))
    out["footprint_m2"] = {"value": round(total, 3), "lo": round(total - half, 3), "hi": round(total + half, 3)}
    return out


def adjacency(rooms: list[dict], tol=0.35, through=0.5) -> list[dict]:
    """Rooms are connected where doors of both line up (the two faces of one doorway), or where a
    door opens onto another room's outline within `through` (a wall's thickness plus a margin):
    the other side's jambs are not always seen, a hallway's often are not."""
    doors = [(r["id"], o) for r in rooms for o in r["openings"] if o["type"] == "door"]
    links, seen = [], set()
    for i, (ra, a) in enumerate(doors):
        ca = (np.array(a["start"]) + np.array(a["end"])) / 2
        for rb, b in doors[i + 1:]:
            if ra == rb or (ra, rb) in seen:
                continue
            cb = (np.array(b["start"]) + np.array(b["end"])) / 2
            if np.linalg.norm(ca - cb) < tol:
                links.append({"rooms": [ra, rb], "via": [a["id"], b["id"]]})
                seen |= {(ra, rb), (rb, ra)}
    for ra, a in doors:
        ca = (np.array(a["start"]) + np.array(a["end"])) / 2
        for r in rooms:
            rb = r["id"]
            if rb == ra or (ra, rb) in seen:
                continue
            if _distance_to_outline(ca, np.array(r["vertices"])) < through:
                links.append({"rooms": [ra, rb], "via": [a["id"]]})
                seen |= {(ra, rb), (rb, ra)}
    return links


def _distance_to_outline(p, poly):
    a, b = poly, np.roll(poly, -1, axis=0)
    d = b - a
    t = np.clip(np.einsum("ij,ij->i", p - a, d) / np.maximum(np.einsum("ij,ij->i", d, d), 1e-12), 0, 1)
    return float(np.min(np.linalg.norm(a + t[:, None] * d - p, axis=1)))


def _r(p):
    return np.round(p, 4).tolist()


def write_json(data: dict, path: Path):
    Path(path).write_text(json.dumps(data, indent=2))


def write_svg(rooms: list[Room], path: Path, title="Floor plan", px_per_m=110.0, data: dict | None = None):
    """`data`: the to_dict() output, for the intervals on dimensions and for damage marks."""
    pts = np.concatenate([r.polygon for r in rooms])
    lo, hi = pts.min(0), pts.max(0)
    pad = 0.9
    W = (hi[0] - lo[0] + 2 * pad) * px_per_m
    H = (hi[1] - lo[1] + 2 * pad) * px_per_m + 40

    def P(p):  # world (y up) -> svg (y down)
        return ((p[0] - lo[0] + pad) * px_per_m, (hi[1] - p[1] + pad) * px_per_m + 40)

    el = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W:.0f}" height="{H:.0f}" '
          f'viewBox="0 0 {W:.0f} {H:.0f}" font-family="Helvetica, Arial, sans-serif">',
          '<rect width="100%" height="100%" fill="#fbfaf7"/>',
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
        for e, p0, p1 in r.door_segments("windows"):  # window: thin double line across the gap
            (x0, y0), (x1, y1) = P(p0), P(p1)
            el.append(f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" '
                      f'stroke="#fbfaf7" stroke-width="7"/>')
            el.append(f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" '
                      f'stroke="#2f80c2" stroke-width="4"/>')
            el.append(f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" '
                      f'stroke="#fbfaf7" stroke-width="1.5"/>')
    for k, r in enumerate(rooms):
        dr = data["rooms"][k] if data else None
        for i, (a, b) in enumerate(r.edges()):
            pm = None
            if dr:
                m = dr["walls"][i]["length_m"]
                pm = (m["hi"] - m["lo"]) / 2
            el += _dimension(a, b, P, px_per_m, pm=pm)
        c = _label_point(r.polygon)
        cx, cy = P(c)
        h = f" · h {r.height:.2f} m" if r.height else ""
        el.append(f'<text x="{cx:.1f}" y="{cy - 4:.1f}" text-anchor="middle" font-size="15" '
                  f'font-weight="600" fill="#1f2933">{r.name}</text>')
        el.append(f'<text x="{cx:.1f}" y="{cy + 14:.1f}" text-anchor="middle" font-size="12" '
                  f'fill="#52606d">{r.area:.2f} m²{h}</text>')
    for d in (data or {}).get("damage", []):  # damage: a numbered marker at the region
        cx, cy = P(np.array(d["center"][:2]))
        el.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="9" fill="#d64545" opacity="0.9"/>')
        el.append(f'<text x="{cx:.1f}" y="{cy + 4:.1f}" text-anchor="middle" font-size="10" '
                  f'font-weight="700" fill="#fff">{d["id"]}</text>')
    x0, y0 = 16, H - 16  # 1 m scale bar
    el.append(f'<line x1="{x0}" y1="{y0}" x2="{x0 + px_per_m:.0f}" y2="{y0}" stroke="#222" stroke-width="3"/>')
    el.append(f'<text x="{x0 + px_per_m + 8:.0f}" y="{y0 + 4}" font-size="12" fill="#222">1 m</text>')
    el.append("</svg>")
    Path(path).write_text("\n".join(el))


def _dimension(a, b, P, px_per_m, offset=0.28, pm=None):
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
            f'dominant-baseline="middle" transform="rotate({ang:.1f} {mx:.1f} {my:.1f})">{L * 100:.1f}{"" if pm is None else f" ±{pm * 100:.1f}"} cm</text>']


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
