"""Score plans against tape/laser ground truth, and against each other (repeatability).

Ground truth is what a person can measure with a tape, no positions needed (see
benchmark/README.md):

    {"rooms": [{"name": "kitchen",
                "walls_m": [3.62, 2.91, 3.61, 2.90],       # going round the room, any start
                "ceiling_m": 2.58,
                "area_m2": 10.5,                             # optional, else from walls if 4
                "openings": [{"type": "door", "width_m": 0.82},
                             {"type": "window", "width_m": 1.21}]}],
     "footprint_m2": 42.0}                                   # optional

`walls_m`, `ceiling_m` and `openings` are each optional: whatever is left out is not scored.
Rooms are matched by name (a photo folder is named after its room), else by area.
Walls are matched by the cyclic order and direction that fits best. Openings are matched by
type and width; a missed or a phantom opening counts as a failure, as the gates require.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Pass/fail thresholds. Relative gates are the spec's; LiDAR walls use the Round 1 style
# absolute gate. Each value is (absolute m, relative): pass if within either.
GATES = {
    "wall": {"lidar": (0.02, 0.0), "video": (0.0, 0.03), "photos": (0.0, 0.08)},
    "opening": (0.02, 0.85),  # within 2 cm, on at least 85% of openings
    "ceiling": 0.015,
    "footprint": 0.08,
    "repeat_wall": (0.01, 0.005),
    "repeat_ceiling": 0.01,
}


def _inside(m, truth):
    return m["lo"] <= truth <= m["hi"]


def _match_walls(pred, gt):
    """Best cyclic shift / direction of the predicted wall list against the ground truth."""
    best = None
    n = len(pred)
    for d in (1, -1):
        seq = pred[::d]
        for k in range(n):
            rot = seq[k:] + seq[:k]
            m = min(n, len(gt))
            err = sum(abs(rot[i]["length_m"]["value"] - gt[i]) for i in range(m))
            if best is None or err < best[0]:
                best = (err, rot)
    return best[1]


def _match_rooms(rooms, gt_rooms):
    by_name = {r["name"].lower(): r for r in rooms}
    out, left = [], list(rooms)
    for g in gt_rooms:
        r = by_name.get(g["name"].lower())
        if r is None and left:  # fall back to the closest area
            ga = g.get("area_m2") or _gt_area(g)
            r = min(left, key=lambda r: abs(r["floor_area_m2"]["value"] - ga)) if ga else None
        if r in left:
            left.remove(r)
        out.append((g, r))
    return out


def _gt_area(g):
    if g.get("area_m2"):
        return g["area_m2"]
    w = g.get("walls_m", [])
    if len(w) == 4:
        return (w[0] + w[2]) / 2 * (w[1] + w[3]) / 2
    return None


def score(plan: dict, gt: dict) -> dict:
    tier = plan["tier"]
    wall_abs, wall_rel = GATES["wall"][tier]
    walls, openings, ceilings, areas, cover = [], [], [], [], []
    missing_rooms = []
    for g, r in _match_rooms(plan["rooms"], gt["rooms"]):
        if r is None:
            missing_rooms.append(g["name"])
            continue
        gw = g.get("walls_m", [])
        if gw:
            pw = _match_walls(r["walls"], gw)
            for i, t in enumerate(gw):
                if i >= len(pw):
                    walls.append({"room": g["name"], "truth": t, "pred": None, "err": None, "pass": False})
                    continue
                m = pw[i]["length_m"]
                e = m["value"] - t
                ok = abs(e) <= max(wall_abs, wall_rel * t)
                walls.append({"room": g["name"], "wall": pw[i]["id"], "truth": t, "pred": m["value"],
                              "err": e, "pass": ok})
                cover.append(_inside(m, t))
            for extra in pw[len(gw):]:  # predicted walls that do not exist
                walls.append({"room": g["name"], "wall": extra["id"], "truth": None,
                              "pred": extra["length_m"]["value"], "err": None, "pass": False})
        if g.get("ceiling_m") is not None:
            m = r["ceiling_height_m"]
            e = None if m is None else m["value"] - g["ceiling_m"]
            ceilings.append({"room": g["name"], "truth": g["ceiling_m"], "pred": m and m["value"], "err": e,
                             "pass": e is not None and abs(e) <= GATES["ceiling"]})
            if m:
                cover.append(_inside(m, g["ceiling_m"]))
        ga = _gt_area(g)
        if ga:
            m = r["floor_area_m2"]
            areas.append({"room": g["name"], "truth": ga, "pred": m["value"], "err_pct": 100 * (m["value"] - ga) / ga})
            cover.append(_inside(m, ga))
        # openings: greedy by width within each type; leftovers on either side are misses.
        # No "openings" key means they were not measured (e.g. sizes read off a floor plan), so none are scored.
        for kind in ("door", "window") if "openings" in g else ():
            gws = sorted(o["width_m"] for o in g.get("openings", []) if o["type"] == kind)
            pws = [o for o in r["openings"] if o["type"] == kind]
            for t in gws:
                if not pws:
                    openings.append({"room": g["name"], "type": kind, "truth": t, "pred": None, "pass": False})
                    continue
                o = min(pws, key=lambda o: abs(o["width_m"]["value"] - t))
                pws.remove(o)
                e = o["width_m"]["value"] - t
                openings.append({"room": g["name"], "type": kind, "truth": t, "pred": o["width_m"]["value"],
                                 "err": e, "pass": abs(e) <= GATES["opening"][0]})
                cover.append(_inside(o["width_m"], t))
            for o in pws:  # phantom
                openings.append({"room": g["name"], "type": kind, "truth": None,
                                 "pred": o["width_m"]["value"], "pass": False})
    fp = None
    if gt.get("footprint_m2") or all(_gt_area(g) for g in gt["rooms"]):
        t = gt.get("footprint_m2") or sum(_gt_area(g) for g in gt["rooms"])
        m = plan["footprint_m2"]
        fp = {"truth": t, "pred": m["value"], "err_pct": 100 * (m["value"] - t) / t,
              "in_interval": _inside(m, t), "pass": abs(m["value"] - t) <= GATES["footprint"] * t}
    n_open = len(openings)
    summary = {
        "tier": tier,
        "rooms_found": f"{len(gt['rooms']) - len(missing_rooms)}/{len(gt['rooms'])}",
        "walls_pass": f"{sum(w['pass'] for w in walls)}/{len(walls)}",
        "wall_mae_cm": _mae([w["err"] for w in walls]),
        "wall_max_cm": _max([w["err"] for w in walls]),
        "openings_pass_pct": round(100 * sum(o["pass"] for o in openings) / n_open, 1) if n_open else None,
        "openings_gate": (sum(o["pass"] for o in openings) / n_open >= GATES["opening"][1]) if n_open else None,
        "ceiling_mae_cm": _mae([c["err"] for c in ceilings]),
        "ceiling_bias_cm": round(100 * float(np.mean([c["err"] for c in ceilings if c["err"] is not None])), 2)
        if any(c["err"] is not None for c in ceilings) else None,
        "ceiling_gate": all(c["pass"] for c in ceilings) if ceilings else None,
        "area_err_pct_mean": round(float(np.mean([abs(a["err_pct"]) for a in areas])), 2) if areas else None,
        "footprint": fp,
        "interval_coverage_pct": round(100 * float(np.mean(cover)), 1) if cover else None,
        "adjacent_pairs": len(plan.get("adjacency", [])),
    }
    return {"summary": summary, "walls": walls, "openings": openings, "ceilings": ceilings,
            "areas": areas, "missing_rooms": missing_rooms}


def walk_check(cap) -> dict:
    """Truth-free check of a walkthrough capture. The phone never leaves the home, so every
    camera position should fall inside some room (hallways included), and the rooms should link
    up through their doorways into one plan. Needs no tape: it runs on any walkthrough."""
    import cv2

    from .export import to_dict
    if cap.path is None or len(cap.path) < 2:
        raise ValueError(f"{cap.source}: not a walkthrough (no camera path); use a Stray folder or a video")
    data = to_dict(cap.rooms)
    C = np.asarray(cap.path)[:, :2]
    polys = [r.polygon.astype(np.float32) for r in cap.rooms]
    inside = np.array([any(cv2.pointPolygonTest(q, (float(x), float(y)), False) >= 0 for q in polys) for x, y in C])
    step = np.r_[0.0, np.linalg.norm(np.diff(C, axis=0), axis=1)]  # metres walked into each frame
    parent = {r["id"]: r["id"] for r in data["rooms"]}

    def root(a):
        while parent[a] != a:
            a = parent[a]
        return a
    for link in data["adjacency"]:
        parent[root(link["rooms"][0])] = root(link["rooms"][1])
    return {"capture": cap.source, "rooms": len(cap.rooms), "walk_m": round(float(step.sum()), 1),
            "walk_inside_pct": round(100 * float(step[inside].sum() / max(step.sum(), 1e-9)), 1),
            "adjacent_pairs": len(data["adjacency"]), "components": len({root(k) for k in parent})}


def repeatability(a: dict, b: dict) -> dict:
    """Two captures of the same room(s) at the same tier: per-wall and ceiling agreement."""
    rows = []
    for ra in a["rooms"]:
        rb = next((r for r in b["rooms"] if r["name"] == ra["name"]), None)
        if rb is None:
            rb = min(b["rooms"], key=lambda r: abs(r["floor_area_m2"]["value"] - ra["floor_area_m2"]["value"]))
        wb = _match_walls(rb["walls"], [w["length_m"]["value"] for w in ra["walls"]])
        for w1, w2 in zip(ra["walls"], wb):
            L1, L2 = w1["length_m"]["value"], w2["length_m"]["value"]
            d = abs(L1 - L2)
            rows.append({"room": ra["name"], "wall": w1["id"], "a": L1, "b": L2, "diff_cm": round(100 * d, 2),
                         "pass": d <= GATES["repeat_wall"][0] or d <= GATES["repeat_wall"][1] * L1})
        ha, hb = ra["ceiling_height_m"], rb["ceiling_height_m"]
        if ha and hb:
            d = abs(ha["value"] - hb["value"])
            rows.append({"room": ra["name"], "wall": "ceiling", "a": ha["value"], "b": hb["value"],
                         "diff_cm": round(100 * d, 2), "pass": d <= GATES["repeat_ceiling"]})
    return {"pass": all(r["pass"] for r in rows), "rows": rows}


def _mae(errs):
    e = [abs(x) for x in errs if x is not None]
    return round(100 * float(np.mean(e)), 2) if e else None


def _max(errs):
    e = [abs(x) for x in errs if x is not None]
    return round(100 * float(np.max(e)), 2) if e else None


def to_json(obj) -> str:
    """json.dumps that accepts numpy scalars."""
    return json.dumps(obj, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def load(path: Path) -> dict:
    return json.loads(Path(path).read_text())
