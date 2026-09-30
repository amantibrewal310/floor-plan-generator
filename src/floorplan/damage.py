"""Damage regions, concealed-damage flags and scope line items, all keyed to surfaces.

1. Detect: OWLv2 (Google, Apache-2.0, open vocabulary) finds candidate regions in the
   capture's images from text prompts, one set of prompts per damage class.
2. Locate: every detection box selects the 3D points seen through it (each point remembers the
   pixel it came from). The points vote for a surface of the room (a wall, floor or ceiling)
   and give the region's metric extent on that surface.
3. Merge: the same stain seen from several views is one region.
4. Rules: concealed-damage flags (what the visible damage implies behind the surface) and
   scope line items (what to do about it), both as small explicit tables below.
"""

from __future__ import annotations

import numpy as np

from .uncertainty import TIERS, interval

DETECTOR_ID = "google/owlv2-base-patch16-ensemble"
PROMPTS = {
    "water_stain": ["a brown water stain on a ceiling", "a water stain on a wall", "water damage"],
    "mold": ["black mold on a wall", "mold spots"],
    "crack": ["a crack in a wall", "a crack in plaster"],
    "peeling_paint": ["peeling paint", "bubbling paint"],
    "hole": ["a hole in drywall", "a hole in a wall"],
}
MIN_SCORE = 0.30
_DET = None


def load_detector():
    import torch
    from transformers import Owlv2ForObjectDetection, Owlv2Processor

    global _DET
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if _DET is None:
        _DET = (Owlv2Processor.from_pretrained(DETECTOR_ID),
                Owlv2ForObjectDetection.from_pretrained(DETECTOR_ID).to(dev).eval(), dev)
    return _DET


def detect(images: list[np.ndarray], min_score=MIN_SCORE) -> list[list[tuple[str, float, np.ndarray]]]:
    """Per image: [(class, score, box (x0, y0, x1, y1) normalised to 0..1)]."""
    import torch

    proc, model, dev = load_detector()
    labels = [(c, p) for c, ps in PROMPTS.items() for p in ps]
    out = []
    for img in images:
        inp = proc(text=[[p for _, p in labels]], images=img, return_tensors="pt").to(dev)
        with torch.no_grad():
            res = model(**inp)
        # OWLv2 pads to a square: boxes are relative to the padded side
        side = max(img.shape[:2])
        r = proc.post_process_grounded_object_detection(res, threshold=min_score,
                                                        target_sizes=[(side, side)])[0]
        dets = []
        for s, lab, box in zip(r["scores"].tolist(), r["labels"].tolist(), r["boxes"].cpu().numpy()):
            b = np.clip(box / [img.shape[1], img.shape[0], img.shape[1], img.shape[0]], 0, 1)
            if (b[2] - b[0]) * (b[3] - b[1]) > 0.8:  # "the whole picture is damage": not a region
                continue
            dets.append((labels[lab][0], float(s), b))
        out.append(_nms(dets))
    return out


def _nms(dets, iou=0.5, contain=0.7):
    """Drop boxes that overlap a stronger one, or mostly contain / sit inside one (the same
    stain found again by another prompt, with a looser box)."""
    dets = sorted(dets, key=lambda d: -d[1])
    keep = []
    for d in dets:
        if all(_iou(d[2], k[2]) < iou and _inside(d[2], k[2]) < contain and _inside(k[2], d[2]) < contain
               for k in keep):
            keep.append(d)
    return keep


def _inside(a, b):
    """Fraction of box a that lies inside box b."""
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    return max(0, x1 - x0) * max(0, y1 - y0) / ((a[2] - a[0]) * (a[3] - a[1]) + 1e-12)


def _iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-12)


# ---------------------------------------------------------------- locate on surfaces

def locate(dets, frames, rooms, view_names) -> None:
    """Attach detections to the rooms' surfaces (room.extra["damage"]).

    frames: per view (points (N,3), camera, uv (N,2) in 0..1) in the rooms' frame, floor at z=0."""
    for (P, _, uv), vd, name in zip(frames, dets, view_names):
        for cls, score, box in vd:
            sel = (uv[:, 0] >= box[0]) & (uv[:, 0] <= box[2]) & (uv[:, 1] >= box[1]) & (uv[:, 1] <= box[3])
            X = P[sel]
            if len(X) < 20:
                continue
            hit = _surface(X, rooms)
            if hit is None:
                continue
            room, surf, coords = hit
            lo, hi = np.percentile(coords, [2, 98], axis=0)
            region = {"class": cls, "score": score, "surface": surf, "lo": lo, "hi": hi,
                      "center": np.median(X, axis=0), "views": [name]}
            _merge(room.extra.setdefault("damage", []), region)


def _surface(X, rooms, tol=0.12):
    """(room, surface, 2-D coordinates on it) for the surface most of X lies on."""
    best = None
    z = X[:, 2]
    for room in rooms:
        H = room.height or 2.5
        cands = [("floor", np.abs(z) < tol, X[:, :2]), ("ceiling", np.abs(z - H) < tol, X[:, :2])]
        for e, (a, b) in enumerate(room.edges()):
            L = np.linalg.norm(b - a)
            u = (b - a) / L
            rel = X[:, :2] - a
            along, perp = rel @ u, rel @ np.array([u[1], -u[0]])
            on = (np.abs(perp) < tol) & (along > -0.05) & (along < L + 0.05)
            cands.append((e, on, np.column_stack([along, z])))
        for surf, on, coords in cands:
            n = int(on.sum())
            if n >= 0.5 * len(X) and (best is None or n > best[0]):
                best = (n, room, surf, coords[on])
    return None if best is None else best[1:]


def _merge(regions, new):
    for r in regions:
        if r["class"] != new["class"] or r["surface"] != new["surface"]:
            continue
        overlap = np.minimum(r["hi"], new["hi"]) - np.maximum(r["lo"], new["lo"])
        if (overlap > -0.1).all():  # same region seen again: keep a robust average of the extents
            n = len(r["views"])
            r["lo"] = (r["lo"] * n + new["lo"]) / (n + 1)
            r["hi"] = (r["hi"] * n + new["hi"]) / (n + 1)
            r["score"] = max(r["score"], new["score"])
            r["views"] = r["views"] + new["views"]
            return
    regions.append(new)


# ---------------------------------------------------------------- output: regions, flags, scope

def regions(rooms, data, tier) -> None:
    """Fill data["damage"], data["concealed_flags"] and data["scope"] from room.extra."""
    p = TIERS[tier]
    for room, dr in zip(rooms, data["rooms"]):
        for reg in room.extra.get("damage", []):
            w, h = np.maximum(reg["hi"] - reg["lo"], 0.02)
            surf = f"{dr['id']}.W{reg['surface'] + 1}" if isinstance(reg["surface"], int) \
                else f"{dr['id']}.{reg['surface']}"
            # box edges from a detector are good to ~10% of the region, plus the tier's error
            sw, sh = np.hypot(0.1 * w, p["opening"]), np.hypot(0.1 * h, p["opening"])
            data["damage"].append({
                "id": f"D{len(data['damage']) + 1}", "surface": surf, "class": reg["class"],
                "score": round(reg["score"], 3),
                "width_m": interval(float(w), sw, 3), "height_m": interval(float(h), sh, 3),
                "area_m2": interval(float(w * h), float(np.hypot(w * sh, h * sw)), 3),
                "center": np.round(reg["center"], 3).tolist(), "views": reg["views"],
                "_room": dr, "_lo": reg["lo"], "_hi": reg["hi"],
            })
    data["concealed_flags"] = concealed_flags(data)
    data["scope"] = scope(data)
    for d in data["damage"]:
        for k in ("_room", "_lo", "_hi"):
            d.pop(k)


# Each rule: (name, test(damage, all damage in the room) -> bool, reason).
RULES = [
    ("R1 ceiling water stain",
     lambda d, same: d["class"] in ("water_stain", "mold") and d["surface"].endswith("ceiling"),
     "water on a ceiling comes from above: roof, plumbing or a wet room overhead; the ceiling "
     "cavity is likely wet beyond the visible stain"),
    ("R2 damp at wall base",
     lambda d, same: d["class"] in ("water_stain", "mold", "peeling_paint") and ".W" in d["surface"]
     and d["_lo"][1] < 0.3,
     "staining or peeling within 30 cm of the floor points to rising damp or a leak inside the "
     "wall: check the cavity, skirting and subfloor"),
    ("R3 mould next to an opening",
     lambda d, same: d["class"] in ("mold", "water_stain") and _near_opening(d, 0.4),
     "moisture beside a window or door frame suggests a failed seal or flashing; water may be "
     "tracking inside the wall"),
    ("R4 crack from an opening corner",
     lambda d, same: d["class"] == "crack" and _near_opening(d, 0.3),
     "cracks running from door or window corners are a classic sign of structural movement "
     "(settlement, a failing lintel)"),
    ("R5 repeated moisture in one room",
     lambda d, same: d["class"] in ("water_stain", "mold")
     and sum(o["class"] in ("water_stain", "mold") for o in same) >= 2,
     "several separate moisture regions in one room suggest an active source rather than an "
     "old one-off event"),
]


def _near_opening(d, dist):
    if ".W" not in d["surface"]:
        return False
    room = d["_room"]
    for o in room["openings"]:
        if o["wall"] != d["surface"]:
            continue
        w = next(w for w in room["walls"] if w["id"] == o["wall"])
        a = np.array(w["start"])
        s0 = float(np.linalg.norm(np.array(o["start"]) - a))
        s1 = float(np.linalg.norm(np.array(o["end"]) - a))
        s0, s1 = min(s0, s1), max(s0, s1)
        if d["_lo"][0] < s1 + dist and d["_hi"][0] > s0 - dist:
            return True
    return False


def concealed_flags(data) -> list[dict]:
    flags = []
    for d in data["damage"]:
        same = [o for o in data["damage"] if o["_room"] is d["_room"]]
        for name, test, reason in RULES:
            if not test(d, same):
                continue
            if any(f["rule"] == name and f["surface"] == d["surface"] for f in flags):
                next(f for f in flags if f["rule"] == name and f["surface"] == d["surface"])["evidence"].append(d["id"])
                continue
            flags.append({"id": f"C{len(flags) + 1}", "surface": d["surface"], "rule": name,
                          "reason": reason, "evidence": [d["id"]]})
    return flags


# class -> [(line item, unit, quantity basis)]
#   region: the damaged area plus a 0.3 m margin all round; surface: the whole surface
#   (a patch never matches old paint); length: the region's longest side; each: one per region
SCOPE = {
    "water_stain": [("Stain-block primer on affected area", "m2", "region"),
                    ("Repaint whole surface", "m2", "surface")],
    "mold": [("Mould remediation: clean, biocide treat and dry", "m2", "region"),
             ("Repaint whole surface with mould-resistant paint", "m2", "surface")],
    "crack": [("Rake out, tape and fill crack", "m", "length"), ("Repaint whole surface", "m2", "surface")],
    "peeling_paint": [("Scrape, sand and prime", "m2", "region"), ("Repaint whole surface", "m2", "surface")],
    "hole": [("Patch drywall hole", "each", "each"), ("Repaint whole surface", "m2", "surface")],
}


def scope(data) -> list[dict]:
    items = {}
    for d in data["damage"]:
        for item, unit, basis in SCOPE[d["class"]]:
            w, h = d["width_m"], d["height_m"]
            if basis == "region":
                q = interval((w["value"] + 0.6) * (h["value"] + 0.6),
                             (w["hi"] - w["lo"] + h["hi"] - h["lo"]) / 2 / 1.645 * 2, 2)
            elif basis == "length":
                v = max(w["value"], h["value"])
                q = interval(v, (max(w["hi"], h["hi"]) - v) / 1.645, 2)
            elif basis == "each":
                q = {"value": 1, "lo": 1, "hi": 1}
            else:
                q = _surface_area(d["_room"], d["surface"])
            key = (d["surface"], item)
            if key in items:  # one line per surface and task
                it = items[key]
                it["from"].append(d["id"])
                if basis in ("region", "length", "each"):
                    for k in ("value", "lo", "hi"):
                        it["quantity"][k] = round(it["quantity"][k] + q[k], 2)
                continue
            items[key] = {"surface": d["surface"], "item": item, "quantity": q, "unit": unit, "from": [d["id"]]}
    out = []
    for it in items.values():
        out.append({"id": f"S{len(out) + 1}", **it})
    for f in data.get("concealed_flags", []):
        out.append({"id": f"S{len(out) + 1}", "surface": f["surface"],
                    "item": f"Investigate ({f['rule']}): moisture meter / opening-up survey",
                    "quantity": {"value": 1, "lo": 1, "hi": 1}, "unit": "each", "from": [f["id"]]})
    return out


def _surface_area(room, surface) -> dict:
    h = room["ceiling_height_m"] or {"value": 2.5, "lo": 2.3, "hi": 2.7}
    if surface.endswith(("floor", "ceiling")):
        a = room["floor_area_m2"]
        return {k: round(a[k], 2) for k in ("value", "lo", "hi")}
    w = next(w for w in room["walls"] if w["id"] == surface)
    L = w["length_m"]
    return {k: round(L[k] * h[k], 2) for k in ("value", "lo", "hi")}
