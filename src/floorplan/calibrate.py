"""Calibrate the 90% intervals on a benchmark: `floorplan calibrate benchmark/out [--write]`.

For every measurement with ground truth (wall, room area, opening width, ceiling height) take
the error in units of the interval's half-width, as the prior model alone would have drawn it
(any factor already applied is divided out). The factor for a tier and quantity is the 90%
quantile of those ratios, with the finite-sample correction of split conformal prediction:
with n measurements, the ceil((n + 1) * 0.9)-th smallest. Scaling the prior's sigma by it puts
at least 90% of the benchmark's truth inside the interval.

A factor below 1 narrows the intervals. That needs evidence: with fewer than MIN_TO_NARROW
measurements a tier is only ever widened.
"""

from __future__ import annotations

import datetime
import json
import math
from collections import defaultdict
from pathlib import Path

from . import uncertainty
from .uncertainty import QUANTITIES

TARGET = 0.9
MIN_TO_NARROW = 20


def residuals(results: dict) -> dict:
    """{(tier, quantity): [|error| / prior half-width, ...]} from a `floorplan bench` results.json."""
    out = defaultdict(list)
    for r in results.values():
        if "plan" not in r or "score" not in r:
            continue
        plan, sc = r["plan"], r["score"]
        tier, used = plan["tier"], plan.get("interval_calibration", {})
        rooms = {room["id"]: room for room in plan["rooms"]}
        walls = {w["id"]: w["length_m"] for room in plan["rooms"] for w in room["walls"]}
        openings = {o["id"]: o["width_m"] for room in plan["rooms"] for o in room["openings"]}

        def add(q, m, truth):
            half = (m["hi"] - m["lo"]) / 2 if m else 0
            if half > 0 and truth:
                out[(tier, q)].append(abs(m["value"] - truth) / half * used.get(q, 1.0))

        for w in sc["walls"]:
            add("wall", walls.get(w.get("wall")), w.get("truth"))
        for a in sc["areas"]:
            add("area", rooms.get(a.get("id"), {}).get("floor_area_m2"), a["truth"])
        for c in sc["ceilings"]:
            add("height", rooms.get(c.get("id"), {}).get("ceiling_height_m"), c["truth"])
        for o in sc["openings"]:
            add("opening", openings.get(o.get("id")), o.get("truth"))
    return out


def fit(ratios: list[float]) -> float:
    n = len(ratios)
    k = min(n, math.ceil((n + 1) * TARGET))
    f = sorted(ratios)[k - 1]
    return f if n >= MIN_TO_NARROW else max(f, 1.0)


def calibrate(sources: list[Path], write=False) -> dict:
    res = defaultdict(list)
    for src in sources:
        path = Path(src) / "results.json" if Path(src).is_dir() else Path(src)
        for key, v in residuals(json.loads(path.read_text())).items():
            res[key] += v
    cal = {"target_coverage": TARGET, "sources": [str(s) for s in sources],
           "fitted": datetime.date.today().isoformat(), "tiers": {}, "n": {}}
    rows = ["| tier | quantity | n | inside the prior interval | factor | inside after |", "|---|---|---|---|---|---|"]
    for (tier, q), v in sorted(res.items()):
        f = fit(v)
        cal["tiers"].setdefault(tier, {})[q] = round(f, 3)
        cal["n"].setdefault(tier, {})[q] = len(v)
        inside = lambda s: f"{100 * sum(x <= s for x in v) / len(v):.0f}%"
        rows.append(f"| {tier} | {q} | {len(v)} | {inside(1.0)} | {f:.2f} | {inside(f)} |")
    missing = [f"{t}/{q}" for t in {t for t, _ in res} for q in QUANTITIES if (t, q) not in res]
    if missing:
        rows.append(f"\nno ground truth for {', '.join(sorted(missing))}: those stay at the prior (factor 1)")
    print("\n".join(rows))
    if write:
        uncertainty.CALIBRATION.write_text(json.dumps(cal, indent=1) + "\n")
        print(f"\nwrote {uncertainty.CALIBRATION}")
    return cal
