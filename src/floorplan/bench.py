"""Real-capture benchmark: regenerate every reported number from raw inputs.

`floorplan bench benchmark/manifest.json` runs each capture listed in the manifest, scores it
against its ground truth and writes benchmark/out/report.md plus the raw JSON. Paths in the
manifest are relative to the manifest file.

    {"captures": [
      {"id": "flat-lidar", "tier": "lidar", "inputs": ["raw/flat/lidar"],
       "truth": "raw/flat/truth.json", "ablate_drift": true},
      {"id": "flat-photos", "tier": "photos", "inputs": ["raw/flat/photos/kitchen", "..."],
       "truth": "raw/flat/truth.json"},
      {"id": "bed-lidar-2", "tier": "lidar", "inputs": ["raw/bed/lidar_2"],
       "truth": "raw/bed/truth.json", "repeat_of": "bed-lidar-1"},
      {"id": "bed-lidar-1", "...": "...", "competitor": {"app": "Polycam 4.1 (free)",
                                                          "measured": "raw/bed/polycam.json"}}
    ]}

A competitor's numbers are written down from its export in the same format as ground truth.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .evaluate import load, repeatability, score, to_json
from .pipeline import run


def run_bench(manifest: Path, out: Path | None = None, damage=True) -> dict:
    manifest = Path(manifest)
    root = manifest.parent
    out = Path(out) if out else root / "out"
    spec = json.loads(manifest.read_text())
    results = {}
    for c in spec["captures"]:
        d = out / c["id"]
        t = time.time()
        try:
            plan = run(c["tier"], [root / p for p in c["inputs"]], d, find_damage=damage)
        except ValueError as e:  # e.g. no enclosed room: a result, not a reason to stop the run
            results[c["id"]] = {"id": c["id"], "tier": c["tier"], "seconds": round(time.time() - t, 1),
                                "error": str(e)}
            print(f"[{c['id']}] failed: {e}", flush=True)
            continue
        r = {"id": c["id"], "tier": c["tier"], "seconds": round(time.time() - t, 1), "plan": plan}
        if c.get("truth"):
            r["score"] = score(plan, load(root / c["truth"]))
            (d / "score.json").write_text(to_json(r["score"]))
        if c.get("ablate_drift"):
            off = run(c["tier"], [root / p for p in c["inputs"]], d / "drift_off", drift_correction=False,
                      find_damage=False)
            r["drift_off"] = {"plan": off, "score": score(off, load(root / c["truth"])) if c.get("truth") else None}
        if c.get("competitor"):
            r["competitor"] = head_to_head(plan, load(root / c["competitor"]["measured"]),
                                           load(root / c["truth"]), c["competitor"]["app"])
        results[c["id"]] = r
        print(f"[{c['id']}] {r['seconds']} s", flush=True)
    for c in spec["captures"]:
        if c.get("repeat_of") and "plan" in results.get(c["repeat_of"], {}) and "plan" in results[c["id"]]:
            results[c["id"]]["repeat"] = repeatability(results[c["repeat_of"]]["plan"], results[c["id"]]["plan"])
    (out / "results.json").write_text(to_json(results))
    report = _report(results)
    (out / "report.md").write_text(report)
    print(report)
    return results


def head_to_head(plan: dict, theirs: dict, truth: dict, app: str) -> dict:
    """Dimension by dimension: our error and the app's against the same ground truth."""
    ours = score(plan, truth)
    # the competitor's numbers, in ground-truth format, scored as if they were a plan
    as_plan = {"tier": plan["tier"], "footprint_m2": {"value": 0, "lo": 0, "hi": 0}, "rooms": [
        {"name": r["name"], "floor_area_m2": _m(r.get("area_m2") or 0),
         "ceiling_height_m": _m(r["ceiling_m"]) if r.get("ceiling_m") else None,
         "walls": [{"id": f"W{i + 1}", "length_m": _m(w)} for i, w in enumerate(r.get("walls_m", []))],
         "openings": [{"type": o["type"], "width_m": _m(o["width_m"])} for o in r.get("openings", [])]}
        for r in theirs["rooms"]]}
    them = score(as_plan, truth)
    rows = []
    for kind in ("walls", "ceilings", "openings"):
        # both lists follow the ground truth's order; phantoms (truth None) are appended, so drop them
        pairs = zip([x for x in ours[kind] if x.get("truth") is not None],
                    [x for x in them[kind] if x.get("truth") is not None])
        for a, b in pairs:
            if a.get("err") is None or b.get("err") is None:
                continue
            ea, eb = abs(a["err"]), abs(b["err"])
            rows.append({"dimension": f"{a['room']} {kind[:-1]}", "truth": a["truth"],
                         "ours_err_cm": round(100 * ea, 2), "theirs_err_cm": round(100 * eb, 2),
                         "beat_or_tie": ea <= eb + 0.005})  # tie: within 5 mm
    n = len(rows)
    return {"app": app, "rows": rows,
            "beat_or_tie_pct": round(100 * sum(r["beat_or_tie"] for r in rows) / n, 1) if n else None}


def _m(v):
    return {"value": v, "lo": v, "hi": v}


def _report(results) -> str:
    L = ["# Benchmark report", "", "Generated by `floorplan bench`. Errors in cm unless noted; "
         "intervals are 90%.", "", "## Gates per capture", "",
         "| capture | tier | rooms | walls pass | wall MAE | wall max | openings ≤2 cm | ceiling MAE | "
         "ceiling bias | footprint err % | interval coverage % | time s |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    L[4:4] = _overall(results)
    for r in results.values():
        s = r.get("score", {}).get("summary")
        if not s:
            continue
        fp = s["footprint"]["err_pct"] if s["footprint"] else None
        L.append(f"| {r['id']} | {r['tier']} | {s['rooms_found']} | {s['walls_pass']} | {s['wall_mae_cm']} | "
                 f"{s['wall_max_cm']} | {s['openings_pass_pct']} | {s['ceiling_mae_cm']} | {s['ceiling_bias_cm']} | "
                 f"{'-' if fp is None else round(fp, 2)} | {s['interval_coverage_pct']} | {r['seconds']} |")
    failed = [r for r in results.values() if "error" in r]
    if failed:
        L += ["", "## Failed captures", "", "| capture | tier | error |", "|---|---|---|"]
        L += [f"| {r['id']} | {r['tier']} | {r['error']} |" for r in failed]
    rep = [r for r in results.values() if "repeat" in r]
    if rep:
        L += ["", "## Repeatability (same room, same tier)", "", "| capture | room | surface | a cm | b cm | diff cm | pass |",
              "|---|---|---|---|---|---|---|"]
        for r in rep:
            for row in r["repeat"]["rows"]:
                L.append(f"| {r['id']} | {row['room']} | {row['wall']} | {row['a'] * 100:.1f} | {row['b'] * 100:.1f} | "
                         f"{row['diff_cm']} | {'yes' if row['pass'] else 'NO'} |")
    abl = [r for r in results.values() if "drift_off" in r]
    if abl:
        L += ["", "## Drift ablation (stitched footprint)", "",
              "| capture | drift correction | footprint m² | footprint err % | wall MAE cm | walls pass |",
              "|---|---|---|---|---|---|"]
        for r in abl:
            for label, plan, sc in (("on", r["plan"], r.get("score")), ("off", r["drift_off"]["plan"], r["drift_off"]["score"])):
                s = sc["summary"] if sc else {}
                fp = s.get("footprint")
                L.append(f"| {r['id']} | {label} | {plan['footprint_m2']['value']:.2f} | "
                         f"{'-' if not fp else round(fp['err_pct'], 2)} | {s.get('wall_mae_cm')} | {s.get('walls_pass')} |")
    h2h = [r for r in results.values() if "competitor" in r]
    if h2h:
        for r in h2h:
            c = r["competitor"]
            L += ["", f"## Head-to-head: {r['id']} vs {c['app']}", "",
                  f"Beat or tie on {c['beat_or_tie_pct']}% of shared dimensions (gate: 70%).", "",
                  "| dimension | truth cm | our error | their error | beat/tie |", "|---|---|---|---|---|"]
            for row in c["rows"]:
                L.append(f"| {row['dimension']} | {row['truth'] * 100:.1f} | {row['ours_err_cm']} | "
                         f"{row['theirs_err_cm']} | {'yes' if row['beat_or_tie'] else 'no'} |")
    return "\n".join(L) + "\n"


def _overall(results) -> list[str]:
    """One row per tier, pooling every wall and room across captures (a failed capture counts
    its rooms as not found)."""
    L = ["## Overall per tier", "",
         "| tier | captures | failed | walls pass | wall MAE cm | wall median rel err % | "
         "area median abs err % | area bias % | interval coverage % |",
         "|---|---|---|---|---|---|---|---|---|"]
    for tier in sorted({r["tier"] for r in results.values()}):
        rs = [r for r in results.values() if r["tier"] == tier]
        sc = [r["score"] for r in rs if "score" in r]
        walls = [w for s in sc for w in s["walls"] if w.get("truth") is not None]
        errs = [w for w in walls if w.get("err") is not None]
        areas = [a["err_pct"] for s in sc for a in s["areas"]]
        cov = [s["summary"]["interval_coverage_pct"] for s in sc if s["summary"]["interval_coverage_pct"] is not None]
        med = lambda v: round(float(np.median(v)), 1) if v else None
        L.append(f"| {tier} | {len(rs)} | {sum('error' in r for r in rs)} | "
                 f"{sum(w['pass'] for w in walls)}/{len(walls)} | "
                 f"{round(100 * float(np.mean([abs(w['err']) for w in errs])), 1) if errs else None} | "
                 f"{med([100 * abs(w['err']) / w['truth'] for w in errs])} | {med([abs(a) for a in areas])} | "
                 f"{med(areas)} | {round(float(np.mean(cov)), 1) if cov else None} |")
    return L + [""]
