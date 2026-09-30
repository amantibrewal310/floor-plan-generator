"""Accuracy benchmark: every tier on synthetic scenes with known ground truth."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import synth
from .pipeline import run

# (case name, scene, tier, how to create inputs)
CASES = [
    ("lidar/rect", "rect", "lidar", lambda sc, d: [synth.write_lidar_ply(sc, d / "scan.ply", seed=1)]),
    ("lidar/apartment (2 scans, door stitch)", "apartment", "lidar",
     lambda sc, d: [synth.write_lidar_ply(sc, d / f"scan_{i}.ply", rooms=[i], seed=i) for i in (0, 1)]),
    ("roomplan/lshape", "lshape", "lidar", lambda sc, d: [synth.write_roomplan_json(sc, d / "room.json")]),
    ("photos/rect", "rect", "photos", lambda sc, d: [synth.write_photos(sc, d / "photos", 0)]),
    ("photos/apartment (2 captures)", "apartment", "photos",
     lambda sc, d: [synth.write_photos(sc, d / f"photos_{i}", i) for i in (0, 1)]),
    ("video/lshape", "lshape", "video", lambda sc, d: [synth.write_video(sc, d / "walk.mp4")]),
    ("video/apartment (1 walkthrough)", "apartment", "video", lambda sc, d: [synth.write_video(sc, d / "walk.mp4")]),
]


def _rot90(p, k):
    for _ in range(k):
        p = np.column_stack([-p[:, 1], p[:, 0]])
    return p


def _rigid(src, dst):
    ms, md = src.mean(0), dst.mean(0)
    U, _, Vt = np.linalg.svd((src - ms).T @ (dst - md))
    R = (U @ Vt).T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = (U @ Vt).T
    return R, md - R @ ms


def evaluate(result: dict, gt: dict) -> dict:
    """Compare a plan with ground truth after the best rigid alignment."""
    gt_rooms = [np.array(r) for r in gt["rooms"]]
    pred_rooms = [np.array(r["vertices"]) for r in result["rooms"]]
    G = np.concatenate(gt_rooms)
    gt_doors = [np.array(d["center"]) for d in gt["doors"]]
    best = None
    for k in range(4):  # the plan's orientation is only defined up to 90 degrees
        P = [_rot90(p, k) for p in pred_rooms]
        allp = np.concatenate(P)
        P = [p - allp.min(0) + G.min(0) for p in P]
        allp = np.concatenate(P)
        for _ in range(3):  # ICP on vertices
            idx = np.argmin(np.linalg.norm(G[:, None] - allp[None], axis=2), axis=1)
            R, t = _rigid(allp[idx], G)
            P = [p @ R.T + t for p in P]
            allp = np.concatenate(P)
        d = np.linalg.norm(G[:, None] - allp[None], axis=2)
        idx = np.argmin(d, axis=1)
        err = d[np.arange(len(G)), idx]
        doors = []  # predicted door centres in the aligned frame
        for r, p in zip(result["rooms"], P):
            Rr, tr = _rigid(np.array(r["vertices"]), p)
            for dd in r["openings"]:
                if dd["type"] != "door":
                    continue
                c = (np.array(dd["start"]) + np.array(dd["end"])) / 2
                doors.append((Rr @ c + tr, dd["width_m"]["value"]))
        # symmetric rooms fit equally well in several orientations: let the doors decide
        miss = np.mean([min([np.linalg.norm(c - g) for c, _ in doors] + [1.0]) for g in gt_doors]) if gt_doors else 0
        score = np.median(err) + miss
        if best is None or score < best[0]:
            best = (score, P, err, idx, allp, doors)
    _, P, err, idx, allp, pred_doors = best
    matched = err < 0.25

    walls, off = [], 0
    for g in gt_rooms:
        n = len(g)
        for i in range(n):
            a, b = off + i, off + (i + 1) % n
            if matched[a] and matched[b]:
                walls.append(abs(np.linalg.norm(allp[idx[a]] - allp[idx[b]]) - np.linalg.norm(g[i] - g[(i + 1) % n])))
        off += n
    n_walls = sum(len(g) for g in gt_rooms)

    areas = []
    for g in gt_rooms:
        ga = abs(_area(g))
        c = g.mean(0)
        j = int(np.argmin([np.linalg.norm(p.mean(0) - c) for p in P]))
        areas.append(abs(abs(_area(P[j])) - ga) / ga)

    door_err = []
    for dg in gt["doors"]:
        if not pred_doors:
            break
        dists = [np.linalg.norm(c - dg["center"]) for c, _ in pred_doors]
        j = int(np.argmin(dists))
        if dists[j] < 0.3:
            door_err.append(abs(pred_doors[j][1] - dg["width"]))
    return {
        "rooms": f"{len(pred_rooms)}/{len(gt_rooms)}",
        "walls_matched": f"{len(walls)}/{n_walls}",
        "wall_len_mae_cm": 100 * float(np.mean(walls)) if walls else None,
        "wall_len_max_cm": 100 * float(np.max(walls)) if walls else None,
        "corner_rmse_cm": 100 * float(np.sqrt(np.mean(err[matched] ** 2))) if matched.any() else None,
        "area_err_pct": 100 * float(np.mean(areas)),
        "doors": f"{len(door_err)}/{len(gt['doors'])}",
        "door_width_mae_cm": 100 * float(np.mean(door_err)) if door_err else None,
    }


def _area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def run_benchmark(out: Path, quick=False, case: str | None = None):
    out = Path(out)
    rows = []
    for name, scene, tier, make in CASES:
        if (quick and tier != "lidar") or (case and case not in name):
            continue
        sc = synth.make_scene(scene)
        d = out / name.split(" ")[0].replace("/", "_")
        d.mkdir(parents=True, exist_ok=True)
        print(f"[{name}] generating synthetic capture...", flush=True)
        inputs = make(sc, d)
        t = time.time()
        try:
            result = run(tier, inputs, d / "result", marker_size=sc.marker_size)
            m = evaluate(result, sc.ground_truth())
            m["stitching"] = "; ".join(result["stitching"][1:]) or "-"
        except Exception as e:  # report and keep going
            m = {"error": f"{type(e).__name__}: {e}"}
        m["case"], m["seconds"] = name, round(time.time() - t, 1)
        print(f"[{name}] {m}", flush=True)
        rows.append(m)
    (out / "results.json").write_text(json.dumps(rows, indent=2))
    table = _table(rows)
    (out / "results.md").write_text(table)
    print("\n" + table)
    return rows


def _table(rows):
    cols = [("case", "case"), ("rooms", "rooms"), ("walls_matched", "walls"),
            ("wall_len_mae_cm", "wall MAE cm"), ("wall_len_max_cm", "wall max cm"),
            ("corner_rmse_cm", "corner RMSE cm"), ("area_err_pct", "area err %"),
            ("doors", "doors"), ("door_width_mae_cm", "door MAE cm"), ("seconds", "time s")]
    lines = ["| " + " | ".join(h for _, h in cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        if "error" in r:
            lines.append(f"| {r['case']} | FAILED: {r['error']} |" + " |" * (len(cols) - 2))
            continue
        cells = []
        for k, _ in cols:
            v = r.get(k)
            cells.append("-" if v is None else f"{v:.2f}" if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
