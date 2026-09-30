import json
import math

import numpy as np
import pytest

from floorplan import synth
from floorplan.benchmark import evaluate
from floorplan.cli import main
from floorplan.pipeline import run


def _check(result, scene, wall_cm, corner_cm, door_cm=3.0):
    m = evaluate(result, scene.ground_truth())
    n_gt = len(scene.rooms)
    assert m["rooms"] == f"{n_gt}/{n_gt}", m
    assert m["walls_matched"].split("/")[0] == m["walls_matched"].split("/")[1], m
    assert m["wall_len_max_cm"] < wall_cm, m
    assert m["corner_rmse_cm"] < corner_cm, m
    assert m["doors"] == f"{len(scene.doors)}/{len(scene.doors)}", m
    assert m["door_width_mae_cm"] < door_cm, m
    return m


@pytest.mark.parametrize("name", synth.SCENES)
def test_lidar_single_scan(tmp_path, name):
    sc = synth.make_scene(name)
    ply = synth.write_lidar_ply(sc, tmp_path / "scan.ply", seed=3)
    _check(run("lidar", [ply], tmp_path / "out"), sc, wall_cm=1.0, corner_cm=1.0)
    assert (tmp_path / "out" / "plan.svg").read_text().startswith("<svg")


def test_lidar_two_scans_stitched_by_doorway(tmp_path):
    sc = synth.make_scene("apartment")
    scans = [synth.write_lidar_ply(sc, tmp_path / f"s{i}.ply", rooms=[i], seed=10 + i) for i in (0, 1)]
    result = run("lidar", scans, tmp_path / "out")
    assert "matching doorway" in result["stitching"][1]
    _check(result, sc, wall_cm=1.0, corner_cm=2.0)


def test_roomplan_json(tmp_path):
    sc = synth.make_scene("lshape")
    js = synth.write_roomplan_json(sc, tmp_path / "room.json")
    _check(run("lidar", [js], tmp_path / "out"), sc, wall_cm=0.5, corner_cm=0.5)


def test_markers_command(tmp_path):
    main(["markers", "-o", str(tmp_path), "--count", "2"])
    assert sorted(p.name for p in tmp_path.iterdir()) == ["marker_0.png", "marker_1.png"]


@pytest.mark.slow
def test_photos_six_stills_with_marker(tmp_path):
    """Photo tier end to end: 6 unposed stills of one room (runs MapAnything, needs weights)."""
    sc = synth.make_scene("rect")
    r = synth.Renderer(sc)
    folder = tmp_path / "living"
    folder.mkdir()
    c = sc.rooms[0].mean(0)
    for i, p in enumerate([(0.5, 0.5), (3.7, 0.5), (3.7, 3.0), (0.5, 3.0), (2.1, 0.4), (2.1, 3.1)]):
        img = r.render(np.array([p[0], p[1], 1.45]), math.atan2(c[1] - p[1], c[0] - p[0]), -0.15)
        synth._save_jpeg_with_focal(img, folder / f"IMG_{i}.jpg", r.focal_35mm)
    result = run("photos", [folder], tmp_path / "out", marker_size=sc.marker_size)
    room = result["rooms"][0]
    assert room["name"] == "living"
    area = room["floor_area_m2"]["value"]
    assert abs(area - 4.2 * 3.5) / (4.2 * 3.5) < 0.2, area


def test_stray_lidar_drift_correction(tmp_path):
    """Raw depth + drifting poses: correction must restore the plan that poses-as-is bend."""
    sc = synth.make_scene("apartment")
    stray = synth.write_stray(sc, tmp_path / "walk", drift_deg_per_m=0.8)
    off = evaluate(run("lidar", [stray], tmp_path / "off", drift_correction=False, find_damage=False),
                   sc.ground_truth())
    on = run("lidar", [stray], tmp_path / "on", find_damage=False)
    assert on["captures"][0]["drift"]["final_yaw_correction_deg"] < -10  # it undid real drift
    m = _check(on, sc, wall_cm=3.0, corner_cm=3.0)
    assert off["corner_rmse_cm"] > 2 * m["corner_rmse_cm"], (off, m)


def test_walkcheck_on_a_synthetic_walkthrough(tmp_path):
    """Two rooms joined by a door, walked through: the walk stays inside rooms and the rooms link."""
    from floorplan.evaluate import walk_check
    from floorplan.pipeline import lidar_capture
    sc = synth.make_scene("apartment")
    w = walk_check(lidar_capture(synth.write_stray(sc, tmp_path / "walk"), find_damage=False))
    assert w["rooms"] == 2 and w["walk_inside_pct"] >= 95 and w["components"] == 1, w


def test_hallway_is_a_room_and_links_the_rooms_off_it(tmp_path):
    """Two rooms off a 1 m hallway, walked in one go. Closing doorways used to fill the hallway
    as wall, so it was never a room, 57% of the walk fell outside every room and nothing was
    adjacent. The walked space now becomes a room, and the doors link through it."""
    from floorplan.evaluate import walk_check
    from floorplan.pipeline import lidar_capture
    sc = synth.make_scene("hall")
    w = walk_check(lidar_capture(synth.write_stray(sc, tmp_path / "walk"), find_damage=False))
    assert w["rooms"] == 3 and w["walk_inside_pct"] >= 95 and w["components"] == 1, w


def test_output_matches_published_schema(tmp_path):
    import jsonschema
    from pathlib import Path
    sc = synth.make_scene("apartment")
    scans = [synth.write_lidar_ply(sc, tmp_path / f"s{i}.ply", rooms=[i], seed=10 + i) for i in (0, 1)]
    result = run("lidar", scans, tmp_path / "out")
    schema = json.loads((Path(__file__).parents[1] / "schema" / "plan.schema.json").read_text())
    jsonschema.validate(json.loads((tmp_path / "out" / "plan.json").read_text()), schema)
    assert result["adjacency"] == [{"rooms": ["R1", "R2"], "via": result["adjacency"][0]["via"]}]
    for r in result["rooms"]:
        for w in r["walls"]:
            assert w["length_m"]["lo"] < w["length_m"]["value"] < w["length_m"]["hi"]


def test_damage_lands_on_surface_with_metric_extent_and_rules():
    """A detection box over a known wall patch -> right wall, right size, rule R2, scope."""
    from floorplan import damage, export
    from floorplan.plan import Room
    room = Room(np.array([[0, 0], [4.0, 0], [4.0, 3.0], [0, 3.0]]), height=2.5)
    # one view looking at wall 0 (y = 0): pixel u <-> x in [0, 4], v <-> z in [2.5, 0]
    u, v = np.meshgrid(np.linspace(0, 1, 200), np.linspace(0, 1, 120))
    X = np.column_stack([u.ravel() * 4.0, np.zeros(u.size), (1 - v.ravel()) * 2.5])
    uv = np.column_stack([u.ravel(), v.ravel()])
    box = np.array([1.0 / 4, 1 - 0.5 / 2.5, 2.0 / 4, 1 - 0.1 / 2.5])  # x 1..2 m, z 0.1..0.5 m
    damage.locate([[("water_stain", 0.8, box)], [("water_stain", 0.5, box)]], [(X, None, uv)] * 2, [room],
                  ["IMG_1.jpg", "IMG_2.jpg"])
    data = export.to_dict([room], {"stitching": [], "captures": []}, tier="lidar")
    damage.regions([room], data, "lidar")
    (d,) = data["damage"]
    assert d["surface"] == "R1.W1"
    assert abs(d["width_m"]["value"] - 1.0) < 0.06 and abs(d["height_m"]["value"] - 0.4) < 0.04, d
    assert [f["rule"] for f in data["concealed_flags"]] == ["R2 damp at wall base"]
    items = {s["item"] for s in data["scope"]}
    assert "Repaint whole surface" in items and any(i.startswith("Investigate") for i in items)


def test_score_against_tape_truth_and_repeatability(tmp_path):
    from floorplan.evaluate import repeatability, score
    sc = synth.make_scene("lshape")
    a = run("lidar", [synth.write_lidar_ply(sc, tmp_path / "a.ply", seed=1)], tmp_path / "a")
    b = run("lidar", [synth.write_lidar_ply(sc, tmp_path / "b.ply", seed=2)], tmp_path / "b")
    s = score(a, synth.tape_truth(sc))["summary"]
    assert s["rooms_found"] == "1/1" and s["walls_pass"] == "6/6", s
    assert s["ceiling_gate"] and s["openings_pass_pct"] == 100.0, s
    assert s["interval_coverage_pct"] >= 80, s
    assert repeatability(a, b)["pass"]
    # truth without openings: they were not measured, so none are scored
    unmeasured = synth.tape_truth(sc)
    for r in unmeasured["rooms"]:
        del r["openings"]
    assert score(a, unmeasured)["summary"]["openings_pass_pct"] is None


def test_bench_manifest_end_to_end(tmp_path):
    from floorplan.bench import run_bench
    sc = synth.make_scene("apartment")
    synth.write_stray(sc, tmp_path / "raw" / "walk", drift_deg_per_m=0.8)
    for i in (0, 1):
        synth.write_lidar_ply(sc, tmp_path / "raw" / f"scan{i}.ply", seed=i)
    truth = synth.tape_truth(sc)
    (tmp_path / "truth.json").write_text(json.dumps(truth))
    import trimesh
    trimesh.PointCloud(np.random.default_rng(0).random((200, 3))).export(tmp_path / "raw" / "empty.ply")
    theirs = json.loads(json.dumps(truth))
    for r in theirs["rooms"]:
        r["walls_m"] = [w + 0.03 for w in r["walls_m"]]  # an app that is 3 cm long everywhere
    (tmp_path / "app.json").write_text(json.dumps(theirs))
    (tmp_path / "manifest.json").write_text(json.dumps({"captures": [
        {"id": "walk", "tier": "lidar", "inputs": ["raw/walk"], "truth": "truth.json", "ablate_drift": True},
        {"id": "scan0", "tier": "lidar", "inputs": ["raw/scan0.ply"], "truth": "truth.json",
         "competitor": {"app": "SomeApp 1.0", "measured": "app.json"}},
        {"id": "scan1", "tier": "lidar", "inputs": ["raw/scan1.ply"], "truth": "truth.json", "repeat_of": "scan0"},
        {"id": "empty", "tier": "lidar", "inputs": ["raw/empty.ply"], "truth": "truth.json"},
    ]}))
    res = run_bench(tmp_path / "manifest.json", damage=False)
    report = (tmp_path / "out" / "report.md").read_text()
    for section in ("Overall per tier", "Gates per capture", "Failed captures", "Repeatability",
                    "Drift ablation", "Head-to-head"):
        assert section in report
    assert "too few wall points" in res["empty"]["error"]  # a failed capture is reported, not fatal
    assert res["scan0"]["competitor"]["beat_or_tie_pct"] >= 70
    assert res["scan1"]["repeat"]["pass"]


def test_room_seen_on_two_sides_is_not_closed_by_the_grid_border():
    """Two photos of a room often show only two walls. The gap-closing fallback used to turn the
    flood-fill seed pixel (0, 0) into wall (an anti-diagonal kernel reaches only outside the
    image there, where OpenCV's erosion counts as wall), so the empty grid padding came back as
    a room 2 m larger than anything seen."""
    from floorplan.plan import extract_rooms
    g = np.random.default_rng(0)
    s = np.arange(0, 3.6, 0.02)
    z = np.arange(0.0, 2.6, 0.02)
    back = np.array([(x, 3.2, h) for x in s for h in z])  # the wall along y = 3.2
    side = np.array([(3.6, y, h) for y in s[s <= 3.2] for h in z])  # the wall along x = 3.6
    P = np.concatenate([back, side]) + g.normal(0, 0.005, (len(back) + len(side), 3))
    try:
        rooms = extract_rooms(P, floor_z=0.0, ceiling_z=2.6, seeds=np.array([[1.8, 1.6, 1.4]]))
    except ValueError as e:
        assert "no enclosed room" in str(e)
        return
    lo, hi = P[:, :2].min(0), P[:, :2].max(0)
    for r in rooms:
        assert (r.polygon >= lo - 0.05).all() and (r.polygon <= hi + 0.05).all(), r.polygon


def test_room_seen_on_two_sides_becomes_the_rectangle_its_walls_span():
    """A photo folder is one room: when only two walls were seen, return the rectangle they span,
    flagged, with the two unseen walls marked as unsupported (so their intervals widen)."""
    from floorplan.plan import extract_rooms
    g = np.random.default_rng(0)
    s = np.arange(0, 3.6, 0.02)
    z = np.arange(0.0, 2.6, 0.02)
    back = np.array([(x, 3.2, h) for x in s for h in z])
    side = np.array([(3.6, y, h) for y in s[s <= 3.2] for h in z])
    P = np.concatenate([back, side]) + g.normal(0, 0.005, (len(back) + len(side), 3))
    (room,) = extract_rooms(P, floor_z=0.0, ceiling_z=2.6, seeds=np.array([[1.8, 1.6, 1.4]]), open_fallback=True)
    assert room.extra.get("unclosed")
    dims = np.sort(room.polygon.max(0) - room.polygon.min(0))
    assert np.allclose(dims, [3.2, 3.6], rtol=0.06), dims
    support = sorted(c for c, _ in room.wall_support)
    assert support[1] < 0.5 < support[2], room.wall_support  # two walls seen, two not


def test_webp_photos_reach_the_model(tmp_path):
    """Photos saved from the web are often .webp, which MapAnything's loader skips without a word."""
    from PIL import Image
    from floorplan import recon
    Image.new("RGB", (64, 48), (200, 10, 10)).save(tmp_path / "a.webp")
    (p,) = recon.list_images(tmp_path)
    png = recon._readable(p, tmp_path / "a.png")
    assert png.endswith(".png") and Image.open(png).size == (64, 48)


def test_damage_seen_once_needs_a_confident_detector():
    from floorplan import damage, export
    from floorplan.plan import Room
    room = Room(np.array([[0, 0], [4.0, 0], [4.0, 3.0], [0, 3.0]]), height=2.5)
    u, v = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 60))
    X = np.column_stack([u.ravel() * 4.0, np.zeros(u.size), (1 - v.ravel()) * 2.5])
    uv = np.column_stack([u.ravel(), v.ravel()])
    damage.locate([[("crack", 0.45, np.array([0.1, 0.1, 0.3, 0.5]))]], [(X, None, uv)], [room], ["a.jpg"])
    data = export.to_dict([room], {"stitching": [], "captures": []}, tier="video")
    damage.regions([room], data, "video")
    assert data["damage"] == [] and data["scope"] == []


def test_floor_is_lowest_layer_even_when_furniture_tops_outnumber_it():
    """Real video of a furnished room: more up-facing points on bed and sofa tops than on the
    floor. The floor is still the lowest big layer, not the biggest one."""
    from floorplan.plan import layer
    rng = np.random.default_rng(0)
    floor = rng.normal(0.00, 0.01, 40_000)
    bed = rng.normal(0.52, 0.01, 60_000)
    table = rng.normal(0.75, 0.01, 20_000)
    noise = rng.uniform(-0.3, 1.2, 3_000)
    z = np.concatenate([floor, bed, table, noise]) - 1.3
    assert abs(layer(z, lowest=True) - (-1.3)) < 0.01
    ceiling = np.concatenate([rng.normal(2.85, 0.01, 5_000), rng.normal(2.30, 0.01, 4_000)])  # + a bulkhead
    assert abs(layer(ceiling, lowest=False) - 2.85) < 0.01
