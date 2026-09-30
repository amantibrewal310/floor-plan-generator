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
    damage.locate([[("water_stain", 0.8, box)]], [(X, None, uv)], [room], ["IMG_1.jpg"])
    data = export.to_dict([room], {"stitching": [], "captures": []}, tier="lidar")
    damage.regions([room], data, "lidar")
    (d,) = data["damage"]
    assert d["surface"] == "R1.W1"
    assert abs(d["width_m"]["value"] - 1.0) < 0.06 and abs(d["height_m"]["value"] - 0.4) < 0.04, d
    assert [f["rule"] for f in data["concealed_flags"]] == ["R2 damp at wall base"]
    items = {s["item"] for s in data["scope"]}
    assert "Repaint whole surface" in items and any(i.startswith("Investigate") for i in items)
