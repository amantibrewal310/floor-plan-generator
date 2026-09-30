import json
import math

import cv2
import numpy as np
import pytest

from floorplan import sfm, synth
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


def test_marker_edge_refinement_removes_corner_bias():
    sc = synth.make_scene("rect")
    r = synth.Renderer(sc)
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(synth.ARUCO_DICT), sfm._aruco_params())
    ratios = []
    for cam, yaw, pitch in [((1.0, 0.9, 1.45), 0.4, -0.5), ((3.2, 2.6, 1.45), -2.5, -0.45)]:
        cam = np.array(cam)
        gray = cv2.cvtColor(r.render(cam, yaw, pitch), cv2.COLOR_BGR2GRAY)
        fwd = np.array([math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), math.sin(pitch)])
        right = np.array([math.sin(yaw), -math.cos(yaw), 0])
        down = np.cross(fwd, right)
        corners, ids, _ = det.detectMarkers(gray)
        for c, i in zip(corners, ids.ravel()):
            _, mx, my, myaw = sc.markers[i]
            h = sc.marker_size / 2
            gt = []
            for lx, ly in [(-h, h), (h, h), (h, -h), (-h, -h)]:
                d = np.array([mx + lx * math.cos(myaw) - ly * math.sin(myaw),
                              my + lx * math.sin(myaw) + ly * math.cos(myaw), 0]) - cam
                gt.append([r.f * (d @ right) / (d @ fwd), r.f * (d @ down) / (d @ fwd)])
            gt = np.array(gt)
            rc = sfm.refine_corners(gray, c.reshape(4, 2).astype(float))
            side = lambda q: np.mean([np.linalg.norm(q[(k + 1) % 4] - q[k]) for k in range(4)])
            ratios.append(side(rc) / side(gt))
    assert len(ratios) >= 3
    assert abs(np.mean(ratios) - 1) < 0.0015, ratios


def test_markers_command(tmp_path):
    main(["markers", "-o", str(tmp_path), "--count", "2"])
    assert sorted(p.name for p in tmp_path.iterdir()) == ["marker_0.png", "marker_1.png"]


@pytest.mark.slow
def test_photos_rect(tmp_path):
    sc = synth.make_scene("rect")
    photos = synth.write_photos(sc, tmp_path / "photos", 0)
    result = run("photos", [photos], tmp_path / "out", marker_size=sc.marker_size)
    _check(result, sc, wall_cm=2.0, corner_cm=2.0, door_cm=4.0)
    saved = json.loads((tmp_path / "out" / "plan.json").read_text())
    assert saved["captures"][0]["registered"] == saved["captures"][0]["images"]
