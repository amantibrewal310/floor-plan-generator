"""Photo and video tiers: structure-from-motion (COLMAP) made metric with printed ArUco markers.

Monocular reconstructions have no scale and no notion of "down". Markers of known size lying
on the floor fix both: their triangulated corners give the scale (known edge length), the floor
plane (normal and height) and a stable origin. Markers shared between captures also stitch
them together exactly (see stitch.py).
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ARUCO_DICT = cv2.aruco.DICT_4X4_50
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".tif", ".tiff"}


@dataclass
class MetricCloud:
    points: np.ndarray  # (N, 3) metres, z up, floor at z = 0
    markers: dict[int, tuple[np.ndarray, float]]  # id -> (floor xy, yaw)
    cameras: np.ndarray  # (M, 3) camera centres
    scale_spread: float  # relative std of marker edge lengths: a sanity check on the scale
    n_images: int
    n_registered: int


# ---------------------------------------------------------------- inputs

def prepare_photos(src: Path, dst: Path, max_side=1600) -> float | None:
    """Copy/resize photos into dst. Returns the focal length in pixels from EXIF, if present."""
    dst.mkdir(parents=True, exist_ok=True)
    focal = None
    files = sorted(p for p in Path(src).iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not files:
        raise ValueError(f"no images found in {src}")
    for i, p in enumerate(files):
        im = Image.open(p)
        f35 = im.getexif().get_ifd(0x8769).get(0xA405)  # FocalLengthIn35mmFilm
        im = im.convert("RGB")
        s = min(1.0, max_side / max(im.size))
        if s < 1:
            im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
        if f35 and focal is None:
            focal = float(f35) / 36.0 * max(im.size)
        im.save(dst / f"{i:04d}.jpg", quality=95)
    return focal


def extract_keyframes(video: Path, dst: Path, fps=3.0, max_frames=200, max_side=1280) -> int:
    """Pick the sharpest frame in every 1/fps-second window (motion blur kills SfM)."""
    dst.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"cannot open video {video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 10**9
    window = max(1, int(round(src_fps / min(fps, max_frames * src_fps / total))))
    best, best_score, n, i = None, -1.0, 0, 0
    while True:
        ok, frame = cap.read()
        if ok:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            score = cv2.Laplacian(gray, cv2.CV_64F).var()
            if score > best_score:
                best, best_score = frame, score
        if (not ok or (i + 1) % window == 0) and best is not None:
            s = min(1.0, max_side / max(best.shape[:2]))
            if s < 1:
                best = cv2.resize(best, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(dst / f"{n:04d}.jpg"), best, [cv2.IMWRITE_JPEG_QUALITY, 95])
            best, best_score, n = None, -1.0, n + 1
        if not ok:
            break
        i += 1
    cap.release()
    if n < 10:
        raise ValueError(f"only {n} frames extracted from {video}; the video is too short")
    return n


# ---------------------------------------------------------------- reconstruction

def reconstruct(image_dir: Path, work: Path, sequential: bool, focal_px: float | None = None):
    import pycolmap

    if not os.environ.get("FLOORPLAN_VERBOSE"):
        pycolmap.logging.minloglevel = 2  # errors only; COLMAP is very chatty
    work.mkdir(parents=True, exist_ok=True)
    db = work / "database.db"
    if db.exists():
        db.unlink()
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "SIMPLE_RADIAL"
    if focal_px:
        w, h = Image.open(next(iter(sorted(image_dir.iterdir())))).size
        reader.camera_params = f"{focal_px},{w / 2},{h / 2},0"
    extraction = pycolmap.FeatureExtractionOptions()
    extraction.sift.max_num_features = 8192
    pycolmap.extract_features(db, image_dir, camera_mode=pycolmap.CameraMode.SINGLE,
                              reader_options=reader, extraction_options=extraction)
    if sequential:
        pairing = pycolmap.SequentialPairingOptions()
        pairing.overlap = 12
        pycolmap.match_sequential(db, pairing_options=pairing)
    else:
        pycolmap.match_exhaustive(db)
    opts = pycolmap.IncrementalPipelineOptions()
    opts.multiple_models = False
    if focal_px:
        opts.ba_refine_focal_length = True
    out = work / "sparse"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir()
    recs = pycolmap.incremental_mapping(db, image_dir, out, options=opts)
    if not recs:
        raise RuntimeError("structure-from-motion failed: not enough overlap/texture between images")
    return max(recs.values(), key=lambda r: r.num_reg_images())


def _pose(image):
    cfw = image.cam_from_world() if callable(image.cam_from_world) else image.cam_from_world
    return cfw.matrix()  # 3x4


def detect_markers(rec, image_dir: Path):
    """{marker_id: [(image, corners(4,2)), ...]} for every registered image."""
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(ARUCO_DICT), _aruco_params())
    obs: dict[int, list] = {}
    for img in rec.images.values():
        if not img.has_pose:
            continue
        gray = cv2.imread(str(image_dir / img.name), cv2.IMREAD_GRAYSCALE)
        corners, ids, _ = det.detectMarkers(gray)
        if ids is None:
            continue
        for c, i in zip(corners, ids.ravel()):
            c = refine_corners(gray, c.reshape(4, 2).astype(np.float64))
            if c is not None:  # +0.5: OpenCV puts pixel centres at integers, COLMAP at .5
                obs.setdefault(int(i), []).append((img, c + 0.5))
    return obs


def refine_corners(gray: np.ndarray, c: np.ndarray, half=4.0) -> np.ndarray | None:
    """Re-estimate marker corners as intersections of lines fitted to the four outer edges.

    Corner detectors are biased inward on blurred convex corners (~0.5% of the marker size,
    which becomes a 0.5% scale error). Edges are not: the steepest point of a symmetrically
    blurred step is the true edge."""
    img = cv2.GaussianBlur(gray, (0, 0), 0.8).astype(np.float32)
    offs = np.arange(-half, half + 1e-9, 0.25)
    lines = []
    for k in range(4):
        a, b = c[k], c[(k + 1) % 4]
        L = np.linalg.norm(b - a)
        if L < 20:  # too small for sub-pixel edges: a bit cell would be ~3 px
            return None
        u = (b - a) / L
        n = np.array([-u[1], u[0]])
        ts = np.linspace(0.12, 0.88, max(8, int(L / 3)))
        base = a + np.outer(ts * L, u)
        sx = (base[:, None, 0] + offs[None, :] * n[0]).astype(np.float32)
        sy = (base[:, None, 1] + offs[None, :] * n[1]).astype(np.float32)
        prof = cv2.remap(img, sx, sy, cv2.INTER_LINEAR)
        g = np.abs(np.diff(prof, axis=1))
        j = np.clip(np.argmax(g, axis=1), 1, g.shape[1] - 2)
        rows = np.arange(len(j))
        g0, g1, g2 = g[rows, j - 1], g[rows, j], g[rows, j + 1]
        den = g0 - 2 * g1 + g2
        sub = np.divide(0.5 * (g0 - g2), den, out=np.zeros_like(den), where=np.abs(den) > 1e-6)
        d = offs[0] + (j + 0.5 + np.clip(sub, -0.5, 0.5)) * 0.25
        pts = base + np.outer(d, n)
        strong = g1 > 0.5 * np.median(g1)
        pts = pts[strong]
        if len(pts) < 5:
            return None
        m = pts.mean(0)
        dirv = np.linalg.svd(pts - m)[2][0]
        lines.append((m, dirv))
    out = []
    for k in range(4):
        (p1, d1), (p2, d2) = lines[k - 1], lines[k]
        A = np.column_stack([d1, -d2])
        if abs(np.linalg.det(A)) < 1e-6:
            return None
        s = np.linalg.solve(A, p2 - p1)
        out.append(p1 + s[0] * d1)
    out = np.array(out)
    return out if np.max(np.linalg.norm(out - c, axis=1)) < 3.0 else None


def _aruco_params():
    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return p


def _triangulate(views) -> np.ndarray | None:
    """Linear (DLT) triangulation from [(3x4 pose, normalised xy)]."""
    if len(views) < 2:
        return None
    A = []
    for P, (x, y) in views:
        A += [x * P[2] - P[0], y * P[2] - P[1]]
    X = np.linalg.svd(np.array(A))[2][-1]
    return X[:3] / X[3]


def metric_cloud(rec, image_dir: Path, marker_size: float) -> MetricCloud:
    obs = detect_markers(rec, image_dir)
    corners = {}  # id -> (4, 3) in SfM units
    for mid, views in obs.items():
        if len(views) < 3:
            continue
        pts = []
        for k in range(4):
            v = []
            for img, c in views:
                xy = img.camera.cam_from_img(c[k][None])[0]
                v.append((_pose(img), xy))
            pts.append(_triangulate(v))
        if all(p is not None for p in pts):
            corners[mid] = np.array(pts)
    if not corners:
        raise RuntimeError(
            "no ArUco marker was seen in 3+ registered images. Print markers with "
            "`floorplan markers`, lay them flat on the floor and keep them in view.")

    # A well-triangulated marker is a square: its four sides agree. Markers seen from far away
    # or in few views are not, and must not set the scale of the whole plan.
    quality = {}
    for mid, c in list(corners.items()):
        sides = np.array([np.linalg.norm(c[(k + 1) % 4] - c[k]) for k in range(4)])
        cv = sides.std() / sides.mean()
        if cv > 0.02:
            del corners[mid]
        else:
            quality[mid] = (sides.mean(), 1.0 / max(cv, 0.002) ** 2)
    if not corners:
        raise RuntimeError("markers were seen but none could be triangulated reliably; "
                           "take a few closer shots of the markers on the floor")
    w = np.array([q[1] for q in quality.values()])
    side = np.sum(w * np.array([q[0] for q in quality.values()])) / w.sum()
    scale = marker_size / side
    spread = float(np.std([q[0] for q in quality.values()]) / side) if len(quality) > 1 else 0.0

    allc = np.concatenate(list(corners.values()))
    centre = allc.mean(0)
    normal = np.linalg.svd(allc - centre)[2][2]
    cams = np.array([img.projection_center() for img in rec.images.values() if img.has_pose])
    if np.dot(cams.mean(0) - centre, normal) < 0:
        normal = -normal
    first = corners[min(corners)]
    x = first[1] - first[0]
    x = x - normal * np.dot(x, normal)
    x /= np.linalg.norm(x)
    R = np.stack([x, np.cross(normal, x), normal])  # rows: new axes

    def to_metric(p):
        return scale * (p - centre) @ R.T

    pts = []
    for p in rec.points3D.values():
        if p.track.length() >= 3 and p.error < 2.0:
            pts.append(p.xyz)
    pts = to_metric(np.array(pts))
    markers = {}
    for mid, c in corners.items():
        m = to_metric(c)
        e = m[1] - m[0]
        markers[mid] = (m[:, :2].mean(0), float(np.arctan2(e[1], e[0])))
    return MetricCloud(pts, markers, to_metric(cams), spread, len(rec.images), rec.num_reg_images())


def write_marker_sheet(path: Path, ids=range(6), size_mm=180):
    """A4 pages (as one PNG per marker at 300 dpi) with the marker printed at `size_mm`."""
    d = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    path.mkdir(parents=True, exist_ok=True)
    px = int(round(size_mm / 25.4 * 300))
    for i in ids:
        m = cv2.aruco.generateImageMarker(d, i, px)
        page = np.full((3508, 2480), 255, np.uint8)  # A4 @ 300 dpi
        y0, x0 = (3508 - px) // 2, (2480 - px) // 2
        page[y0:y0 + px, x0:x0 + px] = m
        cv2.putText(page, f"ArUco 4x4 #{i}  -  print at 100% scale, black square = {size_mm} mm",
                    (120, 3380), cv2.FONT_HERSHEY_SIMPLEX, 1.6, 0, 3)
        Image.fromarray(page).save(path / f"marker_{i}.png", dpi=(300, 300))
