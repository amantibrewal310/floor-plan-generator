"""Photo and video tiers: unposed images -> metric, gravity-aligned point cloud.

MapAnything (Meta, Apache-2.0 weights `facebook/map-anything-apache`) predicts a camera pose,
intrinsics and a metric depth map for every image in one forward pass, from images alone.
That is what makes 2-8 photos per room workable: there is no feature matching to fail on
plain walls, and no printed marker is needed for scale.

What is left for us:
  * gravity: the floor/ceiling normals, seeded by the average camera "up";
  * optional scale correction from an ArUco marker of known size, when one is in view;
  * a cache of the raw predictions, so a rerun replays deterministically without a GPU.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

MODEL_ID = "facebook/map-anything-apache"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp"}
# what MapAnything's loader opens; it silently skips anything else
MODEL_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".heif"}
CACHE_DIR = Path(os.environ.get("FLOORPLAN_CACHE", Path.home() / ".cache" / "floorplan"))


@dataclass
class Prediction:
    pts: np.ndarray  # (V, H, W, 3) world points, metres, model frame (OpenCV: y down)
    mask: np.ndarray  # (V, H, W) valid pixels
    conf: np.ndarray  # (V, H, W) confidence, >= 1
    poses: np.ndarray  # (V, 4, 4) cam-to-world
    K: np.ndarray  # (V, 3, 3) intrinsics at model resolution
    img: np.ndarray  # (V, H, W, 3) uint8 RGB at model resolution
    names: list[str]


def list_images(folder: Path) -> list[Path]:
    files = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not files:
        raise ValueError(f"no images found in {folder}")
    return files


def is_portrait(path: Path) -> bool:
    from PIL import Image
    from pillow_heif import register_heif_opener

    register_heif_opener()
    with Image.open(path) as im:
        w, h = im.size
        if im.getexif().get(0x0112, 1) in (5, 6, 7, 8):  # EXIF orientation: rotated 90 degrees
            w, h = h, w
    return h > w


def predict(images: list[Path], cache_dir: Path = CACHE_DIR) -> Prediction:
    """Run MapAnything on `images`, or replay the cached result for exactly these files."""
    h = hashlib.sha256(MODEL_ID.encode())
    for p in images:
        h.update(Path(p).read_bytes())
    cache = Path(cache_dir) / f"{h.hexdigest()[:24]}.npz"
    if cache.exists():
        d = np.load(cache)
        return Prediction(d["pts"], d["mask"], d["conf"], d["poses"], d["K"], d["img"],
                          [Path(p).name for p in images])
    pred = _infer(images)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, pts=pred.pts, mask=pred.mask, conf=pred.conf, poses=pred.poses,
                        K=pred.K, img=pred.img)
    return pred


_MODEL = None


def load_model():
    """MapAnything on the best available device (downloads the weights the first time)."""
    import torch
    from mapanything.models import MapAnything

    global _MODEL
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if _MODEL is None:
        _MODEL = MapAnything.from_pretrained(MODEL_ID).to(dev).eval()
    return _MODEL, dev


def _infer(images: list[Path]) -> Prediction:
    from mapanything.utils.image import load_images

    model, dev = load_model()
    with tempfile.TemporaryDirectory() as tmp:
        views = load_images([_readable(p, Path(tmp) / f"{i}.png") for i, p in enumerate(images)])
    out = model.infer(views, memory_efficient_inference=True, use_amp=dev != "cpu",
                       amp_dtype="bf16" if dev == "cuda" else "fp16", apply_mask=True, mask_edges=True)

    def stack(key):
        return np.stack([o[key][0].float().cpu().numpy() for o in out])

    img = np.clip(stack("img_no_norm") * 255, 0, 255).astype(np.uint8)
    return Prediction(stack("pts3d"), stack("mask")[..., 0] > 0.5, stack("conf"), stack("camera_poses"),
                      stack("intrinsics"), img, [Path(p).name for p in images])


def _readable(path: Path, png: Path) -> str:
    """A path MapAnything will load: other formats (.webp) re-encoded losslessly."""
    if Path(path).suffix.lower() in MODEL_EXTS:
        return str(path)
    from PIL import Image

    with Image.open(path) as im:
        im.convert("RGB").save(png)
    return str(png)


# ---------------------------------------------------------------- metric, gravity-aligned cloud

def gravity_rotation(pred: Prediction) -> np.ndarray:
    """Rows = new x, y, z axes with z up.

    People hold the phone roughly upright, so the mean camera up (-y in OpenCV) is a good seed.
    It is refined with the normals of floor and ceiling pixels, which are exactly vertical."""
    up = -pred.poses[:, :3, 1].mean(0)
    up /= np.linalg.norm(up)
    P = pred.pts
    n = np.cross(P[:, 1:-1, 2:] - P[:, 1:-1, :-2], P[:, 2:, 1:-1] - P[:, :-2, 1:-1])
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12
    n = n[pred.mask[:, 1:-1, 1:-1]]
    for tol in (25, 10, 5):
        sel = np.abs(n @ up) > np.cos(np.radians(tol))
        if sel.sum() < 1000:
            break
        v = (n[sel] * np.sign(n[sel] @ up)[:, None]).mean(0)
        up = v / np.linalg.norm(v)
    x = np.cross([0.0, 0.0, 1.0] if abs(up[2]) < 0.9 else [1.0, 0.0, 0.0], up)
    x /= np.linalg.norm(x)
    return np.stack([x, np.cross(up, x), up])


def marker_scale(pred: Prediction, marker_size: float) -> tuple[float, int] | None:
    """Scale correction from ArUco markers of known size seen in any view: (factor, n_views)."""
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
                                  cv2.aruco.DetectorParameters())
    sides = []
    for v in range(len(pred.img)):
        corners, ids, _ = det.detectMarkers(cv2.cvtColor(pred.img[v], cv2.COLOR_RGB2GRAY))
        if ids is None:
            continue
        for c in corners:
            c = c.reshape(4, 2)
            ij = np.round(c).astype(int)
            h, w = pred.mask[v].shape
            if (ij < 0).any() or (ij[:, 0] >= w).any() or (ij[:, 1] >= h).any():
                continue
            if not pred.mask[v][ij[:, 1], ij[:, 0]].all():
                continue
            X = pred.pts[v][ij[:, 1], ij[:, 0]]
            sides.append(np.mean([np.linalg.norm(X[(k + 1) % 4] - X[k]) for k in range(4)]))
    if not sides:
        return None
    return marker_size / float(np.median(sides)), len(sides)


def view_points(pred: Prediction, R: np.ndarray, v: int, conf_pct=10, stride=2, tol_deg=15):
    """Points of view v rotated by R (z up), the pixel each came from (normalised to 0..1), and
    which way its surface faces: 1 up (floor, bed, table top), 2 down (ceiling), 0 otherwise.

    A normal taken from the image grid points away from the camera, so a floor seen from above
    has a normal pointing down, and a ceiling seen from below one pointing up."""
    thr = np.percentile(pred.conf[pred.mask], conf_pct)
    ok = pred.mask[v] & (pred.conf[v] >= thr)
    ok[1::stride] = False
    ok[:, 1::stride] = False
    ok[0, :] = ok[-1, :] = False  # the grid normal needs both neighbours
    ok[:, 0] = ok[:, -1] = False
    G = pred.pts[v]
    n = np.zeros_like(G)
    n[1:-1, 1:-1] = np.cross(G[1:-1, 2:] - G[1:-1, :-2], G[2:, 1:-1] - G[:-2, 1:-1])
    nz = (n[ok] @ R[2]) / (np.linalg.norm(n[ok], axis=1) + 1e-12)
    c = np.cos(np.radians(tol_deg))
    facing = np.where(nz < -c, 1, np.where(nz > c, 2, 0)).astype(np.uint8)
    y, x = np.nonzero(ok)
    h, w = ok.shape
    return G[ok] @ R.T, np.column_stack([(x + 0.5) / w, (y + 0.5) / h]), facing


def metric_points(pred: Prediction, R: np.ndarray, views=None, max_points=1_500_000,
                  conf_pct=10) -> np.ndarray:
    """(N, 3) points rotated by R (z up), of the chosen views (default all), low-confidence
    pixels dropped."""
    views = range(len(pred.pts)) if views is None else views
    thr = np.percentile(pred.conf[pred.mask], conf_pct)
    P = np.concatenate([pred.pts[v][pred.mask[v] & (pred.conf[v] >= thr)] for v in views]) @ R.T
    if len(P) > max_points:
        P = P[np.random.default_rng(0).choice(len(P), max_points, replace=False)]
    return P
