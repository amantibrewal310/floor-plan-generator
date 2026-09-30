"""Media helpers for the photo and video tiers: keyframes from a walkthrough video and
printable ArUco markers (an optional scale reference, see recon.marker_scale)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ARUCO_DICT = cv2.aruco.DICT_4X4_50


def extract_keyframes(video: Path, dst: Path, fps=2.0, max_frames=60, max_side=1280) -> int:
    """Pick the sharpest frame in every 1/fps-second window (motion blur hurts depth and pose)."""
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
    if n < 4:
        raise ValueError(f"only {n} frames extracted from {video}; the video is too short")
    return n


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
