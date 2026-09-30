"""90% intervals on every measurement.

An error model per tier, 1-sigma:
  scale   relative error that grows with length (metric scale of depth, drift)
  abs     fixed error on each wall face position, per end (fit noise, grid)
  opening error on each jamb of a door or window
  height  relative error of a ceiling height

The tier constants are priors; `floorplan evaluate` measures how often ground truth lands
inside the interval (calibration) and they are tuned so that it is about 90%. A wall only
partly seen gets a wider interval: its ends are extrapolated rather than measured.
"""

from __future__ import annotations

import math

Z90 = 1.645

# Priors until the real benchmark calibrates them. The photo and video numbers are deliberately
# wide: a learned metric scale on a room it has never seen can be several percent off, and a
# confident wrong answer costs more than an honest wide one.
TIERS = {
    "lidar": {"scale": 0.002, "abs": 0.005, "opening": 0.008, "height": 0.003},
    "video": {"scale": 0.025, "abs": 0.015, "opening": 0.025, "height": 0.025},
    "photos": {"scale": 0.040, "abs": 0.025, "opening": 0.035, "height": 0.040},
}
# a marker of known size replaces the model's scale estimate
MARKER_SCALE = 0.006


def interval(value: float, sigma: float, digits=4) -> dict:
    h = Z90 * sigma
    return {"value": round(value, digits), "lo": round(value - h, digits), "hi": round(value + h, digits)}


class ErrorModel:
    def __init__(self, tier: str, marker_scale=False):
        self.p = dict(TIERS[tier])
        if marker_scale:
            self.p["scale"] = min(self.p["scale"], MARKER_SCALE)

    def wall(self, length: float, coverage=1.0, spread=0.0) -> dict:
        # each end is a wall face: fit noise plus extrapolation over the unseen part
        end = math.hypot(self.p["abs"], spread) + 0.25 * (1 - coverage) * length
        return interval(length, math.hypot(self.p["scale"] * length, math.sqrt(2) * end))

    def opening(self, width: float) -> dict:
        return interval(width, math.hypot(self.p["scale"] * width, math.sqrt(2) * self.p["opening"]))

    def height(self, h: float | None) -> dict | None:
        if h is None:
            return None
        return interval(h, math.hypot(self.p["height"] * h, self.p["abs"]))

    def area(self, area: float, perimeter: float, coverage=1.0) -> dict:
        # scale error: 2x relative; face error: perimeter times the face offset
        face = self.p["abs"] + 0.1 * (1 - coverage)
        return interval(area, math.hypot(2 * self.p["scale"] * area, perimeter * face / math.sqrt(2)), 3)
