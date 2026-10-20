"""Ridge (slope) lift from 2-D potential flow over a cylindrical ridge.

The ridge is an infinite half-cylinder of radius ``R`` whose crest line passes
through ``crest`` along the horizontal direction ``axis_heading``. Wind of
speed ``U`` blows across it. In the cross-section plane, with ``x`` downwind
from the crest axis and ``z`` up from the ground, the complex potential
``U (zeta + R^2 / zeta)`` gives

    u = U (1 - R^2 (x^2 - z^2) / r^4)
    w = -2 U R^2 x z / r^4

so the windward face (``x < 0``) rises and the lee side sinks. This is the
standard model of the orographic-soaring literature (e.g. Langelaan 2007).
Points inside the ridge return zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from geneticwing.wind.base import WindField


@dataclass(frozen=True)
class RidgeLift(WindField):
    crest: tuple[float, float]  # m, a point on the crest line (ground level)
    axis_heading_deg: float  # compass heading of the crest line
    radius: float = 50.0  # m
    wind_speed: float = 8.0  # m/s, positive blows towards the right of the crest heading

    def _frame(self):
        h = np.radians(self.axis_heading_deg)
        along = np.array([np.sin(h), np.cos(h)])
        across = np.array([along[1], -along[0]])  # right of the crest heading
        return along, across

    def __call__(self, position, t=0.0):
        position = np.asarray(position, dtype=float)
        _, across = self._frame()
        rel = position[..., :2] - np.asarray(self.crest, dtype=float)
        x = rel @ across
        z = position[..., 2]
        r4 = np.maximum((x**2 + z**2) ** 2, 1e-12)
        r_sq = self.radius**2
        u = self.wind_speed * (1.0 - r_sq * (x**2 - z**2) / r4)
        w = -2.0 * self.wind_speed * r_sq * x * z / r4
        outside = (x**2 + z**2 >= r_sq) & (z >= 0.0)
        out = np.zeros(position.shape)
        out[..., 0] = np.where(outside, u * across[0], 0.0)
        out[..., 1] = np.where(outside, u * across[1], 0.0)
        out[..., 2] = np.where(outside, w, 0.0)
        return out
