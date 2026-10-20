"""Thermal updraft with Allen's vertical scaling and a Gaussian radial shape.

Vertical scaling follows Allen, "Updraft model for development of autonomous
soaring uninhabited air vehicles" (NASA/TM-2006-214382 / AIAA 2006-1510):

    w_bar(z) = w* (z / z_i)^(1/3) (1 - 1.1 z / z_i)                    mean updraft
    r2(z)    = max(10, 0.102 (z / z_i)^(1/3) (1 - 0.25 z / z_i) z_i)    outer radius

Allen's bell-shaped radial profile and downdraft ring are replaced here by a
Gaussian core of radius ``r2 / 2`` whose mean over the disk of radius ``r2``
equals ``w_bar``, plus an optional uniform environmental sink. The scaling
formulas should be checked against the original report before publication.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from geneticwing.wind.base import WindField

# Mean of exp(-(r / (R/2))^2) over the disk of radius R.
_GAUSSIAN_DISK_MEAN = (1.0 - np.exp(-4.0)) / 4.0


@dataclass(frozen=True)
class Thermal(WindField):
    center: tuple[float, float]  # m, position of the core at t = 0
    w_star: float = 2.0  # m/s, convective velocity scale
    z_i: float = 1000.0  # m, convective boundary-layer depth
    drift: tuple[float, float] = (0.0, 0.0)  # m/s, core drift (usually the horizontal wind)
    environment_sink: float = 0.0  # m/s, positive down, applied everywhere below z_i

    def mean_updraft(self, z: np.ndarray) -> np.ndarray:
        ratio = np.clip(np.asarray(z, dtype=float) / self.z_i, 0.0, None)
        return np.clip(self.w_star * np.cbrt(ratio) * (1.0 - 1.1 * ratio), 0.0, None)

    def outer_radius(self, z: np.ndarray) -> np.ndarray:
        ratio = np.clip(np.asarray(z, dtype=float) / self.z_i, 0.0, None)
        return np.maximum(10.0, 0.102 * np.cbrt(ratio) * (1.0 - 0.25 * ratio) * self.z_i)

    def peak_updraft(self, z: np.ndarray) -> np.ndarray:
        return self.mean_updraft(z) / _GAUSSIAN_DISK_MEAN

    def __call__(self, position, t=0.0):
        position = np.asarray(position, dtype=float)
        t = np.asarray(t, dtype=float)
        z = position[..., 2]
        cx = self.center[0] + self.drift[0] * t
        cy = self.center[1] + self.drift[1] * t
        r2 = (position[..., 0] - cx) ** 2 + (position[..., 1] - cy) ** 2
        core_radius = self.outer_radius(z) / 2.0
        w = self.peak_updraft(z) * np.exp(-r2 / core_radius**2)
        inside_layer = (z > 0.0) & (z < self.z_i)
        w = np.where(inside_layer, w - self.environment_sink, 0.0)
        out = np.zeros(np.broadcast_shapes(position.shape, w.shape + (3,)))
        out[..., 2] = w
        return out
