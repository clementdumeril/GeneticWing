"""Thermal updraft with Allen's vertical scaling and a Gaussian radial shape.

Vertical scaling follows Allen, "Updraft model for development of autonomous
soaring uninhabited air vehicles" (NASA/TM-2006-214382 / AIAA 2006-1510):

    w_bar(z) = w* (z / z_i)^(1/3) (1 - 1.1 z / z_i)                    mean updraft
    r2(z)    = max(10, 0.102 (z / z_i)^(1/3) (1 - 0.25 z / z_i) z_i)    outer radius

With ``profile="gaussian"`` (the default, the S1 family) Allen's bell-shaped
radial profile and downdraft ring are replaced by a Gaussian core of radius
``r2 / 2`` whose mean over the disk of radius ``r2`` equals ``w_bar``, plus an
optional uniform environmental sink. With ``profile="allen"`` the updraft is
Allen's own (:func:`allen_updraft`), the profile of the real-weather thermals of
:mod:`geneticwing.weather` and of the weather family; :class:`EnvironmentSink`
is the sinking air between such thermals.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from geneticwing.wind.base import WindField

# Mean of exp(-(r / (R/2))^2) over the disk of radius R.
_GAUSSIAN_DISK_MEAN = (1.0 - np.exp(-4.0)) / 4.0


def allen_radius(z, z_i):
    """Allen's outer updraft radius, Eq. (12)."""
    ratio = np.clip(np.asarray(z, float) / z_i, 0.0, None)
    return np.maximum(10.0, 0.102 * np.cbrt(ratio) * (1.0 - 0.25 * ratio) * z_i)


# Allen's shape constants k1..k4 by r1/r2 (Table 3, with k4 as in his MATLAB code, Appendix B).
ALLEN_R1R2 = np.array([0.14, 0.25, 0.36, 0.47, 0.58, 0.69, 0.80])
ALLEN_K = np.array([[1.5352, 2.5826, -0.0113, -0.1950], [1.5265, 3.6054, -0.0176, -0.1265],
                    [1.4866, 4.8356, -0.0320, -0.0818], [1.2042, 7.7904, 0.0848, -0.0445],
                    [0.8816, 13.9720, 0.3404, -0.0216], [0.7067, 23.9940, 0.5689, -0.0099],
                    [0.6189, 42.7965, 0.7157, -0.0033]])


def allen_updraft(r, z, w_star, z_i):
    """Allen's updraft (NASA/TM-2006-214382, Eqs. 11-18 as in his MATLAB code): bell-shaped core and,
    above half the layer, a downdraft ring. ``r`` is the distance to the core, all arrays broadcast."""
    ratio = np.clip(np.asarray(z, float) / z_i, 0.0, None)
    w_bar = w_star * np.cbrt(ratio) * (1.0 - 1.1 * ratio)
    r2 = np.maximum(10.0, 0.102 * np.cbrt(ratio) * (1.0 - 0.25 * ratio) * z_i)
    r1r2 = np.where(r2 < 600.0, 0.0011 * r2 + 0.14, 0.8)
    r1 = r1r2 * r2
    w_peak = 3.0 * w_bar * (r2**3 - r2**2 * r1) / (r2**3 - r1**3)
    k = ALLEN_K[np.abs(r1r2[..., None] - ALLEN_R1R2).argmin(axis=-1)]
    rr = r / r2
    shape = np.clip(1.0 / (1.0 + (k[..., 0] * np.abs(rr + k[..., 2])) ** k[..., 1]) + k[..., 3] * rr, 0.0, None)
    ring = np.where((r > r1) & (rr < 2.0), np.pi / 6.0 * np.sin(np.pi * rr), 0.0)
    s_wd = np.where((ratio > 0.5) & (ratio <= 0.9), 2.5 * (ratio - 0.5), 0.0)
    w = shape * w_peak + np.minimum(s_wd * ring, 0.0) * w_bar
    return np.where((ratio > 0.0) & (ratio < 1.0), w, 0.0)


def allen_peak(z, w_star, z_i):
    """Peak of Allen's updraft (at the core) at height ``z``."""
    return allen_updraft(np.zeros_like(np.asarray(z, float)), z, w_star, z_i)


def thermal_density(z_i):
    """Thermals per m^2, Allen's Eq. (20) N = 0.6 X Y / (z_i r2), with r2 at z = 0.4 z_i (Lenschow's height)."""
    return 0.6 / (z_i * allen_radius(0.4 * z_i, z_i))


@dataclass(frozen=True)
class Thermal(WindField):
    center: tuple[float, float]  # m, position of the core at t = 0
    w_star: float = 2.0  # m/s, convective velocity scale
    z_i: float = 1000.0  # m, convective boundary-layer depth
    drift: tuple[float, float] = (0.0, 0.0)  # m/s, core drift (usually the horizontal wind)
    environment_sink: float = 0.0  # m/s, positive down, applied everywhere below z_i
    profile: str = "gaussian"  # radial shape: "gaussian" core, or "allen" (Allen's bell and downdraft ring)

    def mean_updraft(self, z: np.ndarray) -> np.ndarray:
        ratio = np.clip(np.asarray(z, dtype=float) / self.z_i, 0.0, None)
        return np.clip(self.w_star * np.cbrt(ratio) * (1.0 - 1.1 * ratio), 0.0, None)

    def outer_radius(self, z: np.ndarray) -> np.ndarray:
        ratio = np.clip(np.asarray(z, dtype=float) / self.z_i, 0.0, None)
        return np.maximum(10.0, 0.102 * np.cbrt(ratio) * (1.0 - 0.25 * ratio) * self.z_i)

    def peak_updraft(self, z: np.ndarray) -> np.ndarray:
        if self.profile == "allen":
            return allen_peak(z, self.w_star, self.z_i)
        return self.mean_updraft(z) / _GAUSSIAN_DISK_MEAN

    def __call__(self, position, t=0.0):
        position = np.asarray(position, dtype=float)
        t = np.asarray(t, dtype=float)
        z = position[..., 2]
        cx = self.center[0] + self.drift[0] * t
        cy = self.center[1] + self.drift[1] * t
        r2 = (position[..., 0] - cx) ** 2 + (position[..., 1] - cy) ** 2
        if self.profile == "allen":
            w = allen_updraft(np.sqrt(r2), z, self.w_star, self.z_i)
        else:
            core_radius = self.outer_radius(z) / 2.0
            w = self.peak_updraft(z) * np.exp(-r2 / core_radius**2)
        inside_layer = (z > 0.0) & (z < self.z_i)
        w = np.where(inside_layer, w - self.environment_sink, 0.0)
        out = np.zeros(np.broadcast_shapes(position.shape, w.shape + (3,)))
        out[..., 2] = w
        return out


@dataclass(frozen=True)
class EnvironmentSink(WindField):
    """Sinking air between the thermals that returns their mass flux (Allen's MATLAB code, after Eq. 21).

    w_e(z) = -A_t w_bar / (A - A_t), with A_t / A = n pi r2(z)^2 the area fraction of the updrafts.
    """

    density: float  # thermals per m^2
    w_star: float
    z_i: float

    def __call__(self, position, t=0.0):
        position = np.asarray(position, float)
        z = position[..., 2]
        ratio = np.clip(z / self.z_i, 0.0, None)
        w_bar = np.clip(self.w_star * np.cbrt(ratio) * (1.0 - 1.1 * ratio), 0.0, None)
        frac = np.clip(self.density * np.pi * allen_radius(z, self.z_i) ** 2, 0.0, 0.5)
        out = np.zeros(position.shape)
        out[..., 2] = np.where((z > 0.0) & (z < self.z_i), -frac * w_bar / (1.0 - frac), 0.0)
        return out
