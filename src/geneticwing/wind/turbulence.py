"""Dryden turbulence (MIL-F-8785C low-altitude model), frozen along time.

Gusts are generated once, for a nominal airspeed and altitude, by passing
white noise through the Dryden shaping filters

    H_u(s) = 1 / (1 + (L_u / V) s)
    H_v(s) = (1 + sqrt(3) (L_v / V) s) / (1 + (L_v / V) s)^2      (same for w)

and each component is rescaled to its target intensity. This sidesteps the
spectral normalization conventions: the filters set the spectral shape, the
rescaling sets sigma exactly. The field then depends on time only, which is
what a planner needs to replay one turbulence realization against many
candidate trajectories.

Low-altitude scales (h in feet inside the formulas, converted to metres):

    L_w = h,  L_u = L_v = h / (0.177 + 0.000823 h)^1.2
    sigma_w = 0.1 W20,  sigma_u = sigma_v = sigma_w / (0.177 + 0.000823 h)^0.4

``W20`` is the mean wind speed 20 ft above ground.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal

from geneticwing.wind.base import WindField

_FT = 0.3048


def dryden_scales(altitude: float, w20: float) -> tuple[np.ndarray, np.ndarray]:
    """Length scales ``(L_u, L_v, L_w)`` in m and intensities ``(s_u, s_v, s_w)`` in m/s."""
    h_ft = float(np.clip(altitude / _FT, 10.0, 1000.0))
    denom = 0.177 + 0.000823 * h_ft
    l_w = h_ft * _FT
    l_uv = h_ft / denom**1.2 * _FT
    s_w = 0.1 * w20
    s_uv = s_w / denom**0.4
    return np.array([l_uv, l_uv, l_w]), np.array([s_uv, s_uv, s_w])


@dataclass(frozen=True)
class DrydenTurbulence(WindField):
    duration: float  # s
    airspeed: float  # m/s, nominal, sets the temporal scale L / V
    altitude: float  # m, nominal
    w20: float  # m/s, mean wind 20 ft above ground
    heading_deg: float = 0.0  # compass heading of the longitudinal (u) axis
    dt: float = 0.02  # s
    seed: int = 0
    _t: np.ndarray = field(init=False, repr=False, compare=False)
    _gusts: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        lengths, sigmas = dryden_scales(self.altitude, self.w20)
        t = np.arange(0.0, self.duration + self.dt, self.dt)
        rng = np.random.default_rng(self.seed)
        gusts = np.empty((t.size, 3))
        for i, (length, sigma) in enumerate(zip(lengths, sigmas)):
            tau = length / self.airspeed
            if i == 0:
                num, den = [1.0], [tau, 1.0]
            else:
                num, den = [np.sqrt(3.0) * tau, 1.0], [tau**2, 2.0 * tau, 1.0]
            # Run past the filter's settling time so the record starts stationary.
            warmup = int(np.ceil(10.0 * tau / self.dt))
            noise = rng.standard_normal(t.size + warmup)
            _, y, _ = signal.lsim((num, den), noise, np.arange(noise.size) * self.dt)
            y = y[warmup:]
            y = y - y.mean()
            gusts[:, i] = sigma * y / max(y.std(), 1e-12)
        object.__setattr__(self, "_t", t)
        object.__setattr__(self, "_gusts", gusts)

    def __call__(self, position, t=0.0):
        position = np.asarray(position, dtype=float)
        t = np.broadcast_to(np.asarray(t, dtype=float), position.shape[:-1])
        u, v, w = (np.interp(t, self._t, self._gusts[:, i]) for i in range(3))
        h = np.radians(self.heading_deg)
        along = np.array([np.sin(h), np.cos(h)])
        left = np.array([-along[1], along[0]])
        out = np.empty(position.shape)
        out[..., 0] = u * along[0] + v * left[0]
        out[..., 1] = u * along[1] + v * left[1]
        out[..., 2] = w
        return out
