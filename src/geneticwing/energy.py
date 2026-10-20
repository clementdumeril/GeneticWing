"""Specific energy and the netto variometer.

Same definitions as Morando et al. (arXiv 2605.15619, Eqs. 10, 11, 13) and
ArduSoar: the netto reading is the specific-energy rate with the aircraft's
own still-air sink added back, so it reads 0 in a steady glide through still
air and the vertical speed of the air mass inside a thermal.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

from geneticwing.aircraft import Aircraft
from geneticwing.flatness import FlatState


def specific_energy(altitude: np.ndarray, airspeed: np.ndarray, g: float) -> np.ndarray:
    """Total specific energy (m): ``z + V_a^2 / 2g``."""
    return np.asarray(altitude) + np.asarray(airspeed) ** 2 / (2.0 * g)


def specific_energy_rate(state: FlatState, g: float, window_s: float | None = None) -> np.ndarray:
    """Time derivative of :func:`specific_energy` along ``t`` (uniform sampling assumed).

    With ``window_s`` the derivative comes from a Savitzky-Golay filter of
    that width, as the ARPL netto variometer does for ``dV_a/dt``; without it,
    from plain finite differences.
    """
    energy = specific_energy(state.position[..., 2], state.airspeed, g)
    if window_s is None:
        return np.gradient(energy, state.t, axis=-1)
    dt = float(state.t[1] - state.t[0])
    window = max(5, int(round(window_s / dt)) | 1)
    return savgol_filter(energy, window, polyorder=2, deriv=1, delta=dt, axis=-1, mode="interp")


def netto(aircraft: Aircraft, state: FlatState, window_s: float | None = None) -> np.ndarray:
    """Netto variometer (m/s): energy rate plus the nominal still-air sink."""
    return specific_energy_rate(state, aircraft.g, window_s) + aircraft.nominal_sink_rate(
        state.airspeed, state.bank
    )
