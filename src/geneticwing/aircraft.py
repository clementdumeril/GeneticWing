"""Point-mass aircraft parameters with a parabolic drag polar.

The polar ``C_D = C_D0 + k C_L^2`` is the same model as Morando et al.
(arXiv 2605.15619), whose sink-rate form ``V_z = V_a (P / C_L + B C_L / cos^2 phi)``
is recovered with ``P = C_D0`` and ``B = k``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GRAVITY = 9.80665  # m/s^2
AIR_DENSITY = 1.225  # kg/m^3, ISA sea level


@dataclass(frozen=True)
class Aircraft:
    name: str
    mass: float  # kg
    wing_area: float  # m^2
    cd0: float
    k: float  # induced-drag factor, 1 / (pi AR e)
    cl_max: float
    max_thrust: float = 0.0  # N, 0 for a pure glider
    rho: float = AIR_DENSITY
    g: float = GRAVITY

    @property
    def weight(self) -> float:
        return self.mass * self.g

    def drag_coefficient(self, cl: np.ndarray) -> np.ndarray:
        return self.cd0 + self.k * np.asarray(cl) ** 2

    def dynamic_pressure(self, airspeed: np.ndarray) -> np.ndarray:
        return 0.5 * self.rho * np.asarray(airspeed) ** 2

    @property
    def max_glide_ratio(self) -> float:
        """Maximum lift-to-drag ratio of the parabolic polar."""
        return 0.5 / np.sqrt(self.k * self.cd0)

    @property
    def min_drag_airspeed(self) -> float:
        """Airspeed of maximum L/D in wings-level flight (small path angle)."""
        cl_star = np.sqrt(self.cd0 / self.k)
        return float(np.sqrt(2.0 * self.weight / (self.rho * self.wing_area * cl_star)))

    def steady_glide(self, airspeed: np.ndarray, bank: np.ndarray = 0.0):
        """Exact steady, unpowered glide in still air.

        Balancing ``L cos(phi) = W cos(gamma)`` and ``D = -W sin(gamma)`` gives a
        quadratic in ``sin(gamma)`` with one physical (descending) root.
        Returns ``(gamma, sink_rate, cl)``; ``sink_rate`` is positive downward.
        """
        airspeed = np.asarray(airspeed, dtype=float)
        cos_bank = np.cos(np.asarray(bank, dtype=float))
        q_s = self.dynamic_pressure(airspeed) * self.wing_area
        a = q_s * self.cd0 / self.weight
        b = self.k * self.weight / (q_s * cos_bank**2)
        sin_gamma = (1.0 - np.sqrt(1.0 + 4.0 * b * (a + b))) / (2.0 * b)
        gamma = np.arcsin(sin_gamma)
        cl = self.weight * np.cos(gamma) / (q_s * cos_bank)
        return gamma, -airspeed * sin_gamma, cl

    def nominal_sink_rate(self, airspeed: np.ndarray, bank: np.ndarray = 0.0) -> np.ndarray:
        """Still-air sink rate (m/s, positive down) used by the netto variometer."""
        return self.steady_glide(airspeed, bank)[1]


def identify_parabolic_polar(
    mass: float,
    wing_area: float,
    best_glide_airspeed: float,
    best_glide_sink_rate: float,
    rho: float = AIR_DENSITY,
    g: float = GRAVITY,
) -> tuple[float, float]:
    """``(C_D0, k)`` such that the best-glide point sits at the given airspeed and sink.

    At maximum L/D the two drag terms are equal, ``C_D0 = k C_L*^2``, and
    ``C_D* = C_L* / E_max``. Small-angle approximation for ``C_L*``.
    """
    e_max = np.sqrt(best_glide_airspeed**2 - best_glide_sink_rate**2) / best_glide_sink_rate
    cl_star = 2.0 * mass * g / (rho * best_glide_airspeed**2 * wing_area)
    return cl_star / (2.0 * e_max), 1.0 / (2.0 * e_max * cl_star)


# Strix Stratosurfer, the ARPL glide platform. Mass and wing area are NOT
# published in arXiv 2605.15619: the values below are placeholders. The polar
# is then fitted to the best-glide point that paper reports (about 12 m/s for
# a 1.51 m/s sink). Replace with the lab's identified parameters.
_STRATOSURFER_MASS = 1.3  # kg, placeholder
_STRATOSURFER_WING_AREA = 0.30  # m^2, placeholder
_STRATOSURFER_CD0, _STRATOSURFER_K = identify_parabolic_polar(
    _STRATOSURFER_MASS, _STRATOSURFER_WING_AREA, 12.0, 1.51
)

STRATOSURFER_APPROX = Aircraft(
    name="Strix Stratosurfer (approximate)",
    mass=_STRATOSURFER_MASS,
    wing_area=_STRATOSURFER_WING_AREA,
    cd0=_STRATOSURFER_CD0,
    k=_STRATOSURFER_K,
    cl_max=1.1,  # placeholder
    max_thrust=8.0,  # N, placeholder
)


# SB-XC, a 4.6 m RC sailplane used as a soaring UAV in the literature. Mass,
# wing area and aspect ratio come from the Model Aviation review (Sept. 1992):
# 9.3 lb, 1656 sq in, AR 19.8. No polar is published: C_D0 and the Oswald
# factor below are engineering estimates to refine (XFLR5 or glide tests).
_SBXC_ASPECT_RATIO = 19.8
_SBXC_OSWALD = 0.85  # estimate

SBXC_APPROX = Aircraft(
    name="SB-XC sailplane (estimated polar)",
    mass=9.3 * 0.45359237,
    wing_area=1656 * 0.0254**2,
    cd0=0.018,  # estimate
    k=1.0 / (np.pi * _SBXC_ASPECT_RATIO * _SBXC_OSWALD),
    cl_max=1.2,  # estimate
)
