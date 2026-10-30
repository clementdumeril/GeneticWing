"""Strip-theory wing in a steady turn: how tight can the sailplane really circle?

The point-mass model sees one lift coefficient for the whole wing. In a turn
of radius R the wing yaws at ``Omega cos(bank)`` (``Omega = V / R``), so a
strip at spanwise position ``y`` (positive towards the outer wing) flies at

    V(y) = V (1 + y cos(bank) / R).

On a 4.6 m span circling at 15 m, the inner tip is about 15 % slower than the
outer tip, so it lifts about 28 % less. The outer wing then rolls the aircraft
further into the turn (the overbanking tendency), and the ailerons must hold
the bank by adding lift on the inner side. Two physical limits follow:

* the inner wing's local lift coefficient reaches stall before the mean one does;
* the ailerons run out of authority.

And deflected ailerons cost drag, i.e. energy.

The model: the wing is cut into strips with a trapezoidal chord. Each strip
carries ``CL0 + delta * s(y)``, where ``s = +1`` on the inner aileron, ``-1`` on
the outer one, 0 elsewhere. Total lift must equal what the point mass needs,
and the rolling moment must vanish: two linear equations in ``CL0`` and
``delta``, solved in closed form for whole batches. Induced-drag and
downwash coupling between strips are neglected (strip theory), as are
washout and the thermal's lift gradient across the span. The geometry and the
aileron numbers below are estimates for the SB-XC, to refine.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np

from geneticwing.aircraft import Aircraft


@dataclass(frozen=True)
class StripWing:
    span: float  # m
    root_chord: float  # m
    tip_chord: float  # m
    aileron_start: float = 0.6  # fraction of the semispan where the ailerons begin
    aileron_cl_max: float = 0.5  # largest lift-coefficient change the ailerons produce
    aileron_drag: float = 0.1  # extra profile drag coefficient per unit (delta CL)^2
    local_cl_max: float = 1.2  # section stall, same as the aircraft's C_L max
    n_strips: int = 40

    @classmethod
    def trapezoid(cls, area: float, aspect_ratio: float, taper: float = 0.5, **kwargs) -> "StripWing":
        span = np.sqrt(aspect_ratio * area)
        root = 2.0 * area / (span * (1.0 + taper))
        return cls(span=span, root_chord=root, tip_chord=taper * root, **kwargs)

    @cached_property
    def stations(self):
        """Strip centers ``y`` (m), chords and widths, ordered from the inner tip to the outer tip."""
        half = self.span / 2.0
        edges = np.linspace(-half, half, self.n_strips + 1)
        y = 0.5 * (edges[1:] + edges[:-1])
        chord = self.root_chord + (self.tip_chord - self.root_chord) * np.abs(y) / half
        width = np.diff(edges)
        aileron = np.abs(y) >= self.aileron_start * half
        side = np.where(aileron, -np.sign(y), 0.0)  # +1 inner aileron, -1 outer aileron
        return y, chord, width, side

    @property
    def area(self) -> float:
        _, chord, width, _ = self.stations
        return float(np.sum(chord * width))


@dataclass
class TurnLoads:
    cl_body: np.ndarray  # symmetric part of the lift coefficient
    aileron: np.ndarray  # aileron lift-coefficient change needed to hold the bank
    cl_max_local: np.ndarray  # highest local lift coefficient along the span
    extra_drag: np.ndarray  # N, profile drag of the deflected ailerons


def turn_loads(wing: StripWing, aircraft: Aircraft, airspeed, bank, cl) -> TurnLoads:
    """Spanwise loads of a coordinated turn, batched over any array shape.

    ``cl`` is the point-mass lift coefficient (lift over dynamic pressure times
    wing area); the turn radius follows from the bank, ``V^2 / (g tan|bank|)``.
    """
    airspeed, bank, cl = np.broadcast_arrays(*(np.asarray(x, dtype=float) for x in (airspeed, bank, cl)))
    y, chord, width, side = wing.stations
    tan_b = np.abs(np.tan(bank))
    curvature = aircraft.g * tan_b / np.maximum(airspeed**2, 1e-6)  # 1 / R
    ratio = 1.0 + y * (np.cos(bank) * curvature)[..., None]  # V(y) / V
    weight = chord * width * ratio**2  # dynamic-pressure-weighted strip areas / q_mean
    a = weight.sum(-1)
    b = (weight * side).sum(-1)
    c = (weight * y).sum(-1)
    d = (weight * side * y).sum(-1)
    cl_body = cl * aircraft.wing_area / (a - b * c / d)
    delta = -c * cl_body / d
    local = cl_body[..., None] + delta[..., None] * side
    q = aircraft.dynamic_pressure(airspeed)
    extra = q * wing.aileron_drag * delta**2 * (weight * np.abs(side)).sum(-1)
    return TurnLoads(cl_body=cl_body, aileron=delta, cl_max_local=local.max(-1), extra_drag=extra)


def minimum_turn_radius(
    wing: StripWing, aircraft: Aircraft, airspeed: float, cl_limit: float, banks_deg=np.linspace(1, 45, 450)
) -> dict:
    """Smallest steady-turn radius at ``airspeed`` (bank up to 45 deg), from the point mass and from the strips.

    The point mass only needs its mean C_L below ``cl_limit``; the strip wing
    also needs its highest local C_L below ``cl_limit`` and the ailerons within
    their authority. Turns are level-equivalent (``n = 1 / cos(bank)``).
    """
    bank = np.radians(banks_deg)
    q_s = aircraft.dynamic_pressure(airspeed) * aircraft.wing_area
    cl = aircraft.weight / (np.cos(bank) * q_s)
    radius = airspeed**2 / (aircraft.g * np.tan(bank))
    loads = turn_loads(wing, aircraft, airspeed, bank, cl)
    point_ok = cl <= cl_limit
    strip_ok = point_ok & (loads.cl_max_local <= cl_limit) & (np.abs(loads.aileron) <= wing.aileron_cl_max)
    return {
        "point_mass": float(radius[point_ok].min()) if point_ok.any() else np.inf,
        "strip": float(radius[strip_ok].min()) if strip_ok.any() else np.inf,
    }


def sbxc_wing(**kwargs) -> StripWing:
    """SB-XC: 1.07 m^2, aspect ratio 19.8 (Model Aviation, 1992); taper and ailerons estimated."""
    return StripWing.trapezoid(area=1656 * 0.0254**2, aspect_ratio=19.8, taper=0.5, **kwargs)


def protected_cl(wing: StripWing, aircraft: Aircraft, airspeed: float, bank: float, cl: float, fraction: float = 0.9) -> float:
    """The largest C_L, up to ``cl``, that keeps every wing section below ``fraction`` of its stall.

    An autopilot's angle-of-attack protection works on the mean C_L; in a slow
    turn the inner wing works harder than the mean, so the cap shrinks with
    bank and with low airspeed. The ratio of the local peak to the mean C_L is
    taken at ``cl`` (it hardly depends on it).
    """
    if cl <= 0.0:
        return cl
    peak = float(turn_loads(wing, aircraft, airspeed, bank, cl).cl_max_local)
    return min(cl, fraction * wing.local_cl_max * cl / max(peak, 1e-9))
