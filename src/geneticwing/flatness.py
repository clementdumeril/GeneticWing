"""Flat evaluation of a coordinated point-mass fixed-wing in a known wind field.

Given the inertial position trajectory and its derivatives, every quantity the
planner needs follows algebraically, with no integration:

    v_a   = p_dot - W(p, t)                       air-relative velocity
    F     = m (p_ddot - g)                        total aerodynamic + thrust force
    L     = F - (F . v_a_hat) v_a_hat             lift, normal to v_a
    T_req = F . v_a_hat + D(C_L)                  thrust needed along v_a

Thrust is assumed aligned with the air-relative velocity (small angle of
attack) and flight is coordinated (no sideslip), as in Morando et al.
(ICRA 2025, arXiv 2502.00581).

With thrust, position is a flat output. Without thrust it is not: a glider
has two inputs (C_L, bank) for three outputs, and ``T_req = 0`` becomes a
constraint on the trajectory. That condition generalizes the sink-rate
constraint of arXiv 2605.15619 to non-uniform wind.

Bank sign: positive is right wing down, i.e. a turn to the right.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from geneticwing.aircraft import Aircraft

_UP = np.array([0.0, 0.0, 1.0])
_EPS = 1e-9


@dataclass(frozen=True)
class FlatState:
    """Quantities along a trajectory, each shaped ``(..., N)`` or ``(..., N, 3)``."""

    t: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    wind: np.ndarray
    air_velocity: np.ndarray
    airspeed: np.ndarray
    air_path_angle: np.ndarray  # rad, positive climbing through the air mass
    lift: np.ndarray  # N, vector
    cl: np.ndarray
    load_factor: np.ndarray
    bank: np.ndarray  # rad
    drag: np.ndarray  # N
    thrust_required: np.ndarray  # N, negative means excess energy (must be bled or stored)

    @property
    def sink_rate_air(self) -> np.ndarray:
        """Descent rate relative to the air mass, m/s, positive down."""
        return -self.air_velocity[..., 2]

    @property
    def ground_speed(self) -> np.ndarray:
        return np.linalg.norm(self.velocity, axis=-1)


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), _EPS)


def flat_state(
    aircraft: Aircraft,
    t: np.ndarray,
    position: np.ndarray,
    velocity: np.ndarray,
    acceleration: np.ndarray,
    wind: np.ndarray,
) -> FlatState:
    """Evaluate the flat map. ``wind`` is ``W(position, t)``, already sampled."""
    position, velocity, acceleration, wind = np.broadcast_arrays(
        *(np.asarray(x, dtype=float) for x in (position, velocity, acceleration, wind))
    )
    air_velocity = velocity - wind
    airspeed = np.linalg.norm(air_velocity, axis=-1)
    v_hat = _unit(air_velocity)

    force = aircraft.mass * (acceleration + aircraft.g * _UP)
    axial = np.sum(force * v_hat, axis=-1)
    lift = force - axial[..., None] * v_hat
    lift_norm = np.linalg.norm(lift, axis=-1)

    q_s = aircraft.dynamic_pressure(airspeed) * aircraft.wing_area
    cl = lift_norm / np.maximum(q_s, _EPS)
    drag = q_s * aircraft.drag_coefficient(cl)

    up_perp = _unit(_UP - np.sum(_UP * v_hat, axis=-1, keepdims=True) * v_hat)
    right = np.cross(v_hat, up_perp)
    bank = np.arctan2(np.sum(lift * right, axis=-1), np.sum(lift * up_perp, axis=-1))

    return FlatState(
        t=np.asarray(t, dtype=float),
        position=position,
        velocity=velocity,
        acceleration=acceleration,
        wind=wind,
        air_velocity=air_velocity,
        airspeed=airspeed,
        air_path_angle=np.arcsin(np.clip(v_hat[..., 2], -1.0, 1.0)),
        lift=lift,
        cl=cl,
        load_factor=lift_norm / aircraft.weight,
        bank=bank,
        drag=drag,
        thrust_required=axial + drag,
    )
