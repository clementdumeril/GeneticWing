"""Closed-loop point-mass simulation: a planned trajectory flown in wind.

The tracker follows the reference as a **path**, not a clock: at each step the
reference point is the closest sample ahead of the previous one, as the ARPL
planners do by parameterizing the reference with path length (Morando et al.,
ICRA 2025 and arXiv 2605.15619). A glider cannot catch up with a time-indexed
reference after a headwind gust, so time tracking would be the wrong baseline.

Around that point the commanded acceleration is

    a_c = a_r + k_v e_v + k_p e_p + k_z e_z z_hat + k_speed (V_a - V_ref) z_hat,

with ``e_p, e_v`` the position and velocity errors normal to the path,
``e_z`` the altitude error, and the last term trading altitude for airspeed
(descend when slow). The flat map turns ``a_c`` into lift and bank commands,
which reach the airframe through a first-order lag and a bank-rate limit
standing in for the autopilot's attitude loops. Without an engine the axial
force is drag alone.

The simulator returns a :class:`~geneticwing.flatness.FlatState` rebuilt from
the integrated states, so the same metrics apply to plans and to flights.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from geneticwing.aircraft import Aircraft
from geneticwing.flatness import FlatState, flat_state
from geneticwing.wind.base import WindField

_UP = np.array([0.0, 0.0, 1.0])


@dataclass(frozen=True)
class TrackerConfig:
    k_p: float = 0.3  # 1/s^2, cross-track position
    k_v: float = 1.0  # 1/s, cross-track velocity
    k_z: float = 0.05  # 1/s^2, altitude, kept soft: altitude is the energy buffer
    k_speed: float = 0.8  # 1/s, airspeed error -> vertical acceleration
    attitude_time_constant: float = 0.3  # s, lag on C_L and bank
    bank_rate_max_deg: float = 60.0  # deg/s
    bank_max_deg: float = 45.0
    powered: bool = False
    thrust_max: float | None = None  # N, motor limit when powered; None = the aircraft's max_thrust
    cl_max_fraction: float = 1.0  # angle-of-attack protection: C_L commands capped at this fraction of C_L max
    search_window: int = 50  # reference samples scanned ahead of the last match


def _air_frame(v_air: np.ndarray):
    """Unit air velocity, 'up' normal to it, and right wing direction; batched over ``(..., 3)``."""
    v_hat = v_air / np.maximum(np.linalg.norm(v_air, axis=-1, keepdims=True), 1e-9)
    up_perp = _UP - np.sum(_UP * v_hat, axis=-1, keepdims=True) * v_hat
    up_perp /= np.maximum(np.linalg.norm(up_perp, axis=-1, keepdims=True), 1e-9)
    return v_hat, up_perp, np.cross(v_hat, up_perp)


def aerodynamic_acceleration(
    aircraft: Aircraft, v_air: np.ndarray, cl: np.ndarray, bank: np.ndarray, thrust: np.ndarray
) -> np.ndarray:
    """Inertial acceleration of the point mass for given lift coefficient, bank and thrust (batched)."""
    v_hat, up_perp, right = _air_frame(v_air)
    q_s = aircraft.dynamic_pressure(np.linalg.norm(v_air, axis=-1)) * aircraft.wing_area
    cl, bank, thrust = (np.asarray(x, dtype=float)[..., None] for x in (cl, bank, thrust))
    lift = q_s[..., None] * cl * (np.cos(bank) * up_perp + np.sin(bank) * right)
    drag = q_s[..., None] * aircraft.drag_coefficient(cl)
    return (lift + (thrust - drag) * v_hat) / aircraft.mass - aircraft.g * _UP


def rk4_step(accel, p, v, t: float, dt: float):
    """One classical Runge-Kutta step of ``p' = v, v' = accel(p, v, t)``.

    Returns the new position and velocity, and the acceleration at the start of
    the step (the one a scorer should pair with ``(p, v)``). Explicit Euler is
    not good enough here: lift turns the velocity, and every Euler step adds
    ``(a dt)^2`` to its square, which a search learns to harvest as free
    energy in tight turns. Works on NumPy arrays and PyTorch tensors alike.
    """
    a1 = accel(p, v, t)
    v2 = v + 0.5 * dt * a1
    a2 = accel(p + 0.5 * dt * v, v2, t + 0.5 * dt)
    v3 = v + 0.5 * dt * a2
    a3 = accel(p + 0.5 * dt * v2, v3, t + 0.5 * dt)
    v4 = v + dt * a3
    a4 = accel(p + dt * v3, v4, t + dt)
    p_new = p + dt / 6.0 * (v + 2.0 * v2 + 2.0 * v3 + v4)
    v_new = v + dt / 6.0 * (a1 + 2.0 * a2 + 2.0 * a3 + a4)
    return p_new, v_new, a1


def simulate_tracking(
    aircraft: Aircraft,
    reference: FlatState,
    wind: WindField,
    initial_position: np.ndarray,
    initial_velocity: np.ndarray,
    config: TrackerConfig = TrackerConfig(),
    dt: float = 0.01,
    max_duration: float | None = None,
    initial_attitude: tuple[float, float] | None = None,
    extra_drag=None,
) -> FlatState:
    """Fly ``reference`` (a densely sampled plan) until its end point is passed.

    Commands are updated at ``1 / dt`` Hz and held over each RK4 step. The
    attitude loops start from ``initial_attitude`` (C_L, bank in rad), or from
    the reference's first sample; a replanning loop passes the attitude the
    previous segment ended with. ``extra_drag(airspeed, bank, cl)`` (N) adds a
    drag the point mass does not have, e.g. the aileron drag of a strip wing.
    """
    max_duration = 2.0 * float(reference.t[-1]) if max_duration is None else max_duration
    steps = int(round(max_duration / dt))
    p = np.empty((steps + 1, 3))
    v = np.empty((steps + 1, 3))
    acc = np.empty((steps + 1, 3))
    p[0], v[0] = initial_position, initial_velocity

    bank_max = np.radians(config.bank_max_deg)
    bank_rate_max = np.radians(config.bank_rate_max_deg)
    alpha = dt / (config.attitude_time_constant + dt)
    last = reference.position.shape[0] - 1
    end_tangent = reference.velocity[last] / np.linalg.norm(reference.velocity[last])

    idx, cl, bank = 0, float(reference.cl[0]), float(reference.bank[0])
    if initial_attitude is not None:
        cl, bank = (float(x) for x in initial_attitude)
    thrust_max = aircraft.max_thrust if config.thrust_max is None else config.thrust_max
    k = 0
    for k in range(steps + 1):
        t = k * dt
        window = slice(idx, min(idx + config.search_window, last) + 1)
        idx += int(np.argmin(np.linalg.norm(reference.position[window] - p[k], axis=-1)))
        p_r, v_r, a_r = reference.position[idx], reference.velocity[idx], reference.acceleration[idx]
        tangent = v_r / np.linalg.norm(v_r)

        def normal(x):
            return x - (x @ tangent) * tangent

        w = wind(p[k], t)
        airspeed = np.linalg.norm(v[k] - w)
        a_c = (
            a_r
            + config.k_v * normal(v_r - v[k])
            + config.k_p * normal(p_r - p[k])
            + (config.k_z * (p_r[2] - p[k, 2]) + config.k_speed * (airspeed - reference.airspeed[idx])) * _UP
        )
        c = flat_state(aircraft, t, p[k], v[k], a_c, w)
        cl += alpha * (np.clip(float(c.cl), 0.0, config.cl_max_fraction * aircraft.cl_max) - cl)
        bank_step = alpha * (np.clip(float(c.bank), -bank_max, bank_max) - bank)
        bank += np.clip(bank_step, -bank_rate_max * dt, bank_rate_max * dt)
        thrust = np.clip(float(c.thrust_required), 0.0, thrust_max) if config.powered else 0.0
        if extra_drag is not None:
            thrust -= float(extra_drag(airspeed, bank, cl))

        def accel(pos, vel, time, cl=cl, bank=bank, thrust=thrust):
            return aerodynamic_acceleration(aircraft, vel - wind(pos, time), cl, bank, thrust)

        p_next, v_next, acc[k] = rk4_step(accel, p[k], v[k], t, dt)

        passed_end = idx == last and (p[k] - reference.position[last]) @ end_tangent >= 0.0
        if passed_end or k == steps:
            break
        p[k + 1], v[k + 1] = p_next, v_next

    n = k + 1
    t = np.arange(n) * dt
    return flat_state(aircraft, t, p[:n], v[:n], acc[:n], wind(p[:n], t))
