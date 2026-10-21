"""Reactive soaring law tuned offline: the "a la Zohdi" baseline.

T. Zohdi's drone-swarm work optimizes, offline and by a genetic algorithm, the
few parameters of the equations each agent follows, then runs those equations
as they are. This module is that approach applied to soaring: a fixed
guidance law with ten parameters, simulated for a whole population of
parameter sets at once (one agent per parameter set), and tuned by
:class:`~geneticwing.evolution.ZohdiGA` over training instances.

The law knows where the lift is (thermal centers and drift, ridge position),
like the trajectory planners it is compared with:

* **orbit** an anchor point (lift source + tuned offset) with a vector-field
  course law (Beard and McLain, Small Unmanned Aircraft, ch. 10);
* in S2, **hop** from thermal to thermal, leaving each one above a tuned
  altitude or when the remaining time only just covers the distance to the
  goal, then fly to the goal and orbit it;
* hold an orbit or cruise **airspeed** by trading altitude, and run the
  **motor** only below a tuned altitude floor.

Like the trajectory tracker, the law commands an acceleration and lets the
flat map turn it into C_L, bank and thrust; a load limiter caps C_L.

Commands reach the airframe through the same attitude lag and bank-rate limit
as the trajectory tracker, and flights are scored by
:meth:`SoaringProblem.score`, the code that scores planned trajectories.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from geneticwing.evolution import ZohdiGA
from geneticwing.flatness import flat_state
from geneticwing.simulation import aerodynamic_acceleration
from geneticwing.soaring import Evaluation, GoalMission, SoaringProblem
from geneticwing.wind.base import CompositeWind
from geneticwing.wind.ridge import RidgeLift
from geneticwing.wind.thermal import Thermal

N_PARAMS = 10


@dataclass(frozen=True)
class LawSettings:
    attitude_time_constant: float = 0.3
    bank_rate_max_deg: float = 60.0
    bank_max_deg: float = 40.0  # below the planners' 45 deg constraint
    k_energy: float = 0.5  # vertical speed (m/s) traded per m/s of airspeed error
    k_vertical: float = 1.0  # 1/s, vertical speed loop
    k_altitude: float = 0.2  # 1/s, climb rate commanded per metre below the floor
    substeps: int = 8  # integration steps per scoring sample


def lift_anchors(problem: SoaringProblem) -> list[tuple[np.ndarray, np.ndarray]]:
    """``(center at t = 0, drift velocity)`` of each lift source, in flight order."""
    wind = problem.scenario.wind
    parts = wind.components if isinstance(wind, CompositeWind) else (wind,)
    anchors = []
    for part in parts:
        if isinstance(part, Thermal):
            anchors.append((np.asarray(part.center, float), np.asarray(part.drift, float)))
        elif isinstance(part, RidgeLift):
            anchors.append((np.asarray(part.crest, float), np.zeros(2)))
    anchors.sort(key=lambda a: a[0][0])
    return anchors


def decode(theta: np.ndarray, problem: SoaringProblem) -> dict[str, np.ndarray]:
    v_md = problem.scenario.aircraft.min_drag_airspeed
    th = np.clip(np.atleast_2d(theta), 0.0, 1.0)

    def span(i, lo, hi):
        return lo + th[:, i] * (hi - lo)

    return {
        "radius": span(0, 20.0, 150.0),
        "v_orbit": span(1, 0.85 * v_md, 1.5 * v_md),
        "v_cruise": span(2, 0.9 * v_md, 1.8 * v_md),
        "k_orbit": span(3, 0.5, 4.0),
        "k_course": span(4, 0.3, 2.0),
        "z_floor": problem.scenario.start[2] + span(5, -100.0, 20.0),
        "direction": np.where(th[:, 6] > 0.5, 1.0, -1.0),  # +1 counter-clockwise
        "dx": span(7, -120.0, 120.0),
        "dy": span(8, -120.0, 120.0),
        "z_exit": problem.scenario.start[2] + span(9, 0.0, 200.0),
    }


def _wrap(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def simulate_law(problem: SoaringProblem, theta: np.ndarray, settings: LawSettings = LawSettings()) -> Evaluation:
    """Fly every parameter set in ``theta (P, 10)`` through the scenario and score the flights."""
    sc, ac = problem.scenario, problem.scenario.aircraft
    prm = decode(theta, problem)
    n_agents = prm["radius"].size
    anchors = lift_anchors(problem)
    goal = sc.mission.goal if isinstance(sc.mission, GoalMission) else None
    n_targets = len(anchors) + (goal is not None)

    n_samples, m = problem.t.size, settings.substeps
    dt = sc.horizon / ((n_samples - 1) * m)
    p = np.tile(np.asarray(sc.start, float), (n_agents, 1))
    v = np.tile(problem.v0, (n_agents, 1))
    target = np.zeros(n_agents, dtype=int)
    cl = np.full(n_agents, 2 * ac.weight / (ac.rho * ac.wing_area * sc.airspeed**2))
    bank = np.zeros(n_agents)
    alpha = dt / (settings.attitude_time_constant + dt)
    bank_max = np.radians(settings.bank_max_deg)
    bank_rate = np.radians(settings.bank_rate_max_deg) * dt
    record = {"p": [], "v": [], "a": []}
    crashed = np.zeros(n_agents, dtype=bool)

    for k in range((n_samples - 1) * m + 1):
        t = k * dt
        w = sc.wind(p, t)
        v_air = v - w
        airspeed = np.linalg.norm(v_air, axis=-1)
        ground_speed = np.maximum(np.linalg.norm(v[:, :2], axis=-1), 1.0)
        course = np.arctan2(v[:, 1], v[:, 0])

        # Current target point: lift anchor (+ tuned offset) or the goal.
        center = np.empty((n_agents, 2))
        orbit_radius = prm["radius"].copy()
        for j, (c0, drift) in enumerate(anchors):
            sel = target == j
            center[sel] = c0 + drift * t + np.stack([prm["dx"][sel], prm["dy"][sel]], axis=-1)
        if goal is not None:
            sel = target == len(anchors)
            center[sel] = goal
            orbit_radius[sel] = 0.5 * sc.mission.radius

        rel = p[:, :2] - center
        distance = np.linalg.norm(rel, axis=-1)
        theta_pos = np.arctan2(rel[:, 1], rel[:, 0])
        lam = prm["direction"]
        course_cmd = theta_pos + lam * (np.pi / 2 + np.arctan(prm["k_orbit"] * (distance - orbit_radius) / orbit_radius))
        near = distance < 2.0 * orbit_radius
        feedforward = np.where(near, lam * ground_speed / orbit_radius, 0.0)
        rate_cmd = prm["k_course"] * _wrap(course_cmd - course) + feedforward
        v_ref = np.where(near & (target < len(anchors)), prm["v_orbit"], prm["v_cruise"])

        if goal is not None:  # thermal hopping: leave high enough, or when time runs short
            to_goal = np.linalg.norm(p[:, :2] - np.asarray(goal), axis=-1)
            late = (sc.horizon - t) < 1.15 * to_goal / prm["v_cruise"]
            done = (p[:, 2] >= prm["z_exit"]) | late
            leave = (target < len(anchors)) & done
            target = np.where(leave, np.where(late, len(anchors), target + 1), target)
            target = np.minimum(target, n_targets - 1)

        # Commanded acceleration: turn at the commanded course rate; vertically, follow the
        # still-air glide at v_ref through the moving air, descending faster when slow and
        # climbing when fast (energy trade). Below the floor the motor climbs back.
        vz_cmd = w[:, 2] - ac.nominal_sink_rate(v_ref) + settings.k_energy * (airspeed - v_ref)
        below = p[:, 2] < prm["z_floor"]
        vz_cmd = np.where(below, np.maximum(vz_cmd, settings.k_altitude * (prm["z_floor"] - p[:, 2])), vz_cmd)
        track = v[:, :2] / ground_speed[:, None]
        a_cmd = np.zeros_like(v)
        a_cmd[:, 0] = -ground_speed * rate_cmd * track[:, 1]
        a_cmd[:, 1] = ground_speed * rate_cmd * track[:, 0]
        a_cmd[:, 2] = settings.k_vertical * (vz_cmd - v[:, 2])
        c = flat_state(ac, t, p, v, a_cmd, w)

        q_s = np.maximum(ac.dynamic_pressure(airspeed) * ac.wing_area, 1e-6)
        cl_limit = np.minimum(sc.cl_margin * ac.cl_max, 0.95 * sc.load_max * ac.weight / q_s)  # load limiter
        cl += alpha * (np.clip(c.cl, 0.0, cl_limit) - cl)
        bank += np.clip(alpha * (np.clip(c.bank, -bank_max, bank_max) - bank), -bank_rate, bank_rate)
        thrust = np.where(below, np.clip(c.thrust_required, 0.0, sc.thrust_max), 0.0)
        a = aerodynamic_acceleration(ac, v_air, cl, bank, thrust)
        # Agents that hit the ground or overspeed stop where they are; their scores are infeasible.
        crashed |= (p[:, 2] < 0.0) | (airspeed > 3.0 * sc.airspeed_max)
        a[crashed] = 0.0
        v[crashed] = 0.0

        if k % m == 0:
            record["p"].append(p.copy())
            record["v"].append(v.copy())
            record["a"].append(a)
        v = v + a * dt
        p = p + v * dt

    P, V, A = (np.stack(record[key], axis=1) for key in ("p", "v", "a"))
    t = np.broadcast_to(problem.t, P.shape[:-1])
    return problem.score(flat_state(ac, t, P, V, A, sc.wind(P, t)), keep_state=True)


def tune_law(
    problems: list[SoaringProblem], budget: int, rng: np.random.Generator, population: int = 40
) -> tuple[np.ndarray, float]:
    """Zohdi-style offline tuning: one parameter set for all training instances.

    The objective averages the Deb fitness over instances, so a set that is
    infeasible anywhere ranks below every set feasible everywhere.
    """

    def objective(theta: np.ndarray) -> np.ndarray:
        return np.mean([simulate_law(pb, theta).fitness for pb in problems], axis=0)

    result = ZohdiGA(population=population, parents=10).minimize(objective, N_PARAMS, budget, rng)
    return result.x, result.f
