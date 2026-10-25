"""Soaring problems: genome, flat evaluation and energy fitness, batched over a population.

Genome
------
A trajectory is a clamped B-spline over a fixed horizon. Its first three
control points are fixed by the initial state; every later control point is
placed "turtle style" from the previous one, by three genes:

* a turn rate ``omega`` (heading change = ``omega`` times the step duration,
  so a constant rate is a constant-curvature turn even where the clamped
  spline's steps shorten),
* a ground speed ``V`` (step length = ``V`` times the Greville spacing),
* a path angle ``gamma`` (climb or descent).

A circle is a run of equal turn rates and a straight glide is zeros, so
random genomes already look like flyable paths, and crossover swaps whole
manoeuvre segments. Genes are normalized to [0, 1] for the optimizers.

Fitness
-------
The energy objective of the GeneticWing design document, in joules:

    net = E_prop - m g (E_spec(T) - E_spec(0)) / eta
    E_prop = int max(0, T_req) V_a dt / eta

i.e. onboard energy spent by the motor minus the specific energy stored in
the airframe (valued at what the motor would have paid for it). It is
normalized by the energy of level cruise at best L/D over the same horizon,
so ``cost = 1`` is "flew level with the motor" and ``cost < 0`` is a net gain.

Constraints (thrust range, C_L margin, airspeed, bank, load factor, altitude,
terrain, mission) are hinge functions, positive when violated. Ranking uses
Deb's feasibility rules: any feasible trajectory beats any infeasible one,
and infeasible ones are ranked by total violation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from geneticwing.aircraft import Aircraft
from geneticwing.flatness import FlatState, flat_state
from geneticwing.spline import ClampedBSpline
from geneticwing.wind.base import WindField

INFEASIBLE_OFFSET = 1e3


@dataclass(frozen=True)
class LoiterMission:
    """Stay within ``radius`` (m) of ``center`` for the whole horizon."""

    center: tuple[float, float]
    radius: float


@dataclass(frozen=True)
class GoalMission:
    """End the horizon within ``radius`` (m) of ``goal``."""

    goal: tuple[float, float]
    radius: float


@dataclass(frozen=True)
class SoaringScenario:
    name: str
    aircraft: Aircraft
    wind: WindField
    start: tuple[float, float, float]
    heading_deg: float  # compass heading of the initial air velocity
    airspeed: float  # initial airspeed, m/s, level flight
    horizon: float  # s
    mission: LoiterMission | GoalMission
    thrust_max: float  # N, motor available on top of the airframe
    z_min: float = 30.0  # m
    clearance: Callable[[np.ndarray], np.ndarray] | None = None  # signed distance to terrain, m
    clearance_min: float = 15.0  # m
    n_control: int = 24
    degree: int = 5
    n_samples: int = 150
    turn_rate_max_deg: float = 30.0  # deg/s; 20 m circles at 9 m/s need about 26 deg/s
    climb_max_deg: float = 12.0
    ground_speed_bounds: tuple[float, float] = (4.0, 22.0)
    airspeed_max: float = 25.0
    cl_margin: float = 0.9
    bank_max_deg: float = 45.0
    load_max: float = 2.5
    turn_radius_min: float | None = None  # m, coordinated-turn radius V_a^2 / (g tan|bank|); None = no limit
    brake_capacity: float = 0.05  # fraction of weight the airframe can shed (spoilers, slips)
    efficiency: float = 0.5  # propulsive efficiency
    ks_sharpness: float = 200.0  # KS aggregation of time-series constraints (conservative by log(N) / rho)


@dataclass
class Evaluation:
    """Per-individual results, each ``(P,)`` unless noted."""

    cost: np.ndarray  # normalized net energy
    net_energy: np.ndarray  # J
    propulsive_energy: np.ndarray  # J
    stored_energy: np.ndarray  # J, m g delta E_spec
    altitude_gain: np.ndarray  # m
    violation: np.ndarray
    constraints: np.ndarray  # (P, C), KS-aggregated, feasible when <= 0
    fitness: np.ndarray  # Deb ranking key, lower is better
    state: FlatState | None = field(default=None, repr=False)

    @property
    def feasible(self) -> np.ndarray:
        return self.violation <= 0.0


class SoaringProblem:
    """Batched decoding and evaluation of genomes for one scenario."""

    CONSTRAINTS = (
        "thrust_max",
        "brake",
        "cl",
        "airspeed_max",
        "bank",
        "load",
        "turn_radius",
        "altitude",
        "terrain",
        "mission",
    )

    def __init__(self, scenario: SoaringScenario):
        self.scenario = sc = scenario
        ac = sc.aircraft
        self.spline = ClampedBSpline(sc.n_control, sc.degree)
        self.tau = np.linspace(0.0, 1.0, sc.n_samples)
        self.t = self.tau * sc.horizon
        self._bases = [self.spline.basis(self.tau, k) / sc.horizon**k for k in range(3)]

        # Initial state: level flight at the given airspeed and heading, through the wind.
        h = np.radians(sc.heading_deg)
        v_air = sc.airspeed * np.array([np.sin(h), np.cos(h), 0.0])
        p0 = np.asarray(sc.start, dtype=float)
        self.v0 = v_air + sc.wind(p0, 0.0)
        self.fixed = self.spline.initial_control_points(p0, self.v0, np.zeros(3), sc.horizon)
        self.n_free = sc.n_control - 3
        self.dim = 3 * self.n_free
        steps = np.diff(self.spline.greville) * sc.horizon
        self.step_durations = steps[2:]  # c_2 -> c_3, ..., c_{n-2} -> c_{n-1}
        d = self.fixed[2] - self.fixed[1]
        self.heading0 = float(np.arctan2(d[1], d[0]))  # math angle from +x

        turn, climb = np.radians(sc.turn_rate_max_deg), np.radians(sc.climb_max_deg)
        v_lo, v_hi = sc.ground_speed_bounds
        self.lower = np.tile([-turn, v_lo, -climb], self.n_free)
        self.upper = np.tile([turn, v_hi, climb], self.n_free)

        best_drag = ac.weight / ac.max_glide_ratio
        self.reference_energy = best_drag * ac.min_drag_airspeed * sc.horizon / sc.efficiency

    # -- genome -----------------------------------------------------------------
    def to_physical(self, x: np.ndarray) -> np.ndarray:
        return self.lower + np.clip(x, 0.0, 1.0) * (self.upper - self.lower)

    def to_normalized(self, genes: np.ndarray) -> np.ndarray:
        return (genes - self.lower) / (self.upper - self.lower)

    def straight_genome(self) -> np.ndarray:
        """Normalized genome continuing the initial velocity in a straight line."""
        speed = np.linalg.norm(self.v0)
        gamma = np.arcsin(self.v0[2] / speed)
        genes = np.tile([0.0, speed, gamma], self.n_free)
        return np.clip(self.to_normalized(genes), 0.0, 1.0)

    def decode(self, x: np.ndarray) -> np.ndarray:
        """Normalized genomes ``(P, dim)`` -> control points ``(P, n_control, 3)``."""
        g = self.to_physical(np.atleast_2d(x)).reshape(-1, self.n_free, 3)
        heading = self.heading0 + np.cumsum(g[..., 0] * self.step_durations, axis=-1)
        step = g[..., 1] * self.step_durations
        cos_gamma = np.cos(g[..., 2])
        increments = np.stack(
            [step * cos_gamma * np.cos(heading), step * cos_gamma * np.sin(heading), step * np.sin(g[..., 2])],
            axis=-1,
        )
        free = self.fixed[2] + np.cumsum(increments, axis=1)
        fixed = np.broadcast_to(self.fixed, (free.shape[0], 3, 3))
        return np.concatenate([fixed, free], axis=1)

    # -- evaluation ---------------------------------------------------------------
    def flat(self, x: np.ndarray) -> FlatState:
        cp = self.decode(x)
        p, v, a = (np.einsum("nm,pmd->pnd", b, cp) for b in self._bases)
        t = np.broadcast_to(self.t, p.shape[:-1])
        return flat_state(self.scenario.aircraft, t, p, v, a, self.scenario.wind(p, t))

    def hinges(self, s: FlatState) -> dict[str, np.ndarray]:
        """Normalized constraint functions, ``(P, N)`` each, positive when violated."""
        sc, ac = self.scenario, self.scenario.aircraft
        w = ac.weight
        p = s.position
        h = {
            "thrust_max": (s.thrust_required - sc.thrust_max) / w,
            "brake": (-s.thrust_required - sc.brake_capacity * w) / w,
            "cl": (s.cl - sc.cl_margin * ac.cl_max) / ac.cl_max,
            "airspeed_max": (s.airspeed - sc.airspeed_max) / sc.airspeed_max,
            "bank": (np.abs(s.bank) - np.radians(sc.bank_max_deg)) / np.radians(sc.bank_max_deg),
            "load": (s.load_factor - sc.load_max) / sc.load_max,
            "altitude": (sc.z_min - p[..., 2]) / 50.0,
        }
        if sc.clearance is not None:
            h["terrain"] = (sc.clearance_min - sc.clearance(p)) / 50.0
        else:
            h["terrain"] = np.full(p.shape[:-1], -1.0)
        if sc.turn_radius_min is not None:
            # R = V^2 / (g tan|bank|) >= R_min  <=>  g tan|bank| <= V^2 / R_min
            h["turn_radius"] = (ac.g * np.tan(np.abs(s.bank)) - s.airspeed**2 / sc.turn_radius_min) / ac.g
        else:
            h["turn_radius"] = np.full(p.shape[:-1], -1.0)
        m = sc.mission
        if isinstance(m, LoiterMission):
            r = np.linalg.norm(p[..., :2] - np.asarray(m.center), axis=-1)
            h["mission"] = (r - m.radius) / m.radius
        else:
            d = np.linalg.norm(p[..., -1, :2] - np.asarray(m.goal), axis=-1)
            h["mission"] = ((d - m.radius) / m.radius)[..., None]
        return h

    def evaluate(self, x: np.ndarray, keep_state: bool = False) -> Evaluation:
        sc, ac = self.scenario, self.scenario.aircraft
        return self.score(self.flat(x), keep_state)

    def score(self, s: FlatState, keep_state: bool = False) -> Evaluation:
        """Energy and constraints of any batch of trajectories sampled on ``s.t``.

        Planned trajectories and simulated flights are scored by the same code.
        """
        sc, ac = self.scenario, self.scenario.aircraft
        power = np.maximum(s.thrust_required, 0.0) * s.airspeed / sc.efficiency
        propulsive = np.trapezoid(power, s.t, axis=-1)
        e_spec = s.position[..., 2] + s.airspeed**2 / (2.0 * ac.g)
        stored = ac.weight * (e_spec[..., -1] - e_spec[..., 0])
        net = propulsive - stored / sc.efficiency
        cost = net / self.reference_energy

        hinges = self.hinges(s)
        violation = np.zeros(cost.shape)
        aggregated = []
        for name in self.CONSTRAINTS:
            hv = hinges[name]
            positive = np.maximum(hv, 0.0)
            violation += np.mean(positive**2, axis=-1) + np.max(positive, axis=-1)
            rho = sc.ks_sharpness
            peak = np.max(hv, axis=-1, keepdims=True)
            aggregated.append(peak[..., 0] + np.log(np.sum(np.exp(rho * (hv - peak)), axis=-1)) / rho)
        constraints = np.stack(aggregated, axis=-1)
        fitness = np.where(violation <= 0.0, cost, INFEASIBLE_OFFSET + violation)
        return Evaluation(
            cost=cost,
            net_energy=net,
            propulsive_energy=propulsive,
            stored_energy=stored,
            altitude_gain=s.position[..., -1, 2] - s.position[..., 0, 2],
            violation=violation,
            constraints=constraints,
            fitness=fitness,
            state=s if keep_state else None,
        )

    def fitness(self, x: np.ndarray) -> np.ndarray:
        """Ranking key for the evolutionary optimizers, lower is better."""
        return self.evaluate(x).fitness
