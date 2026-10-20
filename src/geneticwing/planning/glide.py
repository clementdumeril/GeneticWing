"""Local (NLP) planner for an unpowered glide between two points.

This is the baseline of scenario S0: the counterpart, in this package's flat
formulation, of the gliding mode of Morando et al. (arXiv 2605.15619). That
paper minimizes jerk under a sink-rate band around the still-air polar and an
airspeed band around ``V_a,ref``, solved with CasADi/IPOPT. Here the sink-rate
band is replaced by its generalization ``|T_req| <= eps * W`` (no thrust
needed), the airspeed band is kept, and the problem is solved with SciPy's
SLSQP on one Bernstein segment.

Decision variables, for a segment of degree ``n``:

* control points ``c_3 .. c_{n-1}``;
* the altitude of ``c_n`` (its horizontal position is the goal);
* the log of the duration.

``c_0, c_1, c_2`` are fixed by the initial position, an initial steady-glide
velocity along the track to the goal, and zero initial acceleration.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from geneticwing.aircraft import Aircraft
from geneticwing.bernstein import evaluate
from geneticwing.flatness import FlatState, flat_state
from geneticwing.wind.base import WindField


@dataclass(frozen=True)
class GlideProblem:
    aircraft: Aircraft
    wind: WindField
    start: tuple[float, float, float]
    goal_xy: tuple[float, float]
    airspeed_ref: float
    degree: int = 8
    n_samples: int = 60
    airspeed_tol: float = 0.3  # m/s, half-width of the airspeed band
    thrust_tol: float = 0.02  # fraction of weight
    bank_max_deg: float = 35.0
    cl_margin: float = 0.9  # fraction of cl_max allowed

    @property
    def track(self) -> np.ndarray:
        d = np.asarray(self.goal_xy, dtype=float) - np.asarray(self.start[:2], dtype=float)
        return d / np.linalg.norm(d)

    def initial_velocity(self) -> np.ndarray:
        """Ground velocity of a steady glide at ``airspeed_ref`` whose ground track points at the goal."""
        gamma, _, _ = self.aircraft.steady_glide(self.airspeed_ref)
        wind = self.wind(np.asarray(self.start, dtype=float), 0.0)
        h = self.track
        w_h = wind[:2]
        v_air_h = self.airspeed_ref * np.cos(gamma)
        along = h @ w_h
        discriminant = along**2 - w_h @ w_h + v_air_h**2
        if discriminant < 0:
            raise ValueError("crosswind exceeds the horizontal airspeed: the goal track is not flyable")
        ground_speed = along + np.sqrt(discriminant)
        return np.array([*(ground_speed * h), wind[2] + self.airspeed_ref * np.sin(gamma)])


@dataclass
class GlidePlan:
    problem: GlideProblem
    control_points: np.ndarray  # (n + 1, 3)
    duration: float
    success: bool
    message: str
    iterations: int
    cost: float = field(default=np.nan)

    def sample(self, n: int = 400, wind: WindField | None = None) -> FlatState:
        """Flat state along the plan, optionally re-evaluated in another wind field."""
        return _flat(self.problem, self.control_points, self.duration, np.linspace(0, 1, n), wind)

def _flat(problem: GlideProblem, cp, duration, tau, wind=None) -> FlatState:
    wind = problem.wind if wind is None else wind
    p, v, a = (evaluate(cp, duration, tau, k) for k in range(3))
    t = tau * duration
    return flat_state(problem.aircraft, t, p, v, a, wind(p, t))


class _Layout:
    """Maps the decision vector to control points and back."""

    def __init__(self, problem: GlideProblem):
        self.problem = problem
        self.n = problem.degree
        if self.n < 4:
            raise ValueError("degree must be at least 4")
        self.start = np.asarray(problem.start, dtype=float)
        self.v0 = problem.initial_velocity()
        self.n_free = self.n - 3  # c_3 .. c_{n-1}

    def control_points(self, x: np.ndarray) -> tuple[np.ndarray, float]:
        duration = float(np.exp(x[-1]))
        c0 = self.start
        c1 = c0 + self.v0 * duration / self.n
        c2 = 2.0 * c1 - c0
        free = x[: 3 * self.n_free].reshape(self.n_free, 3)
        cn = np.array([*self.problem.goal_xy, x[3 * self.n_free]])
        return np.vstack([c0, c1, c2, free, cn]), duration

    def vector(self, cp: np.ndarray, duration: float) -> np.ndarray:
        return np.concatenate([cp[3:-1].ravel(), [cp[-1, 2], np.log(duration)]])

    def straight_glide(self) -> tuple[np.ndarray, float]:
        """Exact solution in uniform wind: the straight steady glide, as an initial guess."""
        horizontal = np.linalg.norm(np.asarray(self.problem.goal_xy) - self.start[:2])
        duration = horizontal / np.linalg.norm(self.v0[:2])
        end = self.start + self.v0 * duration
        alphas = np.linspace(0.0, 1.0, self.n + 1)[:, None]
        return self.start + alphas * (end - self.start), duration


def plan_glide(
    problem: GlideProblem,
    initial: tuple[np.ndarray, float] | None = None,
    max_iterations: int = 300,
) -> GlidePlan:
    """Minimum-jerk glide to the goal under no-thrust, airspeed, C_L and bank constraints."""
    layout = _Layout(problem)
    tau = np.linspace(0.0, 1.0, problem.n_samples)
    ac = problem.aircraft
    bank_max = np.radians(problem.bank_max_deg)
    cp0, duration0 = layout.straight_glide() if initial is None else initial
    x0 = layout.vector(cp0, duration0)

    def jerk_cost(x):
        cp, duration = layout.control_points(x)
        jerk = evaluate(cp, duration, tau, 3)
        return float(np.mean(np.sum(jerk**2, axis=-1)) * duration)

    def constraints(x):
        cp, duration = layout.control_points(x)
        s = _flat(problem, cp, duration, tau)
        thrust = s.thrust_required / ac.weight
        dv = s.airspeed - problem.airspeed_ref
        return np.concatenate(
            [
                problem.thrust_tol - thrust,
                problem.thrust_tol + thrust,
                problem.airspeed_tol - dv,
                problem.airspeed_tol + dv,
                problem.cl_margin * ac.cl_max - s.cl,
                bank_max**2 - s.bank**2,
            ]
        )

    result = minimize(
        jerk_cost,
        x0,
        method="SLSQP",
        constraints=[{"type": "ineq", "fun": constraints}],
        options={"maxiter": max_iterations, "ftol": 1e-9},
    )
    cp, duration = layout.control_points(result.x)
    feasible = bool(np.all(constraints(result.x) >= -1e-6))
    return GlidePlan(
        problem=problem,
        control_points=cp,
        duration=duration,
        success=bool(result.success) and feasible,
        message=str(result.message),
        iterations=int(result.nit),
        cost=float(result.fun),
    )
