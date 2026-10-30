"""Phase 4: plans flown in the real air and replanned from the state reached.

In phase 3 every planner is scored in the air it planned in. Here the two part:

* the **air** is the S1 situation (strip wing included) plus, optionally, Dryden
  turbulence (:class:`~geneticwing.wind.turbulence.DrydenTurbulence`);
* the planner only has a **belief**: the thermal's centre and strength and the
  wind, each off by a sampled error (:func:`sample_belief`).

Every ``replan_every`` seconds (open loop: never) the planner plans the next
60 s in its belief, from the state the aircraft has reached; the path tracker
of :mod:`geneticwing.simulation` flies that plan in the real air (attitude lag,
bank-rate limit, motor), its attitude carried over from one segment to the
next. The whole flight is then scored like a plan, in the real air under the
nominal constraints, and :func:`stall_flags` says whether it actually stalled
(beyond the margins the scores use).

Replanning only helps if it costs nothing when the belief is right, so every
plan aims at the energy at the end of the flight (a shrinking horizon: a plan
scored 60 s ahead puts its gain at the end of its window, and replanning every
10 s would push that gain away for ever), starts from the state reached
(velocity, acceleration, attitude), keeps the orbit direction already taken,
is warm-started from the rest of the previous plan, and has that rest as a
candidate: in its belief a new plan is never worse than carrying on.

Planners take a belief scenario and the :class:`Start` and return a
:class:`Plan` in the world frame: :class:`TeacherPlanner` (the dataset
teacher: law seeds, then the NLP with margins), :class:`LatentPlanner` (the GA
over the command VAE) and :class:`PriorPlanner` (the best of a few prior
samples, no search).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, fields, replace

import numpy as np

from geneticwing.benchmarks import MOTOR_THRUST, canonical_scenario, heading_frame_context, with_margins
from geneticwing.flatness import FlatState, flat_state
from geneticwing.planning.local import solve_local
from geneticwing.policies import law_seeds, seed_theta, simulate_law
from geneticwing.seeding import fit_genome
from geneticwing.simulation import TrackerConfig, simulate_tracking
from geneticwing.soaring import Evaluation, SoaringProblem, SoaringScenario
from geneticwing.strip import turn_loads
from geneticwing.wind.base import UniformWind, WindField
from geneticwing.wind.thermal import Thermal
from geneticwing.wind.turbulence import DrydenTurbulence


# -- belief ------------------------------------------------------------------------------
@dataclass(frozen=True)
class Belief:
    """What the planner thinks the air is: the thermal (core at t = 0, world frame) and the wind."""

    center: tuple[float, float]  # m
    w_star: float  # m/s
    wind: tuple[float, float]  # m/s, horizontal; the thermal is believed to drift with it


def _air_parts(situation: SoaringScenario) -> tuple[Thermal, np.ndarray]:
    parts = situation.wind.components
    thermal = next(p for p in parts if isinstance(p, Thermal))
    wind = next(p for p in parts if isinstance(p, UniformWind)).velocity
    return thermal, np.asarray(wind[:2], dtype=float)


def exact_belief(situation: SoaringScenario) -> Belief:
    thermal, wind = _air_parts(situation)
    return Belief(tuple(thermal.center), float(thermal.w_star), tuple(wind))


def sample_belief(
    situation: SoaringScenario,
    rng: np.random.Generator,
    center_sigma: float = 0.0,
    w_star_sigma: float = 0.0,
    wind_sigma: float = 0.0,
) -> Belief:
    """Belief off by Gaussian errors: centre (m per axis), log of w* and wind (m/s per axis)."""
    thermal, wind = _air_parts(situation)
    center = np.asarray(thermal.center) + center_sigma * rng.standard_normal(2)
    w_star = thermal.w_star * np.exp(w_star_sigma * rng.standard_normal())
    wind = wind + wind_sigma * rng.standard_normal(2)
    return Belief(tuple(center), float(w_star), tuple(wind))


def belief_scenario(
    situation: SoaringScenario,
    belief: Belief,
    t0: float,
    position: np.ndarray,
    velocity: np.ndarray,
    acceleration: np.ndarray | None = None,
    horizon: float | None = None,
) -> SoaringScenario:
    """The planning problem at time ``t0``: the situation as believed, starting from the state reached.

    The believed thermal has drifted with the believed wind since t = 0. The
    initial airspeed and heading are the current ones through the believed air;
    with ``acceleration`` (in flight) the plan also starts from the current
    velocity and acceleration. ``horizon`` is the time left (the flight's end).
    """
    thermal, _ = _air_parts(situation)
    wind = np.array([belief.wind[0], belief.wind[1], 0.0])
    center = np.asarray(belief.center) + wind[:2] * t0
    air = UniformWind(tuple(wind)) + replace(thermal, center=tuple(center), w_star=belief.w_star, drift=tuple(wind[:2]))
    v_air = np.asarray(velocity, dtype=float) - air(np.asarray(position, dtype=float), 0.0)
    in_flight = acceleration is not None
    return replace(
        situation,
        wind=air,
        start=tuple(float(x) for x in position),
        heading_deg=float(np.degrees(np.arctan2(v_air[0], v_air[1])) % 360.0),
        airspeed=float(np.linalg.norm(v_air)),
        initial_velocity=tuple(float(x) for x in velocity) if in_flight else None,
        initial_acceleration=tuple(float(x) for x in acceleration) if in_flight else None,
        horizon=situation.horizon if horizon is None else float(horizon),
    )


@dataclass(frozen=True)
class Shifted(WindField):
    """``base`` seen from a clock started at ``t0``."""

    base: WindField
    t0: float

    def __call__(self, position, t=0.0):
        return self.base(position, np.asarray(t, dtype=float) + self.t0)


def turbulent_air(situation: SoaringScenario, w20: float, duration: float, seed: int) -> WindField:
    """The situation's wind plus Dryden turbulence of intensity ``w20`` (0: none)."""
    if w20 <= 0.0:
        return situation.wind
    _, wind = _air_parts(situation)
    gusts = DrydenTurbulence(
        duration=duration + 10.0,
        airspeed=situation.airspeed,
        altitude=situation.start[2],
        w20=w20,
        heading_deg=float(np.degrees(np.arctan2(wind[0], wind[1]))) if np.any(wind) else 0.0,
        seed=seed,
    )
    return situation.wind + gusts


# -- planners ------------------------------------------------------------------------------
def _select(state: FlatState, k: int) -> FlatState:
    return FlatState(**{f.name: getattr(state, f.name)[k] for f in fields(state)})


@dataclass
class Plan:
    reference: FlatState  # world frame, clock starting at the planning time
    predicted_cost: float  # in the belief
    predicted_feasible: bool
    commands: np.ndarray | None = None  # unit-box commands (knots * 3) of the command planners


@dataclass(frozen=True)
class Start:
    """What a plan made in flight starts from, besides the position, velocity and acceleration."""

    cl: float
    bank: float  # rad, positive right wing down (clockwise turn)
    thrust: float  # N
    previous: Plan | None = None  # the plan being flown
    elapsed: float = 0.0  # s flown since it was made


def _directions(start: Start | None, threshold_deg: float = 10.0) -> tuple[bool, ...]:
    """Orbit directions to plan in (True = counter-clockwise): the current one once a turn is taken."""
    if start is None or start.previous is None or abs(start.bank) < np.radians(threshold_deg):
        return (True, False)
    return (bool(start.bank < 0.0),)


def _remainder(plan: Plan, elapsed: float, horizon: float) -> tuple[np.ndarray, np.ndarray]:
    """Times and positions of the rest of ``plan`` after ``elapsed``, carried on straight up to ``horizon``."""
    ref = plan.reference
    keep = ref.t > elapsed
    here = np.array([np.interp(elapsed, ref.t, ref.position[:, i]) for i in range(3)])
    t, p = np.concatenate([[0.0], ref.t[keep] - elapsed]), np.vstack([here, ref.position[keep]])
    extra = np.linspace(t[-1], horizon, 6)[1:]
    return np.concatenate([t, extra]), np.vstack([p, p[-1] + (extra - t[-1])[:, None] * ref.velocity[-1]])


class TeacherPlanner:
    """The dataset's teacher: each law seed fitted and refined by the NLP with margins.

    In flight the orbit direction already taken is kept, and the rest of the
    plan being flown is one more NLP start.
    """

    def __init__(self, theta: np.ndarray, nlp_budget: int = 6_000):
        self.theta, self.nlp_budget = np.asarray(theta), nlp_budget

    def __call__(self, scenario: SoaringScenario, rng: np.random.Generator, start: Start | None = None) -> Plan:
        robust = SoaringProblem(with_margins(scenario))
        genomes = []
        for ccw in _directions(start):
            for overrides in law_seeds(scenario.strip_wing is not None):
                law = simulate_law(robust, seed_theta(self.theta, ccw, overrides)[None])
                genomes.append(fit_genome(robust, law.state.t[0], law.state.position[0]))
        if start is not None and start.previous is not None:
            genomes.append(fit_genome(robust, *_remainder(start.previous, start.elapsed, scenario.horizon)))
        best = None
        for genome in genomes:
            refined, _ = solve_local(robust, genome, self.nlp_budget)
            if best is None or refined.f < best.f:  # Deb ranking: feasible first, then cost
                best = refined
        ev = SoaringProblem(scenario).evaluate(best.x[None], keep_state=True)
        return Plan(_select(ev.state, 0), float(ev.cost[0]), bool(ev.feasible[0]))


class _FirstKnot:
    """A command VAE whose decoded plans start from the current commands (first knot overridden)."""

    def __init__(self, vae, first: np.ndarray):
        self.vae, self.first, self.latent_dim = vae, first, vae.latent_dim

    def decode(self, z, context):
        u = np.array(self.vae.decode(z, context))
        u[..., :3] = self.first
        return u


class _CommandPlanner:
    """Shared by the command-VAE planners: canonical frame in, world frame out, warm start in flight."""

    def __init__(self, vae, cfg):
        self.vae, self.cfg = vae, cfg

    def _problem(self, scenario: SoaringScenario):
        context = heading_frame_context(scenario)
        cfg = replace(self.cfg, initial_airspeed=scenario.airspeed)
        canonical = canonical_scenario(context, airspeed=scenario.airspeed, strip_wing=scenario.strip_wing,
                                       aircraft=scenario.aircraft, horizon=scenario.horizon)
        return context, cfg, SoaringProblem(canonical)

    def _in_flight(self, start: Start | None, context: np.ndarray, cfg):
        """Decoder to use, the rest of the previous plan's commands (a candidate), and its latent code."""
        from geneticwing.physvae import to_unit_commands

        if start is None or start.previous is None or start.previous.commands is None:
            return self.vae, None, None
        first = to_unit_commands(np.array([[[start.cl, start.bank, start.thrust]]]), cfg)[0]
        knots = start.previous.commands.reshape(cfg.knots, 3)
        shift = min(int(round(start.elapsed / (cfg.horizon / (cfg.knots - 1)))), cfg.knots - 1)
        rest = np.vstack([knots[shift:], np.repeat(knots[-1:], shift, axis=0)]).ravel()
        rest[:3] = first
        return _FirstKnot(self.vae, first), rest[None], self.vae.encode(rest[None], context[None])

    def _to_world(self, ev: Evaluation, k: int, scenario: SoaringScenario, commands: np.ndarray) -> Plan:
        s = _select(ev.state, k)
        h = np.radians(scenario.heading_deg)
        rot = np.array([[np.sin(h), -np.cos(h), 0.0], [np.cos(h), np.sin(h), 0.0], [0.0, 0.0, 1.0]])
        offset = np.array([scenario.start[0], scenario.start[1], 0.0])
        p, v, a = s.position @ rot.T + offset, s.velocity @ rot.T, s.acceleration @ rot.T
        ref = flat_state(scenario.aircraft, s.t, p, v, a, scenario.wind(p, s.t))
        return Plan(ref, float(ev.cost[k]), bool(ev.feasible[k]), np.asarray(commands))


class LatentPlanner(_CommandPlanner):
    """The GA over the command VAE's latent vector (``budget`` flights per plan)."""

    def __init__(self, vae, cfg, budget: int = 3_000):
        super().__init__(vae, cfg)
        self.budget = budget

    def __call__(self, scenario: SoaringScenario, rng: np.random.Generator, start: Start | None = None) -> Plan:
        from geneticwing.physvae import command_latent_plan, fly

        context, cfg, problem = self._problem(scenario)
        decoder, rest, z_rest = self._in_flight(start, context, cfg)
        seeds = np.zeros((1, self.vae.latent_dim))
        if z_rest is not None:
            seeds = np.vstack([seeds, z_rest])
        ev, commands = command_latent_plan(decoder, context, cfg, problem, self.budget, rng, seeds)
        if rest is not None:  # carrying on with the plan being flown is a candidate too
            carry = fly(rest, context, cfg, problem)
            if carry.fitness[0] < ev.fitness[0]:
                ev, commands = carry, rest[0]
        return self._to_world(ev, 0, scenario, commands)


class PriorPlanner(_CommandPlanner):
    """The best of ``samples`` prior samples of the command VAE, flown in the belief: no search."""

    def __init__(self, vae, cfg, samples: int = 64):
        super().__init__(vae, cfg)
        self.samples = samples

    def __call__(self, scenario: SoaringScenario, rng: np.random.Generator, start: Start | None = None) -> Plan:
        from geneticwing.physvae import fly

        context, cfg, problem = self._problem(scenario)
        decoder, rest, z_rest = self._in_flight(start, context, cfg)
        z = rng.standard_normal((self.samples, self.vae.latent_dim))
        candidates = decoder.decode(z if z_rest is None else np.vstack([z, z_rest]), context)
        if rest is not None:
            candidates = np.vstack([candidates, rest])
        ev = fly(candidates, context, cfg, problem)
        k = int(np.argmin(ev.fitness))
        return self._to_world(ev, k, scenario, candidates[k])


# -- the loop ---------------------------------------------------------------------------------
@dataclass(frozen=True)
class LoopConfig:
    duration: float = 60.0  # s
    replan_every: float | None = 10.0  # s; None = plan once and fly it (open loop)
    dt: float = 0.02  # s, tracker step
    # The autopilot: motor, angle-of-attack protection at the nominal C_L margin, bank kept under the 45 deg limit.
    tracker: TrackerConfig = field(default_factory=lambda: TrackerConfig(
        powered=True, thrust_max=MOTOR_THRUST, cl_max_fraction=0.9, bank_max_deg=40.0))


@dataclass
class Flight:
    state: FlatState  # (1, N) samples, world frame, real air
    evaluation: Evaluation  # real air, nominal constraints, normalized by the flight's duration
    stalled: bool  # the C_L or a wing section beyond its stall limit at some point
    plans: list[Plan]
    plan_seconds: list[float]
    violated: list[str]  # constraints exceeded by more than FLIGHT_TOLERANCE

    @property
    def feasible(self) -> bool:
        return not self.violated


# A flown constraint counts as violated beyond 0.1 % of its scale (weight, C_L max...):
# a motor saturated at its limit, or a C_L held at the protection, is not a violation.
FLIGHT_TOLERANCE = 1e-3


def stall_flags(problem: SoaringProblem, state: FlatState) -> np.ndarray:
    """Whether each flight stalls: mean C_L beyond C_L max, or a strip of the wing beyond its section stall."""
    sc, ac = problem.scenario, problem.scenario.aircraft
    stalled = np.max(state.cl, axis=-1) > ac.cl_max
    if sc.strip_wing is not None:
        loads = turn_loads(sc.strip_wing, ac, state.airspeed, state.bank, state.cl)
        stalled |= np.max(loads.cl_max_local, axis=-1) > sc.strip_wing.local_cl_max
    return stalled


def fly_closed_loop(
    situation: SoaringScenario,
    air: WindField,
    belief: Belief,
    planner,
    config: LoopConfig = LoopConfig(),
    rng: np.random.Generator | None = None,
    initial: tuple[np.ndarray, np.ndarray, float, float, float] | None = None,
) -> Flight:
    """Fly ``situation`` in ``air`` for ``config.duration`` seconds, replanning in ``belief``.

    ``initial = (velocity, acceleration, C_L, bank, thrust)`` starts the flight from
    a state reached in flight (a cross-country climb) instead of level flight;
    the first plan then starts from it too.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    ac = situation.aircraft
    thrust_max = config.tracker.thrust_max if config.tracker.thrust_max is not None else ac.max_thrust
    p = np.asarray(situation.start, dtype=float)
    h = np.radians(situation.heading_deg)
    v = situation.airspeed * np.array([np.sin(h), np.cos(h), 0.0]) + air(p, 0.0)
    a, start, attitude = None, None, None
    if initial is not None:
        v, a = np.asarray(initial[0], dtype=float), np.asarray(initial[1], dtype=float)
        attitude = (float(initial[2]), float(initial[3]))
        start = Start(attitude[0], attitude[1], float(initial[4]))
    wing = situation.strip_wing
    extra_drag = None if wing is None else (lambda va, bank, cl: turn_loads(wing, ac, va, bank, cl).extra_drag)
    t0, pieces, plans, seconds = 0.0, [], [], []
    while config.duration - t0 > 2.0 * config.dt:
        tic = time.perf_counter()
        left = config.duration - t0
        plan = planner(belief_scenario(situation, belief, t0, p, v, a, min(left, situation.horizon)), rng, start)
        seconds.append(time.perf_counter() - tic)
        plans.append(plan)
        span = left if config.replan_every is None else min(config.replan_every, left)
        seg = simulate_tracking(ac, plan.reference, Shifted(air, t0), p, v, config.tracker, dt=config.dt,
                                max_duration=span, initial_attitude=attitude, extra_drag=extra_drag)
        pieces.append((seg, t0))
        p, v, a = seg.position[-1], seg.velocity[-1], seg.acceleration[-1]
        attitude = (float(seg.cl[-1]), float(seg.bank[-1]))
        start = Start(attitude[0], attitude[1], float(np.clip(seg.thrust_required[-1], 0.0, thrust_max)),
                      plan, float(seg.t[-1]))
        t0 += float(seg.t[-1])
        if seg.t[-1] < span - 1.5 * config.dt:  # the plan ended before the segment did
            break

    # Stitch the segments: each one starts where the previous ended.
    t = np.concatenate([piece.t[(i > 0):] + t_start for i, (piece, t_start) in enumerate(pieces)])
    pos, vel, acc = (np.concatenate([getattr(piece, name)[(i > 0):] for i, (piece, _) in enumerate(pieces)])
                     for name in ("position", "velocity", "acceleration"))
    state = flat_state(ac, t[None], pos[None], vel[None], acc[None], air(pos, t)[None])
    problem = SoaringProblem(replace(situation, wind=air, horizon=float(t[-1])))
    ev = problem.score(state)
    hinges = problem.hinges(state)
    violated = [name for name in SoaringProblem.CONSTRAINTS if np.max(hinges[name]) > FLIGHT_TOLERANCE]
    return Flight(state, ev, bool(stall_flags(problem, state)[0]), plans, seconds, violated)
