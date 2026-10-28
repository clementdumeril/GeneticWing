"""Physics decoder: a VAE over flight commands, decoded through the simulator.

The VAE of :mod:`geneticwing.latent` decodes positions; their small errors
become large errors in acceleration, and only about 20 % of the teacher's own
trajectories survive its round trip. Here the VAE encodes and decodes the
teacher's **commands** (C_L, bank, thrust at regular knots), read off its
trajectory by the flat map, and the point-mass simulator of
:mod:`geneticwing.torchsim` turns decoded commands into the trajectory.
Commands are bounded and the simulator caps the load factor, so C_L, bank,
thrust and load factor hold by construction; only the geometry (zone,
altitude) and the energy are left to the search.

Flown open loop, the teacher's commands drift (about 125 m after 60 s) but
stay feasible in about 90 % of cases at nearly the same energy, so the
command space is a good space for the GA to search.

Training through the simulator (differentiable physics) is also possible
with :func:`geneticwing.torchsim.simulate`, but backpropagating through
60 s of flight is slow and hard to optimize; imitating the commands trains
like an ordinary VAE. The searches fly with the NumPy twin,
:func:`geneticwing.torchsim.simulate_flights`.
"""

from __future__ import annotations

import numpy as np

from geneticwing.benchmarks import canonical_scenario, mirror
from geneticwing.evolution import RealGA
from geneticwing.flatness import flat_state
from geneticwing.latent import ConditionalVAE, to_box, to_latent
from geneticwing.planning.local import solve_local
from geneticwing.seeding import fit_genome
from geneticwing.soaring import Evaluation, SoaringProblem
from geneticwing.torchsim import SimConfig, simulate_flights


def _limits(cfg: SimConfig):
    ac = cfg.aircraft
    return cfg.safety * cfg.cl_limit * ac.cl_max, cfg.safety * np.radians(cfg.bank_max_deg), cfg.thrust_max


def to_unit_commands(commands: np.ndarray, cfg: SimConfig) -> np.ndarray:
    """Physical commands ``(..., knots, 3)`` -> flat unit-box vectors ``(..., knots * 3)``."""
    cl_max, bank_max, thrust_max = _limits(cfg)
    u = np.stack(
        [commands[..., 0] / cl_max, (commands[..., 1] / bank_max + 1.0) / 2.0, commands[..., 2] / thrust_max],
        axis=-1,
    )
    return np.clip(u, 0.0, 1.0).reshape(*u.shape[:-2], -1)


def from_unit_commands(u: np.ndarray, cfg: SimConfig) -> np.ndarray:
    cl_max, bank_max, thrust_max = _limits(cfg)
    u = np.clip(np.asarray(u), 0.0, 1.0).reshape(*np.shape(u)[:-1], cfg.knots, 3)
    return np.stack([u[..., 0] * cl_max, (2.0 * u[..., 1] - 1.0) * bank_max, u[..., 2] * thrust_max], axis=-1)


def mirror_commands(contexts: np.ndarray, unit_commands: np.ndarray):
    """Left/right mirror: lateral context terms and bank change sign (bank unit x -> 1 - x)."""
    c, _ = mirror(contexts, np.zeros((len(contexts), 3)))
    u = unit_commands.copy()
    u[:, 1::3] = 1.0 - u[:, 1::3]
    return c, u


def teacher_commands(genomes: np.ndarray, contexts: np.ndarray, cfg: SimConfig) -> np.ndarray:
    """The commands flying each teacher trajectory, read off by the flat map at the knots."""
    knot_t = np.linspace(0.0, cfg.horizon, cfg.knots)
    ac = cfg.aircraft
    out = np.empty((len(genomes), cfg.knots, 3))
    for i, (g, c) in enumerate(zip(genomes, contexts)):
        pb = SoaringProblem(canonical_scenario(c))
        cp = pb.decode(g[None])[0]
        p, v, a = (pb.spline.evaluate(cp, pb.scenario.horizon, knot_t / pb.scenario.horizon, k) for k in range(3))
        s = flat_state(ac, knot_t, p, v, a, pb.scenario.wind(p, knot_t))
        out[i] = np.stack([s.cl, s.bank, np.clip(s.thrust_required, 0.0, cfg.thrust_max)], axis=-1)
    return to_unit_commands(out, cfg)


def fly(unit_commands: np.ndarray, context: np.ndarray, cfg: SimConfig, problem: SoaringProblem) -> Evaluation:
    """Simulate unit-box command vectors in one situation's canonical frame and score the flights."""
    t, p, v, a = simulate_flights(from_unit_commands(np.atleast_2d(unit_commands), cfg), context, cfg)
    tt = np.broadcast_to(t, p.shape[:-1])
    sc = problem.scenario
    return problem.score(flat_state(sc.aircraft, tt, p, v, a, sc.wind(p, tt)), keep_state=True)


def command_latent_search(
    vae: ConditionalVAE,
    context: np.ndarray,
    cfg: SimConfig,
    budget: int,
    rng: np.random.Generator,
    seeds_z: np.ndarray | None = None,
    nlp_budget: int = 0,
    **scenario_overrides,
):
    """GA over the latent vector of the command VAE; every individual is flown by the simulator.

    With ``nlp_budget``, the best flight is fitted to a genome and polished by
    SLSQP. Extra keyword arguments override fields of the canonical scenario
    (``turn_radius_min``, ``strip_wing``). Returns ``(fitness, cost, feasible)``
    of the final trajectory.
    """
    problem = SoaringProblem(canonical_scenario(context, **scenario_overrides))

    def objective(x):
        return fly(vae.decode(to_latent(x), context), context, cfg, problem).fitness

    seeds = None if seeds_z is None else to_box(seeds_z)
    search = RealGA(population=50).minimize(objective, vae.latent_dim, budget, rng, seeds)
    best = fly(vae.decode(to_latent(search.x[None]), context), context, cfg, problem)
    fitness, cost, feasible = float(best.fitness[0]), float(best.cost[0]), bool(best.feasible[0])
    if nlp_budget:
        s = best.state
        genome = fit_genome(problem, s.t[0], s.position[0])
        polished, _ = solve_local(problem, genome, nlp_budget)
        if polished.f < fitness:
            ev = problem.evaluate(polished.x[None])
            fitness, cost, feasible = float(ev.fitness[0]), float(ev.cost[0]), bool(ev.feasible[0])
    return fitness, cost, feasible
