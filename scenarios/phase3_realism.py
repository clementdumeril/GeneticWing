"""Realism test of the physics decoder's gain: one minimum turn radius for every method.

The latent GA with the physics decoder beat the teacher by circling tighter
than the teacher's genome allowed. Here every method faces the same limit on
the coordinated-turn radius ``V_a^2 / (g tan|bank|)``, for several radii:

* Teacher       - re-solved under the limit: law flight, fitted genome, NLP with margins
                  (under the strip wing, also from the faster orbit, as the dataset's teacher)
* VAE prior x64 - 64 prior samples of the command VAE, flown, best feasible kept
* Physics GA    - the latent GA over the command VAE, scored under the limit

If the gain only came from turns too tight for a real sailplane, it shrinks
as the radius grows. With ``--strip``, the limit is physical instead: the
strip-theory wing of ``geneticwing.strip`` (inner-wing stall, aileron
authority, aileron drag) applies to every method. The command VAE is the one trained by phase3_physics.py
(``--margins``), loaded from ``--vae``. ``--airframes`` repeats the test on
SB-XC variants (``cd0:1.2``, ``cl_max:0.8``, ``mass:1.2``...): every method,
teacher included, flies the modified airframe, while the command VAE stays
the one trained on the nominal one -- a sensitivity test of the estimated polar.

Run: ``python scenarios/phase3_realism.py --data D/dataset.npz --theta D/theta.npy --vae D/physics/command_vae_d16.pt``
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from geneticwing.aircraft import SBXC_APPROX
from geneticwing.benchmarks import MOTOR_THRUST, canonical_scenario, s1_family, with_margins
from geneticwing.latent import ConditionalVAE
from geneticwing.physvae import command_latent_search, fly
from geneticwing.planning.local import solve_local
from geneticwing.policies import law_seeds, seed_theta, simulate_law
from geneticwing.seeding import fit_genome
from geneticwing.soaring import SoaringProblem
from geneticwing.strip import sbxc_wing
from geneticwing.torchsim import SimConfig

_VAE: ConditionalVAE | None = None
_CFG: dict = {}


def _sim(cfg, aircraft=SBXC_APPROX) -> SimConfig:
    return SimConfig(aircraft, dt=cfg["dt"], knots=cfg["knots"], cl_limit=0.75, bank_max_deg=38.0,
                     load_max=2.1, thrust_max=0.8 * MOTOR_THRUST)


def _init_worker(state, cfg):
    global _VAE, _CFG
    torch.set_num_threads(1)
    c = state["config"]
    _VAE = ConditionalVAE(c["dim"], cfg["context_dim"], c["latent_dim"], cfg["width"], cfg["depth"], c["beta"], device="cpu")
    _VAE.load(state)
    _CFG = cfg


def _radius(r):
    return None if r <= 0 else float(r)


AIRFRAME_PARAMS = ("cd0", "cl_max", "mass")


def airframe(spec: str):
    """``nominal`` or ``<param>:<factor>`` (param in AIRFRAME_PARAMS) -> (aircraft, C_L max factor)."""
    if spec == "nominal":
        return SBXC_APPROX, 1.0
    name, factor = spec.split(":")
    if name not in AIRFRAME_PARAMS:
        raise ValueError(f"unknown airframe parameter {name!r}, expected one of {AIRFRAME_PARAMS}")
    factor = float(factor)
    aircraft = replace(SBXC_APPROX, **{name: getattr(SBXC_APPROX, name) * factor})
    return aircraft, (factor if name == "cl_max" else 1.0)


def _overrides(radius_m, strip: bool, spec: str = "nominal") -> dict:
    aircraft, cl_factor = airframe(spec)
    wing = None
    if strip:  # the sections stall at the aircraft's C_L max
        wing = sbxc_wing()
        wing = replace(wing, local_cl_max=wing.local_cl_max * cl_factor)
    return {"turn_radius_min": _radius(radius_m), "strip_wing": wing, "aircraft": aircraft}


def teacher_under_limit(instance, overrides, theta, nlp_budget):
    scenario = replace(s1_family(instance)[0], **overrides)
    nominal, robust = SoaringProblem(scenario), SoaringProblem(with_margins(scenario))
    best = np.nan
    for ccw in (True, False):
        for overrides in law_seeds(scenario.strip_wing is not None):  # the dataset teacher's seeds
            law = simulate_law(robust, seed_theta(theta, ccw, overrides)[None])
            genome = fit_genome(robust, law.state.t[0], law.state.position[0])
            refined, _ = solve_local(robust, genome, nlp_budget)
            ev = nominal.evaluate(refined.x[None])
            if ev.feasible[0] and not ev.cost[0] >= best:
                best = float(ev.cost[0])
    return best


def evaluate(job):
    instance, context, radius_m, spec = job
    overrides = _overrides(radius_m, _CFG["strip"], spec)
    sim = _sim(_CFG, overrides["aircraft"])
    out = {}
    t0 = time.perf_counter()
    out["Teacher"] = (teacher_under_limit(instance, overrides, _CFG["theta"], _CFG["nlp_budget"]), time.perf_counter() - t0)

    problem = SoaringProblem(canonical_scenario(context, **overrides))
    rng = np.random.default_rng(instance)
    t0 = time.perf_counter()
    ev = fly(_VAE.decode(rng.standard_normal((64, _VAE.latent_dim)), context), context, sim, problem)
    k = int(np.argmin(ev.fitness))
    out["VAE prior x64"] = (float(ev.cost[k]) if ev.feasible[k] else np.nan, time.perf_counter() - t0)

    t0 = time.perf_counter()
    _, cost, feasible = command_latent_search(_VAE, context, sim, _CFG["search_budget"], rng,
                                             np.zeros((1, _VAE.latent_dim)), **overrides)
    out["Physics latent GA"] = (cost if feasible else np.nan, time.perf_counter() - t0)
    return instance, (radius_m, spec), out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--theta", type=Path, required=True)
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/phase3_realism"))
    parser.add_argument("--radii", type=float, nargs="+", default=[0, 15, 20, 25], help="0 = no limit")
    parser.add_argument("--max-test", type=int)
    parser.add_argument("--search-budget", type=int, default=3_000)
    parser.add_argument("--nlp-budget", type=int, default=6_000)
    parser.add_argument("--knots", type=int, default=61)
    parser.add_argument("--dt", type=float, default=0.4)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--strip", action="store_true", help="apply the strip-theory wing to every method")
    parser.add_argument("--airframes", nargs="+", default=["nominal"],
                        help="SB-XC variants for every method: nominal or <param>:<factor>, param in cd0, cl_max, mass")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    d = np.load(args.data)
    test_mask = d["rec_split"] == "test"
    test = sorted(set(d["rec_instance"][test_mask].tolist()))[: args.max_test]
    contexts = {i: d["rec_heading_context"][test_mask & (d["rec_instance"] == i)][0] for i in test}
    state = torch.load(args.vae, weights_only=False, map_location="cpu")
    cfg = {"context_dim": 6, "width": args.width, "depth": args.depth, "dt": args.dt, "knots": args.knots,
           "theta": np.load(args.theta), "nlp_budget": args.nlp_budget, "search_budget": args.search_budget,
           "strip": args.strip}
    for spec in args.airframes:
        airframe(spec)  # fail before the pool starts
    conditions = [(r, a) for r in args.radii for a in args.airframes]
    jobs = [(i, contexts[i], r, a) for r, a in conditions for i in test]
    print(f"{len(test)} situations x {len(args.radii)} radii x {len(args.airframes)} airframes, "
          f"{args.workers} workers", flush=True)

    results: dict = {c: {} for c in conditions}
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker, initargs=(state, cfg)) as pool:
        for k, (instance, condition, out) in enumerate(pool.map(evaluate, jobs, chunksize=1)):
            results[condition][instance] = out
            if k % 25 == 0:
                print(f"[{time.perf_counter() - start:6.0f}s] {k + 1}/{len(jobs)}", flush=True)

    lines = [f"{'R_min':>6} | {'airframe':<12} | {'method':<18}{'solved':>8}{'mean cost':>11}{'gap to teacher':>16}{'median s':>10}"]
    summary = {}
    for r, a in conditions:
        teacher = np.array([results[r, a][i]["Teacher"][0] for i in test])
        for m in ("Teacher", "VAE prior x64", "Physics latent GA"):
            cost = np.array([results[r, a][i][m][0] for i in test])
            both = np.isfinite(cost) & np.isfinite(teacher)
            row = {
                "solved": float(np.isfinite(cost).mean()),
                "mean_cost": float(np.nanmean(cost)) if np.isfinite(cost).any() else None,
                "mean_gap": float(np.mean(cost[both] - teacher[both])) if both.any() else None,
                "median_s": float(np.median([results[r, a][i][m][1] for i in test])),
            }
            summary[f"{r:g} | {a} | {m}"] = row
            label = ("strip" if args.strip else "none") if r <= 0 else f"{r:g} m"
            lines.append(
                f"{label:>6} | {a:<12} | {m:<18}{100 * row['solved']:>7.0f}%"
                f"{(row['mean_cost'] if row['mean_cost'] is not None else float('nan')):>11.2f}"
                f"{(row['mean_gap'] if row['mean_gap'] is not None else float('nan')):>16.2f}{row['median_s']:>10.1f}"
            )
    text = "\n".join(lines)
    print("\n" + text)
    (args.out / "realism_summary.txt").write_text(text)
    (args.out / "realism_summary.json").write_text(json.dumps(summary, indent=1))
    print(f"done in {time.perf_counter() - start:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
