"""Piecewise-constant command genes against the learned genes, at equal budget.

T. Zohdi's group plans solar-sail trajectories with a GA over a few physical
genes: the sail angles held constant over a handful of time segments, whose
lengths are genes too (Sedky et al., CMAME 446, 2025, 118247). This study asks
whether such genes do as well as the 16 learned numbers of the command VAE.

Every method searches a different gene space but ends in the same place: unit
command vectors (C_L, bank, thrust at 61 knots) flown by the same simulator,
scored by the same cost and constraints (strip wing, the teacher's margins,
Deb's rule), with 3,000 simulated flights per climb, on the same 49 held-out
strip-wing climbs as the d = 16 row of ``phase3_physics.py --strip --margins``.

Gene spaces:

* latent d=16      - the command VAE's 16 numbers (decoded by the network), the GA seeded with z = 0 as before
* piecewise S      - S segments of equal length, 3 values each (C_L, bank, thrust): 3 S genes
* piecewise S var  - the same with the S segment lengths as genes too, as in the solar-sail paper: 4 S genes
* raw commands     - every knot's three values: 183 genes

GAs: the real-coded GA of the latent study (population 50), and the Zohdi GA
with the solar-sail paper's settings (30 strings, 10 parents, 10 offspring, 10
new random strings each generation).

Run: ``python scenarios/piecewise_baseline.py --data results/phase3_strip/dataset.npz
--vae results/phase3_strip/physics/command_vae_d16.pt --workers 4``
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from geneticwing.aircraft import SBXC_APPROX
from geneticwing.benchmarks import MOTOR_THRUST, canonical_scenario
from geneticwing.evolution import RealGA, ZohdiGA
from geneticwing.physvae import fly
from geneticwing.soaring import INFEASIBLE_OFFSET, SoaringProblem
from geneticwing.strip import sbxc_wing
from geneticwing.torchsim import SimConfig

_VAE = None  # the command VAE (torch is only needed for the latent methods)
_CFG: dict = {}

KNOTS = 61
HORIZON = 60.0


def sim_config(dt: float = 0.4, knots: int = KNOTS) -> SimConfig:
    # The teacher's margins, as phase3_physics.py --margins: every method faces the same feasible set.
    return SimConfig(SBXC_APPROX, dt=dt, knots=knots, cl_limit=0.75, bank_max_deg=38.0, load_max=2.1,
                     thrust_max=0.8 * MOTOR_THRUST)


def piecewise_commands(x: np.ndarray, segments: int, variable: bool, knots: int = KNOTS) -> np.ndarray:
    """Genes in the unit box ``(P, dim)`` -> unit commands ``(P, knots * 3)`` held constant on each segment.

    Fixed lengths: ``dim = 3 S`` (C_L, bank, thrust per segment). Variable
    lengths: ``dim = 4 S``, the first S genes set the segment lengths (each at
    least 1/11 of the longest one), the next 3 S the values.
    """
    x = np.atleast_2d(x)
    t = np.linspace(0.0, HORIZON, knots)
    if variable:
        w = 0.1 + x[:, :segments]
        ends = HORIZON * np.cumsum(w, axis=1) / w.sum(axis=1, keepdims=True)
        seg = np.stack([np.searchsorted(e, t, side="right") for e in ends])
        values = x[:, segments:].reshape(len(x), segments, 3)
    else:
        seg = np.broadcast_to(np.minimum((t / HORIZON * segments).astype(int), segments - 1), (len(x), knots))
        values = x.reshape(len(x), segments, 3)
    seg = np.minimum(seg, segments - 1)
    return np.take_along_axis(values, seg[:, :, None], axis=1).reshape(len(x), knots * 3)


def gene_spaces(vae_dim: int | None) -> dict:
    """name -> (dim, decode(x, context) -> unit commands, seeds in the unit box or None)."""
    spaces = {}
    if vae_dim:
        from geneticwing.latent import to_box, to_latent

        spaces["latent d=16"] = (vae_dim, lambda x, c: _VAE.decode(to_latent(x), c), to_box(np.zeros((1, vae_dim))))
    for s in (4, 6, 10):
        spaces[f"piecewise {s}"] = (3 * s, lambda x, c, s=s: piecewise_commands(x, s, False), None)
    spaces["piecewise 6 var"] = (24, lambda x, c: piecewise_commands(x, 6, True), None)
    spaces["raw commands"] = (3 * KNOTS, lambda x, c: np.atleast_2d(x), None)
    return spaces


GAS = {
    "real-coded GA": lambda: RealGA(population=50),
    "Zohdi GA": lambda: ZohdiGA(population=30, parents=10),
}

# (gene space, GA) pairs: the real-coded GA on every space, the Zohdi GA on the latent and piecewise ones.
RUNS = [(s, "real-coded GA") for s in ("latent d=16", "piecewise 4", "piecewise 6", "piecewise 10", "piecewise 6 var",
                                       "raw commands")]
RUNS += [(s, "Zohdi GA") for s in ("latent d=16", "piecewise 6", "piecewise 6 var")]


def _init_worker(state, cfg):
    global _VAE, _CFG
    _CFG = cfg
    if state is not None:
        import torch

        from geneticwing.latent import ConditionalVAE

        torch.set_num_threads(1)
        c = state["config"]
        _VAE = ConditionalVAE(c["dim"], cfg["context_dim"], c["latent_dim"], cfg["width"], cfg["depth"], c["beta"],
                              device="cpu")
        _VAE.load(state)


def evaluate_instance(job):
    instance, context, seed = job
    sim = sim_config(_CFG["dt"])
    problem = SoaringProblem(canonical_scenario(context, strip_wing=sbxc_wing()))
    spaces = gene_spaces(_VAE.latent_dim if _VAE is not None else None)
    out = {}
    for space, ga in RUNS:
        if space not in spaces or (_CFG["only"] and space not in _CFG["only"]):
            continue
        dim, decode, seeds = spaces[space]
        # Same random stream per (situation, seed) for every method. (The published latent-GA run drew its prior
        # samples from this stream first, so its numbers match this one's in distribution, not draw for draw.)
        rng = np.random.default_rng(instance if seed == 0 else [instance, seed])
        t0 = time.perf_counter()
        res = GAS[ga]().minimize(lambda x: fly(decode(x, context), context, sim, problem).fitness, dim,
                                 _CFG["budget"], rng, seeds)
        ev = fly(decode(res.x[None], context), context, sim, problem)
        f, c = float(ev.fitness[0]), float(ev.cost[0])
        out[f"{space} | {ga}"] = {"feasible": bool(f < INFEASIBLE_OFFSET), "cost": c if f < INFEASIBLE_OFFSET else None,
                                  "genes": dim, "evaluations": res.evaluations, "s": time.perf_counter() - t0,
                                  "curve": [(int(e), float(b)) for e, b in res.history[:: max(1, len(res.history) // 30)]]}
    return instance, seed, out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=Path("results/phase3_strip/dataset.npz"))
    parser.add_argument("--vae", type=Path, default=Path("results/phase3_strip/physics/command_vae_d16.pt"),
                        help="the command VAE; pass --vae '' to skip the latent methods (no torch needed)")
    parser.add_argument("--only", nargs="+", help="gene spaces to run (default: all)")
    parser.add_argument("--merge", type=Path, nargs="+", help="only merge these runs.json files and summarize")
    parser.add_argument("--out", type=Path, default=Path("results/piecewise_baseline"))
    parser.add_argument("--budget", type=int, default=3_000)
    parser.add_argument("--seeds", type=int, default=1, help="independent GA runs per climb")
    parser.add_argument("--max-test", type=int)
    parser.add_argument("--dt", type=float, default=0.4)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count()))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    if args.merge:
        merged = {}
        for f in args.merge:
            for r in json.loads(f.read_text()):
                key = (r["instance"], r["seed"])
                merged.setdefault(key, {**r, "methods": {}})["methods"].update(r["methods"])
        rows = [merged[k] for k in sorted(merged)]
        order = [f"{s} | {g}" for s, g in RUNS]
        for r in rows:
            r["methods"] = {m: r["methods"][m] for m in order if m in r["methods"]}
        (args.out / "runs.json").write_text(json.dumps(rows))
        summary = summarize(rows)
        print(summary)
        (args.out / "summary.txt").write_text(summary)
        return

    d = np.load(args.data)
    teacher = dict(zip(d["teacher_instance"].tolist(), d["teacher_best_cost"].tolist()))
    test_mask = d["rec_split"] == "test"
    test = [i for i in sorted(set(d["rec_instance"][test_mask].tolist())) if np.isfinite(teacher[i])][: args.max_test]
    contexts = {i: d["rec_heading_context"][test_mask & (d["rec_instance"] == i)][0] for i in test}
    print(f"{len(test)} test climbs, teacher mean cost {np.mean([teacher[i] for i in test]):.2f}, "
          f"{len(RUNS)} methods x {args.seeds} seed(s), budget {args.budget}", flush=True)

    state = None
    if args.vae and str(args.vae) not in ("", "."):
        import torch

        state = torch.load(args.vae, map_location="cpu", weights_only=False)
    cfg = {"context_dim": len(next(iter(contexts.values()))), "width": args.width, "depth": args.depth,
           "dt": args.dt, "budget": args.budget, "only": args.only}
    jobs = [(i, contexts[i], s) for s in range(args.seeds) for i in test]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker, initargs=(state, cfg)) as pool:
        for n, (i, s, out) in enumerate(pool.map(evaluate_instance, jobs, chunksize=1), 1):
            rows.append({"instance": i, "seed": s, "teacher": teacher[i], "methods": out})
            (args.out / "runs.json").write_text(json.dumps(rows))
            print(f"[{time.perf_counter() - start:6.0f}s] {n}/{len(jobs)} climb {i} seed {s}: "
                  + ", ".join(f"{m.split(' | ')[0]}/{m.split(' | ')[1][:4]} "
                              + (f"{r['cost']:.2f}" if r["feasible"] else "inf") for m, r in out.items()), flush=True)

    summary = summarize(rows)
    print("\n" + summary)
    (args.out / "summary.txt").write_text(summary)
    print(f"done in {time.perf_counter() - start:.0f}s -> {args.out}")


def summarize(rows) -> str:
    methods = list(rows[0]["methods"])
    t = np.array([r["teacher"] for r in rows])
    lines = [f"{len({r['instance'] for r in rows})} climbs x {len({r['seed'] for r in rows})} seed(s), "
             f"teacher mean cost {t.mean():.2f}",
             f"{'gene space | GA':<36}{'genes':>6}{'solved':>8}{'mean cost':>11}{'vs teacher':>12}{'beats teacher':>15}"
             f"{'s / plan':>10}"]
    for m in methods:
        rs = [r["methods"][m] for r in rows]
        ok = np.array([x["feasible"] for x in rs])
        cost = np.array([x["cost"] if x["feasible"] else np.nan for x in rs])
        lines.append(f"{m:<36}{rs[0]['genes']:>6}{100 * ok.mean():>7.0f}%{np.nanmean(cost):>11.2f}"
                     f"{np.nanmean(cost - t):>12.2f}{100 * np.mean(ok & (cost < t)):>14.0f}%"
                     f"{np.median([x['s'] for x in rs]):>10.1f}")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
