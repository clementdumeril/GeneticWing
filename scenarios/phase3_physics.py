"""Phase 3c: latent GA with the physics decoder (commands flown by the simulator).

Same protocol as ``phase3_latent.py``, but the VAE encodes and decodes the
teacher's commands (C_L, bank, thrust at regular knots) and every decoded
command sequence is flown by the point-mass simulator, so dynamic feasibility
holds by construction. Methods on the held-out test situations:

* Teacher commands flown  - the teacher's own commands, flown open loop (ceiling of the representation)
* Teacher round trip      - those commands encoded, decoded and flown
* VAE prior x64           - 64 prior samples decoded and flown, best feasible kept
* Physics latent GA       - GA (population 50) over the latent vector
* Physics latent GA + NLP - the same, then the best flight fitted to a genome and polished

Run: ``python scenarios/phase3_physics.py --data results/phase3/dataset.npz``
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from geneticwing.aircraft import SBXC_APPROX
from geneticwing.benchmarks import MOTOR_THRUST, canonical_scenario
from geneticwing.latent import ConditionalVAE
from geneticwing.physvae import command_latent_search, fly, mirror_commands, teacher_commands
from geneticwing.soaring import INFEASIBLE_OFFSET, SoaringProblem
from geneticwing.torchsim import SimConfig

_VAE: ConditionalVAE | None = None
_CFG: dict = {}


def _sim_config(args) -> SimConfig:
    if args.get("margins"):
        # The teacher's constraint margins (benchmarks.with_margins): same feasible set as the teacher.
        return SimConfig(SBXC_APPROX, dt=args["dt"], knots=args["knots"], cl_limit=0.75,
                         bank_max_deg=38.0, load_max=2.1, thrust_max=0.8 * MOTOR_THRUST)
    return SimConfig(SBXC_APPROX, dt=args["dt"], knots=args["knots"])


def _init_worker(state, cfg):
    global _VAE, _CFG
    torch.set_num_threads(1)
    c = state["config"]
    _VAE = ConditionalVAE(c["dim"], cfg["context_dim"], c["latent_dim"], cfg["width"], cfg["depth"], c["beta"], device="cpu")
    _VAE.load(state)
    _CFG = cfg


def _row(fitness, cost, seconds):
    feasible = fitness < INFEASIBLE_OFFSET
    return {"feasible": bool(feasible), "cost": float(cost) if feasible else float("nan"), "ms": 1e3 * seconds}


def evaluate_instance(job):
    instance, context, teacher_units = job
    sim = _sim_config(_CFG)
    pb = SoaringProblem(canonical_scenario(context))
    rng = np.random.default_rng(instance)
    out = {}

    t0 = time.perf_counter()
    ev = fly(teacher_units, context, sim, pb)
    k = int(np.argmin(ev.fitness))
    out["Teacher commands flown"] = _row(ev.fitness[k], ev.cost[k], time.perf_counter() - t0)

    t0 = time.perf_counter()
    z = _VAE.encode(teacher_units, np.tile(context, (len(teacher_units), 1)))
    ev = fly(np.vstack([_VAE.decode(zi[None], context) for zi in z]), context, sim, pb)
    k = int(np.argmin(ev.fitness))
    out["Teacher round trip"] = _row(ev.fitness[k], ev.cost[k], time.perf_counter() - t0)

    t0 = time.perf_counter()
    ev = fly(_VAE.decode(rng.standard_normal((64, _VAE.latent_dim)), context), context, sim, pb)
    k = int(np.argmin(ev.fitness))
    out["VAE prior x64"] = _row(ev.fitness[k], ev.cost[k], time.perf_counter() - t0)

    t0 = time.perf_counter()
    f, c, _ = command_latent_search(_VAE, context, sim, _CFG["search_budget"], rng, np.zeros((1, _VAE.latent_dim)))
    out["Physics latent GA"] = _row(f, c, time.perf_counter() - t0)

    t0 = time.perf_counter()
    f, c, _ = command_latent_search(_VAE, context, sim, _CFG["search_budget"], np.random.default_rng(instance),
                                    np.zeros((1, _VAE.latent_dim)), nlp_budget=_CFG["nlp_budget"])
    out["Physics latent GA + NLP"] = _row(f, c, time.perf_counter() - t0)
    return instance, out


def summarize(results, teacher, label):
    methods = list(next(iter(results.values())))
    rows = {}
    for m in methods:
        rs = [(results[i][m], teacher[i]) for i in results]
        feasible = np.array([r["feasible"] for r, _ in rs])
        rows[f"{label} | {m}"] = {
            "solved": float(feasible.mean()),
            "mean_cost": float(np.nanmean([r["cost"] for r, _ in rs])) if feasible.any() else None,
            "mean_gap": float(np.nanmean([r["cost"] - t for r, t in rs])) if feasible.any() else None,
            "median_ms": float(np.median([r["ms"] for r, _ in rs])),
        }
    return rows


def format_rows(rows):
    lines = [f"{'run | method':<46}{'solved':>8}{'mean cost':>11}{'mean gap':>10}{'median ms':>11}"]
    for name, r in rows.items():
        cost = r["mean_cost"] if r["mean_cost"] is not None else float("nan")
        gap = r["mean_gap"] if r["mean_gap"] is not None else float("nan")
        lines.append(f"{name:<46}{100 * r['solved']:>7.0f}%{cost:>11.2f}{gap:>10.2f}{r['median_ms']:>11.0f}")
    return "\n".join(lines)


def _cpu_state(vae: ConditionalVAE) -> dict:
    state = vae.state()
    for part in ("encoder", "decoder"):
        state[part] = {k: v.detach().cpu() for k, v in state[part].items()}
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=Path("results/phase3/dataset.npz"))
    parser.add_argument("--out", type=Path, default=Path("results/phase3_physics"))
    parser.add_argument("--latent-dims", type=int, nargs="+", default=[16])
    parser.add_argument("--knots", type=int, default=61)
    parser.add_argument("--dt", type=float, default=0.4)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--max-train", type=int)
    parser.add_argument("--max-test", type=int)
    parser.add_argument("--search-budget", type=int, default=3_000)
    parser.add_argument("--nlp-budget", type=int, default=2_000)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--margins", action="store_true",
                        help="bound the commands by the teacher's margins instead of the nominal limits")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    start = time.perf_counter()
    sim = _sim_config({"dt": args.dt, "knots": args.knots, "margins": args.margins})

    d = np.load(args.data)
    train = d["rec_split"] == "train"
    if args.max_train is not None:
        kept = np.unique(d["rec_instance"][train])[: args.max_train]
        train &= np.isin(d["rec_instance"], kept)
    contexts = d["rec_heading_context"][train]
    units = teacher_commands(d["rec_genome"][train], contexts, sim)
    mc, mu = mirror_commands(contexts, units)
    units_all, contexts_all = np.vstack([units, mu]), np.vstack([contexts, mc])
    teacher = dict(zip(d["teacher_instance"].tolist(), d["teacher_best_cost"].tolist()))
    test_mask = d["rec_split"] == "test"
    test = [i for i in sorted(set(d["rec_instance"][test_mask].tolist())) if np.isfinite(teacher[i])][: args.max_test]
    test_jobs = []
    for i in test:
        sel = test_mask & (d["rec_instance"] == i)
        ctx = d["rec_heading_context"][sel][0]
        test_jobs.append((i, ctx, teacher_commands(d["rec_genome"][sel], d["rec_heading_context"][sel], sim)))
    print(f"[{time.perf_counter() - start:5.0f}s] {len(units)} teacher command sequences ({units.shape[1]} values), "
          f"{len(test)} test situations, teacher test cost {np.mean([teacher[i] for i in test]):.2f}", flush=True)

    cfg = {"context_dim": contexts.shape[1], "width": args.width, "depth": args.depth, "dt": args.dt,
           "knots": args.knots, "margins": args.margins,
           "search_budget": args.search_budget, "nlp_budget": args.nlp_budget}
    rows, report = {}, {"args": {k: str(v) for k, v in vars(args).items()}, "results": {}}
    for latent_dim in args.latent_dims:
        vae = ConditionalVAE(units.shape[1], contexts.shape[1], latent_dim, args.width, args.depth, args.beta, device=args.device)
        log = vae.fit(units_all, contexts_all, steps=args.steps)
        print(f"[{time.perf_counter() - start:5.0f}s] command VAE d={latent_dim}: {log.seconds:.0f}s, "
              f"loss {np.mean(log.losses[-200:]):.2f}", flush=True)
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker, initargs=(_cpu_state(vae), cfg)) as pool:
            results = dict(pool.map(evaluate_instance, test_jobs, chunksize=1))
        stage = summarize(results, teacher, f"physics d={latent_dim}{', margins' if args.margins else ''}")
        rows.update(stage)
        report["results"][latent_dim] = stage
        torch.save(vae.state(), args.out / f"command_vae_d{latent_dim}.pt")
        print(format_rows(stage), flush=True)

    summary = (f"{len(test)} test situations, teacher mean cost {np.mean([teacher[i] for i in test]):.2f}\n\n"
               + format_rows(rows))
    print("\n" + summary)
    (args.out / "physics_summary.txt").write_text(summary)
    (args.out / "physics_summary.json").write_text(json.dumps(report, indent=1))
    print(f"done in {time.perf_counter() - start:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
