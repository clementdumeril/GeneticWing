"""Phase 3b: genetic search in the latent space of a conditional VAE (H1 reformulated, H6).

Stage 1 - one conditional VAE per latent size, trained on the teacher's
solutions (heading-frame context, left/right mirrors). On the held-out test
situations, each method gets the same online budget:

* Teacher round trip  - the teacher's own solution encoded then decoded: what the
                        latent space can express (an upper bound, not a method)
* VAE prior x64       - 64 latent vectors drawn from the prior, decoded, best feasible kept
* Latent GA           - GA (population 50) over the latent vector, decoded and checked
* Latent GA + NLP     - the same, then SLSQP polishing of the best decoded trajectory
* Direct GA + NLP     - the phase 2 GA on the 87 genes with the same budget (baseline)

Stage 2 - improvement loop with the best latent size. Each round, the latent
GA starts from the teacher's solution of training situations (its encoding)
and tries to beat it; every improvement replaces the teacher's label, then the
VAE is fine-tuned on the improved labels and re-evaluated on the test set.

Gap = method cost - teacher cost on the same test situation (> 0 is worse).

Run: ``python scenarios/phase3_latent.py --data results/phase3/dataset.npz``
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

from geneticwing.benchmarks import mirror, s1_family
from geneticwing.evolution import RealGA
from geneticwing.latent import ConditionalVAE, latent_search
from geneticwing.planning.local import solve_local
from geneticwing.soaring import INFEASIBLE_OFFSET, SoaringProblem
from geneticwing.student import smooth_genomes

_VAE: ConditionalVAE | None = None
_CFG: dict = {}


def _init_worker(state, cfg):
    """Each worker rebuilds the VAE on CPU, single-threaded."""
    global _VAE, _CFG
    os.environ["OMP_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    c = state["config"]
    _VAE = ConditionalVAE(c["dim"], cfg["context_dim"], c["latent_dim"], cfg["width"], cfg["depth"], c["beta"], device="cpu")
    _VAE.load(state)
    _CFG = cfg


def _problem(instance):
    from geneticwing.benchmarks import heading_frame_context

    scenario, _ = s1_family(instance)
    return SoaringProblem(scenario), heading_frame_context(scenario)


def _row(fitness, seconds):
    feasible = fitness < INFEASIBLE_OFFSET
    return {"feasible": bool(feasible), "cost": float(fitness) if feasible else float("nan"), "ms": 1e3 * seconds}


def evaluate_instance(job):
    """All latent methods on one test situation."""
    instance, teacher_genomes = job
    pb, ctx = _problem(instance)
    rng = np.random.default_rng(instance)
    out = {}
    t0 = time.perf_counter()
    round_trip = smooth_genomes(_VAE.decode(_VAE.encode(teacher_genomes, np.tile(ctx, (len(teacher_genomes), 1))), ctx))
    out["Teacher round trip"] = _row(float(np.min(pb.evaluate(round_trip).fitness)), time.perf_counter() - t0)
    t0 = time.perf_counter()
    prior = smooth_genomes(_VAE.decode(rng.standard_normal((64, _VAE.latent_dim)), ctx))
    out["VAE prior x64"] = _row(float(np.min(pb.evaluate(prior).fitness)), time.perf_counter() - t0)

    t0 = time.perf_counter()
    found = latent_search(_VAE, pb, ctx, _CFG["search_budget"], rng, np.zeros((1, _VAE.latent_dim)))
    out["Latent GA"] = _row(found.fitness, time.perf_counter() - t0)
    t1 = time.perf_counter()
    polished, _ = solve_local(pb, found.genome, _CFG["nlp_budget"])
    out["Latent GA + NLP"] = _row(min(found.fitness, polished.f), time.perf_counter() - t1 + out["Latent GA"]["ms"] / 1e3)
    return instance, out


def direct_ga_instance(job):
    """Phase 2 GA on the 87 genes, same online budget, then the same polishing."""
    instance, search_budget, nlp_budget = job
    pb, _ = _problem(instance)
    t0 = time.perf_counter()
    ga = RealGA(population=50).minimize(pb.fitness, pb.dim, search_budget, np.random.default_rng(instance), pb.straight_genome()[None])
    polished, _ = solve_local(pb, ga.x, nlp_budget)
    return instance, _row(min(ga.f, polished.f), time.perf_counter() - t0)


def improve_instance(job):
    """Latent GA seeded with the encoding of the current label; returns a better label if found."""
    row, instance, genome, cost = job
    pb, ctx = _problem(instance)
    seed = _VAE.encode(genome, ctx)
    found = latent_search(_VAE, pb, ctx, _CFG["search_budget"], np.random.default_rng(row), seed, nlp_budget=_CFG["nlp_budget"])
    if found.fitness < INFEASIBLE_OFFSET:
        new_cost = float(pb.evaluate(found.genome[None]).cost[0])
        if new_cost < cost:
            return row, found.genome, new_cost
    return row, None, cost


def summarize(results, teacher, label):
    methods = list(next(iter(results.values())))
    rows = {}
    for m in methods:
        rs = [(results[i][m], teacher[i]) for i in results]
        feasible = np.array([r["feasible"] for r, _ in rs])
        gaps = np.array([r["cost"] - t for r, t in rs])
        rows[f"{label} | {m}"] = {
            "solved": float(feasible.mean()),
            "mean_cost": float(np.nanmean([r["cost"] for r, _ in rs])) if feasible.any() else None,
            "mean_gap": float(np.nanmean(gaps)) if feasible.any() else None,
            "median_ms": float(np.median([r["ms"] for r, _ in rs])),
        }
    return rows


def format_rows(rows):
    lines = [f"{'run | method':<44}{'solved':>8}{'mean cost':>11}{'mean gap':>10}{'median ms':>11}"]
    for name, r in rows.items():
        cost = r["mean_cost"] if r["mean_cost"] is not None else float("nan")
        gap = r["mean_gap"] if r["mean_gap"] is not None else float("nan")
        lines.append(f"{name:<44}{100 * r['solved']:>7.0f}%{cost:>11.2f}{gap:>10.2f}{r['median_ms']:>11.0f}")
    return "\n".join(lines)


def run_pool(fn, jobs, state, cfg, workers):
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(state, cfg)) as pool:
        return list(pool.map(fn, jobs, chunksize=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=Path("results/phase3/dataset.npz"))
    parser.add_argument("--out", type=Path, default=Path("results/phase3_latent"))
    parser.add_argument("--latent-dims", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--finetune-steps", type=int, default=5_000)
    parser.add_argument("--max-train", type=int, help="training instances to use (default: all)")
    parser.add_argument("--max-test", type=int)
    parser.add_argument("--search-budget", type=int, default=3_000)
    parser.add_argument("--nlp-budget", type=int, default=2_000)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--loop-solutions", type=int, default=2_000, help="training solutions revisited per round")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--threads", type=int, default=8, help="torch threads while training")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    start = time.perf_counter()

    d = np.load(args.data)
    train = d["rec_split"] == "train"
    if args.max_train is not None:
        kept = np.unique(d["rec_instance"][train])[: args.max_train]
        train &= np.isin(d["rec_instance"], kept)
    labels = d["rec_genome"][train].copy()
    label_cost = d["rec_cost"][train].copy()
    label_instance = d["rec_instance"][train]
    label_context = d["rec_heading_context"][train]
    teacher = dict(zip(d["teacher_instance"].tolist(), d["teacher_best_cost"].tolist()))
    test = sorted(set(d["rec_instance"][d["rec_split"] == "test"].tolist()))
    test = [i for i in test if np.isfinite(teacher[i])][: args.max_test]
    test_split = d["rec_split"] == "test"
    test_jobs = [(i, d["rec_genome"][test_split & (d["rec_instance"] == i)]) for i in test]
    print(f"{len(labels)} training solutions, {len(test)} test situations, teacher test cost "
          f"{np.mean([teacher[i] for i in test]):.2f}", flush=True)

    def augmented():
        mc, mg = mirror(label_context, labels)
        return np.vstack([labels, mg]), np.vstack([label_context, mc])

    cfg = {"context_dim": label_context.shape[1], "width": args.width, "depth": args.depth,
           "search_budget": args.search_budget, "nlp_budget": args.nlp_budget}
    rows, report = {}, {"args": {k: str(v) for k, v in vars(args).items()}, "stage1": {}, "stage2": []}

    # Baseline: the GA directly on the genes, same budget.
    direct = dict(_direct(test, args))
    rows.update(summarize({i: {"Direct GA + NLP": direct[i]} for i in test}, teacher, "baseline"))
    print(format_rows(rows), flush=True)

    # Stage 1: latent sizes.
    vaes = {}
    for latent_dim in args.latent_dims:
        vae = ConditionalVAE(labels.shape[1], cfg["context_dim"], latent_dim, args.width, args.depth, args.beta, device=args.device)
        g, c = augmented()
        log = vae.fit(g, c, steps=args.steps)
        print(f"[{time.perf_counter() - start:6.0f}s] VAE d={latent_dim}: {log.seconds:.0f}s, loss {np.mean(log.losses[-200:]):.2f}", flush=True)
        results = dict(run_pool(evaluate_instance, test_jobs, _cpu_state(vae), cfg, args.workers))
        stage = summarize(results, teacher, f"d={latent_dim}, round 0")
        rows.update(stage)
        report["stage1"][latent_dim] = stage
        vaes[latent_dim] = vae
        print(format_rows(stage), flush=True)

    # Stage 2: improvement loop with the best latent size.
    best_dim = min(args.latent_dims, key=lambda k: _gap(report["stage1"][k], f"d={k}, round 0 | Latent GA + NLP"))
    vae = vaes[best_dim]
    rng = np.random.default_rng(0)
    for r in range(1, args.rounds + 1):
        picked = rng.choice(len(labels), size=min(args.loop_solutions, len(labels)), replace=False)
        jobs = [(int(k), int(label_instance[k]), labels[k], float(label_cost[k])) for k in picked]
        before = float(np.mean(label_cost[picked]))
        improved = 0
        for k, genome, cost in run_pool(improve_instance, jobs, _cpu_state(vae), cfg, args.workers):
            if genome is not None:
                labels[k], label_cost[k] = genome, cost
                improved += 1
        after = float(np.mean(label_cost[picked]))
        g, c = augmented()
        vae.fit(g, c, steps=args.finetune_steps, lr=3e-4, seed=r)
        results = dict(run_pool(evaluate_instance, test_jobs, _cpu_state(vae), cfg, args.workers))
        stage = summarize(results, teacher, f"d={best_dim}, round {r}")
        rows.update(stage)
        report["stage2"].append({"round": r, "revisited": len(picked), "improved": improved,
                                 "label_cost_before": before, "label_cost_after": after, "test": stage})
        print(f"[{time.perf_counter() - start:6.0f}s] round {r}: improved {improved}/{len(picked)} teacher labels, "
              f"their mean cost {before:.2f} -> {after:.2f}", flush=True)
        print(format_rows(stage), flush=True)

    text = format_rows(rows)
    loop_lines = [f"round {s['round']}: improved {s['improved']}/{s['revisited']} labels, mean cost "
                  f"{s['label_cost_before']:.2f} -> {s['label_cost_after']:.2f}" for s in report["stage2"]]
    summary = (f"{len(test)} test situations, teacher mean cost {np.mean([teacher[i] for i in test]):.2f}\n\n"
               + text + "\n\nImprovement loop (d=%d):\n" % best_dim + "\n".join(loop_lines))
    print("\n" + summary)
    (args.out / "latent_summary.txt").write_text(summary)
    (args.out / "latent_summary.json").write_text(json.dumps(report, indent=1))
    torch.save(vae.state(), args.out / f"vae_d{best_dim}.pt")
    print(f"done in {time.perf_counter() - start:.0f}s -> {args.out}")


def _direct(test, args):
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        return list(pool.map(direct_ga_instance, [(i, args.search_budget, args.nlp_budget) for i in test]))


def _cpu_state(vae: ConditionalVAE) -> dict:
    state = vae.state()
    for part in ("encoder", "decoder"):
        state[part] = {k: v.detach().cpu() for k, v in state[part].items()}
    return state


def _gap(stage, key):
    gap = stage[key]["mean_gap"]
    return np.inf if gap is None else gap


if __name__ == "__main__":
    main()
