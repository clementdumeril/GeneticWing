"""Phase 6: amortize the latent GA -- a fast generator trained on what the GA finds.

The latent GA over the physics decoder beats the teacher but takes about 13 s
per plan. This study distills it, in the spirit of expert iteration (search
improves the solutions, a network learns to produce them directly):

A. the GA (``--budget`` flights) solves ``--train`` training situations of the
   strip dataset; its best point in the command VAE's latent box is kept;
B. a conditional flow matching model learns those points given the situation's
   context (in the GA's unit box, so samples land where the GA searched);
C. on the dataset's test situations, against the teacher, compare:
   * VAE prior x64        - 64 points from the VAE's own prior, best kept (no search)
   * distilled x1         - one point from the distilled generator
   * distilled x64        - 64 points from it, best kept
   * distilled + short GA - a GA of ``--short-budget`` flights seeded with 8 of its points
   * latent GA            - the full GA (``--budget`` flights), the target being distilled

Every candidate is decoded into commands and flown by the simulator, with the
teacher's margins and the strip wing, as in ``phase3_physics.py --strip --margins``.
Stage A is cached in ``<out>/ga_train.npz``: a rerun only retrains and re-evaluates
(``--cache-from`` reuses another run's). ``--inflight`` adds the GA's climb plans
recorded in flight (``phase5_crosscountry.py --record``), DAgger-style: the
generator then also sees the aircraft's state (bank, airspeed, C_L;
``closedloop.state_features``), level flight for the situations of stage A.

Run: ``python scenarios/phase6_distill.py --data D/dataset.npz --vae D/physics/command_vae_d16.pt --out D/distill``
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from geneticwing.aircraft import SBXC_APPROX  # noqa: E402
from geneticwing.benchmarks import MOTOR_THRUST, canonical_scenario  # noqa: E402
from geneticwing.closedloop import state_features  # noqa: E402
from geneticwing.evolution import RealGA  # noqa: E402
from geneticwing.latent import ConditionalVAE, to_latent  # noqa: E402
from geneticwing.physvae import fly  # noqa: E402
from geneticwing.soaring import INFEASIBLE_OFFSET, SoaringProblem  # noqa: E402
from geneticwing.strip import sbxc_wing  # noqa: E402
from geneticwing.student import FlowMatchingStudent  # noqa: E402
from geneticwing.torchsim import SimConfig  # noqa: E402

_W: dict = {}


def _sim() -> SimConfig:
    # The teacher's margins, as phase3_physics --margins.
    return SimConfig(SBXC_APPROX, dt=0.4, knots=61, cl_limit=0.75, bank_max_deg=38.0, load_max=2.1,
                     thrust_max=0.8 * MOTOR_THRUST)


def _init_worker(vae_state, flow_state, cfg):
    torch.set_num_threads(1)
    c = vae_state["config"]
    vae = ConditionalVAE(c["dim"], 6, c["latent_dim"], cfg["width"], cfg["depth"], c["beta"], device="cpu")
    vae.load(vae_state)
    flow = None
    if flow_state is not None:
        flow = FlowMatchingStudent.from_state(flow_state, c["latent_dim"], flow_state.get("context_dim", 6),
                                              cfg["flow_width"], cfg["flow_depth"])
    _W.update(vae=vae, flow=flow, cfg=cfg, sim=_sim())


def _problem(context):
    return SoaringProblem(canonical_scenario(context, strip_wing=sbxc_wing()))


def _fly_box(x_box, context, problem):
    """Points of the GA's unit box -> latent -> commands -> flights."""
    return fly(_W["vae"].decode(to_latent(np.atleast_2d(x_box)), context), context, _W["sim"], problem)


def _search(context, problem, budget, rng, seeds_box):
    """The latent GA (as physvae.command_latent_plan), returning its best point in the unit box too."""
    result = RealGA(population=50).minimize(lambda x: _fly_box(x, context, problem).fitness,
                                            _W["vae"].latent_dim, budget, rng, seeds_box)
    ev = _fly_box(result.x[None], context, problem)
    return np.asarray(result.x), ev


def _condition(context):
    """The generator's input: the context, and the level-flight state when it was distilled in flight too."""
    if _W["flow"].context_dim == len(context):
        return context
    return np.concatenate([context, state_features(canonical_scenario(context), None)])


def _row(ev, k, seconds):
    feasible = bool(ev.fitness[k] < INFEASIBLE_OFFSET)
    return {"feasible": feasible, "cost": float(ev.cost[k]) if feasible else None, "ms": 1e3 * seconds}


def solve_train(job):
    instance, context = job
    problem = _problem(context)
    x, ev = _search(context, problem, _W["cfg"]["budget"], np.random.default_rng(instance), np.full((1, _W["vae"].latent_dim), 0.5))
    return instance, x, float(ev.fitness[0]), float(ev.cost[0])


def evaluate_test(job):
    instance, context = job
    cfg, vae, flow = _W["cfg"], _W["vae"], _W["flow"]
    problem = _problem(context)
    rng = np.random.default_rng(instance)
    out = {}

    t0 = time.perf_counter()
    ev = fly(vae.decode(rng.standard_normal((64, vae.latent_dim)), context), context, _W["sim"], problem)
    out["VAE prior x64"] = _row(ev, int(np.argmin(ev.fitness)), time.perf_counter() - t0)

    t0 = time.perf_counter()
    condition = _condition(context)
    ev = _fly_box(flow.sample(condition, 1, rng=rng), context, problem)
    out["distilled x1"] = _row(ev, 0, time.perf_counter() - t0)

    t0 = time.perf_counter()
    ev = _fly_box(flow.sample(condition, 64, rng=rng), context, problem)
    out["distilled x64"] = _row(ev, int(np.argmin(ev.fitness)), time.perf_counter() - t0)

    t0 = time.perf_counter()
    _, ev = _search(context, problem, cfg["short_budget"], np.random.default_rng(instance), flow.sample(condition, 8, rng=rng))
    out[f"distilled + GA {cfg['short_budget']}"] = _row(ev, 0, time.perf_counter() - t0)

    t0 = time.perf_counter()
    _, ev = _search(context, problem, cfg["budget"], np.random.default_rng(instance), np.full((1, vae.latent_dim), 0.5))
    out[f"latent GA {cfg['budget']}"] = _row(ev, 0, time.perf_counter() - t0)
    return instance, out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True, help="strip dataset (phase3_strip_v2)")
    parser.add_argument("--vae", type=Path, required=True, help="command VAE (physics/command_vae_d16.pt)")
    parser.add_argument("--out", type=Path, default=Path("results/phase6_distill"))
    parser.add_argument("--train", type=int, default=3_000, help="training situations solved by the GA")
    parser.add_argument("--budget", type=int, default=3_000, help="GA flights per situation (the target)")
    parser.add_argument("--short-budget", type=int, default=300, help="GA flights after the distilled seeds")
    parser.add_argument("--steps", type=int, default=20_000, help="flow matching training steps")
    parser.add_argument("--flow-width", type=int, default=256)
    parser.add_argument("--flow-depth", type=int, default=3)
    parser.add_argument("--max-test", type=int)
    parser.add_argument("--cache-from", type=Path, help="ga_train.npz of an earlier run (skips stage A)")
    parser.add_argument("--inflight", type=Path, nargs="*", default=[],
                        help="ga_inflight.npz files recorded in flight by phase5_crosscountry.py --record")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    log = lambda msg: print(f"[{time.perf_counter() - start:6.0f}s] {msg}", flush=True)  # noqa: E731

    d = np.load(args.data)
    split, inst, ctx = d["rec_split"], d["rec_instance"], d["rec_heading_context"]
    first = {}
    for s, i, c in zip(split, inst, ctx):
        first.setdefault((s, int(i)), c)
    train = sorted(i for (s, i) in first if s == "train")[: args.train]
    test = sorted(i for (s, i) in first if s == "test")[: args.max_test]
    teacher = dict(zip(d["teacher_instance"].tolist(), d["teacher_best_cost"].tolist()))
    vae_state = torch.load(args.vae, weights_only=False, map_location="cpu")
    latent_dim = vae_state["config"]["latent_dim"]
    cfg = {"budget": args.budget, "short_budget": args.short_budget, "width": args.width, "depth": args.depth,
           "flow_width": args.flow_width, "flow_depth": args.flow_depth}

    # A. The GA on the training situations (cached).
    cache = args.cache_from or args.out / "ga_train.npz"
    if cache.exists():
        a = np.load(cache)
        xs, fits, costs, done = a["x"], a["fitness"], a["cost"], a["instance"]
        log(f"GA solutions loaded from {cache}: {len(done)} situations")
    else:
        log(f"A. latent GA ({args.budget} flights) on {len(train)} training situations, {args.workers} workers")
        rows = []
        with ProcessPoolExecutor(args.workers, initializer=_init_worker, initargs=(vae_state, None, cfg)) as pool:
            for k, r in enumerate(pool.map(solve_train, [(i, first[("train", i)]) for i in train], chunksize=2)):
                rows.append(r)
                if k % 200 == 0:
                    log(f"   {k + 1}/{len(train)}")
        done = np.array([r[0] for r in rows])
        xs = np.array([r[1] for r in rows])
        fits, costs = np.array([r[2] for r in rows]), np.array([r[3] for r in rows])
        np.savez(cache, instance=done, x=xs, fitness=fits, cost=costs)
    feasible = fits < INFEASIBLE_OFFSET
    teach_train = np.array([teacher.get(int(i), np.nan) for i in done])
    both = feasible & np.isfinite(teach_train)
    log(f"GA training solutions: {feasible.mean():.0%} feasible, mean cost {np.nanmean(np.where(feasible, costs, np.nan)):.2f} "
        f"(teacher on the same situations {np.nanmean(teach_train[both]):.2f} vs GA {costs[both].mean():.2f})")

    # B. Distill: flow matching from context to the GA's best point.
    contexts = np.array([first[("train", int(i))] for i in done])
    targets, conditions = xs[feasible], contexts[feasible]
    if args.inflight:  # DAgger: the states met in flight, with the aircraft's state in the condition
        level = np.array([state_features(canonical_scenario(c), None) for c in conditions])
        conditions = np.hstack([conditions, level])
        for path in args.inflight:
            r = np.load(path)
            ok = r["fitness"] < INFEASIBLE_OFFSET
            targets = np.vstack([targets, r["x"][ok]])
            conditions = np.vstack([conditions, np.hstack([r["context"][ok], r["state"][ok]])])
            log(f"   + {ok.sum()} GA climb plans recorded in flight ({path})")
    flow = FlowMatchingStudent(latent_dim, conditions.shape[1], args.flow_width, args.flow_depth, device=args.device)
    tlog = flow.fit(targets, conditions, steps=args.steps)
    log(f"B. flow matching trained on {len(targets)} GA solutions: {tlog.seconds:.0f}s, final loss {np.mean(tlog.losses[-200:]):.3f}")
    flow_state = {**flow.state(), "width": args.flow_width, "depth": args.flow_depth, "context_dim": conditions.shape[1]}
    torch.save(flow_state, args.out / "distilled_flow.pt")

    # C. Test.
    log(f"C. {len(test)} test situations")
    results = {}
    with ProcessPoolExecutor(args.workers, initializer=_init_worker, initargs=(vae_state, flow_state, cfg)) as pool:
        for i, out in pool.map(evaluate_test, [(i, first[("test", i)]) for i in test], chunksize=1):
            results[i] = out
    methods = list(next(iter(results.values())))
    tcost = np.array([teacher[i] for i in test])
    lines = [f"{len(test)} test situations, teacher mean cost {np.nanmean(tcost):.2f}",
             f"{'method':<26}{'solved':>8}{'mean cost':>11}{'gap':>8}{'median ms':>11}"]
    summary = {}
    for m in methods:
        rows = [results[i][m] for i in test]
        ok = np.array([r["feasible"] for r in rows])
        cost = np.array([r["cost"] if r["feasible"] else np.nan for r in rows])
        gap = np.nanmean(cost - tcost) if ok.any() else float("nan")
        ms = float(np.median([r["ms"] for r in rows]))
        summary[m] = {"solved": float(ok.mean()), "mean_cost": float(np.nanmean(cost)) if ok.any() else None,
                      "gap": float(gap), "median_ms": ms}
        lines.append(f"{m:<26}{100 * ok.mean():>7.0f}%{np.nanmean(cost) if ok.any() else float('nan'):>11.2f}{gap:>8.2f}{ms:>11.0f}")
    text = "\n".join(lines)
    (args.out / "distill_summary.txt").write_text(text)
    (args.out / "distill.json").write_text(json.dumps({"summary": summary, "per_situation": {str(k): v for k, v in results.items()}}, indent=1))
    print("\n" + text)
    log(f"done -> {args.out}")


if __name__ == "__main__":
    main()
