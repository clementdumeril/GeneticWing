"""Phase 3, step 2: train the students and test them against the teacher (H2, H3, H4).

On held-out instances of the S1 family, each method gets the same context:

* Teacher            - law flights fitted and refined by the NLP with constraint margins
                       (offline, both orbit modes), scored under nominal constraints
* Regression         - deterministic MLP, one genome per context (H3 baseline)
* FM xK, n steps     - flow-matching student, K samples, n Euler steps, smoothed, best feasible kept (H2)
* FM + mutations     - the best samples re-noised and flowed back (learned mutation, H4)
* FM + NLP           - the best sample refined by SLSQP with a small online budget (H4)
* Law (online)       - the tuned reactive law, flown as is
* NLP (online)       - SLSQP from straight flight with the same small budget

Both students see the context in the aircraft's heading frame and train on
the data plus its left/right mirror image (an exact symmetry of the problem).

Cost: net onboard energy normalized by level cruise on the motor (< 0 = gain).
Gap: method cost minus teacher cost on the same instance (> 0 = worse).

Run: ``python scenarios/phase3_student.py [--steps 20000]`` (after phase3_dataset.py).
Scaling study: ``--max-train N`` keeps the first N training instances (both
orbit modes of each), ``--tag`` names the output folder, ``--device cuda``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from geneticwing.benchmarks import heading_frame_context, mirror, s1_family
from geneticwing.planning.local import solve_local
from geneticwing.policies import simulate_law
from geneticwing.soaring import SoaringProblem
from geneticwing.student import FlowMatchingStudent, RegressionStudent, smooth_genomes

ONLINE_NLP_BUDGET = 2_000
K = 64


def evaluate_methods(fm, reg, theta, instance, rng):
    scenario, _ = s1_family(instance)
    context = heading_frame_context(scenario)
    pb = SoaringProblem(scenario)
    out = {}

    def record(name, genomes, seconds):
        ev = pb.evaluate(np.atleast_2d(genomes))
        feasible = ev.feasible
        best = int(np.argmin(ev.fitness))
        out[name] = {
            "feasible_any": bool(feasible.any()),
            "feasible_rate": float(feasible.mean()),
            "cost": float(ev.cost[best]) if feasible[best] else np.nan,
            "genome": np.atleast_2d(genomes)[best],
            "ms": 1e3 * seconds,
        }

    t0 = time.perf_counter()
    g = smooth_genomes(reg.predict(context))
    record("Regression", g, time.perf_counter() - t0)

    for n_steps in (1, 4, 8, 16):
        t0 = time.perf_counter()
        samples = smooth_genomes(fm.sample(context, K, n_steps, rng))
        pb.evaluate(samples)  # verification is part of the online time
        record(f"FM x{K}, {n_steps} steps", samples, time.perf_counter() - t0)

    t0 = time.perf_counter()
    samples = smooth_genomes(fm.sample(context, K, 8, rng))
    fit = pb.evaluate(samples).fitness
    elite = samples[np.argsort(fit)[:8]]
    pool = [samples]
    for _ in range(2):
        children = smooth_genomes(fm.mutate(np.repeat(elite, 3, axis=0), context, rng=rng))
        pool.append(children)
        everyone = np.vstack(pool)
        elite = everyone[np.argsort(pb.evaluate(everyone).fitness)[:8]]
    record("FM + mutations", np.vstack(pool), time.perf_counter() - t0)

    t0 = time.perf_counter()
    refined, _ = solve_local(pb, out["FM + mutations"]["genome"], ONLINE_NLP_BUDGET)
    record("FM + mutations + NLP", np.vstack([refined.x, out["FM + mutations"]["genome"]]),
           time.perf_counter() - t0 + out["FM + mutations"]["ms"] / 1e3)

    t0 = time.perf_counter()
    law = simulate_law(pb, np.asarray(theta)[None])
    out["Law (online)"] = {
        "feasible_any": bool(law.feasible[0]),
        "feasible_rate": float(law.feasible[0]),
        "cost": float(law.cost[0]) if law.feasible[0] else np.nan,
        "genome": None,
        "ms": 1e3 * (time.perf_counter() - t0),
    }

    t0 = time.perf_counter()
    straight, _ = solve_local(pb, pb.straight_genome(), ONLINE_NLP_BUDGET)
    record(f"NLP (online, {ONLINE_NLP_BUDGET})", straight.x, time.perf_counter() - t0)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=Path("results/phase3/dataset.npz"))
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--out", type=Path, default=Path("results/phase3"))
    parser.add_argument("--tag", default="", help="sub-folder of --out for this run")
    parser.add_argument("--device", default="auto", help="auto, cpu or cuda")
    parser.add_argument("--max-train", type=int, help="number of training instances to use (default: all)")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--max-test", type=int, help="number of test instances to evaluate (default: all)")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.tag:
        args.out = args.out / args.tag
    torch.set_num_threads(args.threads)

    d = np.load(args.data)
    train = d["rec_split"] == "train"
    if args.max_train is not None:
        kept = np.unique(d["rec_instance"][train])[: args.max_train]
        train &= np.isin(d["rec_instance"], kept)
    n_train_instances = len(np.unique(d["rec_instance"][train]))
    genomes, contexts = d["rec_genome"][train], d["rec_heading_context"][train]
    mirrored_c, mirrored_g = mirror(contexts, genomes)
    genomes, contexts = np.vstack([genomes, mirrored_g]), np.vstack([contexts, mirrored_c])
    dim, cdim = genomes.shape[1], contexts.shape[1]
    print(f"training on {n_train_instances} instances, {len(genomes)} teacher solutions (with mirrors), "
          f"genome dim {dim}, context dim {cdim}")

    fm = FlowMatchingStudent(dim, cdim, args.width, args.depth, device=args.device)
    print(f"device: {fm.device}", flush=True)
    log_fm = fm.fit(genomes, contexts, steps=args.steps, batch=args.batch)
    reg = RegressionStudent(dim, cdim, args.width, args.depth, device=args.device)
    log_reg = reg.fit(genomes, contexts, steps=args.steps, batch=args.batch)
    print(f"flow matching: {log_fm.seconds:.0f}s, final loss {np.mean(log_fm.losses[-200:]):.3f}; "
          f"regression: {log_reg.seconds:.0f}s, final loss {np.mean(log_reg.losses[-200:]):.3f}")
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save({"fm": fm.model.state_dict(), "fm_scaler": (fm.scaler.mean, fm.scaler.std),
                "reg": reg.model.state_dict(), "reg_scaler": (reg.scaler.mean, reg.scaler.std)},
               args.out / "students.pt")
    torch.set_num_threads(1)

    teacher = dict(zip(d["teacher_instance"].tolist(), d["teacher_best_cost"].tolist()))
    test_instances = sorted(set(d["rec_instance"][d["rec_split"] == "test"].tolist()))
    test_instances = [i for i in test_instances if np.isfinite(teacher[i])][: args.max_test]
    if not test_instances:
        raise SystemExit("no test instance with a feasible teacher solution in the dataset")
    rng = np.random.default_rng(0)
    results = {}
    for k, instance in enumerate(test_instances):
        results[instance] = evaluate_methods(fm, reg, d["theta"], instance, rng)
        if k % 20 == 0:
            print(f"evaluated {k + 1}/{len(test_instances)} test instances", flush=True)

    methods = list(next(iter(results.values())))
    lines = [f"{len(test_instances)} test instances, teacher mean cost {np.mean([teacher[i] for i in test_instances]):.2f}",
             f"{'method':<26}{'feasible':>9}{'samples ok':>12}{'mean cost':>11}{'mean gap':>10}{'median ms':>11}"]
    summary = {}
    for m in methods:
        rows = [results[i][m] for i in test_instances]
        feasible = np.array([r["feasible_any"] for r in rows])
        costs = np.array([r["cost"] for r in rows])
        gaps = np.array([r["cost"] - teacher[i] for r, i in zip(rows, test_instances)])
        summary[m] = {
            "feasible": float(feasible.mean()),
            "sample_feasible_rate": float(np.mean([r["feasible_rate"] for r in rows])),
            "mean_cost": float(np.nanmean(costs)) if feasible.any() else None,
            "mean_gap": float(np.nanmean(gaps)) if feasible.any() else None,
            "median_ms": float(np.median([r["ms"] for r in rows])),
        }
        s = summary[m]
        lines.append(
            f"{m:<26}{100 * s['feasible']:>8.0f}%{100 * s['sample_feasible_rate']:>11.0f}%"
            f"{(s['mean_cost'] if s['mean_cost'] is not None else float('nan')):>11.2f}"
            f"{(s['mean_gap'] if s['mean_gap'] is not None else float('nan')):>10.2f}{s['median_ms']:>11.0f}"
        )
    text = "\n".join(lines)
    print(text)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "phase3_summary.txt").write_text(text)
    (args.out / "phase3_summary.json").write_text(json.dumps(
        {"training": {"fm_seconds": log_fm.seconds, "reg_seconds": log_reg.seconds, "steps": args.steps,
                      "train_instances": n_train_instances, "train_solutions": len(genomes),
                      "width": args.width, "depth": args.depth, "device": str(fm.device)},
         "methods": summary}, indent=1))
    plot(fm, reg, d, test_instances, results, teacher, args.out, log_fm, log_reg)


def plot(fm, reg, d, test_instances, results, teacher, out, log_fm, log_reg):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(20, 5), constrained_layout=True)
    for ax, instance in zip(axes[:3], test_instances[:3]):
        scenario, _ = s1_family(instance)
        context = heading_frame_context(scenario)
        pb = SoaringProblem(scenario)
        sel = d["rec_instance"] == instance
        samples = smooth_genomes(fm.sample(context, 16, 8, np.random.default_rng(1)))
        s = pb.flat(samples)
        for j in range(len(samples)):
            ax.plot(s.position[j, :, 0], s.position[j, :, 1], color="C0", alpha=0.35, lw=0.8,
                    label="flow matching samples" if j == 0 else None)
        for j, g in enumerate(d["rec_genome"][sel]):
            st = pb.flat(g[None])
            ax.plot(st.position[0, :, 0], st.position[0, :, 1], color="k", lw=1.8,
                    label="teacher (both modes)" if j == 0 else None)
        sr = pb.flat(smooth_genomes(reg.predict(context)))
        ax.plot(sr.position[0, :, 0], sr.position[0, :, 1], color="C3", lw=1.5, ls="--", label="regression")
        ax.plot(0, 0, "k^", ms=8)
        ax.plot(*scenario.start[:2], "ko", ms=5)
        ax.set(title=f"test instance {instance} (teacher {teacher[instance]:.2f})", xlabel="x (m)", ylabel="y (m)", aspect="equal")
        ax.legend(fontsize=7)
    ax = axes[3]
    ax.semilogy(np.convolve(log_fm.losses, np.ones(200) / 200, mode="valid"), label="flow matching")
    ax.semilogy(np.convolve(log_reg.losses, np.ones(200) / 200, mode="valid"), label="regression")
    ax.set(title="training loss (standardized genome)", xlabel="step")
    ax.legend()
    fig.savefig(out / "phase3_student.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
