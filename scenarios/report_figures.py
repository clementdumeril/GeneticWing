"""Figures of the Savio phase 3 results, for the README.

Reads the folder copied from Savio (``results/savio_phase3``: students/,
latent/, physics/ and law_rk4.json) and writes two PNGs to ``docs/figures``:

* ``savio_scaling.png`` - the flow-matching student's gap to the teacher
  against the number of training situations;
* ``savio_methods.png`` - net energy gain of every method on the same 77 test
  situations.

Run: ``python scenarios/report_figures.py [--src results/savio_phase3]``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834"]  # categorical slots 1 and 2 of the reference palette
REFERENCE = "#8a8984"


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def scaling(src: Path, out: Path) -> None:
    runs = []
    for folder in sorted((src / "students").iterdir()):
        s = json.loads((folder / "phase3_summary.json").read_text())
        runs.append((s["training"]["train_instances"], s["methods"]))
    runs.sort()
    n = [r[0] for r in runs]
    law = json.loads((src / "law_rk4.json").read_text())

    fig, ax = plt.subplots(figsize=(7.5, 4.2), constrained_layout=True)
    _style(ax)
    series = [("Student, 64 samples, 8 steps", "FM x64, 8 steps"),
              ("Student + learned mutations + NLP", "FM + mutations + NLP")]
    for (label, key), color in zip(series, SERIES):
        gap = [m[key]["mean_gap"] for _, m in runs]
        ax.plot(n, gap, color=color, linewidth=2, marker="o", markersize=7, label=label)
        ax.annotate(f"{gap[-1]:.2f}", (n[-1], gap[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK)
    ax.axhline(law["mean_gap"], color=REFERENCE, linestyle="--", linewidth=1.2)
    ax.text(n[0], law["mean_gap"] + 0.08, f"reactive law, {law['mean_gap']:.2f}", color=MUTED, fontsize=9)
    ax.axhline(0.0, color=INK, linewidth=0.8)
    ax.text(n[0], 0.08, "teacher", color=MUTED, fontsize=9)
    ax.set_xscale("log")
    ax.set_xticks(n, [f"{v:,}" for v in n])
    ax.set_xlabel("training situations", color=MUTED)
    ax.set_ylabel("gap to the teacher (energy units, lower is better)", color=MUTED)
    ax.set_ylim(-0.3, 4.2)
    ax.set_title("More data helps the student up to about 3,000 situations, then plateaus",
                 loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.savefig(out / "savio_scaling.png", dpi=150)
    plt.close(fig)


def methods(src: Path, out: Path) -> None:
    law = json.loads((src / "law_rk4.json").read_text())
    student = json.loads((src / "students" / "n_10000" / "phase3_summary.json").read_text())["methods"]
    latent = json.loads((src / "latent" / "latent_summary.json").read_text())["stage1"]["16"]
    physics = json.loads((src / "physics" / "physics_summary.json").read_text())["results"]
    rows = [
        ("Reactive law (online)", law["mean_cost"], law["solved"], False),
        ("Flow-matching student + NLP", student["FM + mutations + NLP"]["mean_cost"],
         student["FM + mutations + NLP"]["feasible"], False),
        ("Latent GA, position VAE, + NLP", latent["d=16, round 0 | Latent GA + NLP"]["mean_cost"],
         latent["d=16, round 0 | Latent GA + NLP"]["solved"], False),
        ("Physics VAE, 64 prior samples", physics["8"]["physics d=8, margins | VAE prior x64"]["mean_cost"],
         physics["8"]["physics d=8, margins | VAE prior x64"]["solved"], False),
        ("Teacher (offline: law + NLP)", law["teacher"], 1.0, True),
        ("Latent GA, physics VAE", physics["16"]["physics d=16, margins | Physics latent GA"]["mean_cost"],
         physics["16"]["physics d=16, margins | Physics latent GA"]["solved"], False),
    ]
    fig, ax = plt.subplots(figsize=(7.5, 3.8), constrained_layout=True)
    _style(ax)
    ax.grid(axis="y", visible=False)
    for k, (label, cost, solved, is_teacher) in enumerate(rows):
        gain = -cost
        ax.barh(k, gain, height=0.6, color=REFERENCE if is_teacher else SERIES[0])
        ax.text(gain + 0.08, k, f"{gain:.2f}" + ("" if solved >= 0.999 else f"  ({100 * solved:.0f}% solved)"),
                va="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=9, color=INK)
    ax.set_xlabel("net energy gained from the air (-cost, higher is better)", color=MUTED)
    ax.set_xlim(0, 7.2)
    fig.suptitle("With the physics decoder, the latent GA beats the teacher (77 test situations)",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.savefig(out / "savio_methods.png", dpi=150)
    plt.close(fig)


def turn_limits(out: Path) -> None:
    import numpy as np

    from geneticwing.aircraft import SBXC_APPROX
    from geneticwing.strip import minimum_turn_radius, sbxc_wing

    wing, speeds = sbxc_wing(), np.linspace(8.5, 14.0, 23)
    limit = 0.9 * SBXC_APPROX.cl_max
    point = [minimum_turn_radius(wing, SBXC_APPROX, v, limit)["point_mass"] for v in speeds]
    strip = [minimum_turn_radius(wing, SBXC_APPROX, v, limit)["strip"] for v in speeds]
    fig, ax = plt.subplots(figsize=(7.5, 4.2), constrained_layout=True)
    _style(ax)
    ax.plot(speeds, point, color=SERIES[0], linewidth=2, label="point mass (mean C_L only)")
    ax.plot(speeds, strip, color=SERIES[1], linewidth=2, label="strip wing (inner-tip stall, ailerons)")
    ax.set_ylim(0, 40)
    ax.set_xlabel("airspeed (m/s)", color=MUTED)
    ax.set_ylabel("smallest steady turn radius (m)", color=MUTED)
    ax.set_title("At thermalling speeds the inner wingtip stalls long before the mean wing does",
                 loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(out / "turn_limits.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", type=Path, default=Path("results/savio_phase3"))
    parser.add_argument("--out", type=Path, default=Path("docs/figures"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    scaling(args.src, args.out)
    methods(args.src, args.out)
    turn_limits(args.out)
    print(f"wrote savio_scaling.png, savio_methods.png and turn_limits.png to {args.out}")


if __name__ == "__main__":
    main()
