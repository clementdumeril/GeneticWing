"""Figures of the A0 poster (docs/poster/figures), one style for all.

Every figure is drawn at the poster's column width (26 cm) so its text prints at the size set here. The numbers come
from the runs in docs/RESULTS.md; the real-weather ones are read from the pilots' summary tables when present:

    results/weather_chain/<day>/crosscountry_summary.txt  motor, circling rule (thermals known), blind rule
                                                          (variometer + EKF) and the weather-trained blind generator,
                                                          all on the same 48 tasks (Savio job 41248405)

A pilot with no summary is drawn as an empty dashed bar marked "?".
Run: ``PYTHONPATH=src python docs/poster/make_figures.py``

The banner (figures/hero.jpg) is frame 100 of the README animation rendered at 3200 x 1800, rows 420-1220:
``python scenarios/xc_render3d.py results/phase5_view/xc.html docs/poster/figures/hero.gif --fps 10 --climb-step 1
--glide-step 10 --until 1300 --size 3200 1800 --preview 100``, then crop ``hero_frame100.png`` to (0, 420, 3200, 1220).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

from geneticwing.wind.thermal import allen_updraft  # noqa: E402

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)

# Entities keep one colour on the whole poster (reference palette, categorical slots in fixed order).
RULE, GA, GEN, MOTOR = "#2a78d6", "#1baf7a", "#eb6834", "#8a8f98"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
COL_W = 26.0 / 2.54  # inches, one poster column

for f in ("segoeui.ttf", "segoeuib.ttf", "segoeuii.ttf", "segoeuisl.ttf"):
    p = Path("C:/Windows/Fonts") / f
    if p.exists():
        font_manager.fontManager.addfont(str(p))
plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 22, "axes.titlesize": 22, "axes.labelsize": 22,
    "xtick.labelsize": 20, "ytick.labelsize": 20, "legend.fontsize": 19, "axes.edgecolor": INK2,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.linewidth": 1.2, "xtick.major.width": 1.2, "ytick.major.width": 1.2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1.0, "axes.axisbelow": True, "savefig.dpi": 300,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "mathtext.fontset": "custom",
    "mathtext.rm": "Segoe UI", "mathtext.it": "Segoe UI:italic", "pdf.fonttype": 42,
})


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", pad_inches=0.05, dpi=110)
    plt.close(fig)
    print(name)


def bar_labels(ax, bars, values, fmt, pad, inside=False):
    for b, v in zip(bars, values):
        if v is None:
            continue
        y = b.get_height() + pad if not inside else b.get_height() - pad
        ax.text(b.get_x() + b.get_width() / 2, y, fmt.format(v), ha="center", va="bottom" if not inside else "top",
                color=INK, fontsize=19)


# 1. Allen's thermal on a real day (12 Jul 2025: w* 2.9 m/s, z_i 2,300 m): updraft core and downdraft ring.
w_star, z_i = 2.9, 2300.0
r = np.linspace(-420, 420, 600)
z = np.linspace(5, z_i * 0.98, 400)
R, Z = np.meshgrid(r, z)
W = allen_updraft(np.abs(R), Z, w_star, z_i)
lift = LinearSegmentedColormap.from_list(
    "lift", [(0.0, "#1c5cab"), (0.25, "#86b6ef"), (0.5, "#f4f3ef"), (0.65, "#f5b48f"), (0.82, "#eb6834"), (1.0, "#9c3a12")])
fig, ax = plt.subplots(figsize=(COL_W, 0.54 * COL_W))
mesh = ax.pcolormesh(R, Z, W, cmap=lift, norm=TwoSlopeNorm(0.0, -1.5, W.max()), shading="gouraud", rasterized=True)
ax.contour(R, Z, W, levels=[0.0], colors=INK2, linewidths=1.0, linestyles=":")
cb = fig.colorbar(mesh, ax=ax, pad=0.02, aspect=22)
cb.set_label("vertical air speed (m/s)")
cb.outline.set_visible(False)
ax.set_xlabel("distance from the thermal core (m)")
ax.set_ylabel("altitude (m)")
ax.grid(False)
ax.annotate("core: up to 5 m/s", (0, 1100), xytext=(150, 1500), color=INK, fontsize=20,
            arrowprops=dict(arrowstyle="-", color=INK, lw=1.2))
ax.annotate("sinking ring", (-215, 1700), xytext=(-410, 2050), color=INK, fontsize=20,
            arrowprops=dict(arrowstyle="-", color=INK, lw=1.2))
ax.text(400, 120, "Allen's model, 12 Jul 2025\nw* 2.9 m/s, layer 2,300 m", ha="right", va="bottom", fontsize=18,
        color=INK2)
save(fig, "thermal")


# 2. Climb rate on 64 tasks of 20 km, MacCready 0.5: the tasks of the DAgger-like round and 64 unseen ones.
groups = ["64 training tasks", "64 unseen tasks"]
pilots = [("Circling rule", RULE, [1.29, 1.30]), ("Genetic search (13 s)", GA, [2.08, 2.04]),
          ("Generator (0.24 s)", GEN, [2.23, 2.21])]
fig, ax = plt.subplots(figsize=(COL_W, 0.5 * COL_W))
x = np.arange(len(groups))
w = 0.26
for k, (name, color, vals) in enumerate(pilots):
    bars = ax.bar(x + (k - 1) * (w + 0.02), vals, w, color=color, label=name, edgecolor=SURFACE, linewidth=2)
    bar_labels(ax, bars, vals, "{:.2f}", 0.04)
ax.set_xticks(x, groups)
ax.set_ylabel("mean climb rate (m/s)")
ax.set_ylim(0, 2.75)
ax.grid(axis="x", visible=False)
ax.tick_params(axis="x", length=0)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=3, frameon=False, handlelength=1.0,
          columnspacing=1.2)
save(fig, "climbs")


# 3. Quality against planning time, 65 test climbs (energy gained over 60 s; higher is better).
methods = [("VAE prior, best of 64", 0.24, 4.64, MOTOR, (16, -8), "left"),
           ("Generator, 1 sample", 0.14, 3.52, GEN, (16, -8), "left"),
           ("Generator, best of 64", 0.24, 5.72, GEN, (-14, 16), "left"),
           ("Generator + short search", 1.4, 5.59, GEN, (0, -44), "center"),
           ("Genetic search\n3,000 flights", 13.4, 5.66, GA, (16, -40), "left")]
fig, ax = plt.subplots(figsize=(COL_W, 0.55 * COL_W))
ax.axhline(4.88, color=INK2, ls="--", lw=1.6)
ax.text(110, 4.94, "teacher (offline)", ha="right", va="bottom", color=INK2, fontsize=19)
for name, t, e, color, off, ha in methods:
    ax.scatter(t, e, s=260, color=color, edgecolor=SURFACE, linewidth=2.5, zorder=3)
    ax.annotate(name, (t, e), xytext=off, textcoords="offset points", fontsize=19, color=INK, ha=ha)
ax.annotate("", (0.30, 5.78), xytext=(12.0, 5.74),
            arrowprops=dict(arrowstyle="->", color=GEN, lw=2.2, connectionstyle="arc3,rad=0.3"))
ax.text(1.9, 6.42, "56× faster, same quality", color=GEN, fontsize=21, ha="center", weight="bold")
ax.set_xscale("log")
ax.set_xlim(0.09, 120)
ax.set_ylim(3.2, 6.7)
ax.set_xticks([0.1, 1, 10, 100], ["0.1 s", "1 s", "10 s", "100 s"])
ax.set_xlabel("time to plan one climb")
ax.set_ylabel("energy gained (−cost)")
save(fig, "speed_quality")


# 4. Why a learned space: the same genetic search (real-coded, ~17 s per climb) over different genes, 49 climbs.
spaces = [("raw commands (183)", -2.21), ("10 knots (30)", 2.09), ("6 knots (18)", 3.09),
          ("4 knots (12)", 3.77), ("learned space (16)", 5.99)]
fig, ax = plt.subplots(figsize=(COL_W, 0.5 * COL_W))
y = np.arange(len(spaces))
vals = [v for _, v in spaces]
bars = ax.barh(y, vals, 0.62, color=[MOTOR] * 4 + [GA], edgecolor=SURFACE, linewidth=2)
for b, v in zip(bars, vals):
    ax.text(v + (0.12 if v >= 0 else -0.12), b.get_y() + b.get_height() / 2, f"{v:+.2f}", va="center",
            ha="left" if v >= 0 else "right", fontsize=19, color=INK)
ax.axvline(4.97, color=INK2, ls="--", lw=1.6)
ax.text(4.97, len(spaces) - 0.35, " teacher", color=INK2, fontsize=19, va="bottom", ha="center")
ax.axvline(0, color=INK2, lw=1.2)
ax.set_yticks(y, [n for n, _ in spaces])
ax.set_xlim(-3.6, 7.2)
ax.set_xlabel("energy gained (−cost)")
ax.set_ylabel("genes searched")
ax.grid(axis="y", visible=False)
ax.tick_params(axis="y", length=0)
save(fig, "learned_space")


# 5. Real weather, MacCready 1, 48 tasks of 20 km per day: battery used and tasks completed.
DAYS = [("2025-07-12", "12 Jul: strong"), ("2025-08-02", "2 Aug: cumulus"), ("2025-06-22", "22 Jun: windy")]


def summary(path: Path) -> dict:
    rows = {}
    if path.exists():
        for line in path.read_text().splitlines()[1:]:
            f = line.split()
            if len(f) >= 9 and f[1].endswith("%"):  # pilot rows, not the "fast1 - mc1 ..." comparisons
                rows[f[0]] = {"arrived": float(f[1].rstrip("%")), "battery": float(f[3]), "climb": float(f[6])}
    return rows


WEATHER = [("Motor only", MOTOR, "results/weather_chain/{d}", "motor", False),
           ("Rule, thermals known", RULE, "results/weather_chain/{d}", "mc1", False),
           ("Rule, blind", RULE, "results/weather_chain/{d}", "bmc1", True),
           ("Generator, blind", GEN, "results/weather_chain/{d}", "bfast1", True)]
fig, axes = plt.subplots(2, 3, figsize=(COL_W * 1.0, 0.64 * COL_W), sharey="row",
                         gridspec_kw={"height_ratios": [1.5, 1], "hspace": 0.18, "wspace": 0.08})
for j, (day, title) in enumerate(DAYS):
    for row, key in enumerate(("battery", "arrived")):
        ax = axes[row, j]
        for k, (name, color, folder, pilot, blind) in enumerate(WEATHER):
            rec = summary(Path(folder.format(d=day)) / "crosscountry_summary.txt").get(pilot)
            if rec is None:
                ax.bar(k, 100 if key == "arrived" else 60, 0.78, fill=False, edgecolor=color, linestyle="--",
                       linewidth=1.8, label=name if (row, j) == (0, 0) else None)
                ax.text(k, 4, "?", ha="center", va="bottom", color=color, fontsize=24, weight="bold")
                continue
            v = rec[key]
            ax.bar(k, v, 0.78, color=color, hatch="//" if blind else None, edgecolor=SURFACE, linewidth=1.5,
                   label=name if (row, j) == (0, 0) else None)
            ax.text(k, v + 2, f"{v:.1f}" if key == "battery" and v < 10 else f"{v:.0f}", ha="center", va="bottom", fontsize=17,
                    color=INK)
        ax.set_xticks([])
        ax.grid(axis="x", visible=False)
        if row == 0:
            ax.set_title(title, fontsize=20, color=INK)
            ax.set_ylim(0, 140)
        else:
            ax.set_ylim(0, 125)
            ax.set_yticks([0, 50, 100])
    axes[0, 0].set_ylabel("battery (kJ)")
    axes[1, 0].set_ylabel("arrived (%)")
plt.rcParams["hatch.color"] = SURFACE
fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.07), ncol=2, frameon=False, handlelength=1.3,
           columnspacing=1.5)
save(fig, "weather_bars")


# 6. One 20 km task: altitude against distance made good, circling rule and planned climbs (phase 5 viewer data).
def page_data(path):
    text = Path(path).read_text(encoding="utf-8")
    return json.loads(re.search(r"const DATA\s*=\s*(\{.*?\});\s*\n", text, re.S).group(1))


xc = page_data("results/phase5_view/xc.html")
task = xc["task"]
fig, ax = plt.subplots(figsize=(COL_W, 0.42 * COL_W))
for f, color, lab in ((xc["flights"][0], RULE, "circling rule"), (xc["flights"][1], GA, "planned climbs")):
    p = np.array(f["p"])
    ax.plot((task["distance"] - np.array(f["to_goal"])) / 1000, p[:, 2], color=color, lw=2.4,
            label=f"{lab}: {f['duration'] / 60:.0f} min")
ax.axhline(600, ls="--", color=INK2, lw=1.2)
ax.text(19.8, 612, "cloud base", fontsize=17, color=INK2, ha="right", va="bottom")
ax.set_xlabel("distance toward the goal (km)")
ax.set_ylabel("altitude (m)")
ax.set_xlim(0, 20)
ax.set_ylim(100, 700)
ax.legend(loc="lower right", frameon=False)
save(fig, "barogram")


# 7. The real day: forecast w* over Issoudun, the thermals alive at mid-flight and one task flown by the rule.
from geneticwing.crosscountry import Autopilot, MacCreadyPilot, fly_xc  # noqa: E402
from geneticwing.weather import ConvectiveDay, weather_task  # noqa: E402

day = ConvectiveDay.load(Path("data/weather/issoudun_2025-07-12.npz"))
task = weather_task(day, 0)
flight = fly_xc(task, MacCreadyPilot(macready=0.5, wind_aware=True), autopilot=Autopilot(inner_wing_protection=0.9))
t_mid = 0.5 * flight.duration
hour = task.start_hour + t_mid / 3600.0
k = int(np.argmin(np.abs(day.hours - hour)))
km = 1e-3
blues = LinearSegmentedColormap.from_list("blues", ["#eef4fc", "#cde2fb", "#86b6ef", "#2a78d6", "#1c5cab"])
fig, ax = plt.subplots(figsize=(COL_W, 0.78 * COL_W))
mesh = ax.pcolormesh(day.x * km, day.y * km, day.w_star[k], cmap=blues, shading="gouraud", rasterized=True)
cb = fig.colorbar(mesh, ax=ax, orientation="horizontal", fraction=0.05, pad=0.13, aspect=40)
cb.set_label(f"forecast w* (m/s), 12 Jul 2025 {int(hour):02d}:{int(60 * (hour % 1)):02d}")
cb.outline.set_visible(False)
alive = [(i, th) for i, th in enumerate(task.thermals) if task.thermal_at(i, t_mid) is not None]
centres = np.array([th.center for _, th in alive]) + np.array(task.wind) * t_mid
strength = np.array([task.thermal_at(i, t_mid).w_star for i, _ in alive])
m = task.to_map(centres) * km
ax.scatter(m[:, 0], m[:, 1], s=6 * strength**2, color=GEN, alpha=0.85, linewidths=0, label="thermals alive")
path = task.to_map(flight.state.position[0, :, :2]) * km
ax.plot(path[:, 0], path[:, 1], color=INK, lw=2.2, label="one 20 km task")
ends = task.to_map(np.array([[0.0, 0.0], list(task.goal)])) * km
ax.plot(*ends[0], "o", color=INK, ms=11)
ax.plot(*ends[1], "s", color=INK, ms=11)
ax.set_aspect("equal")
ax.grid(False)
ax.set_xlabel("km east of Issoudun-Le Fay")
ax.set_ylabel("km north")
ax.set_xlim(day.x[0] * km, day.x[-1] * km)
ax.set_ylim(day.y[0] * km, day.y[-1] * km)
ax.legend(loc="upper left", frameon=True, facecolor=SURFACE, edgecolor=GRID, markerscale=1.5)
save(fig, "weather_map")
