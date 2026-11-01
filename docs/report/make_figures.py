"""Figures of the GeneticWing report (docs/report/figures), from the models and the exported flights."""

import json
import re
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from geneticwing.aircraft import SBXC_APPROX as AC  # noqa: E402
from geneticwing.crosscountry import speed_to_fly  # noqa: E402
from geneticwing.wind.thermal import Thermal  # noqa: E402

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25, "savefig.dpi": 200})
GREEN, GREY, BLUE, RED, ORANGE = "#17936a", "#8a97a3", "#2f6db0", "#c9502f", "#e39a2d"


def data(path):
    return json.loads(re.search(r"const DATA = (\{.*?\});\n", Path(path).read_text(encoding="utf-8"), re.S).group(1))


# 1. Polar and MacCready construction.
v = np.linspace(7.6, 22, 300)
sink = AC.nominal_sink_rate(v)
fig, ax = plt.subplots(figsize=(6.2, 3.6))
ax.plot(v, -sink, color=BLUE, lw=2, label="SB-XC still-air polar")
for m, c in ((0.0, GREY), (1.0, GREEN), (2.0, RED)):
    vs = speed_to_fly(AC, m)
    s = AC.nominal_sink_rate(vs)
    ax.plot([0, vs, 22], [m, -s, m + (-s - m) * 22 / vs], "--", color=c, lw=1)
    ax.plot(vs, -s, "o", color=c, label=f"M = {m:g} m/s: V = {vs:.1f} m/s")
ax.set_xlim(0, 22); ax.set_ylim(-2.2, 2.2)
ax.set_xlabel("airspeed (m/s)"); ax.set_ylabel("vertical speed (m/s)")
ax.set_title("Speed-to-fly: tangent from the expected climb M to the polar")
ax.legend(fontsize=8, loc="lower left")
fig.tight_layout(); fig.savefig(OUT / "polar_maccready.png"); plt.close(fig)

# 2. Thermal model.
th = Thermal(center=(0.0, 0.0), w_star=2.0)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.2))
r = np.linspace(0, 120, 200)
for z, c in ((150, GREY), (300, ORANGE), (500, RED)):
    w = th(np.column_stack([r, 0 * r, np.full_like(r, z)]))[:, 2]
    a1.plot(r, w, color=c, label=f"z = {z} m")
a1.set_xlabel("distance to the core (m)"); a1.set_ylabel("updraft (m/s)"); a1.legend(fontsize=8)
a1.set_title("Radial profile, w* = 2 m/s")
z = np.linspace(10, 990, 200)
a2.plot(th.mean_updraft(z), z, color=ORANGE, label="mean updraft")
a2.plot(th.outer_radius(z) / 10, z, color=BLUE, label="outer radius / 10 (m)")
a2.set_xlabel("m/s  |  m/10"); a2.set_ylabel("altitude (m)"); a2.legend(fontsize=8)
a2.set_title("Allen scaling with altitude")
fig.tight_layout(); fig.savefig(OUT / "thermal_model.png"); plt.close(fig)

# 3. A 60 s closed-loop flight in 3D (calm air, and misjudged thermal replanned).
d = data("results/phase4_view/flights.html")
fig = plt.figure(figsize=(8.4, 4.2))
for k, key in enumerate(("calm_open", "wrong_mpc")):
    f = next(x for x in d["flights"] if x["key"] == key)
    ax = fig.add_subplot(1, 2, k + 1, projection="3d")
    p = np.array(f["p"]); t = np.array(f["t"])
    vz = np.array(f["v"])[:, 2]
    tc = np.array(d["thermal"]["center"]) + np.array(d["thermal"]["drift"]) * 30.0
    zz = np.linspace(p[:, 2].min() - 20, p[:, 2].max() + 10, 2)
    ang = np.linspace(0, 2 * np.pi, 60)
    for rr in (15, 30):
        X = tc[0] + rr * np.cos(ang)[:, None]; Y = tc[1] + rr * np.sin(ang)[:, None]
        ax.plot_surface(X + 0 * zz, Y + 0 * zz, np.tile(zz, (60, 1)), color=ORANGE, alpha=0.10 if rr == 30 else 0.18, linewidth=0)
    sc = ax.scatter(p[:, 0], p[:, 1], p[:, 2], c=vz, cmap="RdYlGn", vmin=-2, vmax=3, s=3)
    for pl in f["plans"][:1] if key == "calm_open" else f["plans"]:
        q = np.array(pl["p"]); ax.plot(q[:, 0], q[:, 1], q[:, 2], "--", color=BLUE, lw=0.7, alpha=0.7)
    if key == "wrong_mpc":
        b = np.array(f["belief"]["center"]) + np.array(f["belief"]["wind"]) * 30.0
        ax.plot([b[0], b[0]], [b[1], b[1]], zz, "k:", lw=1.5)
    ax.set_title({"calm_open": "Calm air, planned once", "wrong_mpc": "Thermal misjudged by ~15 m, replanned"}[key], fontsize=9)
    ax.set_xlabel("east (m)", fontsize=8); ax.set_ylabel("north (m)", fontsize=8); ax.set_zlabel("altitude (m)", fontsize=8)
    ax.view_init(elev=22, azim=-60); ax.tick_params(labelsize=7)
cb = fig.colorbar(sc, ax=fig.axes, shrink=0.6, pad=0.08); cb.set_label("climb rate (m/s)")
fig.savefig(OUT / "flight60_3d.png", bbox_inches="tight"); plt.close(fig)

# 4. The 20 km flight in 3D.
x = data("results/phase5_view/xc.html")
fig = plt.figure(figsize=(9.5, 4.6))
ax = fig.add_subplot(projection="3d")
for f, c, lw, lab in ((x["flights"][0], GREY, 0.8, "circling rule (MacCready 1)"),
                      (x["flights"][1], GREEN, 1.3, "latent GA climbs (MacCready 1)")):
    p = np.array(f["p"]); ax.plot(p[:, 0] / 1000, p[:, 1] / 1000, p[:, 2], color=c, lw=lw, label=lab)
for t in x["thermals"]:
    if abs(t["c"][1]) < 1100:
        ax.plot([t["c"][0] / 1000] * 2, [t["c"][1] / 1000] * 2, [0, 600], color=ORANGE, alpha=0.15 + 0.4 * (t["w"] - 1.2) / 1.6, lw=2)
ax.set_xlabel("east (km)"); ax.set_ylabel("north (km)"); ax.set_zlabel("altitude (m)")
ax.set_box_aspect((4, 1.0, 1.3)); ax.view_init(elev=20, azim=-70); ax.legend(fontsize=8, loc="upper left")
ax.set_yticks([-1, 0, 1]); ax.set_zticks([0, 200, 400, 600]); ax.tick_params(labelsize=7)
ax.set_ylabel("north (km)", labelpad=2); ax.set_zlabel("altitude (m)", labelpad=6)
ax.set_title("20 km task: climbs in thermals (orange columns) and glides between them", fontsize=9)
fig.savefig(OUT / "xc_3d.png", bbox_inches="tight"); plt.close(fig)

# 5. Summary of the main comparisons.
fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 3.3))
names = ["Teacher", "VAE prior\nx64", "Latent GA", "Distilled\nx64"]
vals = [-4.88, -4.64, -5.66, -5.72]
a1.bar(names, [-v_ for v_ in vals], color=[GREY, BLUE, GREEN, GREEN])
for i, v_ in enumerate(vals):
    a1.text(i, -v_ + 0.05, f"{v_:.2f}", ha="center", fontsize=8)
a1.set_ylabel("energy gained (-cost)"); a1.set_title("65 test situations, strip wing", fontsize=9); a1.set_ylim(4, 6.1)
pil = ["Circling\nrule", "Latent GA", "Distilled\n(S1 only)", "Distilled\n+ DAgger"]
climb = [1.29, 2.08, 1.50, 2.23]
a2.bar(pil, climb, color=[GREY, GREEN, BLUE, GREEN])
for i, c_ in enumerate(climb):
    a2.text(i, c_ + 0.03, f"{c_:.2f}", ha="center", fontsize=8)
a2.set_ylabel("mean climb rate (m/s)"); a2.set_title("64 tasks of 20 km, MacCready 0.5", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "summary_bars.png"); plt.close(fig)
print("figures:", sorted(p.name for p in OUT.iterdir()))

# 6. The 20 km task: top view and barogram (English).
plt.rcParams.update({"font.size": 11})
fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.2, 6.4), gridspec_kw={"height_ratios": [0.8, 1]})
task = x["task"]
for f, c, w, lab in ((x["flights"][0], GREY, 1.5, "circling rule"), (x["flights"][1], GREEN, 2.0, "latent GA climbs")):
    p = np.array(f["p"])
    a1.plot(p[:, 0] / 1000, p[:, 1] / 1000, color=c, lw=w,
            label=f"{lab}: {f['duration'] / 60:.1f} min, {f['speed_kmh']:.1f} km/h, {f['battery_kj']:.0f} kJ")
    a2.plot((task["distance"] - np.array(f["to_goal"])) / 1000, p[:, 2], color=c, lw=w)
for t in x["thermals"]:
    a1.add_patch(plt.Circle((t["c"][0] / 1000, t["c"][1] / 1000), 0.08, color=ORANGE, alpha=0.15 + 0.4 * (t["w"] - 1.2) / 1.6, lw=0))
a1.plot(0, 0, "ko", ms=6); a1.plot(task["goal"][0] / 1000, task["goal"][1] / 1000, "k*", ms=14)
a1.set_xlim(-0.6, 20.6); a1.set_ylim(-1.1, 1.1); a1.set_xlabel("east (km)"); a1.set_ylabel("north (km)")
a1.legend(loc="upper center", bbox_to_anchor=(0.5, 1.42), fontsize=10, frameon=False)
a2.axhline(600, ls="--", color=BLUE, lw=1); a2.axhline(120, ls="--", color=RED, lw=1)
a2.text(0.15, 612, "cloud base 600 m", fontsize=9, color=BLUE); a2.text(0.15, 132, "motor floor 120 m", fontsize=9, color=RED)
a2.set_xlabel("distance made good toward the goal (km)"); a2.set_ylabel("altitude (m)"); a2.set_xlim(0, 20)
fig.tight_layout(); fig.savefig(OUT / "xc_flight_en.png"); plt.close(fig)
print("xc_flight_en.png")
