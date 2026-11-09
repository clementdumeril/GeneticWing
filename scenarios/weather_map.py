"""Figure of the real-weather tasks: the thermal strength over Issoudun, one route flown, and the days through time.

Left: w* at the hour of one task (from the archived ICON-D2 forecast, ``geneticwing.weather``), the thermals alive
at mid-flight (dot area = strength) and the path of the circling-rule pilot (MacCready 0.5) on that task.
Middle and right: w* and z_i at the site through the three days (z_i by the parcel method; GFS dotted).
Run: ``python scenarios/weather_map.py --out docs/figures/weather_issoudun.png``
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from geneticwing.crosscountry import Autopilot, MacCreadyPilot, fly_xc
from geneticwing.weather import ConvectiveDay, weather_task

DAYS = {"2025-07-12": "12 Jul, strong", "2025-08-02": "2 Aug, cumulus", "2025-06-22": "22 Jun, windy"}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3 of the reference palette
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUES = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#86b6ef", "#2a78d6", "#1c5cab", "#0d366b"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--day", default="2025-07-12")
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("docs/figures/weather_issoudun.png"))
    args = parser.parse_args()
    days = {d: ConvectiveDay.load(Path(f"data/weather/issoudun_{d}.npz")) for d in DAYS}
    day = days[args.day]
    task = weather_task(day, args.task)
    flight = fly_xc(task, MacCreadyPilot(macready=0.5, wind_aware=True), autopilot=Autopilot(inner_wing_protection=0.9))
    t_mid = 0.5 * flight.duration

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(13, 4.8), facecolor="#fcfcfb")
    grid = fig.add_gridspec(2, 2, width_ratios=[1.35, 1], height_ratios=[1, 1], wspace=0.18, hspace=0.45)

    ax = fig.add_subplot(grid[:, 0])
    hour = task.start_hour + t_mid / 3600.0
    k = int(np.argmin(np.abs(day.hours - hour)))
    km = 1e-3
    lo, hi = np.floor(10 * day.w_star[k].min()) / 10, np.ceil(10 * day.w_star[k].max()) / 10
    mesh = ax.pcolormesh(day.x * km, day.y * km, day.w_star[k], cmap=BLUES, shading="gouraud", vmin=lo, vmax=hi)
    cbar = fig.colorbar(mesh, ax=ax, orientation="horizontal", fraction=0.05, pad=0.16, aspect=40)
    cbar.set_label("w* (m/s) from the ICON-D2 forecast")
    cbar.outline.set_visible(False)
    alive = [(i, th) for i, th in enumerate(task.thermals) if task.thermal_at(i, t_mid) is not None]
    centres = np.array([th.center for _, th in alive]) + np.array(task.wind) * t_mid
    strength = np.array([task.thermal_at(i, t_mid).w_star for i, _ in alive])
    m = task.to_map(centres) * km
    ax.scatter(m[:, 0], m[:, 1], s=1.5 * strength**2, color="#eb6834", alpha=0.8, linewidths=0,
               label=f"thermals alive at mid-flight ({len(alive)})")
    path = task.to_map(flight.state.position[0, :, :2]) * km
    ax.plot(path[:, 0], path[:, 1], color=INK, lw=1.2, label=f"circling rule, {flight.speed_kmh:.1f} km/h")
    ends = task.to_map(np.array([[0.0, 0.0], list(task.goal)])) * km
    ax.plot(*ends[0], "o", color=INK, ms=5)
    ax.plot(*ends[1], "s", color=INK, ms=5)
    ax.annotate("start", ends[0], xytext=(4, 4), textcoords="offset points", color=INK)
    ax.annotate("goal", ends[1], xytext=(4, 4), textcoords="offset points", color=INK)
    ax.set_aspect("equal")
    ax.set_xlabel("km east of Issoudun-Le Fay")
    ax.set_ylabel("km north")
    ax.set_xlim(day.x[0] * km, day.x[-1] * km)
    ax.set_ylim(day.y[0] * km, day.y[-1] * km)
    ax.set_title(f"{args.day} at {int(hour):02d}:{int(60 * (hour % 1)):02d}: one 20 km task", loc="left", color=INK,
                 fontsize=9.5)
    ax.legend(loc="lower left", frameon=True, facecolor="#fcfcfb", edgecolor=GRID, fontsize=8)

    for row, (field, label) in enumerate((("w_star", "w* (m/s)"), ("z_i", "z_i (m above ground)"))):
        a = fig.add_subplot(grid[row, 1])
        for color, (date, name) in zip(SERIES, DAYS.items()):
            d = days[date]
            j, i = np.argmin(np.abs(d.y)), np.argmin(np.abs(d.x))
            a.plot(d.hours, getattr(d, field)[:, j, i], color=color, lw=2, marker="o", ms=4, label=name)
            if field == "z_i":
                a.plot(d.hours, d.z_i_gfs, color=color, lw=1, ls=":")
        a.set_ylabel(label)
        a.grid(axis="y", color=GRID, lw=0.8)
        a.set_xlim(8, 19)
        if row == 0:
            a.set_title("At the site through the day (z_i: parcel method; GFS dotted)", loc="left", color=INK,
                        fontsize=9.5)
            a.legend(loc="lower center", frameon=False, fontsize=8, ncol=3)
        else:
            a.set_xlabel("local time (h)")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"{args.out}: task {args.task} at {task.start_hour:.2f} h, rule {flight.speed_kmh:.1f} km/h, "
          f"arrived {flight.arrived}, {len(flight.climbs)} climbs")


if __name__ == "__main__":
    main()
