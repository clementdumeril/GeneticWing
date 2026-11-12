"""Thermals from a real weather forecast: a convective day over a real place, for the cross-country tasks.

The thermal strength of a day is read from a numerical weather forecast archived by Open-Meteo
(https://open-meteo.com, historical forecast API), the way soaring forecasts such as RASP do:

* DWD ICON-D2 (2.2 km grid): surface sensible heat flux H, 2 m temperature and dew point, wind and
  temperature on pressure levels;
* the depth of the convective layer z_i, by the parcel method on the ICON-D2 sounding (a parcel lifted
  dry-adiabatically from the 2 m temperature plus 1 K until it is colder than the air), checked against
  the boundary-layer height of NOAA GFS (0.25 deg) at the site;
* Deardorff's convective velocity scale w* = (g / T * H / (rho c_p) * z_i)^(1/3), the input of Allen's
  updraft model (:class:`geneticwing.wind.thermal.Thermal`);
* the cumulus base from the 2 m dew-point spread, 125 m per kelvin.

:func:`fetch_day` downloads a grid around a site once and caches it under ``data/weather/``;
:class:`ConvectiveDay` interpolates it in space and time.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from geneticwing.crosscountry import ThermalField, XCTask
from geneticwing.wind.base import UniformWind, WindField
from geneticwing.wind.thermal import EnvironmentSink, Thermal, allen_radius, allen_updraft, thermal_density

API = "https://historical-forecast-api.open-meteo.com/v1/forecast"
SITES = {"issoudun": (46.89, 2.04)}  # Issoudun-Le Fay gliding centre, flat country in the Centre region of France
LEVELS = (975, 950, 925, 900, 850, 800, 700)
G, CP, RHO = 9.81, 1005.0, 1.15
ICON_VARS = ["sensible_heat_flux", "temperature_2m", "dew_point_2m", "cloud_cover_low", "precipitation"] + [
    f"{name}_{level}hPa" for level in LEVELS for name in ("temperature", "geopotential_height")
] + [f"{name}_925hPa" for name in ("wind_speed", "wind_direction")]


def _get(params: dict, tries: int = 6) -> list[dict]:
    """One API request; the free tier allows about 600 location-variable calls a minute, so wait and retry."""
    query = "&".join(f"{k}={v}" for k, v in params.items())
    for k in range(tries):
        try:
            with urllib.request.urlopen(f"{API}?{query}", timeout=120) as r:
                data = json.load(r)
            return data if isinstance(data, list) else [data]
        except urllib.error.HTTPError as e:
            if e.code != 429 or k == tries - 1:
                raise
            time.sleep(30 * (k + 1))
    raise RuntimeError("unreachable")


def fetch_day(site: str, date: str, box_km=(40.0, 30.0), step_km: float = 2.5, hours=(8, 19),
              cache: Path = Path("data/weather")) -> Path:
    """Download the forecast fields of ``date`` on a grid ``box_km`` (east, north) centred on ``site``."""
    out = cache / f"{site}_{date}.npz"
    if out.exists():
        return out
    lat0, lon0 = SITES[site]
    x = np.arange(-box_km[0] / 2, box_km[0] / 2 + 1e-9, step_km)
    y = np.arange(-box_km[1] / 2, box_km[1] / 2 + 1e-9, step_km)
    lat = lat0 + y / 111.2
    lon = lon0 + x / (111.2 * np.cos(np.radians(lat0)))
    pts = [(la, lo) for la in lat for lo in lon]
    fields: dict[str, list] = {}
    for k in range(0, len(pts), 50):
        chunk = pts[k:k + 50]
        common = {"latitude": ",".join(f"{p[0]:.4f}" for p in chunk), "longitude": ",".join(f"{p[1]:.4f}" for p in chunk),
                  "start_date": date, "end_date": date, "timezone": "Europe%2FParis"}
        for a in _get(common | {"hourly": ",".join(ICON_VARS), "models": "icon_d2"}):
            fields.setdefault("elevation", []).append(a["elevation"])
            for name in ICON_VARS:
                fields.setdefault(name, []).append(a["hourly"][name][hours[0]:hours[1] + 1])
        time.sleep(20.0)
    # GFS boundary-layer height at the site only (0.25 deg grid), to check the parcel method.
    gfs = _get({"latitude": f"{lat0:.4f}", "longitude": f"{lon0:.4f}", "start_date": date, "end_date": date,
                "timezone": "Europe%2FParis", "hourly": "boundary_layer_height", "models": "gfs_seamless"})[0]
    shape = (len(y), len(x), hours[1] - hours[0] + 1)
    elevation = np.array(fields.pop("elevation"), dtype=float).reshape(shape[:2])
    arrays = {k: np.moveaxis(np.array(v, dtype=float).reshape(shape), -1, 0) for k, v in fields.items()}
    arrays["elevation"] = elevation
    arrays["gfs_boundary_layer_height"] = np.array(gfs["hourly"]["boundary_layer_height"][hours[0]:hours[1] + 1], float)
    cache.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, x=x * 1e3, y=y * 1e3, hours=np.arange(hours[0], hours[1] + 1), lat0=lat0, lon0=lon0,
                        site=site, date=date, **arrays)
    return out


def parcel_top(t2: np.ndarray, z_ground: np.ndarray, temps: np.ndarray, heights: np.ndarray, excess: float = 1.0):
    """Height above ground where a parcel at ``t2 + excess`` lifted on the dry adiabat meets the sounding.

    ``temps`` and ``heights`` are (levels, ...) in deg C and m above sea level, ordered by pressure.
    """
    top = np.full(t2.shape, np.nan)
    for k in range(temps.shape[0] - 1):
        z_a, z_b = heights[k], heights[k + 1]
        excess_a = t2 + excess - 9.8e-3 * (z_a - z_ground) - temps[k]
        excess_b = t2 + excess - 9.8e-3 * (z_b - z_ground) - temps[k + 1]
        cross = np.isnan(top) & (z_b > z_ground) & (excess_a > 0) & (excess_b <= 0)
        f = excess_a / np.where(excess_a - excess_b == 0, 1.0, excess_a - excess_b)
        top = np.where(cross, np.maximum(z_a, z_ground) + f * (z_b - np.maximum(z_a, z_ground)) - z_ground, top)
    return np.nan_to_num(top, nan=0.0)


@dataclass
class ConvectiveDay:
    """The convective fields of one day on a grid: x east and y north in m from the site, hours local."""

    site: str
    date: str
    x: np.ndarray
    y: np.ndarray
    hours: np.ndarray
    heat_flux: np.ndarray  # W/m^2, upward positive, (hours, y, x)
    z_i: np.ndarray  # m above ground, parcel method
    z_i_gfs: np.ndarray  # m, GFS boundary-layer height at the site, (hours,)
    w_star: np.ndarray  # m/s
    cloud_base: np.ndarray  # m above ground
    wind: np.ndarray  # m/s, (hours, y, x, 2) east and north, at 925 hPa (the mixed layer)
    low_cloud: np.ndarray  # %
    rain: np.ndarray  # mm/h

    @classmethod
    def load(cls, path: Path) -> ConvectiveDay:
        d = np.load(path)
        heat = -d["sensible_heat_flux"]  # ICON's flux is negative when the ground heats the air
        temps = np.stack([d[f"temperature_{lv}hPa"] for lv in LEVELS])
        heights = np.stack([d[f"geopotential_height_{lv}hPa"] for lv in LEVELS])
        t2 = d["temperature_2m"]
        # The parcel starts at the ground, which can be above the 1000 hPa surface.
        z_ground = np.broadcast_to(d["elevation"], t2.shape)
        z_i = parcel_top(t2, z_ground, temps, heights)
        w_star = np.cbrt(np.clip(G / (t2 + 273.15) * heat / (RHO * CP) * z_i, 0.0, None))
        speed = d["wind_speed_925hPa"] / 3.6
        towards = np.radians(d["wind_direction_925hPa"] + 180.0)  # direction the wind blows to
        wind = np.stack([speed * np.sin(towards), speed * np.cos(towards)], axis=-1)
        return cls(str(d["site"]), str(d["date"]), d["x"], d["y"], d["hours"], heat, z_i,
                   d["gfs_boundary_layer_height"], w_star, 125.0 * (t2 - d["dew_point_2m"]), wind,
                   d["cloud_cover_low"], d["precipitation"])

    def at(self, field: np.ndarray, hour: float, x, y) -> np.ndarray:
        """Bilinear in space, linear in time; ``x``, ``y`` in m from the site, clipped to the grid."""
        x = np.clip(np.asarray(x, float), self.x[0], self.x[-1])
        y = np.clip(np.asarray(y, float), self.y[0], self.y[-1])
        h = float(np.clip(hour, self.hours[0], self.hours[-1]))
        k = min(int(np.searchsorted(self.hours, h, side="right")) - 1, len(self.hours) - 2)
        s = h - self.hours[k]
        i = np.clip(np.searchsorted(self.x, x, side="right") - 1, 0, len(self.x) - 2)
        j = np.clip(np.searchsorted(self.y, y, side="right") - 1, 0, len(self.y) - 2)
        u = (x - self.x[i]) / (self.x[i + 1] - self.x[i])
        v = (y - self.y[j]) / (self.y[j + 1] - self.y[j])

        def plane(f):
            u_, v_ = (u[..., None], v[..., None]) if f.ndim == 3 else (u, v)
            return ((1 - u_) * (1 - v_) * f[j, i] + u_ * (1 - v_) * f[j, i + 1]
                    + (1 - u_) * v_ * f[j + 1, i] + u_ * v_ * f[j + 1, i + 1])

        return (1 - s) * plane(field[k]) + s * plane(field[k + 1])


# -- thermals and tasks on a real day ------------------------------------------------------------------

_r2 = allen_radius  # kept for scenarios/weather_view.py


def envelope(s):
    """Strength through a thermal's life, ``s`` from 0 (birth) to 1 (death): a fifth to grow, a fifth to fade."""
    s = np.asarray(s, float)
    return np.clip(np.minimum(s, 1.0 - s) / 0.2, 0.0, 1.0) * ((s >= 0.0) & (s <= 1.0))


@dataclass(frozen=True)
class LivingThermalField(ThermalField):
    """Thermals with Allen's bell-shaped profile (:func:`allen_updraft`) that are born, grow, fade and die:
    ``births`` and ``lives`` in s, one per thermal. Every thermal adds its updraft (Allen keeps the nearest)."""

    births: tuple[float, ...] = ()
    lives: tuple[float, ...] = ()

    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, "_b", np.asarray(self.births, float))
        object.__setattr__(self, "_l", np.asarray(self.lives, float))
        object.__setattr__(self, "_alive_cache", {})

    def gain(self, t):
        return envelope((t - self._b) / self._l)

    def _local(self, t: float, xy) -> ThermalField:
        """The thermals alive at ``t`` that can reach the 1 km cell around ``xy`` (one subset per 20 s bucket and
        cell, reused step after step: a flight evaluates a few dozen thermals, not the tens of thousands of the day)."""
        cell, span = 1000.0, 20.0
        bucket, ix, iy = int(t // span), int(xy[0] // cell), int(xy[1] // cell)
        key = (bucket, ix, iy)
        cached = self._alive_cache.get(key)
        if cached is None:
            if self._alive_cache and next(iter(self._alive_cache))[0] != bucket:
                self._alive_cache.clear()
            lo, hi = span * bucket, span * (bucket + 1)
            mid = 0.5 * (lo + hi)
            cx, cy = self._c[:, 0] + self._d[:, 0] * mid, self._c[:, 1] + self._d[:, 1] * mid
            # Allen's profile is zero beyond 2 r2 < 0.2 z_i; the centre moves |drift| x span/2 around ``mid``.
            pad = cell / np.sqrt(2.0) + 0.2 * self._zi + np.hypot(self._d[:, 0], self._d[:, 1]) * 0.5 * span + 1.0
            near = np.hypot(cx - (ix + 0.5) * cell, cy - (iy + 0.5) * cell) < pad
            idx = np.nonzero(near & (self._b < hi) & (self._b + self._l > lo))[0]
            cached = LivingThermalField(tuple(self.thermals[i] for i in idx), births=tuple(self._b[idx]),
                                        lives=tuple(self._l[idx]))
            object.__setattr__(cached, "_alive_cache", None)
            self._alive_cache[key] = cached
        return cached

    def __call__(self, position, t=0.0):
        if self._alive_cache is not None and np.ndim(t) == 0 and np.ndim(position) == 1:  # the aircraft, one instant
            cached = self._local(float(t), position)
            return cached(position, t) if cached.thermals else np.zeros(3)
        # Many points (a whole flight): in chunks of consecutive points, each with only the thermals alive during
        # the chunk and within reach of it, so memory stays at (chunk x a few hundred thermals).
        position = np.asarray(position, dtype=float)
        shape = position.shape
        pos = position.reshape(-1, 3)
        times = np.broadcast_to(np.asarray(t, dtype=float), shape[:-1]).reshape(-1)
        out = np.zeros(pos.shape)
        reach = 0.2 * self._zi  # Allen's profile is zero beyond 2 r2 < 0.2 z_i
        for k in range(0, len(pos), 1024):
            p, tk = pos[k:k + 1024], times[k:k + 1024]
            alive = np.nonzero((self._b < tk.max()) & (self._b + self._l > tk.min()))[0]
            if alive.size == 0:
                continue
            cx = self._c[alive, 0] + self._d[alive, 0] * tk[:, None]
            cy = self._c[alive, 1] + self._d[alive, 1] * tk[:, None]
            r = np.hypot(p[:, :1] - cx, p[:, 1:2] - cy)
            near = np.nonzero((r < reach[alive]).any(axis=0))[0]
            if near.size == 0:
                continue
            idx = alive[near]
            w = allen_updraft(r[:, near], p[:, 2:3], self._w[idx], self._zi[idx]) \
                * envelope((tk[:, None] - self._b[idx]) / self._l[idx])
            out[k:k + 1024, 2] = w.sum(axis=-1)
        return out.reshape(shape)


@dataclass(frozen=True)
class SinkField(WindField):
    sink: EnvironmentSink

    def __call__(self, position, t=0.0):
        return self.sink(position, t)


@dataclass(frozen=True)
class WeatherXCTask(XCTask):
    """A cross-country task in the thermals of a real day (see :func:`weather_task`)."""

    births: tuple[float, ...] = ()
    lives: tuple[float, ...] = ()
    sink: EnvironmentSink | None = None
    origin: tuple[float, float] = (0.0, 0.0)  # m, the start in the map frame (east, north from the site)
    bearing: float = 0.0  # rad, direction of the goal in the map frame, from east, counter-clockwise
    start_hour: float = 12.0

    @property
    def air(self) -> WindField:
        air = UniformWind((self.wind[0], self.wind[1], 0.0)) + LivingThermalField(
            self.thermals, births=self.births, lives=self.lives)
        return air if self.sink is None else air + SinkField(self.sink)

    def thermal_at(self, i, t):
        g = float(envelope((t - self.births[i]) / self.lives[i]))
        return None if g < 0.1 else replace(self.thermals[i], w_star=self.thermals[i].w_star * g)

    def to_map(self, q):
        """Task frame (start at 0, goal on +x) to the map frame of the weather grid (east, north from the site)."""
        c, s = np.cos(self.bearing), np.sin(self.bearing)
        q = np.asarray(q, float)
        return np.stack([self.origin[0] + c * q[..., 0] - s * q[..., 1],
                         self.origin[1] + s * q[..., 0] + c * q[..., 1]], axis=-1)


def weather_task(day: ConvectiveDay, instance: int, distance: float = 20_000.0, hours=(11.0, 14.5),
                 lifetime=(300.0, 1800.0), spread: float = 0.25, margin: float = 3_000.0, z_ceiling: float = 600.0):
    """Randomized task on a real day: a route of ``distance`` m inside the weather grid, started between ``hours``.

    The thermals form a Poisson process in space and time whose density follows Allen's Eq. (20) with the local
    z_i; each lives ``lifetime`` s (uniform, Allen's 5 to 30 min), grows and fades (:func:`envelope`), and has
    the local w* at its birth times a log-normal factor of ``spread``. They drift with the mean 925 hPa wind of
    the area; between them the air sinks to return their mass flux (:class:`EnvironmentSink`). The aircraft
    flies in the lowest ``z_ceiling`` m, a small UAV's band, not up to the cloud base.
    """
    rng = np.random.default_rng(90_000 + instance)
    start_hour = float(rng.uniform(*hours))
    half = np.array([day.x[-1], day.y[-1]]) - 1_000.0
    while True:  # a route whose both ends are in the grid, with a kilometre to spare
        bearing = float(rng.uniform(0.0, 2.0 * np.pi))
        origin = rng.uniform(-half, half)
        end = origin + distance * np.array([np.cos(bearing), np.sin(bearing)])
        if np.all(np.abs(end) <= half):
            break
    c, s = np.cos(bearing), np.sin(bearing)

    def to_map(q):
        return np.stack([origin[0] + c * q[..., 0] - s * q[..., 1], origin[1] + s * q[..., 0] + c * q[..., 1]], -1)

    time_limit = distance / 3.5
    # Wind: the mean of the area at mid-flight, turned into the task frame.
    mid = start_hour + time_limit / 7200.0
    wind_map = day.wind[np.argmin(np.abs(day.hours - mid))].mean(axis=(0, 1))
    wind = (c * wind_map[0] + s * wind_map[1], -s * wind_map[0] + c * wind_map[1])

    # Space-time Poisson process by thinning, over the corridor of the route, in the task frame. The births
    # extend upwind by the drift of the longest-lived thermal, so the ones that drift into the corridor exist too:
    # otherwise the upwind side empties as the flight goes on and every thermal lies downwind of the route.
    reach = -np.array(wind) * lifetime[1]
    x0, x1 = -margin + min(reach[0], 0.0), distance + margin + max(reach[0], 0.0)
    y0, y1 = -margin - 1_000.0 + min(reach[1], 0.0), margin + 1_000.0 + max(reach[1], 0.0)
    t0, t1 = -lifetime[1], time_limit
    mean_life = 0.5 * (lifetime[0] + lifetime[1])
    hours_grid = np.linspace(start_hour + t0 / 3600.0, start_hour + t1 / 3600.0, 8)
    qx, qy = np.meshgrid(np.linspace(x0, x1, 25), np.linspace(y0, y1, 9))
    pts = to_map(np.stack([qx, qy], -1))
    n_max = max(float(thermal_density(np.maximum(day.at(day.z_i, h, pts[..., 0], pts[..., 1]), 300.0)).max())
                for h in hours_grid)
    count = rng.poisson(n_max * (x1 - x0) * (y1 - y0) * (t1 - t0) / mean_life)
    q = np.column_stack([rng.uniform(x0, x1, count), rng.uniform(y0, y1, count)])
    tb = rng.uniform(t0, t1, count)
    m = to_map(q)
    thermals, births, lives = [], [], []
    for k in range(count):
        hour = start_hour + tb[k] / 3600.0
        z_i = float(day.at(day.z_i, hour, m[k, 0], m[k, 1]))
        w_star = float(day.at(day.w_star, hour, m[k, 0], m[k, 1]))
        if z_i < 300.0 or w_star < 0.3 or rng.uniform() > thermal_density(z_i) / n_max:
            continue
        life = float(rng.uniform(*lifetime))
        drift_center = q[k] - np.array(wind) * tb[k]  # where it would have been at t = 0
        thermals.append(Thermal(center=(float(drift_center[0]), float(drift_center[1])),
                                w_star=w_star * float(np.exp(spread * rng.standard_normal())), z_i=z_i, drift=wind))
        births.append(float(tb[k]))
        lives.append(life)
    zi_mean = float(day.at(day.z_i, mid, *to_map(np.array([distance / 2, 0.0]))))
    ws_mean = float(day.at(day.w_star, mid, *to_map(np.array([distance / 2, 0.0]))))
    sink = EnvironmentSink(float(thermal_density(max(zi_mean, 300.0))), ws_mean, zi_mean)
    return WeatherXCTask(name=f"{day.site} {day.date} #{instance}", start=(0.0, 0.0, float(rng.uniform(250.0, 350.0))),
                heading_deg=90.0, goal=(distance, 0.0), wind=wind, thermals=tuple(thermals), time_limit=time_limit,
                z_ceiling=z_ceiling, births=tuple(births), lives=tuple(lives), sink=sink,
                origin=(float(origin[0]), float(origin[1])), bearing=bearing, start_hour=start_hour)
