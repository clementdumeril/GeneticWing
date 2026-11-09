"""Thermals of a real day: Allen's profile, the living thermal field and the tasks (no network)."""

import numpy as np

from geneticwing.crosscountry import ThermalField
from geneticwing.weather import (
    ConvectiveDay,
    LivingThermalField,
    allen_updraft,
    envelope,
    parcel_top,
    thermal_density,
    weather_task,
)
from geneticwing.wind.thermal import Thermal


def _day(w_star=2.5, z_i=2000.0, wind=(3.0, 1.0)):
    x, y, hours = np.arange(-20e3, 20e3 + 1, 2.5e3), np.arange(-15e3, 15e3 + 1, 2.5e3), np.arange(8, 20)
    shape = (len(hours), len(y), len(x))
    full = lambda v: np.full(shape, float(v))  # noqa: E731
    return ConvectiveDay("test", "2025-07-12", x, y, hours, full(300.0), full(z_i), np.full(len(hours), z_i),
                         full(w_star), full(1500.0), np.broadcast_to(np.array(wind, float), shape + (2,)).copy(),
                         full(0.0), full(0.0))


def test_allen_profile_mean_is_the_mean_updraft():
    for z, z_i in ((300.0, 1000.0), (400.0, 2300.0), (500.0, 1500.0)):
        th = Thermal(center=(0.0, 0.0), w_star=2.5, z_i=z_i)
        r2 = float(th.outer_radius(z))
        r = np.linspace(0.0, r2, 4001)
        mean = np.trapezoid(allen_updraft(r, z, 2.5, z_i) * 2.0 * r, r) / r2**2
        assert abs(mean - float(th.mean_updraft(z))) < 0.08 * float(th.mean_updraft(z))


def test_allen_profile_is_zero_far_out_and_above_the_layer():
    assert allen_updraft(np.array(0.25 * 2000.0), 400.0, 2.5, 2000.0) == 0.0
    assert allen_updraft(np.array(0.0), 2100.0, 2.5, 2000.0) == 0.0


def test_envelope():
    assert envelope(-0.1) == 0.0 and envelope(1.1) == 0.0
    assert envelope(0.5) == 1.0 and abs(envelope(0.1) - 0.5) < 1e-12


def test_parcel_top_meets_the_sounding():
    # Environment cooling at 6.5 K/km from 20 C at the ground (0 m); a parcel at 21 C cools at 9.8 K/km and
    # meets it after 1 / (9.8 - 6.5) km = 303 m.
    heights = np.array([0.0, 200.0, 400.0, 1000.0])[:, None]
    temps = 20.0 - 6.5e-3 * heights
    top = parcel_top(np.array([20.0]), np.array([0.0]), temps, heights)
    assert abs(top[0] - 303.0) < 1.0


def test_thermal_density_follows_allen():
    z_i = 2000.0
    r2 = float(Thermal(center=(0, 0), z_i=z_i).outer_radius(0.4 * z_i))
    assert abs(thermal_density(z_i) - 0.6 / (z_i * r2)) < 1e-12


def test_living_field_subset_matches_every_thermal():
    task = weather_task(_day(), 3)
    field = LivingThermalField(task.thermals, births=task.births, lives=task.lives)
    rng = np.random.default_rng(1)
    pts = rng.uniform([0.0, -2000.0, 100.0], [20000.0, 2000.0, 600.0], (60, 3))
    for t in (0.0, 900.0, 2500.0):
        centres = field._c + field._d * t
        for q in pts:
            r = np.hypot(q[0] - centres[:, 0], q[1] - centres[:, 1])
            brute = (allen_updraft(r, q[2], field._w, field._zi) * envelope((t - field._b) / field._l)).sum()
            assert abs(field(q, t)[2] - brute) < 1e-12


def test_weather_task():
    task = weather_task(_day(), 0)
    assert task.distance == 20_000.0 and 11.0 <= task.start_hour <= 14.5
    assert len(task.thermals) > 100 and all(isinstance(th, Thermal) for th in task.thermals)
    # The wind turned into the task frame keeps its speed, and every thermal drifts with it.
    assert abs(np.hypot(*task.wind) - np.hypot(3.0, 1.0)) < 1e-9
    assert all(th.drift == task.wind for th in task.thermals)
    # Thermals come and go: some are absent at the start, the rest are scaled by their envelope.
    alive = [task.thermal_at(i, 0.0) for i in range(len(task.thermals))]
    assert any(a is None for a in alive) and any(a is not None for a in alive)
    # The route stays inside the grid.
    ends = task.to_map(np.array([[0.0, 0.0], [20_000.0, 0.0]]))
    assert np.all(np.abs(ends[:, 0]) <= 20e3) and np.all(np.abs(ends[:, 1]) <= 15e3)
    # Between the thermals the air sinks a little.
    assert -0.5 < task.sink(np.array([0.0, 0.0, 300.0]))[2] < 0.0


def test_thermals_stay_even_around_the_route_in_a_crosswind():
    # A strong wind: the thermals that drift across the route must be there on both sides until the end.
    counts = np.zeros(2)
    for instance in range(4):
        task = weather_task(_day(wind=(0.0, 6.0)), instance)
        c = np.array([th.center for th in task.thermals]) + np.array(task.wind) * task.time_limit
        alive = np.array([task.thermal_at(i, task.time_limit) is not None for i in range(len(task.thermals))])
        along = alive & (c[:, 0] > 0.0) & (c[:, 0] < 20_000.0)
        counts += [np.sum(along & (c[:, 1] > -3_000.0) & (c[:, 1] < 0.0)),
                   np.sum(along & (c[:, 1] > 0.0) & (c[:, 1] < 3_000.0))]
    assert counts.min() > 0.75 * counts.max()


def test_constant_thermals_unchanged():
    th = (Thermal(center=(100.0, 0.0), w_star=2.0), Thermal(center=(900.0, 50.0), w_star=1.5))
    p = np.array([120.0, 10.0, 300.0])
    assert abs(ThermalField(th)(p, 5.0)[2] - sum(t(p, 5.0)[2] for t in th)) < 1e-12
