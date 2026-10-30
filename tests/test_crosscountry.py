from dataclasses import replace

import numpy as np

from geneticwing.aircraft import SBXC_APPROX as AC
from geneticwing.crosscountry import (
    MacCreadyPilot,
    MotorPilot,
    fly_xc,
    reference_energy,
    speed_to_fly,
    xc_family,
)
from geneticwing.wind.base import CompositeWind, UniformWind


def test_thermal_field_is_the_sum_of_its_thermals():
    task = xc_family(0, distance=5_000.0)
    slow = CompositeWind((UniformWind((task.wind[0], task.wind[1], 0.0)), *task.thermals))
    rng = np.random.default_rng(0)
    pts = np.column_stack([rng.uniform(-500, 5500, 300), rng.uniform(-1000, 1000, 300), rng.uniform(20, 900, 300)])
    t = rng.uniform(0, 1000, 300)
    np.testing.assert_allclose(task.air(pts, t), slow(pts, t), atol=1e-12)


def test_speed_to_fly_rises_with_the_macready_setting_and_falls_in_lift():
    speeds = [speed_to_fly(AC, m) for m in (0.0, 1.0, 2.0)]
    assert speeds == sorted(speeds) and speeds[0] > 0.9 * AC.min_drag_airspeed
    assert speed_to_fly(AC, 1.0, air_vertical=1.0) < speed_to_fly(AC, 1.0)


def test_motor_flight_scores_about_one():
    task = xc_family(1, distance=2_000.0)
    flight = fly_xc(task, MotorPilot())
    assert flight.arrived and abs(flight.cost - 1.0) < 0.25
    assert flight.battery > 0.8 * reference_energy(task) and not flight.violated


def test_macready_pilot_climbs_and_saves_battery():
    task = replace(xc_family(2, distance=8_000.0), start=(0.0, 0.0, 200.0))
    motor = fly_xc(task, MotorPilot())
    glider = fly_xc(task, MacCreadyPilot(macready=0.5))
    assert glider.arrived and glider.climbs
    assert glider.battery < 0.5 * motor.battery
    assert all(c["z1"] > c["z0"] for c in glider.climbs[:-1])
