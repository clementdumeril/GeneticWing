"""Benchmark soaring scenarios S1 to S3 of the GeneticWing protocol.

Each factory takes an ``instance`` seed that randomizes the start (position
and heading) around a fixed wind field, so methods are compared on the same
set of instances.

The airframe is the SB-XC sailplane fitted with a small motor (15 N): the
Albatross framing, where soaring is measured by the onboard energy it saves.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from geneticwing.aircraft import SBXC_APPROX
from geneticwing.soaring import GoalMission, LoiterMission, SoaringScenario
from geneticwing.wind import RidgeLift, Thermal, UniformWind

MOTOR_THRUST = 15.0  # N, placeholder motor-glider conversion
CRUISE_AIRSPEED = 9.0  # m/s, slightly above best L/D


def s1_single_thermal(instance: int = 0) -> SoaringScenario:
    """S1: stay within 300 m of a known thermal for 60 s, drifting in a 2 m/s breeze."""
    rng = np.random.default_rng(instance)
    wind = UniformWind((2.0, 0.0, 0.0))
    thermal = Thermal(center=(0.0, 0.0), w_star=2.0, z_i=1000.0, drift=(2.0, 0.0))
    bearing = rng.uniform(0.0, 2.0 * np.pi)
    distance = rng.uniform(80.0, 180.0)
    start = (distance * np.cos(bearing), distance * np.sin(bearing), rng.uniform(220.0, 280.0))
    heading = rng.uniform(0.0, 360.0)
    return SoaringScenario(
        name="S1 single thermal",
        aircraft=SBXC_APPROX,
        wind=wind + thermal,
        start=start,
        heading_deg=heading,
        airspeed=CRUISE_AIRSPEED,
        horizon=60.0,
        mission=LoiterMission(center=(0.0, 0.0), radius=300.0),
        thrust_max=MOTOR_THRUST,
        n_control=32,
    )


def s2_thermal_field(instance: int = 0) -> SoaringScenario:
    """S2: reach a goal 1.3 km away in 150 s, with two thermals near the route."""
    rng = np.random.default_rng(1000 + instance)
    wind = UniformWind((1.5, 0.0, 0.0))
    thermals = [
        Thermal(center=(400.0 + rng.normal(0, 40), 80.0 + rng.normal(0, 40)), w_star=2.2, drift=(1.5, 0.0)),
        Thermal(center=(900.0 + rng.normal(0, 40), -60.0 + rng.normal(0, 40)), w_star=1.8, drift=(1.5, 0.0)),
    ]
    field = wind
    for th in thermals:
        field = field + th
    return SoaringScenario(
        name="S2 thermal field to goal",
        aircraft=SBXC_APPROX,
        wind=field,
        start=(0.0, rng.normal(0, 30), 200.0 + rng.normal(0, 15)),
        heading_deg=90.0 + rng.normal(0, 15),
        airspeed=CRUISE_AIRSPEED,
        horizon=150.0,
        mission=GoalMission(goal=(1300.0, 0.0), radius=60.0),
        thrust_max=MOTOR_THRUST,
        n_control=56,
    )


def s3_ridge(instance: int = 0) -> SoaringScenario:
    """S3: loiter for 60 s on the windward side of a ridge, 8 m/s wind across it."""
    rng = np.random.default_rng(2000 + instance)
    # Crest along north; wind blows east (from the west), windward side is x < 0.
    ridge = RidgeLift(crest=(0.0, 0.0), axis_heading_deg=0.0, radius=60.0, wind_speed=8.0)

    def clearance(p: np.ndarray) -> np.ndarray:
        # Distance to the cylindrical ridge surface, or height above flat ground.
        to_axis = np.hypot(p[..., 0], p[..., 2]) - ridge.radius
        return np.minimum(to_axis, p[..., 2])

    return SoaringScenario(
        name="S3 ridge lift",
        aircraft=SBXC_APPROX,
        wind=ridge,
        start=(-110.0 + rng.normal(0, 15), rng.normal(0, 40), 110.0 + rng.normal(0, 10)),
        heading_deg=rng.choice([0.0, 180.0]) + rng.normal(0, 10),
        airspeed=CRUISE_AIRSPEED,
        horizon=60.0,
        mission=LoiterMission(center=(-90.0, 0.0), radius=250.0),
        thrust_max=MOTOR_THRUST,
        z_min=20.0,
        clearance=clearance,
        n_control=32,
    )


SCENARIOS = {"S1": s1_single_thermal, "S2": s2_thermal_field, "S3": s3_ridge}


def with_horizon(scenario: SoaringScenario, horizon: float, n_control: int) -> SoaringScenario:
    return replace(scenario, horizon=horizon, n_control=n_control)
