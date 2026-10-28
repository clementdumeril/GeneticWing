import numpy as np

from geneticwing.aircraft import SBXC_APPROX as AC
from geneticwing.strip import minimum_turn_radius, sbxc_wing, turn_loads


def test_geometry_matches_the_airframe():
    wing = sbxc_wing()
    assert abs(wing.area - AC.wing_area) < 1e-9
    assert abs(wing.span - 4.6) < 0.01


def test_straight_flight_needs_no_aileron_and_keeps_the_mean_cl():
    loads = turn_loads(sbxc_wing(), AC, 9.0, 0.0, 0.8)
    assert abs(loads.aileron) < 1e-12 and abs(loads.extra_drag) < 1e-12
    assert abs(loads.cl_max_local - 0.8) < 1e-12


def test_inner_wing_carries_more_lift_coefficient_in_tighter_turns():
    wing, speed = sbxc_wing(), 9.5
    peaks, ailerons = [], []
    for radius in (30.0, 20.0, 12.0):
        bank = np.arctan(speed**2 / (AC.g * radius))
        cl = AC.weight / (np.cos(bank) * AC.dynamic_pressure(speed) * AC.wing_area)
        loads = turn_loads(wing, AC, speed, bank, cl)
        assert loads.cl_max_local > cl  # the slower inner wing works harder than the mean
        peaks.append(loads.cl_max_local - cl)
        ailerons.append(loads.aileron)
    assert peaks == sorted(peaks) and ailerons == sorted(ailerons)  # both grow as the turn tightens


def test_turn_loads_are_symmetric_in_bank_sign_and_batch():
    wing = sbxc_wing()
    left = turn_loads(wing, AC, np.array([9.0, 11.0]), np.radians([-30.0, -25.0]), np.array([0.9, 0.7]))
    right = turn_loads(wing, AC, np.array([9.0, 11.0]), np.radians([30.0, 25.0]), np.array([0.9, 0.7]))
    np.testing.assert_allclose(left.cl_max_local, right.cl_max_local)
    np.testing.assert_allclose(left.extra_drag, right.extra_drag)


def test_strip_wing_cannot_turn_tighter_than_the_point_mass():
    wing = sbxc_wing()
    for speed in (9.0, 10.0, 12.0):
        r = minimum_turn_radius(wing, AC, speed, 0.9 * AC.cl_max)
        assert r["strip"] >= r["point_mass"] - 1e-9
    slow = minimum_turn_radius(wing, AC, 9.0, 0.9 * AC.cl_max)
    assert slow["strip"] > 1.5 * slow["point_mass"]  # at low speed the inner tip stalls first
