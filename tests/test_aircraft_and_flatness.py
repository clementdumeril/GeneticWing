import numpy as np
import pytest

from geneticwing import STRATOSURFER_APPROX as AC
from geneticwing import flat_state
from geneticwing.aircraft import identify_parabolic_polar
from geneticwing.energy import netto


def straight_steady_glide(airspeed, wind, n=50, duration=10.0):
    gamma, _, _ = AC.steady_glide(airspeed)
    v_air = airspeed * np.array([np.cos(gamma), 0.0, np.sin(gamma)])
    v = v_air + np.asarray(wind)
    t = np.linspace(0, duration, n)
    p = np.array([0.0, 0.0, 200.0]) + t[:, None] * v
    return flat_state(AC, t, p, np.tile(v, (n, 1)), np.zeros((n, 3)), np.tile(wind, (n, 1)))


def test_identified_polar_reproduces_the_best_glide_point():
    cd0, k = identify_parabolic_polar(1.3, 0.30, 12.0, 1.51)
    assert cd0 == pytest.approx(k * (2 * 1.3 * 9.80665 / (1.225 * 144 * 0.30)) ** 2)
    assert AC.min_drag_airspeed == pytest.approx(12.0)
    _, sink, _ = AC.steady_glide(12.0)
    assert sink == pytest.approx(1.51, rel=0.01)


def test_sailplane_geometry_and_performance():
    from geneticwing.aircraft import SBXC_APPROX

    assert SBXC_APPROX.mass / SBXC_APPROX.wing_area == pytest.approx(3.95, abs=0.01)
    assert 20.0 < SBXC_APPROX.max_glide_ratio < 35.0
    # Slowest flyable glide (90 % of C_L max) sinks well under the Stratosurfer's best glide.
    cl = 0.9 * SBXC_APPROX.cl_max
    airspeed = np.sqrt(2 * SBXC_APPROX.weight / (SBXC_APPROX.rho * SBXC_APPROX.wing_area * cl))
    assert SBXC_APPROX.nominal_sink_rate(airspeed) < 0.4


def test_steady_glide_balances_forces():
    for bank in (0.0, np.radians(30)):
        gamma, sink, cl = AC.steady_glide(14.0, bank)
        q_s = AC.dynamic_pressure(14.0) * AC.wing_area
        assert q_s * cl * np.cos(bank) == pytest.approx(AC.weight * np.cos(gamma))
        assert q_s * AC.drag_coefficient(cl) == pytest.approx(-AC.weight * np.sin(gamma))
        assert sink > 0


def test_steady_glide_needs_no_thrust_and_matches_the_polar():
    s = straight_steady_glide(12.5, [0.0, 0.0, 0.0])
    _, sink, cl = AC.steady_glide(12.5)
    np.testing.assert_allclose(s.thrust_required, 0.0, atol=1e-10)
    np.testing.assert_allclose(s.cl, cl)
    np.testing.assert_allclose(s.sink_rate_air, sink)
    np.testing.assert_allclose(s.bank, 0.0, atol=1e-12)


def test_uniform_wind_changes_ground_track_but_not_air_quantities():
    calm = straight_steady_glide(12.5, [0.0, 0.0, 0.0])
    windy = straight_steady_glide(12.5, [3.0, -4.0, 0.0])
    for name in ("airspeed", "cl", "bank", "thrust_required"):
        np.testing.assert_allclose(getattr(windy, name), getattr(calm, name), atol=1e-10)
    assert not np.allclose(windy.ground_speed, calm.ground_speed)


def test_coordinated_level_turn():
    speed, radius, n = 15.0, 60.0, 40
    omega = speed / radius
    t = np.linspace(0, 2 * np.pi / omega, n)
    p = radius * np.stack([np.cos(omega * t), np.sin(omega * t), np.zeros(n)], axis=-1)
    v = speed * np.stack([-np.sin(omega * t), np.cos(omega * t), np.zeros(n)], axis=-1)
    a = -omega**2 * p
    s = flat_state(AC, t, p, v, a, np.zeros((n, 3)))
    bank = np.arctan(speed**2 / (AC.g * radius))
    # Counter-clockwise seen from above is a left turn: negative bank.
    np.testing.assert_allclose(s.bank, -bank)
    np.testing.assert_allclose(s.load_factor, 1 / np.cos(bank))
    np.testing.assert_allclose(s.thrust_required, s.drag)


def test_netto_reads_zero_in_still_air_and_the_updraft_in_rising_air():
    calm = straight_steady_glide(12.5, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(netto(AC, calm), 0.0, atol=1e-9)
    rising = straight_steady_glide(12.5, [0.0, 0.0, 1.2])
    np.testing.assert_allclose(netto(AC, rising), 1.2, atol=1e-9)
    np.testing.assert_allclose(netto(AC, rising, window_s=1.0), 1.2, atol=1e-9)
