import numpy as np
import pytest

from geneticwing.wind import CompositeWind, DrydenTurbulence, RidgeLift, Thermal, UniformWind, dryden_scales


def test_uniform_wind_from_southwest_blows_northeast():
    w = UniformWind.from_direction(5.2, 225.0)(np.zeros((4, 3)))
    np.testing.assert_allclose(w[0], [5.2 / np.sqrt(2), 5.2 / np.sqrt(2), 0.0])
    assert w.shape == (4, 3)


def test_fields_add_up():
    total = UniformWind((1.0, 0.0, 0.0)) + UniformWind((0.0, 2.0, 0.0)) + UniformWind((0.0, 0.0, 3.0))
    assert isinstance(total, CompositeWind) and len(total.components) == 3
    np.testing.assert_allclose(total(np.zeros(3)), [1.0, 2.0, 3.0])


def test_thermal_peaks_at_its_core_and_vanishes_outside_the_layer():
    th = Thermal(center=(0.0, 0.0), w_star=2.5, z_i=1200.0)
    z = 400.0
    core = th(np.array([0.0, 0.0, z]))[2]
    assert core == pytest.approx(th.peak_updraft(z))
    assert th(np.array([3 * th.outer_radius(z), 0.0, z]))[2] < 1e-3 * core
    assert th(np.array([0.0, 0.0, 1300.0]))[2] == 0.0
    np.testing.assert_allclose(th(np.array([0.0, 0.0, z]))[:2], 0.0)


def test_thermal_disk_mean_equals_allen_mean_updraft():
    th = Thermal(center=(0.0, 0.0), w_star=2.0, z_i=1000.0)
    z = 300.0
    radius = float(th.outer_radius(z))
    r = np.linspace(0, radius, 4001)
    w = th(np.stack([r, np.zeros_like(r), np.full_like(r, z)], axis=-1))[:, 2]
    disk_mean = np.trapezoid(w * 2 * np.pi * r, r) / (np.pi * radius**2)
    assert disk_mean == pytest.approx(th.mean_updraft(z), rel=1e-3)


def test_thermal_drifts_with_the_wind():
    th = Thermal(center=(0.0, 0.0), drift=(4.0, 0.0))
    z = 300.0
    assert th(np.array([40.0, 0.0, z]), 10.0)[2] == pytest.approx(th.peak_updraft(z))


def test_ridge_lifts_on_the_windward_face_and_sinks_in_the_lee():
    # Crest along north, wind blowing east: windward is west (x < 0).
    ridge = RidgeLift(crest=(0.0, 0.0), axis_heading_deg=0.0, radius=50.0, wind_speed=8.0)
    windward = ridge(np.array([-40.0, 0.0, 40.0]))
    lee = ridge(np.array([40.0, 0.0, 40.0]))
    assert windward[2] > 0 and lee[2] < 0
    assert windward[2] == pytest.approx(-lee[2])
    np.testing.assert_allclose(ridge(np.array([0.0, 0.0, 10.0])), 0.0)  # inside the ridge
    np.testing.assert_allclose(ridge(np.array([-5000.0, 0.0, 200.0])), [8.0, 0.0, 0.0], atol=1e-2)
    top = ridge(np.array([0.0, 0.0, 50.0]))  # crest top: flow speeds up to 2U
    np.testing.assert_allclose(top, [16.0, 0.0, 0.0], atol=1e-9)


def test_dryden_intensities_and_reproducibility():
    lengths, sigmas = dryden_scales(altitude=100.0, w20=6.0)
    assert lengths[2] == pytest.approx(100.0)
    assert sigmas[2] == pytest.approx(0.6) and sigmas[0] > sigmas[2]
    a = DrydenTurbulence(duration=200.0, airspeed=12.5, altitude=100.0, w20=6.0, heading_deg=90.0, seed=3)
    b = DrydenTurbulence(duration=200.0, airspeed=12.5, altitude=100.0, w20=6.0, heading_deg=90.0, seed=3)
    t = np.linspace(0, 200, 5000)
    gust = a(np.zeros((t.size, 3)), t)
    np.testing.assert_allclose(gust, b(np.zeros((t.size, 3)), t))
    np.testing.assert_allclose(gust.std(axis=0), [sigmas[0], sigmas[1], sigmas[2]], rtol=0.05)
