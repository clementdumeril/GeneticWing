import numpy as np
import pytest

from geneticwing.bernstein import PiecewiseBernstein, basis, derivative_control_points, evaluate


def test_basis_is_a_partition_of_unity():
    tau = np.linspace(0, 1, 50)
    np.testing.assert_allclose(basis(7, tau).sum(axis=1), 1.0)


def test_curve_interpolates_its_end_control_points():
    cp = np.random.default_rng(0).normal(size=(6, 3))
    p = evaluate(cp, 2.0, np.array([0.0, 1.0]))
    np.testing.assert_allclose(p, cp[[0, -1]])


@pytest.mark.parametrize("order", [1, 2, 3])
def test_derivatives_match_finite_differences(order):
    cp = np.random.default_rng(1).normal(size=(8, 3))
    duration = 3.0
    tau = np.linspace(0.1, 0.9, 9)
    h = 1e-4
    lower = evaluate(cp, duration, tau - h / duration, order - 1)
    upper = evaluate(cp, duration, tau + h / duration, order - 1)
    np.testing.assert_allclose(evaluate(cp, duration, tau, order), (upper - lower) / (2 * h), atol=1e-5)


def test_evenly_spaced_control_points_give_constant_velocity():
    start, end, duration = np.zeros(3), np.array([10.0, -4.0, 2.0]), 5.0
    cp = start + np.linspace(0, 1, 6)[:, None] * (end - start)
    tau = np.linspace(0, 1, 11)
    np.testing.assert_allclose(evaluate(cp, duration, tau, 1), np.tile((end - start) / duration, (11, 1)))
    np.testing.assert_allclose(evaluate(cp, duration, tau, 2), 0.0, atol=1e-12)


def test_batch_evaluation_matches_individual_evaluation():
    rng = np.random.default_rng(2)
    cps, durations = rng.normal(size=(4, 6, 3)), rng.uniform(1, 3, size=4)
    tau = np.linspace(0, 1, 7)
    batch = evaluate(cps, durations, tau, 2)
    for i in range(4):
        np.testing.assert_allclose(batch[i], evaluate(cps[i], durations[i], tau, 2))


def test_differentiating_past_the_degree_gives_zero():
    cp = np.ones((3, 2))
    assert np.all(derivative_control_points(cp, 1.0, order=4) == 0.0)


def test_piecewise_sampling_and_continuity():
    line = np.linspace(0, 1, 5)[:, None] * np.array([[4.0, 0.0, 0.0]])
    chain = PiecewiseBernstein(np.stack([line, line + [4.0, 0.0, 0.0]]), np.array([2.0, 2.0]))
    t, (p, v) = chain.sample(11, max_order=1)
    assert t.shape == (21,) and chain.total_duration == 4.0
    np.testing.assert_allclose(p[-1], [8.0, 0.0, 0.0])
    np.testing.assert_allclose(v, np.tile([2.0, 0.0, 0.0], (21, 1)))
    np.testing.assert_allclose(chain.continuity_residual(3), 0.0, atol=1e-12)
