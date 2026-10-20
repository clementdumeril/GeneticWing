import numpy as np
import pytest

from geneticwing import STRATOSURFER_APPROX as AC
from geneticwing.planning import GlideProblem, plan_glide
from geneticwing.simulation import simulate_tracking
from geneticwing.wind import DrydenTurbulence, UniformWind


@pytest.fixture(scope="module")
def problem():
    return GlideProblem(AC, UniformWind((5.2, 0.0, 0.0)), (0.0, 0.0, 120.0), (400.0, 0.0), 12.5)


@pytest.fixture(scope="module")
def plan(problem):
    return plan_glide(problem)


def test_s0_plan_is_the_steady_glide(problem, plan):
    assert plan.success
    s = plan.sample()
    _, sink, _ = AC.steady_glide(12.5)
    np.testing.assert_allclose(s.position[-1, :2], [400.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(s.airspeed, 12.5, atol=1e-6)
    np.testing.assert_allclose(s.sink_rate_air, sink, atol=1e-6)
    np.testing.assert_allclose(s.thrust_required, 0.0, atol=1e-6)


def test_s0_planner_recovers_from_a_perturbed_guess(problem, plan):
    rng = np.random.default_rng(0)
    cp = plan.control_points.copy()
    cp[3:-1] += rng.normal(0.0, 10.0, cp[3:-1].shape)
    perturbed = plan_glide(problem, initial=(cp, 1.1 * plan.duration))
    assert perturbed.success
    s = perturbed.sample()
    assert np.all(np.abs(s.airspeed - 12.5) <= problem.airspeed_tol + 1e-6)
    assert np.all(np.abs(s.thrust_required) <= problem.thrust_tol * AC.weight + 1e-6)
    assert abs(perturbed.control_points[-1, 2] - plan.control_points[-1, 2]) < 10.0


def test_crosswind_crab_keeps_the_ground_track(problem):
    crosswind = GlideProblem(AC, UniformWind((0.0, 4.0, 0.0)), (0.0, 0.0, 120.0), (400.0, 0.0), 12.5)
    v0 = crosswind.initial_velocity()
    assert v0[1] == pytest.approx(0.0, abs=1e-12)
    assert np.linalg.norm(v0 - [0.0, 4.0, 0.0]) == pytest.approx(12.5)


def test_closed_loop_flight_follows_the_plan(problem, plan):
    reference = plan.sample(2000)
    calm = simulate_tracking(AC, reference, problem.wind, reference.position[0], reference.velocity[0])
    np.testing.assert_allclose(calm.position[-1], reference.position[-1], atol=0.5)
    np.testing.assert_allclose(calm.airspeed, 12.5, atol=1e-3)

    gusts = DrydenTurbulence(duration=60.0, airspeed=12.5, altitude=100.0, w20=5.2, heading_deg=90.0)
    rough = simulate_tracking(AC, reference, problem.wind + gusts, reference.position[0], reference.velocity[0])
    assert np.abs(rough.position[-1, 1]) < 2.0
    assert np.abs(rough.position[-1, 2] - reference.position[-1, 2]) < 10.0
    assert np.all(rough.cl < AC.cl_max)
