import numpy as np
import pytest

from geneticwing.benchmarks import SCENARIOS
from geneticwing.evolution import CMAES, RealGA, ZohdiGA
from geneticwing.planning.local import single_start
from geneticwing.policies import N_PARAMS, lift_anchors, simulate_law
from geneticwing.soaring import INFEASIBLE_OFFSET, SoaringProblem
from geneticwing.spline import ClampedBSpline


@pytest.fixture(scope="module")
def s1():
    return SoaringProblem(SCENARIOS["S1"](0))


def test_spline_initial_control_points_impose_the_initial_state():
    spline = ClampedBSpline(12, degree=5)
    p0, v0, a0 = np.array([1.0, 2.0, 3.0]), np.array([9.0, -1.0, 0.5]), np.array([0.2, 0.0, -0.1])
    first = spline.initial_control_points(p0, v0, a0, duration=30.0)
    cp = np.vstack([first, np.random.default_rng(0).normal(size=(9, 3))])
    for order, target in enumerate((p0, v0, a0)):
        np.testing.assert_allclose(spline.evaluate(cp, 30.0, np.array([0.0]), order)[0], target, atol=1e-9)


def test_spline_reproduces_straight_lines_on_greville_points():
    spline = ClampedBSpline(10, degree=5)
    v = np.array([3.0, 4.0, -0.5])
    cp = spline.greville[:, None] * 20.0 * v
    tau = np.linspace(0, 1, 15)
    np.testing.assert_allclose(spline.evaluate(cp, 20.0, tau, 1), np.tile(v, (15, 1)), atol=1e-9)


def test_straight_genome_decodes_to_the_initial_velocity(s1):
    s = s1.flat(s1.straight_genome())
    np.testing.assert_allclose(s.velocity[0], np.tile(s1.v0, (s1.t.size, 1)), atol=1e-6)
    assert s.airspeed[0, 0] == pytest.approx(s1.scenario.airspeed)  # later, the thermal changes the air


def test_constant_turn_genes_draw_a_circle(s1):
    genes = s1.to_physical(s1.straight_genome())
    genes[0::3] = np.radians(20.0)  # constant turn rate, deg/s
    s = s1.flat(s1.to_normalized(genes))
    late = s.bank[0, s1.t > 10.0]
    assert np.all(late > 0) or np.all(late < 0)  # one turning direction throughout
    assert np.ptp(np.degrees(late)) < 8.0  # nearly constant bank, up to the end of the horizon


def test_cost_scale_and_feasibility_ranking(s1):
    ev = s1.evaluate(np.vstack([s1.straight_genome(), np.full(s1.dim, 0.5)]))
    assert ev.cost.shape == (2,) and ev.constraints.shape == (2, len(s1.CONSTRAINTS))
    # Leaving the loiter area is infeasible, and ranks above (worse than) any feasible cost.
    assert not ev.feasible[0] and ev.fitness[0] > INFEASIBLE_OFFSET
    # Gliding straight at 9 m/s with the motor holding altitude costs about one cruise unit.
    assert 0.8 < ev.cost[0] < 1.5


@pytest.mark.parametrize("optimizer", [RealGA(), ZohdiGA(), CMAES()])
def test_optimizers_respect_the_budget_and_minimize(optimizer):
    def sphere(x):
        return np.sum((x - 0.3) ** 2, axis=1)

    result = optimizer.minimize(sphere, 5, 3000, np.random.default_rng(0))
    assert result.evaluations <= 3000
    assert result.f < 0.05
    assert [f for _, f in result.history] == sorted((f for _, f in result.history), reverse=True)


def test_seeds_enter_the_first_population():
    target = np.full(4, 0.123)
    result = RealGA(population=20).minimize(lambda x: np.sum((x - target) ** 2, axis=1), 4, 20, np.random.default_rng(0), target)
    assert result.f == 0.0


def test_local_nlp_respects_its_budget_and_improves(s1):
    result = single_start(s1, 3000)
    assert result.evaluations <= 3000
    straight = s1.evaluate(s1.straight_genome()[None]).fitness[0]
    assert result.f <= straight


def test_reactive_law_flies_and_is_scored(s1):
    assert len(lift_anchors(s1)) == 1
    theta = np.random.default_rng(0).uniform(size=(6, N_PARAMS))
    ev = simulate_law(s1, theta)
    assert ev.cost.shape == (6,) and np.all(np.isfinite(ev.cost))
    np.testing.assert_allclose(ev.state.position[:, 0], np.tile(s1.scenario.start, (6, 1)))
