from dataclasses import replace

import numpy as np

from geneticwing.benchmarks import s1_family
from geneticwing.closedloop import (
    LoopConfig,
    Plan,
    Shifted,
    _remainder,
    _select,
    belief_scenario,
    exact_belief,
    fly_closed_loop,
    sample_belief,
    turbulent_air,
)
from geneticwing.soaring import SoaringProblem
from geneticwing.strip import sbxc_wing


def _situation():
    return replace(s1_family(600)[0], strip_wing=sbxc_wing())


class StraightPlanner:
    """Plans the straight glide from wherever it is called: cheap, and enough to exercise the loop."""

    def __init__(self):
        self.calls = []

    def __call__(self, scenario, rng, start=None):
        self.calls.append((scenario, start))
        problem = SoaringProblem(scenario)
        ev = problem.evaluate(problem.straight_genome()[None], keep_state=True)
        return Plan(_select(ev.state, 0), float(ev.cost[0]), bool(ev.feasible[0]))


def test_zero_errors_give_the_exact_belief():
    sit = _situation()
    assert sample_belief(sit, np.random.default_rng(0)) == exact_belief(sit)
    wrong = sample_belief(sit, np.random.default_rng(0), 15.0, 0.2, 0.5)
    assert wrong.center != exact_belief(sit).center


def test_exact_belief_scenario_is_the_situation_seen_later():
    sit = _situation()
    p, v = np.array([30.0, -40.0, 260.0]), np.array([8.0, 3.0, -0.5])
    sc = belief_scenario(sit, exact_belief(sit), 12.0, p, v, np.zeros(3), horizon=48.0)
    probe = np.array([[10.0, 5.0, 250.0], [-20.0, 40.0, 240.0]])
    np.testing.assert_allclose(sc.wind(probe, 3.0), Shifted(sit.wind, 12.0)(probe, 3.0), atol=1e-12)
    assert sc.horizon == 48.0 and sc.start == tuple(p)
    problem = SoaringProblem(sc)
    np.testing.assert_allclose(problem.v0, v)  # in flight the plan starts from the state reached


def test_remainder_continues_the_plan_from_where_it_is():
    sit = _situation()
    plan = StraightPlanner()(sit, None)
    t, p = _remainder(plan, 10.0, 60.0)
    assert t[0] == 0.0 and t[-1] == 60.0 and np.all(np.diff(t) > 0)
    ref = plan.reference
    np.testing.assert_allclose(p[0], [np.interp(10.0, ref.t, ref.position[:, i]) for i in range(3)])


def test_closed_loop_flies_the_whole_duration_and_replans_on_time():
    sit = _situation()
    planner = StraightPlanner()
    flight = fly_closed_loop(sit, sit.wind, exact_belief(sit), planner, LoopConfig(duration=20.0, replan_every=5.0))
    t = flight.state.t[0]
    assert abs(t[-1] - 20.0) < 0.05 and np.all(np.diff(t) > 0)
    assert len(planner.calls) == 4
    horizons = [sc.horizon for sc, _ in planner.calls]
    np.testing.assert_allclose(horizons, [20.0, 15.0, 10.0, 5.0], atol=0.05)  # aims at the flight's end
    assert planner.calls[0][1] is None and planner.calls[1][1].previous is flight.plans[0]
    assert np.isfinite(flight.evaluation.cost[0])


def test_turbulence_moves_the_flight_off_the_calm_one():
    sit = _situation()
    loop = LoopConfig(duration=15.0, replan_every=None)
    calm = fly_closed_loop(sit, sit.wind, exact_belief(sit), StraightPlanner(), loop)
    rough = fly_closed_loop(sit, turbulent_air(sit, 7.5, 15.0, seed=1), exact_belief(sit), StraightPlanner(), loop)
    assert len(calm.plans) == len(rough.plans) == 1
    assert np.max(np.abs(calm.state.position[0, -1] - rough.state.position[0, -1])) > 0.1
    assert np.std(np.diff(rough.state.wind[0, :, 2])) > np.std(np.diff(calm.state.wind[0, :, 2]))
