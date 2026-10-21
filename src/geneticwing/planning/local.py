"""Local NLP baselines on the soaring genome: single start and multi-start.

The problem is solved in the same normalized gene space as the evolutionary
optimizers, with SciPy's SLSQP: objective = normalized net energy,
constraints = the KS-aggregated hinge functions (feasible when <= 0).
Gradients come from forward finite differences, evaluated as ONE batch of
``dim + 1`` genomes, which is what makes SLSQP affordable here. Every
genome evaluated counts against the same budget as the evolutionary methods.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from geneticwing.evolution import OptimizationResult
from geneticwing.soaring import INFEASIBLE_OFFSET, SoaringProblem

_FD_STEP = 1e-4


class _BudgetExhausted(Exception):
    pass


class _BatchedModel:
    """Values at a point cost 1 evaluation; gradients add one batch of ``dim`` more.

    SLSQP's line search asks for values only, so gradients are computed
    lazily, the first time they are requested at a point.
    """

    def __init__(self, problem: SoaringProblem, budget: int):
        self.problem, self.budget = problem, budget
        self.evaluations = 0
        self._key, self._values, self._gradients = None, None, None
        self.best_x, self.best_fitness = None, np.inf

    def _spend(self, n: int) -> None:
        if self.evaluations + n > self.budget:
            raise _BudgetExhausted
        self.evaluations += n

    def _point(self, x: np.ndarray):
        key = x.tobytes()
        if key != self._key:
            self._spend(1)
            ev = self.problem.evaluate(x[None])
            if ev.fitness[0] < self.best_fitness:
                self.best_fitness, self.best_x = float(ev.fitness[0]), x.copy()
            self._key, self._values, self._gradients = key, (ev.cost[0], ev.constraints[0]), None
        return self._values

    def _grad(self, x: np.ndarray):
        f0, g0 = self._point(x)
        if self._gradients is None:
            self._spend(x.size)
            steps = np.where(x + _FD_STEP <= 1.0, _FD_STEP, -_FD_STEP)
            ev = self.problem.evaluate(x + np.diag(steps))
            self._gradients = ((ev.cost - f0) / steps, ((ev.constraints - g0) / steps[:, None]).T)
        return self._gradients

    def fun(self, x):
        return self._point(x)[0]

    def grad(self, x):
        return self._grad(x)[0]

    def cons(self, x):
        return -self._point(x)[1]

    def cons_jac(self, x):
        return -self._grad(x)[1]


def solve_local(
    problem: SoaringProblem, x0: np.ndarray, budget: int, max_iterations: int = 500
) -> tuple[OptimizationResult, int]:
    """One SLSQP run from ``x0``. Returns the best genome seen and the evaluations spent."""
    model = _BatchedModel(problem, budget)
    try:
        minimize(
            model.fun,
            np.clip(x0, 0.0, 1.0),
            jac=model.grad,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * problem.dim,
            constraints=[{"type": "ineq", "fun": model.cons, "jac": model.cons_jac}],
            options={"maxiter": max_iterations, "ftol": 1e-7},
        )
    except _BudgetExhausted:
        pass
    result = OptimizationResult(model.best_x, model.best_fitness, model.evaluations)
    return result, model.evaluations


def multistart(
    problem: SoaringProblem, budget: int, rng: np.random.Generator, seeds: np.ndarray | None = None
) -> OptimizationResult:
    """SLSQP restarted from the seeds, then from random genomes, until the budget is spent."""
    starts = [] if seeds is None else list(np.atleast_2d(seeds))
    spent, best, history = 0, None, []
    per_run = max(40 * (problem.dim + 1), budget // 4)
    while budget - spent > 2 * (problem.dim + 1):
        x0 = starts.pop(0) if starts else rng.uniform(size=problem.dim)
        run, used = solve_local(problem, x0, min(per_run, budget - spent))
        spent += used
        if best is None or run.f < best.f:
            best = run
        history.append((spent, best.f))
    best.evaluations, best.history = spent, history
    return best


def single_start(problem: SoaringProblem, budget: int, rng=None, seeds=None) -> OptimizationResult:
    """The ARPL-style baseline: one local solve from the straight-glide genome."""
    x0 = problem.straight_genome() if seeds is None else np.atleast_2d(seeds)[0]
    run, used = solve_local(problem, x0, budget)
    run.history = [(used, run.f)]
    return run


def is_feasible(fitness: float) -> bool:
    return fitness < INFEASIBLE_OFFSET
