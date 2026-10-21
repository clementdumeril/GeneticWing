"""Evolutionary optimizers over the unit box, with a common interface.

All optimizers minimize a batched objective ``f(X) -> (P,)`` over
``[0, 1]^dim`` and stop after a fixed number of evaluations, so methods are
compared at equal budget. Optional ``seeds`` are injected into the first
population (every method receives the same ones).

* :class:`RealGA` - real-coded genetic algorithm: binary tournament, SBX
  crossover and polynomial mutation (Deb and Agrawal), elitism.
* :class:`ZohdiGA` - the scheme taught in T. Zohdi's course on genetic
  algorithms: rank the strings, keep the best parents, breed them pairwise by
  random convex combinations, refill the rest with random strings. (Our
  reading of the course; check the details against the course notes.)
* :class:`CMAES` - covariance matrix adaptation (Hansen's tutorial, 2016),
  with box constraints handled by clipping the samples before evaluation and
  updating with the clipped points.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

Objective = Callable[[np.ndarray], np.ndarray]


@dataclass
class OptimizationResult:
    x: np.ndarray
    f: float
    evaluations: int
    history: list[tuple[int, float]] = field(default_factory=list)  # (evaluations, best f so far)


class _Tracker:
    def __init__(self, objective: Objective, budget: int):
        self.objective, self.budget = objective, budget
        self.evaluations = 0
        self.best_x, self.best_f = None, np.inf
        self.history: list[tuple[int, float]] = []

    @property
    def remaining(self) -> int:
        return self.budget - self.evaluations

    def __call__(self, X: np.ndarray) -> np.ndarray:
        f = np.asarray(self.objective(X), dtype=float)
        self.evaluations += len(X)
        i = int(np.argmin(f))
        if f[i] < self.best_f:
            self.best_f, self.best_x = float(f[i]), X[i].copy()
        self.history.append((self.evaluations, self.best_f))
        return f

    def result(self) -> OptimizationResult:
        return OptimizationResult(self.best_x, self.best_f, self.evaluations, self.history)


def _initial_population(n: int, dim: int, rng: np.random.Generator, seeds: np.ndarray | None) -> np.ndarray:
    pop = rng.uniform(size=(n, dim))
    if seeds is not None:
        seeds = np.atleast_2d(seeds)[:n]
        pop[: len(seeds)] = seeds
    return pop


@dataclass
class RealGA:
    population: int = 100
    elite: int = 2
    crossover_rate: float = 0.9
    eta_crossover: float = 15.0
    eta_mutation: float = 20.0
    mutation_rate: float | None = None  # default 1 / dim

    def minimize(self, objective: Objective, dim: int, budget: int, rng: np.random.Generator, seeds=None):
        track = _Tracker(objective, budget)
        pm = self.mutation_rate or 1.0 / dim
        pop = _initial_population(self.population, dim, rng, seeds)
        fit = track(pop)
        while track.remaining >= self.population - self.elite:
            n_children = self.population - self.elite
            # Binary tournament selection of parents.
            a, b = rng.integers(0, self.population, size=(2, n_children + 1))
            parents = pop[np.where(fit[a] <= fit[b], a, b)]
            p1, p2 = parents[:-1], np.roll(parents, 1, axis=0)[:-1]
            children = self._sbx(p1, p2, rng)
            children = self._mutate(children, pm, rng)
            elite = np.argsort(fit)[: self.elite]
            child_fit = track(children)
            pop = np.concatenate([pop[elite], children])
            fit = np.concatenate([fit[elite], child_fit])
        return track.result()

    def _sbx(self, p1, p2, rng):
        u = rng.uniform(size=p1.shape)
        eta = self.eta_crossover
        beta = np.where(u <= 0.5, (2 * u) ** (1 / (eta + 1)), (1 / (2 * (1 - u))) ** (1 / (eta + 1)))
        child = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
        # Each gene comes from the "plus" or "minus" child at random; crossover applies per pair.
        flip = rng.uniform(size=p1.shape) < 0.5
        child = np.where(flip, 0.5 * ((1 - beta) * p1 + (1 + beta) * p2), child)
        apply = rng.uniform(size=(len(p1), 1)) < self.crossover_rate
        return np.clip(np.where(apply, child, p1), 0.0, 1.0)

    def _mutate(self, x, rate, rng):
        u = rng.uniform(size=x.shape)
        eta = self.eta_mutation
        delta = np.where(u < 0.5, (2 * u) ** (1 / (eta + 1)) - 1, 1 - (2 * (1 - u)) ** (1 / (eta + 1)))
        mask = rng.uniform(size=x.shape) < rate
        return np.clip(x + mask * delta, 0.0, 1.0)


@dataclass
class ZohdiGA:
    population: int = 100
    parents: int = 10  # kept each generation, bred pairwise

    def minimize(self, objective: Objective, dim: int, budget: int, rng: np.random.Generator, seeds=None):
        if self.parents % 2:
            raise ValueError("parents must be even")
        track = _Tracker(objective, budget)
        pop = _initial_population(self.population, dim, rng, seeds)
        fit = track(pop)
        n_new = self.population - self.parents
        while track.remaining >= n_new:
            order = np.argsort(fit)
            parents, parent_fit = pop[order[: self.parents]], fit[order[: self.parents]]
            mothers, fathers = parents[0::2], parents[1::2]
            phi = rng.uniform(size=mothers.shape)
            psi = rng.uniform(size=mothers.shape)
            offspring = np.concatenate([phi * mothers + (1 - phi) * fathers, psi * mothers + (1 - psi) * fathers])
            immigrants = rng.uniform(size=(n_new - len(offspring), dim))
            fresh = np.concatenate([offspring, immigrants])
            pop = np.concatenate([parents, fresh])
            fit = np.concatenate([parent_fit, track(fresh)])
        return track.result()


@dataclass
class CMAES:
    population: int | None = None  # default 4 + 3 ln(dim)
    sigma0: float = 0.3

    def minimize(self, objective: Objective, dim: int, budget: int, rng: np.random.Generator, seeds=None):
        n = dim
        lam = self.population or 4 + int(3 * np.log(n))
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = 1.0 / np.sum(w**2)
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0.0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))

        track = _Tracker(objective, budget)
        mean = np.full(n, 0.5)
        if seeds is not None:
            seeds = np.atleast_2d(seeds)
            f_seeds = track(seeds)
            mean = seeds[int(np.argmin(f_seeds))].copy()
        sigma = self.sigma0
        cov = np.eye(n)
        ps, pc = np.zeros(n), np.zeros(n)
        generation = 0
        while track.remaining >= lam:
            eigval, basis = np.linalg.eigh(cov)
            scale = np.sqrt(np.maximum(eigval, 1e-20))
            z = rng.standard_normal((lam, n))
            x = np.clip(mean + sigma * (z * scale) @ basis.T, 0.0, 1.0)
            f = track(x)
            best = np.argsort(f)[:mu]
            y = (x[best] - mean) / sigma
            y_w = w @ y
            mean = mean + sigma * y_w
            inv_sqrt = basis @ np.diag(1.0 / scale) @ basis.T
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * inv_sqrt @ y_w
            generation += 1
            hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * generation)) / chi_n < 1.4 + 2 / (n + 1)
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * y_w
            rank_mu = (w[:, None] * y).T @ y
            cov = (1 - c1 - cmu) * cov + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * cov) + cmu * rank_mu
            cov = 0.5 * (cov + cov.T)
            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chi_n - 1))
            sigma = min(sigma, 1.0)
        return track.result()


OPTIMIZERS = {"GA": RealGA, "ZohdiGA": ZohdiGA, "CMA-ES": CMAES}
