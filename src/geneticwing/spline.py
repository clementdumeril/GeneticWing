"""Clamped uniform B-splines: chains of Bernstein segments with built-in continuity.

Long soaring horizons (several thermal circles) need many polynomial pieces.
A clamped B-spline of degree ``k`` is exactly such a chain: each knot span is
a degree-``k`` polynomial (convertible to Bernstein form), joined with
``C^{k-1}`` continuity by construction, so the convex-hull property and the
flat evaluation of :mod:`geneticwing.bernstein` carry over unchanged.

Time is normalized to ``tau = t / T`` in [0, 1]. The basis matrices depend on
``tau`` only, so a population of control polygons sharing one horizon ``T``
is evaluated with a single matrix product per derivative order.
"""

from __future__ import annotations

from functools import cached_property

import numpy as np
from scipy.interpolate import BSpline


class ClampedBSpline:
    def __init__(self, n_control: int, degree: int = 5):
        if n_control < degree + 1:
            raise ValueError("need at least degree + 1 control points")
        self.n_control = n_control
        self.degree = degree
        interior = np.linspace(0.0, 1.0, n_control - degree + 1)[1:-1]
        self.knots = np.concatenate([np.zeros(degree + 1), interior, np.ones(degree + 1)])
        self._basis = BSpline(self.knots, np.eye(n_control), degree)

    @cached_property
    def greville(self) -> np.ndarray:
        """Greville abscissae: the normalized time each control point is 'attached' to."""
        k = self.degree
        return np.array([self.knots[i + 1 : i + k + 1].mean() for i in range(self.n_control)])

    def basis(self, tau: np.ndarray, order: int = 0) -> np.ndarray:
        """Basis functions (or their ``order``-th tau-derivative) at ``tau`` -> ``(N, n_control)``."""
        spline = self._basis.derivative(order) if order else self._basis
        return spline(np.asarray(tau, dtype=float))

    def evaluate(self, control_points: np.ndarray, duration: float, tau: np.ndarray, order: int = 0) -> np.ndarray:
        """``order``-th time derivative at ``tau`` for control points ``(..., n_control, d)``."""
        return np.einsum("nm,...md->...nd", self.basis(tau, order), control_points) / duration**order

    def initial_control_points(
        self, position: np.ndarray, velocity: np.ndarray, acceleration: np.ndarray, duration: float
    ) -> np.ndarray:
        """First three control points imposing position, velocity and acceleration at ``t = 0``.

        On a clamped spline only ``c_0..c_k`` influence the start, and the
        system is triangular: ``c_0`` sets position, ``c_1`` velocity, ``c_2``
        acceleration.
        """
        b1 = self.basis([0.0], 1)[0] / duration
        b2 = self.basis([0.0], 2)[0] / duration**2
        c0 = np.asarray(position, dtype=float)
        c1 = (np.asarray(velocity, dtype=float) - b1[0] * c0) / b1[1]
        c2 = (np.asarray(acceleration, dtype=float) - b2[0] * c0 - b2[1] * c1) / b2[2]
        return np.stack([c0, c1, c2])
