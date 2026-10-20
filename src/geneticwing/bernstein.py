"""Bernstein polynomials used as the flat-output (position) parameterization.

A segment of degree ``n`` and duration ``T`` is defined by ``n + 1`` control
points ``c_i``::

    p(t) = sum_i c_i * B_{i,n}(t / T),   B_{i,n}(tau) = C(n, i) tau^i (1 - tau)^(n - i)

Its derivative is again a Bernstein polynomial, of degree ``n - 1``, with
control points ``n / T * (c_{i+1} - c_i)``. Two consequences matter here:

* the first ``k + 1`` control points alone fix position and the first ``k``
  derivatives at ``t = 0``, so initial conditions can be imposed by
  construction;
* by the convex-hull property, bounds on the (derivative) control points bound
  the curve (and its derivatives) over the whole segment.

All functions accept an optional leading batch dimension on the control points
and durations, so a population of trajectories is evaluated in one call.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import comb


def basis(degree: int, tau: np.ndarray) -> np.ndarray:
    """Bernstein basis of ``degree`` at normalized times ``tau`` -> ``(N, degree + 1)``."""
    tau = np.asarray(tau, dtype=float)[:, None]
    i = np.arange(degree + 1)
    return comb(degree, i) * tau**i * (1.0 - tau) ** (degree - i)


def derivative_control_points(
    control_points: np.ndarray, duration: np.ndarray | float, order: int = 1
) -> np.ndarray:
    """Control points of the ``order``-th time derivative.

    ``control_points`` is ``(..., n + 1, d)`` and ``duration`` broadcasts
    against its leading dimensions. Differentiating past the degree returns a
    single zero control point.
    """
    cp = np.asarray(control_points, dtype=float)
    scale = np.asarray(duration, dtype=float)[..., None, None]
    for _ in range(order):
        degree = cp.shape[-2] - 1
        if degree == 0:
            return np.zeros_like(cp)
        cp = degree / scale * np.diff(cp, axis=-2)
    return cp


def evaluate(
    control_points: np.ndarray,
    duration: np.ndarray | float,
    tau: np.ndarray,
    order: int = 0,
) -> np.ndarray:
    """Evaluate the ``order``-th derivative at normalized times ``tau`` in [0, 1].

    Returns ``(..., N, d)``. Working in normalized time lets a batch of
    segments with different durations share one basis matrix.
    """
    cp = derivative_control_points(control_points, duration, order)
    return np.einsum("nm,...md->...nd", basis(cp.shape[-2] - 1, tau), cp)


@dataclass(frozen=True)
class PiecewiseBernstein:
    """Chain of Bernstein segments of equal degree.

    ``control_points`` is ``(M, n + 1, d)`` and ``durations`` is ``(M,)``.
    Continuity between segments is not enforced here; see
    :meth:`continuity_residual`.
    """

    control_points: np.ndarray
    durations: np.ndarray

    def __post_init__(self) -> None:
        cp = np.asarray(self.control_points, dtype=float)
        if cp.ndim == 2:
            cp = cp[None]
        durations = np.atleast_1d(np.asarray(self.durations, dtype=float))
        if cp.shape[0] != durations.shape[0]:
            raise ValueError("one duration per segment is required")
        if np.any(durations <= 0):
            raise ValueError("segment durations must be positive")
        object.__setattr__(self, "control_points", cp)
        object.__setattr__(self, "durations", durations)

    @property
    def degree(self) -> int:
        return self.control_points.shape[1] - 1

    @property
    def total_duration(self) -> float:
        return float(self.durations.sum())

    def sample(self, samples_per_segment: int, max_order: int = 3):
        """Sample time and derivatives ``0..max_order`` along the whole chain.

        Returns ``(t, [p, v, a, ...])`` with ``t`` of shape ``(N,)`` and each
        derivative of shape ``(N, d)``. Interior segment joints appear once.
        """
        tau = np.linspace(0.0, 1.0, samples_per_segment)
        starts = np.concatenate([[0.0], np.cumsum(self.durations)[:-1]])
        times, derivs = [], [[] for _ in range(max_order + 1)]
        for j, (cp, duration) in enumerate(zip(self.control_points, self.durations)):
            keep = slice(None) if j == 0 else slice(1, None)
            times.append(starts[j] + tau[keep] * duration)
            for k in range(max_order + 1):
                derivs[k].append(evaluate(cp, duration, tau, k)[keep])
        return np.concatenate(times), [np.concatenate(d) for d in derivs]

    def continuity_residual(self, max_order: int) -> np.ndarray:
        """Jumps of derivatives ``0..max_order`` at each interior joint, ``(M - 1, max_order + 1, d)``."""
        jumps = []
        for k in range(max_order + 1):
            d = derivative_control_points(self.control_points, self.durations, k)
            jumps.append(d[1:, 0] - d[:-1, -1])
        return np.stack(jumps, axis=1)
