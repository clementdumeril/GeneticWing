"""Common interface of wind fields: ``W(p, t)`` in the ENU frame."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class WindField:
    """A wind field maps positions ``(..., 3)`` and times ``(...)`` to air velocity ``(..., 3)``.

    Fields add up with ``+``, so a scenario is written as
    ``UniformWind(...) + Thermal(...) + DrydenTurbulence(...)``.
    """

    def __call__(self, position: np.ndarray, t: np.ndarray | float = 0.0) -> np.ndarray:
        raise NotImplementedError

    def __add__(self, other: WindField) -> WindField:
        return CompositeWind((*_parts(self), *_parts(other)))


def _parts(field: WindField) -> tuple[WindField, ...]:
    return field.components if isinstance(field, CompositeWind) else (field,)


@dataclass(frozen=True)
class CompositeWind(WindField):
    components: tuple[WindField, ...]

    def __call__(self, position, t=0.0):
        return sum(component(position, t) for component in self.components)


@dataclass(frozen=True)
class UniformWind(WindField):
    """Constant wind vector (m/s), the model of arXiv 2605.15619."""

    velocity: tuple[float, float, float]

    @classmethod
    def from_direction(cls, speed: float, from_heading_deg: float, vertical: float = 0.0):
        """Wind of ``speed`` blowing FROM a compass heading (0 = north, 90 = east)."""
        to = np.radians(from_heading_deg + 180.0)
        return cls((speed * np.sin(to), speed * np.cos(to), vertical))

    def __call__(self, position, t=0.0):
        position = np.asarray(position, dtype=float)
        return np.broadcast_to(np.asarray(self.velocity, dtype=float), position.shape).copy()
