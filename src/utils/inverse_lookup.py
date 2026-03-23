from abc import ABC, abstractmethod
from enum import Enum
from typing import Generic, Iterable, SupportsFloat, TypeVar, cast, Callable, Final

from src.utils.prints import ProgressBar

import numpy as np


XType = TypeVar("XType", bound=SupportsFloat)
XType2 = TypeVar("XType2", bound=SupportsFloat)
YType = TypeVar("YType", bound=SupportsFloat)
YType2 = TypeVar("YType2", bound=SupportsFloat)


MONOTONICITY_TOL : Final[float] = 1e-12


class MonotonicDirection(Enum):
    INCREASING = "increasing"
    DECREASING = "decreasing"

class Direction(Enum):
    LEFT = "left"
    RIGHT = "right"


class MonotonicInverseLookup(ABC, Generic[XType, YType]):
    """Inverse lookup table for a monotonic scalar function y=f(x).

    For concrete subclass examples, see tests/test_inverse_lookup.py.
    """
    @classmethod
    def from_function(
        cls: type["MonotonicInverseLookup[XType2, YType2]"],
        f: Callable[[XType2], YType2],
        *,
        x_points: Iterable[SupportsFloat],
        direction: MonotonicDirection = MonotonicDirection.INCREASING,
        progress_bar: bool = False,
    ) -> "MonotonicInverseLookup[XType2, YType2]":
        """Factory method to create a MonotonicInverseLookup subclass from a simple function."""

        class FuncBasedLookup(MonotonicInverseLookup):
            @classmethod
            def func(cls, x: XType2) -> YType2:
                return f(x)

        lookup = cast(
            MonotonicInverseLookup[XType2, YType2],
            FuncBasedLookup(
                x_points=x_points,
                direction=direction,
                progress_bar=progress_bar,
            ),
        )
        return lookup


    def __init__(
        self,
        x_points: Iterable[SupportsFloat],
        direction: MonotonicDirection = MonotonicDirection.INCREASING,
        progress_bar: bool = False,
    ) -> None:
        x_values = np.asarray(list(x_points), dtype=float)
        f = self.__class__.func

        if progress_bar:
            iter_x_values = ProgressBar(x_values, prefix="initial points: ")
        else:
            iter_x_values = x_values

        y_values = np.array(
            [float( f(cast(XType, float(x))) ) for x in iter_x_values], 
            dtype=float
        )
        x_arr, y_arr = self._validated_arrays(x_values, y_values, direction)

        self.x_values = x_arr
        self.y_values = y_arr
        x_diff = np.diff(x_arr)
        self.delta_x = float(np.min(x_diff))
        self.direction = direction

    @classmethod
    @abstractmethod
    def func(cls, x: XType) -> YType:
        """Monotonic function mapping x -> y to be inverted."""
        raise NotImplementedError

    @staticmethod
    def _validated_arrays(
        x_values: np.ndarray,
        y_values: np.ndarray,
        direction: MonotonicDirection,
    ) -> tuple[np.ndarray, np.ndarray]:
        x_arr = np.asarray(x_values, dtype=float)
        y_arr = np.asarray(y_values, dtype=float)

        if x_arr.ndim != 1 or y_arr.ndim != 1:
            raise ValueError("x_values and y_values must be 1D arrays.")
        if x_arr.size != y_arr.size:
            raise ValueError("x_values and y_values must have the same length.")
        if x_arr.size < 2:
            raise ValueError("At least 2 sample points are required.")

        x_diff = np.diff(x_arr)
        if np.any(x_diff <= 0):
            raise ValueError("x_values must be strictly increasing.")

        y_diff = np.diff(y_arr)
        if direction == MonotonicDirection.INCREASING and np.any(y_diff < 0):
            raise ValueError("y_values must be non-decreasing for INCREASING direction.")
        if direction == MonotonicDirection.DECREASING and np.any(y_diff > 0):
            from matplotlib import pyplot as plt
            plt.plot(x_arr, y_arr, marker='o')
            plt.axhline(0.0, color='black', linestyle='--', linewidth=1.0)
            raise ValueError("y_values must be non-increasing for DECREASING direction.")

        return x_arr, y_arr

    @property
    def y_bounds(self) -> tuple[float, float]:
        y_min = float(np.min(self.y_values))
        y_max = float(np.max(self.y_values))
        return y_min, y_max

    def x_from_y(self, y_target: float, *, clamp: bool = True) -> float:
        self._extend_bound_to_reach_target(y_target)
        y_arr, x_arr = self._monotonically_increasing_y_view()
        y = float(y_target)

        if y < y_arr[0]:
            if clamp:
                return float(x_arr[0])
            raise ValueError(f"y_target={y} is below interpolation range ({y_arr[0]}, {y_arr[-1]}).")
        if y > y_arr[-1]:
            if clamp:
                return float(x_arr[-1])
            raise ValueError(f"y_target={y} is above interpolation range ({y_arr[0]}, {y_arr[-1]}).")

        idx = int(np.searchsorted(y_arr, y, side="left"))
        if idx <= 0:
            return float(x_arr[0])
        if idx >= y_arr.size:
            return float(x_arr[-1])

        y0 = y_arr[idx - 1]
        y1 = y_arr[idx]
        x0 = x_arr[idx - 1]
        x1 = x_arr[idx]

        if y1 == y0:
            return float(x1)

        alpha = (y - y0) / (y1 - y0)
        return float(x0 + alpha * (x1 - x0))

    def _extend_bound_to_reach_target(self, y_target: float, max_expand_iters: int = 100000) -> None:
        y = float(y_target)
        for _ in range(max_expand_iters):
            y_min = float(np.min(self.y_values))
            y_max = float(np.max(self.y_values))
            if y_min <= y <= y_max:
                return

            ## Decide expand direction
            expand_left = False
            if self.direction == MonotonicDirection.INCREASING:
                expand_left = y < y_min
            elif self.direction == MonotonicDirection.DECREASING:
                expand_left = y > y_max
            else:
                raise RuntimeError(f"Invalid monotonic direction: {self.direction}")

            if expand_left:
                self._extend_left()
            else:
                self._extend_right()
        else:
            raise RuntimeError(
                f"Failed to bracket y_target={y_target} after {max_expand_iters} expansions."
            )

    def _extend_left(self) -> None:
        x_new = float(self.x_values[0] - self.delta_x)
        y_new = float(self.__class__.func(cast(XType, x_new)))
        self._assert_monotonic_append(y_new, Direction.LEFT)
        self.x_values = np.concatenate(([x_new], self.x_values))
        self.y_values = np.concatenate(([y_new], self.y_values))

    def _extend_right(self) -> None:
        x_new = float(self.x_values[-1] + self.delta_x)
        y_new = float(self.__class__.func(cast(XType, x_new)))
        self._assert_monotonic_append(y_new, Direction.RIGHT)
        self.x_values = np.concatenate((self.x_values, [x_new]))
        self.y_values = np.concatenate((self.y_values, [y_new]))

    def _assert_monotonic_append(self, y_new: float, side: Direction) -> None:
        if self.direction == MonotonicDirection.INCREASING:
            if side == Direction.LEFT and y_new > self.y_values[0] + MONOTONICITY_TOL:
                raise ValueError("Monotonicity violated when extending left (expected non-increasing y).")
            if side == Direction.RIGHT and y_new < self.y_values[-1] - MONOTONICITY_TOL:
                raise ValueError("Monotonicity violated when extending right (expected non-decreasing y).")
        elif self.direction == MonotonicDirection.DECREASING:
            if side == Direction.LEFT and y_new < self.y_values[0] - MONOTONICITY_TOL:
                raise ValueError("Monotonicity violated when extending left (expected non-decreasing y).")
            if side == Direction.RIGHT and y_new > self.y_values[-1] + MONOTONICITY_TOL:
                raise ValueError("Monotonicity violated when extending right (expected non-increasing y).")
        else:
            raise RuntimeError(f"Invalid monotonic direction: {self.direction}")

    def _monotonically_increasing_y_view(self) -> tuple[np.ndarray, np.ndarray]:
        if self.direction == MonotonicDirection.INCREASING:
            return self.y_values, self.x_values
        return self.y_values[::-1], self.x_values[::-1]

