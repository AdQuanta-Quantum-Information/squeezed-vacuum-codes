import numpy as np
import pytest

# Use project-standard path setup
try:
    from tests._bootstrap import add_root_to_path
except ModuleNotFoundError:
    from _bootstrap import add_root_to_path
add_root_to_path()

from src.utils.inverse_lookup import MonotonicDirection, MonotonicInverseLookup


def test_inverse_lookup_from_callable_increasing() -> None:
    lookup = MonotonicInverseLookup.from_function(
        lambda x: x * x,
        x_points=np.arange(0.0, 2.01, 0.01),
        direction=MonotonicDirection.INCREASING,
    )

    x = lookup.x_from_y(2.25)
    assert np.isclose(x, 1.5, atol=2e-2)


def test_inverse_lookup_from_callable_decreasing() -> None:
    lookup = MonotonicInverseLookup.from_function(
        lambda x: 9.0 - 3.0 * x,
        x_points=np.arange(0.0, 4.0, 1.0),
        direction=MonotonicDirection.DECREASING,
    )

    x = lookup.x_from_y(2.5)
    assert np.isclose(x, (9.0 - 2.5) / 3.0, atol=1e-12)


def test_inverse_lookup_out_of_range_without_clamp() -> None:
    class IncreasingDoubleLookup(MonotonicInverseLookup[float, float]):
        @classmethod
        def func(cls, x: float) -> float:
            return 2.0 * x

    lookup = IncreasingDoubleLookup(
        x_points=[0.0, 1.0],
        direction=MonotonicDirection.INCREASING,
    )

    # Lazy table expansion should reach this target and not raise.
    x = lookup.x_from_y(7.0, clamp=False)
    assert np.isclose(x, 3.5, atol=1e-12)


def test_inverse_lookup_expands_until_bracket_found() -> None:
    class IncreasingIdentityLookup(MonotonicInverseLookup[float, float]):
        @classmethod
        def func(cls, x: float) -> float:
            return x

    lookup = IncreasingIdentityLookup(
        x_points=[0.0, 1.0],
        direction=MonotonicDirection.INCREASING,
    )

    x = lookup.x_from_y(5.5, clamp=False)
    assert np.isclose(x, 5.5, atol=1e-12)



if __name__ == "__main__":
    test_inverse_lookup_from_callable_increasing()
    test_inverse_lookup_from_callable_decreasing()
    test_inverse_lookup_out_of_range_without_clamp()
    test_inverse_lookup_expands_until_bracket_found()
    print("\n🎉All inverse_lookup tests passed")