
__all__ = [
	"binary_search_callable_increasing",
]

# ==================================================================================== #
#|                                    Imports                                         |#
# ==================================================================================== #
import numpy as np
from math import floor
from typing import Callable, Tuple, Optional, Union, Any
from collections import deque

from ..utils.prints import ProgressBar

# =================================================================================== #
#|                                 Module globals                                    |#
# =================================================================================== #
DEFAULT_TOL = 1e-15
DEFAULT_MAX_ITERS = 10_000

# ==================================================================================== #
#|                                Declared functions                                  |#
# ==================================================================================== #


class OptimizationConvergenceError(ValueError):
    """
    Error raised when an optimization or search algorithm fails to converge.
    
    Stores the state achieved thus far in the optimization, allowing inspection
    of partial progress even when convergence criteria are not met.
    
    Attributes:
        message: The error message
        achieved_value: The best value achieved during optimization (e.g., x, y, or both)
        iteration: The iteration count when the error occurred
        metadata: Optional dictionary containing additional optimization state
    """
    
    def __init__(
        self,
        message: str,
        achieved_value: Any = None,
        iteration: Optional[int] = None,
        **metadata
    ):
        """
        Initialize the optimization convergence error.
        
        Args:
            message: Description of the convergence failure
            achieved_value: The value(s) achieved at failure point (x, y, or (x, y) tuple)
            iteration: Number of iterations completed before failure
            **metadata: Additional context (e.g., tolerance, bounds, target)
        """
        super().__init__(message)
        self.achieved_value = achieved_value
        self.iteration = iteration
        self.metadata = metadata
    
    def __str__(self) -> str:
        """Enhanced string representation with optimization details."""
        parts = [super().__str__()]
        
        if self.iteration is not None:
            parts.append(f"Iterations completed: {self.iteration}")
        
        if self.achieved_value is not None:
            parts.append(f"Achieved value: {self.achieved_value}")
        
        if self.metadata:
            for key, value in self.metadata.items():
                parts.append(f"{key}: {value}")
        
        return "\n  ".join(parts)



def binary_search_callable_increasing(
	f: Callable[[Any], float],
	target: float,
	x_bounds: tuple[float, float],
	*,
	integer: bool = False,
	x_tol: float = DEFAULT_TOL,
	y_tol: float = DEFAULT_TOL,
	max_iters: int = DEFAULT_MAX_ITERS,
    progress_bar: bool = True
) -> tuple[float, float]:
    """
    Find the minimal x in [lo, hi] such that f(x) >= target, assuming f is
    monotonic non-decreasing.

    Contract:
    - f: Callable mapping x -> y, monotonic non-decreasing on [lo, hi]
    - target: y-threshold we want to cross from below
    - lo, hi: search bounds such that f(lo) <= target <= f(hi)
    - integer: when True, operate on integers; when False operate on reals;
        when None, inferred from the types of lo and hi (int -> True, else False)
    - tol: tolerance for the continuous (float) mode (default: utils.assertions.TOLERANCE if available, else 1e-9)
    - max_iters: safety cap on iterations
    - return_value: when True, also returns f(x_final)

    Returns:
    - (x, f(x))

    Raises:
    - ValueError if preconditions (bracketing) are violated or invalid args.

    Notes:
    - In float mode, the function returns the current upper bound ("hi") at the
        end of the search, which guarantees f(x_out) >= target up to tolerance.
    - In integer mode, the function returns the smallest integer x with f(x) >= target.
    """
    lo, hi = x_bounds

    ## Validate inputs and assumption that f is increasing on [lo, hi]
    _validate_monotonicity_and_inputs(lo, hi, f, target)
    assert max_iters > 0, "max_iters must be positive"

    # Infer integer mode when not explicitly provided
    if integer is None:
        integer = isinstance(lo, int) and isinstance(hi, int)

    ## Initialize variables:
    iters = 0  # set counter
    y : float = None  #type: ignore  
    mid : float = None  #type: ignore

    # Determine stopping condition based on mode
    if integer:
        x_low = int(lo)
        x_high = int(hi)
        # Integer mode: stop when lo and hi are adjacent
        def x_stop_condition(): 
            return x_low >= x_high 
    else:
        x_low = float(lo)
        x_high = float(hi)
        # Float mode: stop when interval is smaller than tolerance
        def x_stop_condition(): 
            return (x_high - x_low) <= x_tol 

    i = -1  # For progress tracking

    def stop_condition() -> bool:
        if i < 1:
            return False
        
        if x_stop_condition():
             return True
        
        if abs(y - target) < y_tol:
            return True
        
        return False


    if progress_bar:
        iterator = ProgressBar.range(max_iters, prefix="Binary search: ", suffix=" iters")  
    else:
        iterator = range(max_iters)

    for i in iterator:
        if stop_condition():
            break

        if progress_bar:
            assert isinstance(iterator, ProgressBar)
            extra_str = f"\n    x={mid}"+\
                        f"\n    f(x)={y}"
            iterator.append_extra_str(extra_str)
        
        ## current x and y:
        mid = (x_low + x_high) / 2
        if integer:
            mid = int(floor(mid))

        y = f(mid)
        
        ## Update bounds based on f(mid) vs target:
        if y < target:
            x_low = mid + (1 if integer else 0)
        else:
            x_high = mid
        iters += 1

        
    else:
         ## Failure to converge within max_iters
         raise OptimizationConvergenceError(
             f"Binary search did not converge within the maximum number of iterations.",
             achieved_value=(mid, y),
             iteration=max_iters,
             target=target,
             x_bounds=(lo, hi),
             x_tol=x_tol,
             y_tol=y_tol
         )

    # Return upper bound (guarantees f(x_out) >= target)
    return mid, y




# ==================================================================================== #
#|                             Internal helper functions                              |#
# ==================================================================================== #

def _validate_monotonicity_and_inputs(lo: float, hi: float, f: Callable[[float], float], target: float) -> None:
	if lo > hi:
		raise ValueError(f"Expected lo <= hi, got lo={lo}, hi={hi}")
	y_low = f(lo)
	y_high = f(hi)
	if y_low > y_high:
		raise ValueError(
			f"Function is not monotonic non-decreasing on [{lo}, {hi}]: "
			f"f(lo)={y_low} > f(hi)={y_high}"
		)

	if y_low > target:
		raise ValueError(
			f"Invalid bracket: f(lo)={y_low} > target={target}. Need f(lo) <= target."
		)
	if y_high < target:
		raise ValueError(
			f"Invalid bracket: f(hi)={y_high} < target={target}. Need f(hi) >= target."
		)
     
    
def _all_close_in_list(list_:list[float], tol:float=1e-6) -> bool:
    first_val = list_[0]
    for val in list_[1:]:
        if not np.isclose(first_val, val, atol=tol):
            return False
    return True