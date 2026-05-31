from typing import Literal
import numpy as np
import math
import functools

Sign = Literal["+", "-"]


def _ensure_int(n):
    if math.floor(n) != n:
        raise ValueError("Input must be an integer value.")
    return int(n)


@functools.cache
def factorial(n:int) -> int:
    n = _ensure_int(n) 
    return factorial(n-1)*n if n>1 else 1


def _sqrt_factorial_large_n(n):
    # Compute the logarithm of the factorial
    log_factorial = math.lgamma(n + 1)
    
    # Compute the logarithm of the square root of the factorial
    log_sqrt_factorial = log_factorial / 2
    
    # Exponentiate to get the square root of the factorial
    # Overflow to inf is intentional for very large n, suppress the RuntimeWarning
    with np.errstate(over='ignore'):
        sqrt_factorial = np.exp(log_sqrt_factorial)
    
    return sqrt_factorial


def log_sqrt_factorial(n: int) -> float:
    """
    Compute log(sqrt(n!)) = log(n!) / 2
    This function works for arbitrarily large n without overflow.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return 0.0  # log(sqrt(0!)) = log(1) = 0
    
    return math.lgamma(n + 1) / 2


def sqrt_factorial(n:int) -> float:
    if n<15:
        return np.sqrt(factorial(n))
    else:
        return _sqrt_factorial_large_n(n)


def power_computed_in_log_space(base:float, power:float) -> float:
    """
    Compute a^p safely for large p.
    This avoids overflow by working in log space.
    """
    if base == 0:
        return 0.0
    if power == 0:
        return 1.0
    if power < 0:
        raise ValueError("Cannot compute sqrt of negative base raised to power")
    if base < 0:
        raise ValueError("Base must be non-negative")
    
    # Work in log space: a^n = exp(n*log(a))
    return np.exp(power * np.log(base))


def safe_sqrt_factorial_ratio(numerator_n: int, denominator_n: int) -> float:
    """
    Compute sqrt(numerator_n!) / sqrt(denominator_n!) safely for large numbers.
    This avoids overflow by working in log space.
    """
    log_ratio = log_sqrt_factorial(numerator_n) - log_sqrt_factorial(denominator_n)
    return np.exp(log_ratio)


def safe_sqrt_factorial_product_ratio(numerator_ns: list[int], denominator_ns: list[int]) -> float:
    """
    Compute (sqrt(n1!) * sqrt(n2!) * ...) / (sqrt(m1!) * sqrt(m2!) * ...) safely for large numbers.
    This avoids overflow by working in log space.
    """
    log_numerator = sum(log_sqrt_factorial(n) for n in numerator_ns)
    log_denominator = sum(log_sqrt_factorial(n) for n in denominator_ns)
    log_ratio = log_numerator - log_denominator
    return np.exp(log_ratio)


def safe_coherent_coefficient(alpha: complex, n: int, normalize: bool = True) -> complex:
    """
    Compute the coefficient for the n-th Fock state in a coherent state expansion.
    Formula: exp(-|alpha|^2/2) * alpha^n / sqrt(n!) if normalize=True
             or alpha^n / sqrt(n!) if normalize=False
    
    This avoids overflow by working in log space for large n.
    """
    if alpha == 0 and n > 0:
        return 0.0
    elif n == 0:
        return np.exp(-abs(alpha)**2/2) if normalize else 1.0
    else:
        # Work in log space
        log_term = n * np.log(abs(alpha)) - log_sqrt_factorial(n)
        if normalize:
            log_term -= abs(alpha)**2/2
            
        # Handle phase separately
        phase = (alpha / abs(alpha)) ** n if abs(alpha) > 0 else 1
        return phase * np.exp(log_term)


def find_closest_int_from_sqrt_including_one_over(x:float, eps:float=1e-5) -> tuple[int, bool, Sign]:
    if x >= 0:
        sign = "+"
    else:
        sign = "-"

    def _matching_int(val:float) -> int|None:
        val_int = int(round(val))
        relative_diff = abs(val - val_int)/val
        if relative_diff < eps:
            return val_int
        return None
    
    # search options:
    val = x*x
    val_int = _matching_int(val)
    if val_int is not None:
        return val_int, False, sign
    
    val = 1/val
    val_int = _matching_int(val)
    if val_int is not None:
        return val_int, True, sign
    
    raise ValueError()


def relu(x:float) -> float:
    return max(0, x)




