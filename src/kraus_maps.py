#%%
import numpy as np
import matplotlib.pyplot as plt
import functools  
import itertools
from scipy.optimize import brentq

import warnings
warnings.simplefilter('error', RuntimeWarning)  #TODO remove

from typing import Literal, Callable, Iterable, TypeAlias, Final, Any
from typing import cast as type_cast

import warnings
from scipy.linalg import LinAlgWarning

import qutip
from qutip import Qobj, basis, qeye, qzero, fidelity
from qutip import destroy, create, num

from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D


if __name__ == "__main__":
    from __init__ import add_project_to_path, add_root_to_path
    # add_project_to_path()
    path = add_root_to_path()

from src.utils.prints import ProgressBar
from src.utils.maths import factorial, sqrt_factorial, power_computed_in_log_space
from src.utils.caches import cache


from src.noise import BosonicNoiseType
from src.bosonic_operators import get_operator
from globals import Globals
from src._numerics import exp, log, sqrt, π


if Globals.PRECISE:
    if "auto_tidyup" in qutip.settings.core:         #type: ignore
        qutip.settings.core["auto_tidyup"] = False   #type: ignore

## For precise math:
import mpmath  
from mpmath import mp
mp.dps = 100  # decimal places for higher precision calculations


## Types:
class KrausTruncationError(ValueError):
    pass

## Constants:
""" When computing the KL cost from Kraus operators, if the cost becomes smaller than this value,
    we stop adding more Kraus operators to save time. """
KRAUS_COST_THRESHOLD : Final[float] = 1e-50 if Globals.PRECISE else 1e-16

""" Number of consecutive Kraus operator contributions below KRAUS_COST_THRESHOLD"""
KRAUS_TOO_SMALL_STREAK_SIZE : Final[int] = 500 if Globals.PRECISE else 5  

""" Threshold for checking Kraus completeness condition Σ Kj†Kj = I """
KRAUS_COMPLETENESS_CHECK_THRESHOLD : Final[float] = 1e-12 if Globals.PRECISE else 1e-5



def _sqrt_gamma_j_over_factorial_mpmath(gamma:float, j:int):
    """
    High precision approach using mpmath
    sqrt(gamma^j / j!) = exp(j/2 * ln(gamma) - 1/2 * ln(j!))
    """
    # Convert to mpmath types
    gamma_mp = mp.mpf(gamma)
    j_mp = mp.mpf(j)
    
    # Compute using mpmath: ln(j!) = ln(Gamma(j+1))
    log_result = (j_mp / 2) * mpmath.log(gamma_mp) - 0.5 * mpmath.log(mpmath.gamma(j_mp + 1))
    result = mpmath.exp(log_result)
    
    # Convert back to float for display
    return result


def _compute_sqrt_factor(γ:float, j:int) -> float:
    """sqrt_factor = np.sqrt(γ**j / factorial(j))"""

    if Globals.PRECISE:
        ## Method2 - log-space and high precision:
        sqrt_factor = _sqrt_gamma_j_over_factorial_mpmath(γ, j)

    else:
        ## Method1 - naive:
        sqrt_γ_j = np.sqrt(γ**j)          
        if sqrt_γ_j == 0.0:
            sqrt_γ_j = power_computed_in_log_space(γ, j/2.0)
        sqrt_factor = sqrt_γ_j / sqrt_factorial(j)

    return sqrt_factor


def _compute_kraus_operator_j(noise_type:BosonicNoiseType, N:int, γ:float, j:int) -> Qobj:
    n = get_operator(num, N)

    # sqrt_factor = np.sqrt(γ**j / factorial(j))
    with warnings.catch_warnings():
        warnings.filterwarnings('error', category=RuntimeWarning)
        try:
            sqrt_factor = _compute_sqrt_factor(γ, j)  # works even for big j
        except RuntimeWarning as warning:
            assert warning.args[0] == 'overflow encountered in exp'
            raise KrausTruncationError(f"j={j} is too large for γ={γ}. Got warning: {warning}")

    ## Create according to noise type:
    if noise_type == "loss":
        # to get (1-γ)^(n/2):
        exponent : Qobj = (log(1 - γ)/2) * n   #type: ignore
        # get a**j:
        a_j = get_operator(destroy, N, to_the_power=j)
        # Compute:
        K_j = sqrt_factor * exponent.expm() * a_j

    elif noise_type == "dephasing":
        # get n**j:
        n_2 = get_operator(num, N, to_the_power=2)
        n_j = get_operator(num, N, to_the_power=j)
        K_j = sqrt_factor * (-(γ/2) * n_2).expm() * n_j

    else:
        raise ValueError(f"not a known type. Got {noise_type!r}")

    ## Check resulting operation:
    if np.any(np.isnan(K_j.full())):
        raise KrausTruncationError(f"Kraus operator K_{j} for noise_type={noise_type}, γ={γ} contains NaNs.")
    if np.abs(K_j.full()).max() == 0.0:
        raise KrausTruncationError(f"Kraus operator K_{j} for noise_type={noise_type}, γ={γ} is zero.")

    return K_j


def _derive_num_kraus_operators(noise_type:BosonicNoiseType, N:int, γ:float) -> int:
    match noise_type:
        case "loss":
            num_operators = N + 1
        case "dephasing":
            γn2 = γ * N**2
            for_good_measure = N/2
            num_operators = int(np.ceil(γn2 + 8*np.sqrt(γn2))) 
            num_operators = int(for_good_measure * num_operators)
            num_operators = max(num_operators, N)
        case _:
            raise ValueError(f"not a known type. Got {noise_type!r}")

    return num_operators


@cache(ram=True, disk=False)
def kraus_operator_j(noise_type:BosonicNoiseType, N:int, γ:float, j:int) -> Qobj:
    ## assert num operators:
    if Globals.DEBUG:
        num_operators = _derive_num_kraus_operators(noise_type, N, γ)    
        assert j <= num_operators, f"j={j} is too large for γ={γ} and N={N}. Max j is {num_operators-1}."

    kraus_op = _compute_kraus_operator_j(noise_type, N, γ, j)
    return kraus_op


def kraus_operators_series(noise_type:BosonicNoiseType, N:int, γ:float, _check:bool=True) -> list[Qobj]:
    """
    Returns the Kraus operators for photon loss and dephasing channels [1]:

    Photon loss channel:
        E_j^loss = sqrt(gamma_1^j / j!) * (1 - gamma_1)^(n/2) * a^j

    Dephasing channel:
        E_j^dephasing = sqrt(gamma_2^j / j!) * exp(-gamma_2/2 * n^2) * n^j

    where:
        - gamma_1, gamma_2: unitless noise-strength coefficients
        - a: annihilation operator
        - n = a.dag() * a: number operator

    References
    ----------
    [1] V. V. Albert et al., "Performance and structure of single-mode bosonic codes", Phys. Rev. A 97, 032346 (2018)
    """
    ## Derive num operators:
    num_operators = _derive_num_kraus_operators(noise_type, N, γ)
    
    kraus_ops = []
    for j in ProgressBar(range(num_operators), prefix="create kraus    "):
        try:
            kraus_op = _compute_kraus_operator_j(noise_type, N, γ, j)
        except KrausTruncationError as e:
            break
        kraus_ops.append(kraus_op)
        
    if _check:
        _assert_correct_kraus_ops(kraus_ops, _check_completeness_cond=True)

    return kraus_ops


def _check_kraus_series_completeness(ops:Iterable[Qobj]):

    # Build sum of Kj†Kj
    sum_ : Qobj = sum((Kj.dag() @ Kj for Kj in ops))  #type: ignore
    dim = sum_.shape[0]

    # Compare:
    I = qeye(dim).full()
    sum_ = sum_.full()

    # Check:
    diff = np.linalg.norm(sum_ - I, ord='fro')
    if diff > KRAUS_COMPLETENESS_CHECK_THRESHOLD:
        raise AssertionError(f"Kraus operators are not complete: Σ Kj†Kj - I = {diff!r}")



def _assert_correct_kraus_ops(kraus_ops, _check_completeness_cond:bool=True) -> None:
    """ Check that the provided Kraus operators are valid.
    This validation checks that:
    1. The operators are a list of Qobj instances.
    2. The operators are square matrices of the same dimension.
    3. The sum of the Kraus operators squared equals the identity operator.
    """
    assert isinstance(kraus_ops, list), "Kraus operators must be provided as a list."
    assert all(isinstance(K, Qobj) for K in kraus_ops), "All Kraus operators must be Qobj instances."
    dim = kraus_ops[0].shape[0]
    assert all(K.shape == (dim, dim) for K in kraus_ops), "All Kraus operators must be square matrices of the same dimension."
    
    #$ Check the completeness condition: sum(K_i^† K_i) = I
    if _check_completeness_cond:
        _check_kraus_series_completeness(kraus_ops)



def _kraus_loss_second_order(gamma:float, a:Qobj, adag:Qobj, _check_completeness_cond:bool=True):
    n_op = adag @ a
    I    = qeye(a.shape[0])

    # coefficients chosen to cancel the O(gamma^2) leakage
    K0 = (I
          - 0.5 * gamma * n_op
          + gamma**2 * (-3/8 * (n_op @ n_op) + 1/4 * n_op))
    K1 = (gamma**0.5) * a
    K2 = (gamma / 2.0**0.5) * (a @ a)
    _assert_correct_kraus_ops([K0, K1, K2], _check_completeness_cond=_check_completeness_cond)
    return [K0, K1, K2]


def _kraus_dephasing_second_order(gamma, a, adag, _check_completeness_cond:bool=True):
    """
    Three-operator Kraus set accurate to O(gamma^3) for a small-step
    pure-dephasing channel (jump operator n = a†a).
    """
    dim  = a.shape[0]
    I    = qeye(dim)
    n_op = adag @ a                      # number operator

    # No-jump operator to second order
    K0 = (I
          - 0.5 * gamma * (n_op @ n_op)
          - 0.375 * gamma**2 * (n_op @ n_op @ n_op @ n_op))   # −3/8 n^4

    # Single- and double-jump operators
    K1 = (gamma**0.5) * n_op
    K2 = (gamma / 2**0.5) * (n_op @ n_op)

    _assert_correct_kraus_ops([K0, K1, K2], _check_completeness_cond=_check_completeness_cond)   # now passes to O(gamma^3)
    return [K0, K1, K2]









## Example:
def _run_test_sqrt_gamma_j_over_factorial(
    gamma = 1e-17,
    j_values = [5, 10, 20, 50, 100, 150, 200, 300, 400]
):
    """
    Test different methods for computing sqrt(gamma^j / j!)
    """
    import numpy as np
    from scipy.special import gammaln
    from math import factorial

    import mpmath  
    from mpmath import mp


    import matplotlib.pyplot as plt
    if Globals.LaTeX_RENDERING:
        plt.rcParams['text.usetex'] = True
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Computer Modern Serif']
        plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'



    # Set mpmath precision (number of decimal places)
    mp.dps = 100  # decimal places

    def mpmath_str(mpmath_result) -> str:
        return mpmath.nstr(mpmath_result, 6, strip_zeros=False)  #type: ignore

    def sqrt_gamma_j_over_factorial_naive(gamma, j):
        """
        Naive approach - prone to overflow/underflow for large j
        """
        return np.sqrt(gamma**j / factorial(j))

    def sqrt_gamma_j_over_factorial_stable(gamma, j):
        """
        Numerically stable approach using logarithms
        sqrt(gamma^j / j!) = exp(j/2 * ln(gamma) - 1/2 * ln(j!))
        """
        log_result = (j / 2) * np.log(gamma) - 0.5 * gammaln(j + 1)
        return np.exp(log_result)

    def sqrt_gamma_j_over_factorial_mpmath(gamma, j):
        """
        High precision approach using mpmath
        sqrt(gamma^j / j!) = exp(j/2 * ln(gamma) - 1/2 * ln(j!))
        """
        # Convert to mpmath types
        gamma_mp = mp.mpf(gamma)
        j_mp = mp.mpf(j)
        
        # Compute using mpmath: ln(j!) = ln(Gamma(j+1))
        log_result = (j_mp / 2) * mpmath.log(gamma_mp) - 0.5 * mpmath.log(mpmath.gamma(j_mp + 1))
        result = mpmath.exp(log_result)
        
        # Convert back to float for display
        return result

    print("Comparison of naive vs stable vs mpmath computation:")
    print(f"{'j':<6} {'Naive':<20} {'Stable':<20} {'MPMath':<20}")
    print("-" * 70)


    naive_results = []
    stable_results = []
    mpmath_results = []

    for j in j_values:
        try:
            naive_result = sqrt_gamma_j_over_factorial_naive(gamma, j)
            naive_str = f"{naive_result:.6e}"
        except (OverflowError, ValueError):
            naive_result = np.nan
            naive_str = "overflow"
        
        stable_result = sqrt_gamma_j_over_factorial_stable(gamma, j)
        mpmath_result = sqrt_gamma_j_over_factorial_mpmath(gamma, j)

        # Add to list for plot:
        naive_results.append(naive_result)
        stable_results.append(stable_result)
        mpmath_results.append(mpmath_result)
        
        print(f"{j:<6} {naive_str:<20} {stable_result:<20.6e} {mpmath_str(mpmath_result)}")


    ## Plot results and comparison visually:
    equation_str = r"$\sqrt{\frac{\gamma^{j} }{ j!}}$"
    fontsize = 16

    def find_last_valid_index(y_values):
        """Find the index of the last valid (non-NaN) value."""
        valid_indices = [i for i, y in enumerate(y_values) if not np.isnan(y)]
        if valid_indices and len(valid_indices) < len(y_values):
            return valid_indices[-1]
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.xlabel("$j$", fontsize=fontsize)
    plt.ylabel(r"$\log_{10}$" + "(" + equation_str + ")", fontsize=fontsize)

    # Convert to log10 for plotting to avoid float underflow
    naive_log = [np.log10(x) if x > 0 and not np.isnan(x) else np.nan for x in naive_results]
    stable_log = [np.log10(x) if x > 0 else np.nan for x in stable_results]
    mpmath_log = [float(mpmath.log10(x)) for x in mpmath_results]  # Keep full precision


    y_values_all = [y for vec in (naive_log, stable_log, mpmath_log) for y in vec if np.isfinite(y)]
    text_y_pos = min(y_values_all)

    # Plot each method

    for y_vec, label, linestyle in zip(
        [naive_log, stable_log, mpmath_log],
        ["Naive", "Stable", "MPMath"],
        ["-", "--", ":"]
    ):
        lines = ax.plot(j_values, y_vec, marker='o', label=label, linewidth=4, linestyle=linestyle)
        color = lines[0].get_color()
        
        # Find where NaNs begin and add vertical line
        last_valid_idx = find_last_valid_index(y_vec)
        if last_valid_idx is not None:
            last_valid_j = j_values[last_valid_idx]
            
            # Add vertical line at the transition point
            ax.axvline(x=last_valid_j, color=color, linestyle='--', alpha=0.8, linewidth=3)
            
            # Add text annotation 
            ax.text(last_valid_j, 0.02, f"{label} fails at $j={last_valid_j}$", 
                   rotation=90, verticalalignment='bottom', horizontalalignment='right',
                   fontsize=12, color=color, alpha=1.0,
                   transform=ax.get_xaxis_transform())

    title = "Comparison of Approaches\n"+\
        r"$\gamma=%s$"%(gamma)
    plt.title(title, fontsize=fontsize)
    plt.legend()
    plt.grid()
    plt.show()


    print("\nTest completed.")


if __name__ == "__main__":
    _run_test_sqrt_gamma_j_over_factorial()