import numpy as np
from qutip import Qobj, basis, expect

import sympy as sp
from sympy import lambdify

from scipy.optimize import minimize

from typing import Final, Literal, Iterable, Callable, cast, TypeVar
from collections import deque

import time


if __name__ == "__main__":
    from __init__ import add_root_to_path
    path = add_root_to_path()

from src.utils.maths import factorial
from src.utils.prints import ProgressBar
from src.utils.searches import binary_search_callable_increasing, DEFAULT_TOL
from src.utils.caches import cache
from src.utils import assertions

from src.visualizations import plot_light_states, plot_fock_distribution
from src.codes_built_in_superposition import simple_m_legged_code, simple_m_legged_state, _CodeTypes
from globals import Globals
from src import bosonic_operators


## Types:
# Type that is either a float or a sympy expression:
_NumberOrSympyExpr = TypeVar('_NumberOrSympyExpr', float, sp.Expr)


TESTS_NUM_MOMENTS : Final[int] = 100
DEFAULT_L_CUT_OFF : Final[int] = 1_000
DEFAULT_UPPER_BOUND_FOR_SEARCH : Final[float] = 10.0

r_symbol = sp.symbols('r')
L_symbol = sp.symbols('L')
alpha_symbol = sp.symbols('α') 


i = sp.I
π = sp.pi



def _number_projector(n: int, dim: int) -> Qobj:
    """Create the number projector |n><n| for a bosonic mode of given dimension.

    Args:
        n (int): The photon number.
        dim (int): The dimension of the bosonic mode.

    Returns:
        Qobj: The number projector |n><n|.
    """
    ket = basis(dim, n)
    return ket * ket.dag()



def qutip_mean_photon_number(state:Qobj) -> float:
    """Calculate the mean photon number for a single-mode bosonic state.

    Args:
        state (Qobj): The quantum state.

    Returns:
        float: The mean photon number.
    """

    N = state.dims[0][0]  # Dimension of the bosonic mode

    ## Method 1: Using number operator directly
    n = bosonic_operators.num(N)
    mean_photon_number1 = expect(n, state)
    mean_photon_number1 = float(mean_photon_number1)  #type: ignore

    if Globals.DEBUG:
        ## Method 2: Using number sum of probabilities per Fock state:
        # Create list of projection operators
        projectors = [_number_projector(n, N) for n in range(N)]
        # Calculate all expectations at once
        probs : np.ndarray = expect(projectors, state)  #type: ignore
        # Calculate mean photon number by weighted sum of probabilities
        mean_photon_number2 = sum(n * prob for n, prob in enumerate(probs))

        assert np.isclose(mean_photon_number1, mean_photon_number2), f"Mean photon number calculations do not match: {mean_photon_number1} vs {mean_photon_number2}"

    return mean_photon_number1


def _common_a_lk_factor(n:int) -> sp.Expr:
    return sp.factorial(2*n) / (
        (4**n) * (sp.factorial(n)**2)
    ) * sp.tanh(r_symbol)**(2*n)


def _analytic_mean_photon_number_for_squeezed_k_state(m:int, k:int, L_threshold:Literal[False]|int=False) -> sp.Expr:
    l_symbol = sp.symbols('l', integer=True)

    k_star = (-k) % m
    
    # Define n in terms of l
    n_expr = l_symbol * m + k_star
    a_kl_expr = _common_a_lk_factor(n_expr)
    
    # Create symbolic sums
    numerator = sp.Sum( 2 * n_expr * a_kl_expr, (l_symbol, 0, L_symbol))
    denominator = sp.Sum(a_kl_expr, (l_symbol, 0, L_symbol))

    result : sp.Expr = numerator / denominator  #type: ignore

    if L_threshold is not False:
        assert isinstance(L_threshold, int)
        result = result.subs({L_symbol: L_threshold})  #type: ignore

    return result


def _analytic_mean_photon_number_for_binomial_k_state(m: int, logical_value: int) -> sp.Expr:
    # For qubit (d=2) binomial codes: <n> = (m/2)*(r) with r ≡ N
    return m * r_symbol / 2


def _analytic_mean_photon_number_for_cat_k_state(m:int, k:int) -> sp.Expr:
    d_symbol = sp.symbols('d', integer=True)
    
    abs_alpha_square = abs(alpha_symbol)**2

    # Define ω 
    ω = sp.exp(i * 2 * π / m)
    exponent = sp.exp(
        -abs_alpha_square * (1 - ω**d_symbol)
    )

    def sum_expression(k_:int) -> sp.Sum:
        return sp.Sum(ω**(k_*d_symbol)*exponent, (d_symbol, 0, m-1))  #type: ignore
    
    return abs_alpha_square * sum_expression(k+1) / sum_expression(k)


def _solve_equation_with_optimization_tools(
    func:Callable[[float], float],  
    target_value:float,
    practical_upper_bounds:float=DEFAULT_UPPER_BOUND_FOR_SEARCH
) -> float:

    x_bounds = (0.0, practical_upper_bounds)

    ## First try with a simple binary search, else, use scipy minimization:
    x_solution, y_solution = binary_search_callable_increasing(func, target_value, x_bounds)

    assert np.isclose(y_solution, target_value, atol=1e-6), f"Binary search did not find a solution close enough to target: y_solution={y_solution} vs target={target_value}"

    return x_solution


def k_from_logical_value(m:int, logical_value:int | Literal['+']) -> int:
    assertions.even(m)

    if logical_value == '+':
        # Special case: superposition |+> = 1/sqrt(2)(|0> + |1>)
        # This is handled at a higher level, not here
        raise ValueError("k_from_logical_value does not support logical_value='+' (superposition). Use get_mean_photon_number directly.")
    elif logical_value == 0:
        k = 0
    elif logical_value == 1:
        k = m // 2
    else:
        raise ValueError("logical_value must be 0 (|0> state), 1 (|1> state), or '+' (|+> superposition state).")
    
    return k

def _strip_imaginary_part_from_symbolic_expr_if_close_to_real(numerical_value:_NumberOrSympyExpr) -> _NumberOrSympyExpr:
    if isinstance(numerical_value, (int, float, complex, np.floating, np.integer)):
        type_ = "native"
    elif isinstance(numerical_value, sp.Expr):
        type_ = "sympy"
    else:
        raise TypeError(f"Unsupported type for numerical_value: {type(numerical_value)!r}")

    if type_ == "native":
        is_real = np.isrealobj(numerical_value)
    elif type_ == "sympy":
        is_real = numerical_value.is_real
    else:
        raise RuntimeError("Unreachable code.")

    if is_real:
        return numerical_value

    if type_ == "native":
        imaginary_part = np.imag(numerical_value)
    elif type_ == "sympy":
        imaginary_part = float(sp.im(numerical_value)) 
        

    assert np.isclose(imaginary_part, 0), f"Mean photon number has non-negligible imaginary part: {imaginary_part}"

    if type_ == "native":
        numerical_value = np.real(numerical_value)
    elif type_ == "sympy":
        numerical_value = sp.re(numerical_value)

    return numerical_value



def _numerical_exact_summation_mean_photon_number_for_cat_codeword(m:int, alpha:float, k:int) -> float:
    """Numerical evaluation (NumPy) of the analytic expression for the
    mean photon number of an m-legged cat codeword for a given logical
    sector k and displacement alpha.

    ``alpha`` (may be real or complex). The function computes
    abs(alpha)**2 * S_{k+1} / S_k where
    S_j = sum_{d=0}^{m-1} omega^{j*d} * exp(-|alpha|^2*(1-omega^d))
    and omega = exp(i*2*pi/m).
    """
    # Basic checks
    assert m > 0
    assert 0 <= k < m

    abs_alpha_square = abs(alpha)**2

    # m-th root of unity
    omega = np.exp(1j * 2.0 * np.pi / float(m))

    numerator = 0+0j
    denominator = 0+0j

    for d in range(m):
        omega_d = omega**d
        # exponent may be complex because of omega_d; use numpy complex exp
        exponent = np.exp(-abs_alpha_square * (1.0 - omega_d))

        numerator += (omega**((k + 1) * d)) * exponent
        denominator += (omega**(k * d)) * exponent

    # Multiply by |alpha|^2 as in analytic expression
    result = abs_alpha_square * numerator

    # Safe division: if denominator is effectively zero, handle cases
    denom_abs = abs(denominator)
    tiny = 1e-14
    if denom_abs < tiny:
        # If numerator is also effectively zero, return 0.0
        if abs(result) < tiny:
            return 0.0
        # Otherwise this is a problematic division (shouldn't usually happen)
        raise ZeroDivisionError(f"Denominator in cat mean-photon expression is numerically zero (m={m}, k={k}, alpha={alpha!r})")

    result = result / denominator

    # Strip tiny imaginary component and return float
    imag_part = np.imag(result)
    real_part = np.real(result)
    relative_imag_part = abs(imag_part) / max(abs(real_part), 1e-15)
    if relative_imag_part > 1e-8:
        # If the imaginary part is unexpectedly large relative to the real part,
        # surface an informative error.
        raise FloatingPointError(
            f"Mean photon number has non-negligible imaginary part: imag={imag_part}, "
            f"real={real_part}, relative={relative_imag_part}"
        )

    return float(np.real(result))


def _numerical_exact_summation_mean_photon_number_for_squeezed_codeword(m:int, r:float, k:int, L_cut_off:int) -> float:
    ## Checks:
    assert L_cut_off > 0
    assert m > 0
    assert 0 <= k < m

    k_star = (-k) % m
    res_tol = DEFAULT_TOL * 1e-3  # Be even more strict than the search tolerance
  

    numerator = 0.0
    denominator = 0.0

    last_result = 0.0
    result = 0.0  # place holder
    diff = float('inf')
    diffs = deque(maxlen=5)  # Keep only the last 5 differences

    def _check_convergence(l:int) -> bool:
        if l < 10:
            return False
        if len(diffs) < diffs.maxlen:  #type: ignore
            return False
        if _all_close_to_zero_in_list(list(diffs), tol=res_tol):
            return True
        return False


    for l in ProgressBar.range(L_cut_off + 1, prefix="l values: ", print_length=100):
        ProgressBar.newest().append_extra_str(f" diff={diff}")
        n = l * m + k_star
        n = int(n)
        a_kl = factorial(2*n) / (
            (4**n) * (factorial(n)**2)
        ) * np.tanh(r)**(2*n)
        
        # Create symbolic sums
        numerator   +=  2 * n * a_kl
        denominator += a_kl


        if abs(denominator) < 1e-300:
            raise ZeroDivisionError(
                f"Degenerate ratio in squeezed mean-photon summation: denominator~0 (m={m}, r={r}, k={k}, l={l})."
            )

        result = numerator / denominator

        diff = abs(result - last_result)
        diffs.append(diff)

        if _check_convergence(l):
            break

        last_result = result


    if False == "False":
        from matplotlib import pyplot as plt
        plt.plot(range(len(results)), results, label="Numerator")
        plt.xlabel("l")
        plt.ylabel("Value")
        plt.title("Convergence")

    return result

def mean_photon_number_for_cat_codeword(m:int, alpha:float, logical_value:int | Literal['+'], analytic_substitution:bool=True) -> float:
    # Handle superposition state |+> = 1/sqrt(2)(|0> + |1>)
    if logical_value == '+':
        mean_photon_0 = mean_photon_number_for_cat_codeword(m, alpha, 0, analytic_substitution=analytic_substitution)
        mean_photon_1 = mean_photon_number_for_cat_codeword(m, alpha, 1, analytic_substitution=analytic_substitution)
        return 0.5 * mean_photon_0 + 0.5 * mean_photon_1
    
    k = k_from_logical_value(m, logical_value)

    if analytic_substitution:
        analytical_expression = _analytic_mean_photon_number_for_cat_k_state(m, k)
        numerical_value : float = analytical_expression.subs({alpha_symbol:alpha}).evalf().doit()  #type: ignore
    else:
        numerical_value = _numerical_exact_summation_mean_photon_number_for_cat_codeword(m, alpha, k)

    numerical_value = _strip_imaginary_part_from_symbolic_expr_if_close_to_real(numerical_value)    
    return float(numerical_value)


def mean_photon_number_for_squeezed_codeword(m:int, r:float, logical_value:int | Literal['+'], analytic_substitution:bool=True, L_cut_off:int=DEFAULT_L_CUT_OFF, _k:int|None=None) -> float:
    ## Handle superposition state |+> = 1/sqrt(2)(|0> + |1>)
    if logical_value == '+' and _k is None:
        mean_photon_0 = mean_photon_number_for_squeezed_codeword(m, r, 0, analytic_substitution=analytic_substitution, L_cut_off=L_cut_off)
        mean_photon_1 = mean_photon_number_for_squeezed_codeword(m, r, 1, analytic_substitution=analytic_substitution, L_cut_off=L_cut_off)
        return 0.5 * mean_photon_0 + 0.5 * mean_photon_1
    
    ## Ignore logical value and use k if provided:
    if _k is not None:
        k = _k
    else:
        k = k_from_logical_value(m, logical_value)

    if analytic_substitution:
        analytical_expression = _analytic_mean_photon_number_for_squeezed_k_state(m, k)
        numerical_value = analytical_expression.subs({r_symbol:r, L_symbol:L_cut_off}).evalf().doit()
        return float(numerical_value)
    
    else: 
        return _numerical_exact_summation_mean_photon_number_for_squeezed_codeword(m, r, k, L_cut_off)


def mean_photon_number_for_binomial_codeword(m:int, r:float, logical_value:int | Literal['+'], analytic_substitution:bool=True, _k:int|None=None) -> float:
    ## Handle superposition state |+> = 1/sqrt(2)(|0> + |1>)
    if logical_value == '+':
        mean_photon_0 = mean_photon_number_for_binomial_codeword(m, r, 0, analytic_substitution=analytic_substitution)
        mean_photon_1 = mean_photon_number_for_binomial_codeword(m, r, 1, analytic_substitution=analytic_substitution)
        return 0.5 * mean_photon_0 + 0.5 * mean_photon_1
    
    ## Ignore logical value and use k if provided:
    if _k is not None:
        raise NotImplementedError("Binomial code mean photon number not implemented for k input.")
    
    # Ignore analytic substitution == False:
    analytic_substitution = True
    analytical_expression = _analytic_mean_photon_number_for_binomial_k_state(m, logical_value)
    numerical_value = analytical_expression.subs({r_symbol:r}).evalf().doit()
    return float(numerical_value)


def mean_photon_number_for_gkp_codeword(m:int, nbar:float, logical_value:int | Literal['+'], analytic_substitution:bool=False) -> float:
    # Handle superposition state |+> = 1/sqrt(2)(|0> + |1>)
    if logical_value == '+':
        # Superposition state has quantum interference effects.
        # For GKP: the superposition state does NOT have mean photon number equal to average of |0> and |1>.
        # e.g., for nbar=2.0: <n>_|0>=1.872, <n>_|1>=2.251, but <n>_|+>=1.555
        # This requires computing from actual GKP states - no simple formula available.
        # TODO: Implement proper GKP superposition state photon number calculation.
        raise NotImplementedError(
            "GKP superposition state photon number requires solving from the actual lattice state. "
            "Use gkp_from_nbar() to construct states and compute expectation of n operator."
        )
    
    # WARNING: Current simplified model - empirical testing shows this is an APPROXIMATION
    # Actual findings from lattice structure:
    #   - GKP |0> and |1> states have DIFFERENT mean photon numbers
    #   - |0> tends to have slightly lower <n> than |1> due to lattice well positions
    # The current implementation treats them as equal, which is a known limitation.
    # TODO: Implement state-dependent GKP photon numbers based on lattice structure analysis.
    return nbar


@cache(ram=True, disk=True)
def find_parameter_for_target_mean_photon_number(
    code_type:_CodeTypes,
    m:int,
    logical_value:int | Literal['+'],
    target_mean_photon_number:float,
) -> float:
    
    # Special case: binomial code is linear, so we can solve directly
    # For binomial: <n> = (m/2) * r, so r = 2*<n>/m
    if code_type == 'binomial':
        return 2.0 * target_mean_photon_number / m

    mean_photon_number_func = get_single_input_function_from_symbolic_expression(
        code_type=code_type,
        m=m,
        logical_value=logical_value
    )

    return _solve_equation_with_optimization_tools(
        mean_photon_number_func,
        target_mean_photon_number
    )


def get_single_input_function_from_symbolic_expression(
    code_type:_CodeTypes,        
    m:int,
    logical_value:int | Literal['+']
) -> Callable[[float], float]:


    def mean_photon_number_func(param: float) -> float:
        return get_mean_photon_number(
            code_type=code_type,
            m=m,
            logical_value=logical_value,
            parameter=param,
            analytic_substitution=False
        )
    
    return mean_photon_number_func


def get_mean_photon_number(
    code_type:_CodeTypes,        
    m:int,
    logical_value:int | Literal['+'],
    parameter:float,
    analytic_substitution:bool=True,
    cut_off:int = DEFAULT_L_CUT_OFF 
) -> float:
    # Handle superposition state |+> = 1/sqrt(2)(|0> + |1>)
    if logical_value == '+':
        mean_photon_0 = get_mean_photon_number(
            code_type=code_type,
            m=m,
            logical_value=0,
            parameter=parameter,
            analytic_substitution=analytic_substitution,
            cut_off=cut_off
        )
        mean_photon_1 = get_mean_photon_number(
            code_type=code_type,
            m=m,
            logical_value=1,
            parameter=parameter,
            analytic_substitution=analytic_substitution,
            cut_off=cut_off
        )
        # Average for superposition state
        return 0.5 * mean_photon_0 + 0.5 * mean_photon_1
    
    match code_type:
        case 'cat':
            return mean_photon_number_for_cat_codeword(m, parameter, logical_value, analytic_substitution=analytic_substitution)
        case 'squeeze':
            return mean_photon_number_for_squeezed_codeword(m, parameter, logical_value, analytic_substitution=analytic_substitution, L_cut_off=cut_off)
        case "binomial":
            return mean_photon_number_for_binomial_codeword(m, parameter, logical_value, analytic_substitution=analytic_substitution)
        case "gkp":
            return mean_photon_number_for_gkp_codeword(m, parameter, logical_value, analytic_substitution=analytic_substitution)
        case _:
            raise ValueError(f"Unknown code type: {code_type!r}")


def _all_close_to_zero_in_list(list_:list[float], tol:float=1e-6) -> bool:
    first_val = list_[0]
    for val in list_[1:]:
        if not np.isclose(0.0, val, atol=tol):
            return False
    return True



def _test1():
    state_1 = basis(TESTS_NUM_MOMENTS, 3)  # Fock state |3>
    state_2 = (basis(TESTS_NUM_MOMENTS, 2) + basis(TESTS_NUM_MOMENTS, 4)).unit()  # Superposition state (|2> + |4>)/sqrt(2)

    mean_photon_number_1 = qutip_mean_photon_number(state_1)
    mean_photon_number_2 = qutip_mean_photon_number(state_2)

    assert np.isclose(mean_photon_number_1, mean_photon_number_2)
    assert np.isclose(mean_photon_number_1, 3)

    # print(f"Mean photon number of (|2> + |4>)/sqrt(2) == |3> == {mean_photon_number_2}")



def _test2_squeezed_codes(
    r_vals = np.linspace(0.0, 6, 50).tolist(),
    m: int = 2,
    analytic_substitution: bool = False
):
    
    from matplotlib import pyplot as plt

    fig = plt.figure()
    

    for k in ProgressBar.range(m, prefix="per k: "):

        qutip_vals = []
        analytic_vals = []

        for r in ProgressBar(r_vals, prefix="per r: "):
            ProgressBar.newest().append_extra_str(f" r={r:.3f}")

            qutip_state = simple_m_legged_state(m, r, num_moments=TESTS_NUM_MOMENTS, code_type='squeeze', 
                                        qubit_logical_value=k,
                                        num_qudit_values=m)
                
            qutip_mean_photons = qutip_mean_photon_number(qutip_state)
            qutip_vals.append(qutip_mean_photons)

            if analytic_substitution:
                analytical_expression = _analytic_mean_photon_number_for_squeezed_k_state(m, k)
                analytical_mean_photons = analytical_expression.subs({r_symbol:r, L_symbol:DEFAULT_L_CUT_OFF}).evalf().doit()
                analytical_mean_photons = float(analytical_mean_photons)
            else:
                if r < 2.0:
                    L_cut_off = DEFAULT_L_CUT_OFF
                elif r < 4.0:
                    L_cut_off = 10_000
                elif r < 6.0:
                    L_cut_off = 100_000
                else:
                    L_cut_off = 1_000_000
                analytical_mean_photons = mean_photon_number_for_squeezed_codeword(
                    m, r, logical_value=-1, analytic_substitution=False, _k=k, L_cut_off=L_cut_off
                )

            analytic_vals.append(analytical_mean_photons)

        ## Plot:
        line = plt.plot(r_vals, qutip_vals, label=f"k={k}", marker='o', linestyle='None')
        plt.plot(r_vals, analytic_vals, linestyle='--', color=line[0].get_color())
        plt.show()
        plt.pause(0.1)

    plt.xlabel("Squeezing Parameter r")
    plt.ylabel("Mean Photon Number")
    plt.title(f"Mean Photon Number vs Squeezing Parameter for m={m}\n")
    plt.legend()
    plt.show()

    print("Test 2 completed.")


def _test3_infinite_vs_finite_series(
    m:int = 2,
    logical_value: int = 0,
    r:float = 2.5,
    with_infinity:bool = False,
    with_analytic_substitution:bool = False
):
    from matplotlib import pyplot as plt


    k = k_from_logical_value(m, logical_value)
    analytical_expression = _analytic_mean_photon_number_for_squeezed_k_state(m, k)
    
    def _get_series(L) -> sp.Expr:
        series = analytical_expression.subs({L_symbol : L})
        return series
        
    def _get_val(series:sp.Expr) -> float:
        val = series.subs({r_symbol : r}).evalf().doit()
        return float(val)
    
    def get_numerical_mean(L:int) -> float:
        return get_mean_photon_number("squeeze", m, logical_value, r, analytic_substitution=False, cut_off=L)

    ## Try to sum to infinity:
    if with_infinity:
        inf_series = _get_series(sp.oo)
        inf_value = _get_val(inf_series)

    ## Sum to a predefined L values: 
    values_from_analytical = []
    values_from_exact_numerical:list[float] = []
    times = []

    Ls = np.linspace(1, 1e3, num=20).tolist()
    for L in ProgressBar(Ls, prefix="L values: ", print_length=100):
        L = int(L)
        ProgressBar.newest().append_extra_str(f" (L={L})")

        t0 = time.perf_counter()
        if with_analytic_substitution:
            num_series = _get_series(L)
            from_analytic_value = _get_val(num_series)
            values_from_analytical.append(from_analytic_value)

        numerical_value = get_numerical_mean(L)
        t1 = time.perf_counter()

        values_from_exact_numerical.append(numerical_value)
        times.append(t1 - t0)

    ## Qutip calculation for comparison:
    state = simple_m_legged_state(m, r, num_moments=TESTS_NUM_MOMENTS, code_type='squeeze', 
                                  qubit_logical_value=logical_value)
    qutip_value = qutip_mean_photon_number(state)

    ## Plot comparison:
    if with_analytic_substitution:
        plt.semilogx(Ls, values_from_analytical, label="from Analytic", marker='o')
    if with_infinity:
        plt.axhline(inf_value, color='r', linestyle='--', label="Sum to Infinity")
    plt.axhline(qutip_value, color='g', linestyle='--', label="Qutip Value")
    plt.semilogx(Ls, values_from_exact_numerical, color='magenta', linestyle='', label="from Exact Numerical", marker='x')
    plt.xlabel("L")
    plt.ylabel("Mean Photon Number")
    plt.title(f"Mean Photon Number vs L for m={m}, logical value={logical_value}, r={r}")
    plt.show()

    ## Twin axis for time:
    ax2 = plt.gca().twinx()
    ax2.plot(Ls, times, color='yellow', label="Computation Time", marker='', linestyle='--')

    plt.legend()

    print("Test 3 completed.")



def _test4_cat_state(
    m:int = 2,
    alpha_vals:list[float] = np.linspace(0.01, 2.5, 15).tolist(),
    num_moments:int = TESTS_NUM_MOMENTS
):
    from matplotlib import pyplot as plt
    
    fig, ax = plt.subplots()    

    ax.set_xlabel("Displacement Parameter α")
    ax.set_ylabel("Mean Photon Number")
    ax.set_title(f"Mean Photon Number vs Displacement Parameter for m={m}\n")   
  
    simplified = [abs(alpha)**2 for alpha in alpha_vals]
    plt.plot(alpha_vals, simplified, marker='None', color="black", alpha=0.8, linestyle='-', label="|α|²")
                    
    for k in ProgressBar.range(m, prefix="per k: "):

        qutip_vals = []
        analytical = []

        for alpha in ProgressBar(alpha_vals, prefix="per α: "):
            state = simple_m_legged_state(m, alpha, num_moments=num_moments, code_type='cat', 
                                        qubit_logical_value=k,
                                        num_qudit_values=m)
            
            _qutip_mean_photons = qutip_mean_photon_number(state)
            
            analytical_expression = _analytic_mean_photon_number_for_cat_k_state(m, k)
            analytical_photon_number = analytical_expression.subs({alpha_symbol:alpha}).evalf().doit()
            analytical_photon_number = _strip_imaginary_part_from_symbolic_expr_if_close_to_real(analytical_photon_number)
            analytical_photon_number = float(analytical_photon_number)

            ## Append to lists:
            analytical.append(analytical_photon_number)
            qutip_vals.append(_qutip_mean_photons)

        ## Plot:
        line = plt.plot(alpha_vals, qutip_vals, label=f"k={k}", marker='o', linestyle='None')
        color = line[0].get_color()
        plt.plot(alpha_vals, analytical, marker='None', color=color, linestyle='--')

        ax.legend()
        plt.show()
        plt.pause(0.1)

    plt.show()
    plt.pause(0.1)
    print("Done.")


def _test5_get_parameter_for_given_mean_photons(
    target_mean_photons = 3.0,
    logical_value = 0    
):

    results = []

    for m in ProgressBar([2, 4, 6] ,prefix="per m: "):
        for code_type in ['squeeze', 'cat']:
            code_type = cast(Literal['squeeze', 'cat'], code_type)

            parameter = find_parameter_for_target_mean_photon_number(
                code_type=code_type,
                m=m,
                logical_value=logical_value,
                target_mean_photon_number=target_mean_photons,
            )

            computed_mean_photons = get_mean_photon_number(
                code_type=code_type,
                m=m,
                logical_value=logical_value,
                parameter=parameter,
                analytic_substitution=False 
            )

            assert np.isclose(computed_mean_photons, target_mean_photons, atol=1e-6)

            results.append( (code_type, m, parameter, computed_mean_photons) )
        
    for res in results:
        code_type, m, parameter, computed_mean_photons = res
        print(f"Code Type: {code_type:<8}: m={m}, Parameter={parameter:.6f} => Mean Photons={computed_mean_photons:.6f}")

    print("Done.")


def _test6_plot_mean_photons_params_for_different_codes(
    m:int = 2,
    logical_value:int|Literal['+'] = "+",
    target_mean_photon_numbers:list[float] = np.linspace(1.1, 5.01, 21).tolist()
) -> None:
    
    from matplotlib import pyplot as plt    

    lists = dict(
        cat=[],
        squeeze=[],
        binomial=[],
        gkp=[]
    )

    for target_mean_photon_number in ProgressBar(target_mean_photon_numbers, prefix="per target mean photons: "):
        for code_type in ProgressBar(_CodeTypes.__args__, prefix="per code type: "):

            parameter = find_parameter_for_target_mean_photon_number(
                code_type=code_type,
                m=m,
                logical_value=logical_value,
                target_mean_photon_number=target_mean_photon_number,
            )

            lists[code_type].append(parameter)

    ## Plot:
    linewidth = 3.0
    plt.figure(figsize=(10, 6))
    plot_order = ['cat', 'squeeze', 'gkp', 'binomial']
    for code_type in plot_order:
        params = lists[code_type]
        linestyle = '--' if code_type in ['cat', 'gkp'] else ':'

        # For m=2, binomial and gkp are effectively identical; draw binomial on top
        # with markers so both traces remain visible.
        marker = 'o' if code_type == 'binomial' else None
        zorder = 5 if code_type == 'binomial' else 3
        plt.plot(
            target_mean_photon_numbers,
            params,
            label=code_type,
            linewidth=linewidth,
            linestyle=linestyle,
            marker=marker,
            markersize=4,
            markevery=2,
            zorder=zorder,
        )
        
    plt.xlabel("Target Mean Photon Number")
    plt.ylabel("Parameter")
    plt.title("Mean Photon Number Parameters for Different Codes")
    plt.legend()
    plt.grid()
    plt.show()

    print("Done.")





def _test7_test_binomial_code(
    m:int = 2,
    strength_vals:list[float] = np.linspace(0.01, 5, 101).tolist(),
    num_moments:int = TESTS_NUM_MOMENTS,
    code_type:_CodeTypes = 'binomial'
):
    from matplotlib import pyplot as plt
    
    fig, ax = plt.subplots()    

    ax.set_xlabel("strength r")
    ax.set_ylabel("Mean Photon Number")
    ax.set_title(f"{code_type}: Mean Photon Number vs Strength for m={m}\n")   
  
                    
    for l in ProgressBar([0, 1], prefix="logical: "):

        qutip_vals = []
        analytical = []

        for r in ProgressBar(strength_vals, prefix="per r : "):
            state = simple_m_legged_state(m, r, num_moments=num_moments, code_type=code_type, 
                                        qubit_logical_value=l,
                                        num_qudit_values=m)
            
            _qutip_mean_photons = qutip_mean_photon_number(state)
            analytical_photon_number = get_mean_photon_number(code_type, m, l, r)

            ## Append to lists:
            qutip_vals.append(_qutip_mean_photons)
            analytical.append(analytical_photon_number)

        ## Plot:
        line = plt.plot(strength_vals, qutip_vals, label=f"l={l}", marker='o', linestyle='None')
        color = line[0].get_color()
        plt.plot(strength_vals, analytical, marker='None', color=color, linestyle='--')

        ax.legend()
        plt.show()
        plt.pause(0.1)

    plt.show()
    plt.pause(0.1)
    print("Done.")



lower_threshold_per_code: dict[_CodeTypes, float] = {
    'cat': 0.5,
    'squeeze': 1.0,
    'binomial': 0.0,
    'gkp': 0.0,
}



def _test8_parameter_solver_error_vs_qutip_mean(
    m: int = 2,
    logical_value: int | Literal['+'] = 0,
    target_mean_photon_numbers: list[float] = np.linspace(0.1, 5.1, 21).tolist(),
    num_moments: int = TESTS_NUM_MOMENTS,
) -> None:
    """For each code family, solve parameter-from-target and compare to Qutip mean photons.

    Plots signed error: (Qutip mean photons) - (target mean photons).
    """
    from matplotlib import pyplot as plt

    code_types: list[_CodeTypes] = ['cat', 'squeeze', 'binomial', 'gkp']
    errors_per_code: dict[_CodeTypes, list[float]] = {code_type: [] for code_type in code_types}
    targets_per_code: dict[_CodeTypes, list[float]] = {code_type: [] for code_type in code_types}

    
    for target_mean_photon_number in ProgressBar(target_mean_photon_numbers, prefix="per target mean photons: "):
        for code_type in ProgressBar(code_types, prefix="per code type: "):
            if target_mean_photon_number < lower_threshold_per_code[code_type]:
                continue

            m_for_code = 1 if code_type == 'gkp' else m

            parameter = find_parameter_for_target_mean_photon_number(
                code_type=code_type,
                m=m_for_code,
                logical_value=logical_value,
                target_mean_photon_number=target_mean_photon_number,
            )

            if logical_value == '+':
                state_0 = simple_m_legged_state(
                    m_for_code,
                    parameter,
                    num_moments=num_moments,
                    code_type=code_type,
                    qubit_logical_value=0,
                    num_qudit_values=2,
                )
                state_1 = simple_m_legged_state(
                    m_for_code,
                    parameter,
                    num_moments=num_moments,
                    code_type=code_type,
                    qubit_logical_value=1,
                    num_qudit_values=2,
                )
                qutip_state = (state_0 + state_1).unit()
            else:
                qutip_state = simple_m_legged_state(
                    m_for_code,
                    parameter,
                    num_moments=num_moments,
                    code_type=code_type,
                    qubit_logical_value=logical_value,
                    num_qudit_values=2,
                )

            qutip_mean_photons = qutip_mean_photon_number(qutip_state)
            signed_error = qutip_mean_photons - target_mean_photon_number
            targets_per_code[code_type].append(target_mean_photon_number)
            errors_per_code[code_type].append(signed_error)

    plt.figure(figsize=(10, 6))
    line_styles = {
        'cat': '-',
        'squeeze': '--',
        'binomial': '-.',
        'gkp': ':',
    }
    markers = {
        'cat': 'o',
        'squeeze': 's',
        'binomial': '^',
        'gkp': 'D',
    }

    for code_type in code_types:
        if len(targets_per_code[code_type]) == 0:
            continue
        plt.plot(
            targets_per_code[code_type],
            errors_per_code[code_type],
            label=f"{code_type}",
            linestyle=line_styles[code_type],
            marker=markers[code_type],
            markersize=4,
            linewidth=2.0,
            markevery=2,
        )

    plt.axhline(0.0, color='black', linestyle=':', linewidth=1.2, alpha=0.8)
    plt.xlabel("Target Mean Photon Number")
    plt.ylabel("Qutip Mean - Target Mean")
    plt.title("Parameter Solver Error vs Qutip Mean Photon Number")
    plt.legend()
    plt.grid(True, alpha=0.35)
    plt.show()

    print("Done.")

    

if __name__ == "__main__":
    # _test1()
    # _test2_squeezed_codes()
    # _test3_infinite_vs_finite_series()
    # _test4_cat_state()
    # _test5_get_parameter_for_given_mean_photons()
    # _test6_plot_mean_photons_params_for_different_codes()
    # _test7_test_binomial_code()
    _test8_parameter_solver_error_vs_qutip_mean()

    print("Done.")

