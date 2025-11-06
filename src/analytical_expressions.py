import sympy as sp
from sympy.physics.quantum import Ket   # optional – lets us attach a |2n⟩ label
import numpy as np

import qutip 

import itertools

from typing import Generator

if __name__ == "__main__":
    from __init__ import add_root_to_path; 
    add_root_to_path()

from projects.controlled_squeezing.src.squeezing_direction import squeezing_direction_to_squeezing_phase
from projects.controlled_squeezing.globals import Globals

from src.utils import assertions
from src.utils.prints import ProgressBar


π = sp.pi
i = sp.I


# Symbols
n = sp.symbols('n', integer=True, nonnegative=True)
r = sp.symbols('r', positive=True)          # real-positive magnitude
φ = sp.symbols('φ', real=True)          # real phase
θ = sp.symbols('θ', real=True)          # real phase
m = sp.symbols('m', integer=True, nonnegative=True)

_term_with_ϕ   = (
    sp.sqrt(sp.factorial(2*n)) /
    (2**n * sp.factorial(n)) *
    (-sp.exp(i*φ) * sp.tanh(r))**n *
    (1 / sp.sqrt(sp.cosh(r)))
) 
single_squeezed_vacuum_state_term = (
    sp.sqrt(sp.factorial(2*n)) /
    (2**n * sp.factorial(n)) *
    (sp.exp(i*2*n*θ)) * (sp.tanh(r)**n) *
    (1 / sp.sqrt(sp.cosh(r)))
)

squeezed_superposition_term = (
    m / sp.sqrt(sp.cosh(r)) * (
        sp.sqrt(
            sp.factorial(2*n)
        )
    ) / (
        (2**n) * sp.factorial(n)
    ) * (
        sp.tanh(r)**n
    )
) 


def remove_zeros(expr: sp.Expr) -> sp.Expr:
    old_coeffs = expr.as_coeff_add()
    new_coeffs = []
    for coeff in old_coeffs[1]:
        amp, fock = coeff.as_coeff_Mul()
        if abs(amp) < 1e-10:  #type:ignore filter out negligible amplitudes
            continue
        new_coeffs.append(coeff)

    fock_sum_no_zeros = sp.Add(*new_coeffs)
    return fock_sum_no_zeros


def fock_rep_of_squeezed_vacuum_in_direction(num_moments:int, r_val:float, θ_val:float) -> sp.Expr:
    # start with an empty sum:
    fock_sum : sp.Expr = 0

    for n_ in ProgressBar.range(num_moments):
        coeff = single_squeezed_vacuum_state_term.subs({n: n_, r: r_val, θ: θ_val})
        fock = coeff * Ket(2*n_)
        fock_sum += fock

    ## substitute values and evaluate
    fock_sum = fock_sum.evalf()
    fock_sum_no_zeros = remove_zeros(fock_sum)

    return fock_sum_no_zeros


def analytic_fock_representation_iterate_fock_nums_and_coeffs(fock_state: sp.Expr) -> Generator[tuple[int, sp.Expr], None, None]:
    for val in fock_state.as_coeff_add()[1]:
        coeff_exp, fock_exp = val.as_coeff_Mul()
        fock = fock_exp.args[0]  
        coeff = coeff_exp
        yield fock, coeff


def _get_coeff_of_fock_representation(fock_state: sp.Expr, fock_num:int) -> sp.Expr:
    for fock, coeff in analytic_fock_representation_iterate_fock_nums_and_coeffs(fock_state):
        if fock == fock_num:
            return coeff
    return sp.S(0)


def _get_max_fox_number(fock_state: sp.Expr) -> int:
    return max(fock for fock, _ in analytic_fock_representation_iterate_fock_nums_and_coeffs(fock_state))


def get_normalization_factor(m:int, r:float, k:int, num_moments:int) -> sp.Expr:
    sum_ : sp.Expr = 0
    k_star = _get_first_nonnegative_k_mod_m(k, m)
    for l in ProgressBar.range(num_moments//m):
        n = l*m + k_star
        term_ : sp.Expr = ( 
            sp.factorial(2*n) 
        ) / ( 
            (4 ** n) * (sp.factorial(n) ** 2) 
        ) * (
            sp.tanh(r) ** (2*n)
        )
        sum_ += term_
    
    sum_ = sum_.evalf().simplify()

    return ( sp.sqrt(sp.cosh(r)) / m ) * (sum_**(-0.5))


def compute_fock_state_norm(state: sp.Expr) -> float:
    sum_ = 0
    for val in state.as_coeff_add()[1]:
        coeff = val.as_coeff_Mul()[0]
        sum_ += sp.Abs(coeff)**2 
    norm_ = sp.sqrt(sum_)
    return float(norm_.evalf())


def _get_first_nonnegative_k_mod_m(k:int, m:int) -> int:
    k_star = (-k) % m
    return k_star


def squeezed_superposition_state(
    m_val:int,  # number of legs
    r_val:float,  # squeezing magnitude
    k_val:int,  # rotation index (0 to m-1)
    num_moments:int=10,  # number of moments in the squeezed state
):
    
    ## Keep all the sums the represent the same overall state.
    # If debug mode is on, we will compare them.
    equal_sums = []
    
    ## Formula 1: Analytical fock representation from paper:
    k_star = _get_first_nonnegative_k_mod_m(k_val, m_val)
    sum1_ : sp.Expr = 0 
    for l in ProgressBar.range(num_moments):
        n_val = l*m_val + k_star
        coeff = squeezed_superposition_term.subs({m: m_val, r: r_val, n: n_val})
        fock_2n = coeff * Ket(2*n_val)
        sum1_ += fock_2n

    equal_sums.append(sum1_)

    if Globals.DEBUG:
    ## Formula 2: By summing legs:
        sum2_ : sp.Expr = 0
        for j in ProgressBar.range(m_val):
            ProgressBar.newest().append_extra_str(f"leg j={j+1}")
            leg_phase = sp.exp(i * 2 * π * j * k_val/ m_val)
            θ = π * j / m_val
            leg = fock_rep_of_squeezed_vacuum_in_direction(num_moments, r_val, θ)
            sum2_ += leg * leg_phase

        equal_sums.append(sum2_)

    normalization_factor = get_normalization_factor(m_val, r_val, k_val, num_moments)

    ## Do a deep check of the analytical expressions:
    equal_states = []
    for i_, sum_ in enumerate(equal_sums):

        state = sum_.evalf()
        state = remove_zeros(state)
        state *= normalization_factor
        state = state.evalf().simplify()

        if Globals.DEBUG:
            norm_ = compute_fock_state_norm(state)
            assert sp.Abs(norm_ - 1) < 1e-4, f"State {i_+1} is not normalized, norm is {norm_}"

        equal_states.append(state)

    if Globals.DEBUG:
        ## Check that all states are equal:
        for state1, state2 in itertools.combinations(equal_states, 2):
            # Get max common fock:
            max_1 = _get_max_fox_number(state1)
            max_2 = _get_max_fox_number(state2)
            common_max_fock = min(max_1, max_2)

            for n_ in range(common_max_fock + 1):
                coeff1 = _get_coeff_of_fock_representation(state1, n_)
                coeff2 = _get_coeff_of_fock_representation(state2, n_)
                diff = sp.Abs(coeff1 - coeff2).evalf()
                assert diff < 1e-5, f"States not equal for m={m_val}, r={r_val}, k={k_val}, num_moments={num_moments}. Diff at |{n}⟩ is {diff}."
                


    return equal_states[0]


def analytical_code_word(
    m:int,  # number of legs
    r:float,  # squeezing magnitude
    l:int,  # logical index
    num_moments=30,  # number of moments in the squeezed state fock representation
):
    if l == 0:
        k = 0
    elif l == 1:
        k = m // 2
    else:
        raise ValueError(f"Logical index l must be 0 or 1 for m={m}, got l={l}.")
    return squeezed_superposition_state(m_val=m, r_val=r, k_val=k, num_moments=num_moments)


def test1_():    
    ## substitute values
    N_ = 10  # Number of moments in the squeezed state
    r_ = 1.5  # Squeezing strength

    s1 = fock_rep_of_squeezed_vacuum_in_direction(N_, r_, 0)
    s2 = fock_rep_of_squeezed_vacuum_in_direction(N_, r_, π/2)

    print(f"(S(0) + S(π/2))|0⟩:", s1 + s2)


def _check_qutip_and_analytical_states_are_equal(analytical:sp.Expr, qutip_state:qutip.Qobj) -> None:
    # Get max fock number:
    max_fock_analytical = _get_max_fox_number(analytical)
    max_fock_qutip = qutip_state.shape[0] - 1
    max_fock = min(max_fock_analytical, max_fock_qutip)

    # qutip data vector
    qutip_data = qutip_state.full()

    # Check each fock component:
    for n_ in ProgressBar.range(max_fock + 1, prefix="Comparing analytical and qutip states: "):
        analytical_coeff = _get_coeff_of_fock_representation(analytical, n_).evalf()
        qutip_coeff = qutip_data[n_, 0]  # get as complex number
        diff = abs(analytical_coeff - qutip_coeff)
        assert diff < 1e-2, f"States differ at |{n_}⟩: analytical={analytical_coeff}, qutip={qutip_coeff}. \ndiff={diff}"


def test2_():
    m_vals = [2, 4, 6, 8]
    r = 1.5
    num_moments = 100

    from projects.controlled_squeezing.src.codes_built_in_superposition import simple_m_legged_state as numeric_squeezed_vacuum_codeword


    for m in ProgressBar(m_vals, prefix="Testing m-legged codewords: "):
        assertions.even(m)
        ProgressBar.newest().append_extra_str(f"m={m}")

        for logical_value in ProgressBar([0, 1], prefix=f"logical values: "):
            ProgressBar.newest().append_extra_str(f"logical_value={logical_value}")

            analytical = analytical_code_word(m, r, l=logical_value, num_moments=num_moments)
            qutip_state = numeric_squeezed_vacuum_codeword(m, s=r, num_moments=num_moments, code_type='squeeze', qubit_logical_value=logical_value)

            _check_qutip_and_analytical_states_are_equal(analytical, qutip_state)
    
    print("Done.")


if __name__ == "__main__":
    # test1_()
    test2_()