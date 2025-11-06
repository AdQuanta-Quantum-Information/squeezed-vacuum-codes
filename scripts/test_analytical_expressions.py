import numpy as np
from qutip import Qobj

from typing import Final

from sympy.physics.quantum import Ket   
import sympy as sp

import functools


## Visualization
import matplotlib.pyplot as plt


if __name__ == "__main__":
    from __init__ import add_root_to_path
    add_root_to_path()

from projects.controlled_squeezing.src.analytical_expressions import analytical_code_word
from projects.controlled_squeezing.src.codes_built_in_superposition import simple_m_legged_code


NUM_MOMENTS : Final[int] = 60  # number of moments in the squeezed state


@functools.cache
def _our_qutip_codewords(m:int, r:float) -> tuple[Qobj, Qobj]:
    ψ0, ψ1 = simple_m_legged_code(m, r, num_moments=NUM_MOMENTS, code_type="squeeze")
    return ψ0, ψ1


def our_qutip_codeword(m:int, r:float, l:int) -> Qobj:
    return _our_qutip_codewords(m, r)[l]


def _get_analytical_fock_coeff(state: sp.Expr, n:int) -> complex:
    d = state.as_coefficients_dict()
    coeff = d.get(Ket(n), 0)
    if isinstance(coeff, (sp.Number, sp.Expr)):
        coeff_val = coeff.evalf()
    elif isinstance(coeff, (int, float, complex)):
        coeff_val = coeff
    else:
        raise TypeError(f"Unexpected type for coefficient: {type(coeff)!r}")

    return complex(coeff_val)


def _get_qutip_fock_coeff(qutips, n) -> complex:
    v = qutips.full()  # (dim,1)
    coeff_qutip = v[n, 0]
    return complex(coeff_qutip)


def compute_logical_codeword_error(
    m:int,  # number of legs
    r:float,  # squeezing magnitude
    l:int,  # logical index
    plot:bool=True,
) -> float:

    # Get the analytical Fock representation of the squeezed vacuum state
    analytical = analytical_code_word(m, r, l, num_moments=NUM_MOMENTS)

    # Get the Fock representation from qutip operations: 
    qutips = our_qutip_codeword(m, r, l)


    a_ceoffs = [_get_analytical_fock_coeff(analytical, n) for n in range(NUM_MOMENTS)]
    b_ceoffs = [_get_qutip_fock_coeff(qutips, n) for n in range(NUM_MOMENTS)]
    error = np.abs(np.array(a_ceoffs) - np.array(b_ceoffs))

    if plot:
        f1 = plt.figure()
        plt.bar(np.arange(NUM_MOMENTS) - 0.2, np.abs(a_ceoffs), 0.4, label='Analytical', color='blue')
        plt.bar(np.arange(NUM_MOMENTS) + 0.2, np.abs(b_ceoffs), 0.4, label='Qutip', color='orange')

        plt.xlabel('Fock State Index')
        plt.ylabel('Coefficient Amplitude')
        plt.title(f'Fock Coefficients for m={m}, r={r}, l={l}')
        plt.legend()
        plt.show()

        f2 = plt.figure()
        plt.plot(error, marker='o')
        plt.yscale('log')
        plt.xlabel('Fock State Index')
        plt.ylabel('Error')
        plt.title(f'Error in Fock Coefficients for m={m}, r={r}, l={l}')
        plt.show()

    overall_error = np.mean(error)
    return overall_error



def test_codewords(
    max_m:int=8,
    r:float=1.5,
):
    errors = []
    for m in range(2, max_m+1, 2):
        for l in range(2):
            print(f"\nCodeword m={m}, r={r}, l={l}")

            error = compute_logical_codeword_error(m, r, l)
            errors.append(error)
        
            print(f"error = {error:.3e}")

    plt.figure()
    plt.plot(errors, marker='o')
    plt.yscale('log')
    plt.xlabel('Codeword Index')
    plt.ylabel('Mean Absolute Error')
    plt.title('Mean Absolute Error between Analytical and Qutip Codewords')

    print("All codewords match!")


def main():
    test_codewords()

if __name__ == "__main__":
    main()