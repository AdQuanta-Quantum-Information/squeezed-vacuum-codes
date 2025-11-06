from typing import Final, Literal

from matplotlib import pyplot as plt
from matplotlib.axes import Axes

import numpy as np

import sympy as sp
π = sp.pi
i = sp.I
from sympy.physics.quantum import Ket
from sympy import pretty_print, print_latex

import functools



if __name__ == "__main__":
    from __init__ import add_root_to_path
    path = add_root_to_path()

from src.utils.visuals.matplotlib_support import save_figure, draw_now
from src.utils.prints import ProgressBar

from src.preparation_circuits import probabilistic_2_legged_code
from src.visualizations import plot_light_states



NUM_MOMENTS : Final[int] = 1_000



def simulated_probability_getting_logical_1(r:float) -> float:
    branches = probabilistic_2_legged_code(r=r, num_moments=NUM_MOMENTS)
    if len(branches) < 2:
        return 0.0    
    return branches[1]["prob"]


ϕ : sp.Symbol = sp.symbols("ϕ", real=True)
r : sp.Symbol = sp.symbols("r", real=True)
n = sp.symbols("n", integer=True)
N = sp.symbols("N", integer=True)


_single_ket_coeff : sp.Expr = sp.Pow(-1, n) * (
        (
            sp.sqrt( 
                sp.factorial(2*n)
            )
        ) / (
            (2**n) * sp.factorial(n) 
        )
    ) * (
        sp.exp(i * n * ϕ) * sp.tanh(r)**n 
    ) / (
        sp.sqrt( sp.cosh(r) )
    )


@functools.cache
def prob_symbolic(L:Literal[0, 1]) -> sp.Expr:
    s_p_per_n = _single_ket_coeff.subs(ϕ, 0)
    s_m_per_n = _single_ket_coeff.subs(ϕ, π)
    if L == 0:
        squeeze_coeff : sp.Expr = (s_p_per_n + s_m_per_n)/2
    elif L == 1:
        squeeze_coeff : sp.Expr = (s_p_per_n - s_m_per_n)/2
    else:
        raise ValueError(f"Logical value {L!r} not supported. Use 0 or 1.")

    bracket_term_per_n = abs(squeeze_coeff)**2
    bracket_term_per_n = bracket_term_per_n.simplify()
    bracket_sum_over_fock_space = sp.Sum(bracket_term_per_n, (n, 0, N))
    bracket_sum_over_fock_space = bracket_sum_over_fock_space.simplify()
    probability_symbolic = bracket_sum_over_fock_space.subs(N, sp.oo).doit()
    probability_symbolic = probability_symbolic.simplify()

    return probability_symbolic


def prob_analytical_evaluated_at(L:Literal[0, 1], r_val:float) -> float:
    probability_symbolic = prob_symbolic(L)
    probability_numeric = probability_symbolic.evalf(subs={r: r_val})
    return float(probability_numeric)  #type: ignore


def verify_sum_to_1(r_val:float=0.2) -> None:
    symbolics : list[sp.Expr] = []
    numerics : list[float] = []

    for logical_int in [0, 1]:
        probability_symbolic = prob_symbolic(logical_int)
        pretty_print(probability_symbolic)
        print_latex(probability_symbolic)
        symbolics.append(probability_symbolic)
        probability_numeric = probability_symbolic.evalf(subs={r: r_val})
        numerics.append(float(probability_numeric))

    sum_symbolic = sum(symbolics).simplify()
    sum_numeric = sum(numerics)

    assert sum_symbolic.equals(1), f"Sum of symbolic probabilities {sum_symbolic} is not equal to 1."
    assert np.isclose(sum_numeric, 1.0), f"Sum of probabilities {sum_numeric} is not close to 1."   

    print("All good! Sum of probabilities is 1.")


def plot_probability_of_preparation_as_function_of_r(
    r_max:float = 10,
    r_num:int = 11
):

    x_analytical = np.linspace(0, r_max, 1001)
    y_analytical_1 = [prob_analytical_evaluated_at(1, r) for r in ProgressBar(x_analytical)]
    y_analytical_0 = [prob_analytical_evaluated_at(0, r) for r in ProgressBar(x_analytical)]

    p1 = plt.plot(x_analytical, y_analytical_1, label="prob 1", color="red", linestyle="--")[0]
    p0 = plt.plot(x_analytical, y_analytical_0, label="prob 0", color="blue", linestyle="--")[0]
    axis : Axes = p0.axes

    r_values = np.linspace(0, r_max, r_num)
    probabilities = [simulated_probability_getting_logical_1(r) for r in ProgressBar(r_values)]
    p = axis.plot(r_values, probabilities, label="simulated prob 1")[0]
    p.set_linewidth(5)

    fontdict = dict(size=14)

    plt.xlabel(r"Squeezing strength $r$", fontdict=fontdict)
    plt.ylabel(r"Probability of getting $|1_L\rangle$", fontdict=fontdict)
    plt.ylim(0, 1)
    plt.xlim(0, r_max)
    plt.tight_layout()
    plt.grid()
    plt.legend()
    plt.show()

    save_figure(file_name="probability_of_preparation", extensions=["png", "pdf"])
    print("Done")


def main():
    # verify_sum_to_1()
    plot_probability_of_preparation_as_function_of_r()

if __name__ == "__main__":
    main()