import numpy as np
import qutip as qt

from typing import Final

sqrt2 = np.sqrt(2)
π = np.pi

import sympy as sp


if __name__ == "__main__":
    from __init__ import add_root_to_path
    path = add_root_to_path()
	
from src.utils.prints import ProgressBar
from src.utils.visuals.matplotlib_support import save_figure, draw_now

from src.codes_built_in_superposition import simple_m_legged_code
from src.visualizations import plot_light_states

## Visuals:
from matplotlib import pyplot as plt
from src.utils.visuals import matplotlib_support

from scripts.deterministic_preparation import from_angels_to_coefficients, get_code_states_using_rotation, LogicalCodewordInfo


from globals import Globals

if Globals.LaTeX_RENDERING:
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Computer Modern Serif']
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'


DEFAULT_NUM_MOMENTS : Final[int] = 100




def logical_codewords(m0:qt.Qobj, m1:qt.Qobj, logical_info:LogicalCodewordInfo) -> tuple[qt.Qobj, qt.Qobj]:
	""" Turn the logical |0⟩, |1⟩ states into the desired logical superposition based on the provided logical_info. """
	a = logical_info.a
	b = logical_info.b
	c0 = a * m0 + b * m1
	
	## C1 state is the orthogonal state to c0:
	c1 = -b.conjugate() * m0 + a.conjugate() * m1
	assert np.isclose(c0.overlap(c1), 0.0)

	return c0, c1


def _pretty_print_symbolic_coefficients(**kwargs):
	for name, value in kwargs.items():
		if isinstance(value, sp.Expr):
			val = sp.simplify(sp.simplify(value))
			val = sp.pretty(val)
		else:
			val = value
		print(f"{name}:\n{val}")


def compare_constructions(
	m:int = 2,
	r:float = 1.5,
	θ:float = π/4,
	φ:float = 0,
	num_moments:int = DEFAULT_NUM_MOMENTS,
	verbose:bool = False
) -> tuple[float, float]:
	
	## Get states:
	a, b = from_angels_to_coefficients(θ, φ)
	logical_info = LogicalCodewordInfo(a=a, b=b)
	ψ0, ψ1 = get_code_states_using_rotation(m, r, logical_info=logical_info, num_moments=num_moments)

	## Compare with simple construction:
	m0, m1 = simple_m_legged_code(m, r, num_moments=num_moments, code_type="squeeze")
	c0, c1 = logical_codewords(m0, m1, logical_info)

	f0 = float(qt.metrics.fidelity(ψ0, c0))
	f1 = float(qt.metrics.fidelity(ψ1, c1))

	if verbose:
		plot_light_states([ψ0, ψ1])
		plot_light_states([c0, c1])
		print(f"fidelity = {[f0, f1]}")
		print("Done.")

	return f0, f1


def plot_construction_prob(
	num_moments:int = 500,
	m_vals:list[int] = [2, 4, 6],
    r_vec = np.linspace(0.1, 3, 31).tolist(),
	θ:float = sp.pi/2,
	φ:float = 0,
	plus_state:bool = True
):

    ## assert inputs
    if plus_state:
        assert θ==sp.pi/2
        assert φ==0.0

    ## requested state coefficients:
    a, b = from_angels_to_coefficients(θ, φ)
    θ, φ = float(θ), float(φ)
    _pretty_print_symbolic_coefficients(a=a, b=b)


    ## Compute:
    per_m_results : dict[int, list[float]] = dict()

    for m in ProgressBar(m_vals, prefix="m: "):
        f0_vec = []
        # f1_vec = []

        for r in ProgressBar(r_vec, prefix="r: "):	
            f0, f1 = compare_constructions(m=m, r=r, θ=θ, φ=φ, num_moments=num_moments)
            f0_vec.append(f0)
            # f1_vec.append(f1)

        per_m_results[m] = f0_vec


    ## Styled fonts:
    plot_kwargs = dict(
        linewidth = 3
	)
	
    if Globals.LaTeX_RENDERING:
        fontsize_label = 16
        fontsize_legend = 12
        xlabel_text = r"$r$"
        ylabel_text = r"Fidelity $|\langle \ell_L|\ell_g\rangle|^2$"
        if plus_state:
            ylabel_text = r"Fidelity $|\langle {+}_L|{+}_g\rangle|^2$"
    else:
        fontsize_label = None
        fontsize_legend = None
        xlabel_text = "r"
        ylabel_text = "Fidelity |<ℓ_L|ℓ_g>|^2"
        if plus_state:
            ylabel_text = "Fidelity |<+_L|+_g>|^2"

    ## Plot:
    linestyles = ["-", "--", "-."]
    for i, (m, f0_vec) in enumerate(per_m_results.items()):
        if Globals.LaTeX_RENDERING:
            label1 = f"$m={m}$"
        else:
            label1 = f"m={m}"
        linestyle = linestyles[i % len(linestyles)]
        plot_kwargs["linestyle"] = linestyle
        p1 = plt.plot(r_vec, f0_vec, label=label1, **plot_kwargs)
        # p2 = plt.plot(r_vec, f2_vec, label=label2, **plot_kwargs)

    plt.xlabel(xlabel_text, fontsize=fontsize_label)
    plt.ylabel(ylabel_text, fontsize=fontsize_label)
        
    # Figure size:
    width = plt.gcf().get_figwidth()
    height = plt.gcf().get_figheight() 
    plt.gcf().set_size_inches(width, height*0.6)
    plt.tight_layout()
    
    plt.legend(fontsize=fontsize_legend, loc="lower right", labelspacing=0.04, handlelength=3.0)
    plt.show()

    file_name = ""\
        + "Fidelity_of_preparation"\
    
    save_figure(plt.gcf(), file_name, dpi=500, transparent=True, extensions=['pdf', 'png', 'svg'])
    print("Saved.")

    print("Done.")


if __name__ == "__main__":
	# compare_constructions(verbose=True)
	plot_construction_prob()