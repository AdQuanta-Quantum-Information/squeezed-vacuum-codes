"""This script generates plots of the overlap between 
simple m-legged squeezed-vacuum codes
as a function of the squeezing parameter `r`.
"""


## Imports:
from typing import Final    

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


if __name__ == "__main__":
    from __init__ import add_root_to_path
    path = add_root_to_path()

from src.utils.visuals.matplotlib_support import save_figure, draw_now
from src.utils.prints import ProgressBar

from src.cost_functions import get_m_legged_states
from src.mean_photon_number import get_mean_photon_number

from src.visualizations import plot_light_states



from globals import Globals

if Globals.LaTeX_RENDERING:
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Computer Modern Serif']
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    TEXT_FONT_SIZE = 16 
else:
    TEXT_FONT_SIZE = 12 

NORMALIZE_LOGICAL_STATES_BEFORE_APPLYING_HADAMARD : Final[bool] = False




def get_overlap_r(
    r:float,
    m:int,
    n_max:int,
    use_dual_code:bool
) -> complex:
    ψ1, ψ2 = get_m_legged_states(
        m, r, n_max, code_type="squeeze", use_dual_code=use_dual_code, 
        normalize_logical_states_before_applying_hadamard=NORMALIZE_LOGICAL_STATES_BEFORE_APPLYING_HADAMARD
    )
    # plot_light_states([ψ1, ψ2])
    overlap = ψ1.overlap(ψ2)
    return overlap


def full_overlap_as_function_of_r_fig(
    n_max:int=300,
    r_max:float=3.0,
    num_r_points:int=51,
    use_dual_code:bool=True,
) -> None:
    r_vec = np.linspace(1e-5, r_max, num_r_points)
    m_list = [2, 4, 6, 8]

    if use_dual_code:
        _ls = ("+","-")
    else:
        _ls = ("0","1")
    

    plt.figure(figsize=(6,4))
    ax: Axes = plt.gca()    
    fig: Figure = ax.get_figure()
    
    if Globals.LaTeX_RENDERING:
        ax.set_xlabel("Squeezing parameter $r$", fontsize=TEXT_FONT_SIZE)
        ax.set_ylabel(r"Overlap $|\langle {%s}_{L}|{%s}_{L} \rangle|$" % (_ls[0], _ls[1]), fontsize=TEXT_FONT_SIZE)
    else:
        ax.set_xlabel("Squeezing parameter r", fontsize=TEXT_FONT_SIZE)
        ax.set_ylabel(f"Overlap |⟨{_ls[0]}|{_ls[1]}⟩|", fontsize=TEXT_FONT_SIZE)
    ax.set_yscale('linear')

    for m in m_list:
        overlap_vec = []
        # print(f"m={m}")

        ## Check mean photon number at max r:
        for logical_value in [0, 1]:
            max_photon_number = get_mean_photon_number(
                "squeeze", m, logical_value=logical_value, parameter=r_max,
                analytic_substitution=False
            )
            print(f"m={m}, logical={logical_value}, r={r_max:.2f} => <n>={max_photon_number:.2f}")
            # ψ1, ψ2 = _get_m_legged_states(m, r_max, n_max, code_type="squeeze", use_dual_code=use_dual_code)
            # plot_light_states([ψ1, ψ2])
            pass


        for r in ProgressBar(r_vec, prefix=f"r: "):
            overlap = get_overlap_r(r, m, n_max=n_max, use_dual_code=use_dual_code)
            overlap_vec.append(abs(overlap))
            
        
        label = f'$m={m}$' if Globals.LaTeX_RENDERING else f'm={m}'
        ax.plot(r_vec, overlap_vec, marker='', label=label)
        
        draw_now(sleep_time=0.5)
        pass
    
    ax.legend(fontsize=TEXT_FONT_SIZE)
    plt.tight_layout()

    print("Saving figure...")
    save_figure(
        fig, f"overlap - dual_basis-{use_dual_code} - normalize_before_hadamard-{NORMALIZE_LOGICAL_STATES_BEFORE_APPLYING_HADAMARD}",
        extensions=["png", "pdf"], dpi=300
    )

    print("Done.")


if __name__ == "__main__":
    full_overlap_as_function_of_r_fig()