"""This script generates plots of the overlap between 
simple m-legged squeezed-vacuum codes
as a function of the squeezing parameter `r`.
"""


## Imports:
from typing import Final, Literal, TypeAlias

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


if __name__ == "__main__":
    from __init__ import add_root_to_path
    path = add_root_to_path()

from src.utils.visuals.matplotlib_support import save_figure, draw_now
from src.utils.prints import ProgressBar

from src.codes_built_in_superposition import get_m_legged_states, _CodeTypes
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

_CodeStatesOptions : TypeAlias = Literal["0_1", "+_-", "0_+"]



def get_overlap_r(
    r:float,
    m:int,
    n_max:int,
    code_states:_CodeStatesOptions,
    code_type:_CodeTypes="squeeze"
) -> complex:
    
    code_states_params = dict(
        m=m, strength=r, num_moments=n_max, code_type=code_type, 
        normalize_logical_states_before_applying_hadamard=NORMALIZE_LOGICAL_STATES_BEFORE_APPLYING_HADAMARD
    )

    ψ0, ψ1 = get_m_legged_states(use_dual_code=False, **code_states_params)
    ψp, ψm = get_m_legged_states(use_dual_code=True,  **code_states_params)

    match code_states:
        case "0_1":
            overlap = ψ0.overlap(ψ1)
        case "+_-":
            overlap = ψp.overlap(ψm)
        case "0_+":
            overlap = ψ0.overlap(ψp)
    # plot_light_states([ψ1, ψ2])
        
    return overlap


def get_overlap_vec_for_code_and_m(
    code_type:_CodeTypes,
    m:int,
    parameter_vec:np.ndarray,
    code_states:_CodeStatesOptions,
    n_max:int
) -> np.ndarray:
    
    overlap_vec = []
    for r in ProgressBar(parameter_vec, prefix=f"r: "):
        overlap = get_overlap_r(r, m, n_max=n_max, code_states=code_states, code_type=code_type)
        overlap_vec.append(abs(overlap))

    return np.array(overlap_vec)        
    

def get_all_overlaps_for_code(
    code_type:_CodeTypes,
    m_list:list[int],
    parameter_vec:np.ndarray,
    code_states:_CodeStatesOptions,
    n_max:int
) -> dict[int, np.ndarray]:

    results = {}
    for m in ProgressBar(m_list, prefix="m: "):
        overlap_vec = get_overlap_vec_for_code_and_m(
            code_type=code_type,
            m=m,
            parameter_vec=parameter_vec,
            code_states=code_states,
            n_max=n_max
        )
        results[m] = overlap_vec

    return results


def full_overlap_as_function_of_r_fig(
    n_max:int=100,
    r_max:float=3.0,
    num_r_points:int=11,
    code_states:_CodeStatesOptions = "0_+",
    code_type:_CodeTypes="cat"
) -> None:
    r_vec = np.linspace(1e-5, r_max, num_r_points)
    m_list = [2, 4, 6]

    results = get_all_overlaps_for_code(
        code_type=code_type,
        m_list=m_list,
        parameter_vec=r_vec,
        code_states=code_states,
        n_max=n_max
    )


    match code_states:
        case "0_1":
            _ls = ("0","1")
        case "+_-":
            _ls = ("+","-")
        case "0_+":
            _ls = ("0","+")

    plt.figure(figsize=(6,4))
    ax: Axes = plt.gca()    
    fig: Figure = ax.get_figure()
    
    if Globals.LaTeX_RENDERING:
        ax.set_xlabel("parameter $r$", fontsize=TEXT_FONT_SIZE)
        ax.set_ylabel(r"Overlap $|\langle {%s}_{L}|{%s}_{L} \rangle|$" % (_ls[0], _ls[1]), fontsize=TEXT_FONT_SIZE)
    else:
        ax.set_xlabel("parameter r", fontsize=TEXT_FONT_SIZE)
        ax.set_ylabel(f"Overlap |⟨{_ls[0]}|{_ls[1]}⟩|", fontsize=TEXT_FONT_SIZE)
    ax.set_yscale('linear')

    for m, overlap_vec in results.items():
        label = f'$m={m}$' if Globals.LaTeX_RENDERING else f'm={m}'
        ax.plot(r_vec, overlap_vec, marker='', label=label)
            
    ax.legend(fontsize=TEXT_FONT_SIZE)
    plt.tight_layout()

    print("Saving figure...")

    file_name = f"overlap - "+\
                f"code-{code_type}"+\
                f"normalize_before_hadamard-{NORMALIZE_LOGICAL_STATES_BEFORE_APPLYING_HADAMARD}"
    save_figure(
        fig, file_name,
        extensions=["png", "pdf"], dpi=300
    )

    print("Done.")


if __name__ == "__main__":
    full_overlap_as_function_of_r_fig()