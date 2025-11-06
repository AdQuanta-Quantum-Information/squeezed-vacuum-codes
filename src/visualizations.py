
from typing import TypedDict, Sequence
from qutip import Qobj, basis, expect

from matplotlib.figure import Figure
from matplotlib.axes import Axes

import numpy as np


if __name__ == "__main__":
	from pathlib import Path
	import sys
	sys.path.append(str(Path(__file__).parents[3]))

# Visuals:
from src.utils.visuals import matplotlib_support
from src.quantum.visualizations.wigner_function import plot_plain_wigner 

from projects.controlled_squeezing.src.measurements import measure_qubit_state, _MeasureStats

from projects.controlled_squeezing.globals import Globals


class VisualizationResDict(TypedDict):
    fig : Figure
    axes : Sequence[Axes]


def plot_light_states(branches:list[_MeasureStats]|list[Qobj], _draw_now:bool=True, num_points:int|None=None) -> VisualizationResDict:
    ncols = len(branches)
    fig, axes = matplotlib_support.new_figure(ncols=ncols)
    fig.set_figwidth(10)
    for i, info in enumerate(branches):
        if isinstance(info, Qobj):
            state = info
            title = f"{i}"
        elif isinstance(info, dict):
            prob = info['prob']
            purity = info['purity']
            state = info['state']
            title = f"prob: {prob}\nPurity: {purity:.2f}"
        if num_points is None:
             other_kwargs = {}
        else:
            other_kwargs = {'num_points': num_points}
        plot_plain_wigner(state, ax=axes[i], title=title, **other_kwargs)

    if _draw_now:
        matplotlib_support.draw_now()

    return {'fig': fig, 'axes': axes}



def plot_fock_distribution(ψ:Qobj, _draw_now:bool=True, title:str|None=None) -> VisualizationResDict:

    ## Create figure:
    fig, axes = matplotlib_support.new_figure()
    fig.set_figwidth(10)
    ax = axes[0]
    
    ## Extract probabilities:
    if ψ.isket:
        fock_vec = ψ.full().real.flatten()
        amp = abs(fock_vec)
        probs : np.ndarray = amp**2
    else:
        ρ = ψ
        ## Get exepectation values on fock states:
        N = ρ.shape[0]
        # Create list of projection operators
        projectors = [basis(N, n) * basis(N, n).dag() for n in range(N)]
        # Calculate all expectations at once
        probs : np.ndarray = expect(projectors, ρ)  #type: ignore
        amp = np.sqrt(probs)

    assert np.isclose(sum(probs), 1.0)  # sanity check

    fontdict = {'fontsize': 16}
    ax.bar(range(len(amp)), amp)
    ax.set_yscale('log')
    xlabel = r'Fock state $|n\rangle$' if Globals.LaTeX_RENDERING else 'Fock state |n⟩'
    ylabel = r'amplitude $|\langle n|\psi\rangle|$' if Globals.LaTeX_RENDERING else 'amplitude |⟨n|ψ⟩|'
    ax.set_xlabel(xlabel, **fontdict)
    ax.set_ylabel(ylabel, **fontdict)

    if title is not None:
        ax.set_title(title)

    if _draw_now:
        matplotlib_support.draw_now()

    return {'fig': fig, 'axes': axes}