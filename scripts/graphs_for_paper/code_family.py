import numpy as np

import matplotlib.pyplot as plt
from matplotlib.axes import Axes


if __name__ == "__main__":
    from __init__ import add_root_to_path
    path = add_root_to_path()

from src.utils.visuals.matplotlib_support import save_figure, draw_now
from src.utils.prints import ProgressBar

from src.quantum.visualizations.wigner_function import plot_plain_wigner 
from src.quantum.qutip_support._common import print_fock, fock_str

from src.codes_built_in_superposition import simple_m_legged_code






r_in_proposal_db = 4   #[dB]
r_in_proposal_strength = r_in_proposal_db / (20 * np.log10(np.e))


COLORLIMS = (-0.2, 0.23)


## Font settings:  Works only if LaTeX is installed
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})

FONT_DICT = {
    "family": "serif",
    "size": 20,
    "weight": "normal"
}




def plot_family(
    max_m:int = 8,
    strength:float = 1.5,  
    fig_size = (12, 6),
    high_resolution:bool = False,
    with_colorbar:bool = False,
    print_states:bool = True,
):
    """
    Plot a 2×((n-1)/2) grid of wigner functions for the simple m-legged code.
    The 1st column shows the two logical code states for the 2-legged code.
    The 2nd column shows the two logical code states for the 3-legged code.
    etc... 

    The grid of wigner functions tights and there is only a single global color bar.
    Also, the x-ticks are needed only on the bottom row, and the y-ticks only on the left column.
    """
    m_vals = list(range(2, max_m+1, 2))
    m_num = len(m_vals)

    if high_resolution:
        num_moments:int = 500
        dpi:int = 800
    else:
        num_moments:int = 100
        dpi:int = 100

    ## Prepare the figure and the grid:
    fig, axes = plt.subplots(nrows=2, ncols=m_num, figsize=fig_size, dpi=dpi)

    for j, m in ProgressBar(enumerate(m_vals), expected_end=m_num):
        # Create the code states
        ψ0, ψ1 = simple_m_legged_code(m=m, strength=strength, num_moments=num_moments, code_type="squeeze")

        if print_states:
            s = f"\nCode states for {m} legs:"
            s += "\n"+fock_str(ψ0, prefix="  |ψ0⟩ = ")
            s += "\n"+fock_str(ψ1, prefix="  |ψ1⟩ = ")
            ProgressBar.newest().append_extra_str(s)

        # Plot the wigner functions
        for i, state in ProgressBar(enumerate([ψ0, ψ1]), expected_end=2):
            ax : Axes = axes[i, j] 

            d = plot_plain_wigner(state, ax=ax, colorlims=COLORLIMS, with_colorbar=with_colorbar)

            # pretty tiks:
            ax.set_title("")
            ax.set_xticks([-5, -2.5, 0, 2.5, 5], labels=["-5", "", "0", "", "5"], minor=False)
            ax.set_yticks([-5, -2.5, 0, 2.5, 5], labels=["-5", "", "0", "", "5"], minor=False)
            # Make the grid slightly transparent
            ax.grid(True, alpha=0.5)
            # Make ticks point inward
            ax.tick_params(direction='in')

            if i == 0:
                # Set y-ticks only on the left column      
                font_dict = FONT_DICT          
                font_dict['size'] = 30
                font_dict['weight'] = 'bold'
                ax.set_title(r"$\mathbf{%s}$\textbf{-legs}"%(m), **font_dict)
                ax.set_xticklabels([])
                ax.set_xlabel("")

            elif i ==1:
                xlabel_str = r"Re$(\alpha)$"
                ax.set_xlabel(xlabel_str, fontdict=FONT_DICT)

            else:
                raise ValueError("This should never happen.")


            # Remove y-ticks for columns other than the first
            if j == 0:
                # ax.set_ylabel(f"|{i}⟩", fontsize=LABEL_FONT_SIZE)
                ylabel_str = r"Im$(\alpha)$"
                ax.set_ylabel(ylabel_str, fontdict=FONT_DICT)
            else:
                ax.set_yticklabels([])
                ax.set_ylabel("")

    fig.tight_layout(pad=0.1)

    print("Saving figure...")
    save_figure(fig, file_name="Code States Grid", extensions=["png", "pdf"], dpi=dpi)
    print("All done!")


def plot_colorbar():
    n=6
    ψ0, ψ1 = simple_m_legged_code(m=6, strength=1.5, num_moments=200, code_type="squeeze")

    ## Plot:
    d = plot_plain_wigner(ψ1, with_colorbar=True, colorlims=COLORLIMS)

    ## Edit the colorbar:
    cb = d["cb"]
    cb.set_ticks([-0.1, -0.05, 0, 0.05, 0.10, 0.15, 0.20])


    save_figure(d['fig'], file_name="Colorbar", extensions=["png", "pdf", "svg"], dpi=800)
    pass


if __name__ == "__main__":
    plot_family()
    plot_colorbar()