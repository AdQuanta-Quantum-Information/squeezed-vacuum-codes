from __init__ import import_src
import_src()

## Get stuff from the general src folder
from src.quantum.visualizations.wigner_function import plot_plain_wigner
from src.utils.visuals import matplotlib_support
from src.utils.visuals.matplotlib_support.mpl_types import Figure, Axes

## Math and quantum:
from src.quantum.qutip_support import qutip 
import numpy as np



def empty_wigner_axes(bare:bool=True) -> Axes:
    """
    Create empty wigner axes for plotting.
    """
    matplotlib_support.turn_latex_on()

    fontdict = dict(        
        fontsize=16
    )

    fig, ax = matplotlib_support.new_figure()
    ax.set_xlabel(r"$\mathrm{Re}(\beta)$", fontdict=fontdict)
    ax.set_ylabel(r"$\mathrm{Im}(\beta)$", fontdict=fontdict)
    ax.set_xticks([-5, 0, 5])
    ax.set_yticks([-5, 0, 5])
    ax.set_aspect('equal', adjustable='box')  # Force 1:1 aspect ratio

    ## if bare then no boundary square for axis and a single x=0 arrow and y=0 arrow:
    if bare:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        ax.xaxis.set_ticks_position('none')  # no ticks
        ax.yaxis.set_ticks_position('none')  # no ticks

        # Draw the x and y axes
        ax.axhline(0, color='black', lw=1)
        ax.axvline(0, color='black', lw=1)

    matplotlib_support.draw_now()

    return ax


def vacuum_wigner() -> Axes:
    """
    Plot the Wigner function of the vacuum state.
    """
    vac = qutip.basis(20, 0)  # Vacuum state |0>
    out = plot_plain_wigner(vac)
    return out['ax']


def squeezed_vacuum_wigner(
    gamma:float = 1.0,
    theta:float = np.pi/2,
    n:int = 0,
    num_moments:int = 50
) -> Axes:
    """
    Plot the Wigner function of the vacuum state.
    """
    num_moments = 50
    fock_n = qutip.basis(num_moments, 0)  # Vacuum state |0>
    squeezing_angle_phase = np.exp(1j * theta)
    squeezing = qutip.squeeze(num_moments, squeezing_angle_phase*gamma)  #
    squeezed = squeezing @ fock_n  # Apply squeezing operator
    out = plot_plain_wigner(squeezed)
    matplotlib_support.draw_now()
    return out['ax']


def main():
    # empty_wigner_axes()
    # vacuum_wigner()
    squeezed_vacuum_wigner()

    pass


if __name__ == "__main__":
    main()