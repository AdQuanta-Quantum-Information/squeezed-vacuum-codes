"""Utilities and imports for interactive GKP studies.

This module intentionally gathers the common scientific stack and GKP helpers
in one place so notebooks/scripts can import from here and start quickly.
"""

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from typing import Iterable


try:
    from src.gkp._bootstrap import add_root_to_path
except ModuleNotFoundError:
    from _bootstrap import add_root_to_path
# Use project-standard path setup from src/__init__.py
add_root_to_path()


## GKP-specific imports:
from src.gkp import (
	gkp_params_from_nbar,
	recommended_N_from_nbar,
	gkp_logical,
	gkp_from_nbar,
)
from src.visualizations import plot_light_states, plot_fock_distribution




def gkp_overlap(
    nbar:float = 2.0,
    N:int = 100,
    verbose:bool = True,
) -> float:
    # Create states:
    ψ0, meta0 = gkp_from_nbar(logical_value=0, N=N, nbar_target=nbar)
    ψ1, meta1 = gkp_from_nbar(logical_value=1, N=N, nbar_target=nbar)

    # Compute overlap:
    overlap = ψ0.overlap(ψ1)  # should be small for good GKP approx

    if verbose:
        print(f"<0|1> = {overlap}")
    
    ## Check number of photons:
    n0 = qt.expect(qt.num(N), ψ0)
    n1 = qt.expect(qt.num(N), ψ1)

    if verbose:
        print(f"<0|1> = {overlap}")
        print(f"n0 = {n0}, n1 = {n1}")

    return float(np.abs(overlap))


def gkp_overlap_vs_nbar(
    nbar_values:Iterable[float] = tuple(np.linspace(0.5, 10.0, 15)),
    N:int|None = 200,
    adaptive_N:bool = True,
    N_safety_factor:float = 1.5,
    plot:bool = True,
    verbose:bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the curve |<0|1>| as a function of nbar.

    Args:
        nbar_values: Iterable of nbar points to evaluate.
        N: Fixed Fock cutoff. If None, uses recommended_N_from_nbar with
           N_safety_factor when adaptive_N=True.
        adaptive_N: Enable per-point cutoff from recommended_N_from_nbar.
        N_safety_factor: Multiplier for adaptive cutoff.
        plot: Plot the resulting curve.
        verbose: Print a small table while sweeping.

    Returns:
        A tuple (nbar_array, overlap_array), where overlap_array stores
        |<0|1>| at each nbar point.
    """
    nbar_array = np.asarray(tuple(float(x) for x in nbar_values), dtype=float)
    overlaps = np.zeros_like(nbar_array)

    if nbar_array.size == 0:
        raise ValueError("nbar_values must contain at least one value")

    if verbose:
        print(f"{'nbar':>8} {'N':>6} {'|<0|1>|':>12} {'n0':>10} {'n1':>10}")
        print("-" * 52)

    for idx, nbar in enumerate(nbar_array):
        if N is not None:
            N_here = int(N)
        elif adaptive_N:
            N_rec = recommended_N_from_nbar(nbar)
            N_here = int(np.ceil(N_safety_factor * float(N_rec)))
        else:
            N_here = 100

        ψ0, _ = gkp_from_nbar(logical_value=0, N=N_here, nbar_target=nbar)
        ψ1, _ = gkp_from_nbar(logical_value=1, N=N_here, nbar_target=nbar)

        overlap_abs = float(np.abs(ψ0.overlap(ψ1)))
        overlaps[idx] = overlap_abs

        if verbose:
            n0 = float(np.real(qt.expect(qt.num(N_here), ψ0)))
            n1 = float(np.real(qt.expect(qt.num(N_here), ψ1)))
            print(f"{nbar:8.3f} {N_here:6d} {overlap_abs:12.6f} {n0:10.4f} {n1:10.4f}")

    if plot:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(nbar_array, overlaps, marker="o", linewidth=1.8)
        ax.set_yscale("log")
        
        ax.set_xlabel("nbar")
        ax.set_ylabel("|<0|1>|")
        ax.set_title("GKP logical overlap vs nbar")
        ax.grid(alpha=0.35)
        fig.tight_layout()

    return nbar_array, overlaps





if __name__ == "__main__":
    gkp_overlap()
    gkp_overlap_vs_nbar()   