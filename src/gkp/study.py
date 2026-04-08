"""Utilities and imports for interactive GKP studies.

This module intentionally gathers the common scientific stack and GKP helpers
in one place so notebooks/scripts can import from here and start quickly.
"""

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from typing import Iterable, Literal


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
from src.quantum.quantum_information import (
    gram_schmidt_orthonormal_pair,
    lowdin_orthonormal_pair,
)
from src.quantum.visualizations.orthonormalization_comparison import (
    compare_orthonormalization_methods_for_states,
)
from src.visualizations import plot_light_states, plot_fock_distribution



def _logical_pair_and_overlap_abs_from_nbar(
    nbar: float,
    N: int,
    normalize: bool = True,
) -> tuple[qt.Qobj, qt.Qobj, float]:
    """Construct logical states and return (ψ0, ψ1, |<0|1>|)."""
    ψ0, _ = gkp_from_nbar(logical_value=0, N=N, nbar_target=nbar)
    ψ1, _ = gkp_from_nbar(logical_value=1, N=N, nbar_target=nbar)
    if normalize:
        ψ0 = ψ0.unit()
        ψ1 = ψ1.unit()
    overlap = ψ0.overlap(ψ1)
    overlap_abs = float(np.abs(overlap))
    return ψ0, ψ1, overlap_abs


def gkp_orthonormal_pair_from_nbar(
    nbar: float = 2.0,
    N: int = 200,
    method: Literal["gram_schmidt", "lowdin"] = "lowdin",
    verbose: bool = True,
    plot: bool = True,
) -> tuple[qt.Qobj, qt.Qobj, dict]:
    """Build physical GKP pair and return a Gram-Schmidt orthonormalized pair.

    Use this when downstream metrics assume an orthonormal logical basis.
    """
    ψ0, ψ1, physical_overlap_abs = _logical_pair_and_overlap_abs_from_nbar(
        nbar=nbar,
        N=N,
        normalize=True,
    )

    if method == "gram_schmidt":
        ϕ0, ϕ1 = gram_schmidt_orthonormal_pair(ψ0, ψ1)
    elif method == "lowdin":
        ϕ0, ϕ1 = lowdin_orthonormal_pair(ψ0, ψ1)
    else:
        raise ValueError("method must be 'gram_schmidt' or 'lowdin'")

    ortho_overlap_abs = float(np.abs(ϕ0.overlap(ϕ1)))

    diagnostics = {
        "nbar": float(nbar),
        "N": int(N),
        "method": method,
        "physical_overlap_abs": float(physical_overlap_abs),
        "orthonormal_overlap_abs": float(ortho_overlap_abs),
        "n0_physical": float(np.real(qt.expect(qt.num(N), ψ0))),
        "n1_physical": float(np.real(qt.expect(qt.num(N), ψ1))),
        "n0_orthonormal": float(np.real(qt.expect(qt.num(N), ϕ0))),
        "n1_orthonormal": float(np.real(qt.expect(qt.num(N), ϕ1))),
    }

    if verbose:
        print(f"Method = {method}")
        print(f"Physical |<0|1>| = {diagnostics['physical_overlap_abs']:.6g}")
        print(f"Orthonormalized |<0|1>| = {diagnostics['orthonormal_overlap_abs']:.6g}")
        print(
            f"n(physical): ({diagnostics['n0_physical']:.4f}, {diagnostics['n1_physical']:.4f})"
        )
        print(
            f"n(orthonormal): ({diagnostics['n0_orthonormal']:.4f}, {diagnostics['n1_orthonormal']:.4f})"
        )

    if plot:
        viz = plot_light_states([ψ0, ψ1, ϕ0, ϕ1], _draw_now=False)
        fig = viz["fig"]
        axes = viz["axes"]

        fig.suptitle(
            f"GKP Physical vs Orthonormalized Basis ({method}, nbar={nbar:.2f}, N={N})"
        )

        subplot_titles = [
            "|0> physical",
            "|1> physical",
            "|0> orthonormal",
            "|1> orthonormal",
        ]
        for ax, title in zip(axes, subplot_titles):
            ax.set_title(title)

        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
        plt.show()

    return ϕ0, ϕ1, diagnostics


def gkp_compare_orthonormalization_methods(
    nbar: float = 2.0,
    N: int = 200,
    verbose: bool = True,
    plot: bool = True,
) -> dict[str, object]:
    """Compare GS and Lowdin for GKP pair by calling generic state comparison."""
    ψ0, ψ1, _ = _logical_pair_and_overlap_abs_from_nbar(
        nbar=nbar,
        N=N,
        normalize=True,
    )

    results = compare_orthonormalization_methods_for_states(
        ψ0,
        ψ1,
        state_labels=("|0>", "|1>"),
        figure_title=(
            f"GKP basis comparison (nbar={nbar:.2f}, N={N})\n"
            f"physical |<0|1>|={{physical_overlap_abs:.3e}}"
        ),
        verbose=verbose,
        plot=plot,
    )

    return {
        "nbar": float(nbar),
        "N": int(N),
        **results,
    }


def _logical_mean_photon_numbers(ψ0: qt.Qobj, ψ1: qt.Qobj, N: int) -> tuple[float, float]:
    """Return mean photon numbers (n0, n1) for the two logical states."""
    n0 = float(np.real(qt.expect(qt.num(N), ψ0)))
    n1 = float(np.real(qt.expect(qt.num(N), ψ1)))
    return n0, n1




def gkp_overlap(
    nbar:float = 2.0,
    N:int = 100,
    verbose:bool = True,
    plot:bool = True
) -> float:
    # Create states:
    ψ0, ψ1, overlap_abs = _logical_pair_and_overlap_abs_from_nbar(
        nbar=nbar,
        N=N,
        normalize=True,
    )

    if verbose:
        print(f"<0|1> = {overlap_abs}")
    
    ## Check number of photons:
    n0, n1 = _logical_mean_photon_numbers(ψ0, ψ1, N)

    if verbose:
        print(f"n0 = {n0}, n1 = {n1}")

    if plot:
        plot_light_states(
            [ψ0, ψ1]
        )
        plot_fock_distribution(ψ0)
        plot_fock_distribution(ψ1)

    return overlap_abs


def gkp_plot_plus_states(
    nbar_targets: Iterable[float] = (1.0, 2.0, 3.0, 4.0),
    N: int = 100,
) -> None:
    """Plot the |+> GKP state for several target nbar values, keeping track of the actual physical nbar."""
    states = []
    titles = []
    
    n_op = qt.num(N)
    
    for nbar in nbar_targets:
        ψ0, _ = gkp_from_nbar(logical_value=0, N=N, nbar_target=nbar, _prog_bar=False)
        ψ1, _ = gkp_from_nbar(logical_value=1, N=N, nbar_target=nbar, _prog_bar=False)
        ψ_plus = (ψ0 + ψ1).unit()
        
        n_actual = float(np.real(qt.expect(n_op, ψ_plus)))
        states.append(ψ_plus)
        titles.append(f"nbar target: {nbar:.2f}\nnbar actual: {n_actual:.2f}")

    viz = plot_light_states(states, _draw_now=False)
    fig = viz["fig"]
    axes = viz["axes"]
    
    fig.suptitle(f"GKP |+> States (N={N})", y=1.05)    
    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=10)
        
    fig.tight_layout()
    plt.show()



def gkp_overlap_vs_nbar(
    nbar_values:Iterable[float] = tuple(np.linspace(0.5, 10.0, 15)),
    N:int|None = 200,
    adaptive_N:bool = True,
    N_safety_factor:float = 1.5,
    cutoff_probe_factor:float = 1.5,
    plot:bool = True,
    verbose:bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the curve |<0|1>| as a function of nbar.

    Args:
        nbar_values: Iterable of nbar points to evaluate.
        N: Fixed Fock cutoff. If None, uses recommended_N_from_nbar with
           N_safety_factor when adaptive_N=True.
        adaptive_N: Enable per-point cutoff from recommended_N_from_nbar.
        N_safety_factor: Multiplier for adaptive cutoff.
        cutoff_probe_factor: Multiplier for a larger probe cutoff used to
            estimate how sensitive the overlap is to Fock truncation.
        plot: Plot the resulting curve.
        verbose: Print a small table while sweeping.

    Returns:
        A tuple (nbar_array, overlap_array, cutoff_sensitivity_pct), where
        cutoff_sensitivity_pct is the percent change in |<0|1>| when N is
        increased to N_probe = ceil(cutoff_probe_factor * N).
    """
    nbar_array = np.asarray(tuple(float(x) for x in nbar_values), dtype=float)
    overlaps = np.zeros_like(nbar_array)
    cutoff_sensitivity_pct = np.zeros_like(nbar_array)

    if nbar_array.size == 0:
        raise ValueError("nbar_values must contain at least one value")
    if cutoff_probe_factor <= 1.0:
        raise ValueError("cutoff_probe_factor must be > 1.0")

    if verbose:
        print(f"{'nbar':>8} {'N':>6} {'|<0|1>|':>12} {'n0':>10} {'n1':>10} {'dN%':>8}")
        print("-" * 61)

    for idx, nbar in enumerate(nbar_array):
        if N is not None:
            N_here = int(N)
        elif adaptive_N:
            N_rec = recommended_N_from_nbar(nbar)
            N_here = int(np.ceil(N_safety_factor * float(N_rec)))
        else:
            N_here = 100

        ψ0, ψ1, overlap_abs = _logical_pair_and_overlap_abs_from_nbar(
            nbar=nbar,
            N=N_here,
            normalize=True,
        )
        overlaps[idx] = overlap_abs

        # Cutoff quality proxy: how much overlap changes at a larger Hilbert cutoff.
        N_probe = int(np.ceil(cutoff_probe_factor * N_here))
        _, _, overlap_probe = _logical_pair_and_overlap_abs_from_nbar(
            nbar=nbar,
            N=N_probe,
            normalize=True,
        )
        denom = max(overlap_probe, 1e-14)
        cutoff_sensitivity_pct[idx] = 100.0 * abs(overlap_probe - overlap_abs) / denom

        if verbose:
            n0, n1 = _logical_mean_photon_numbers(ψ0, ψ1, N_here)
            print(
                f"{nbar:8.3f} {N_here:6d} {overlap_abs:12.6f} "
                f"{n0:10.4f} {n1:10.4f} {cutoff_sensitivity_pct[idx]:8.2f}"
            )

    if plot:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(nbar_array, overlaps, marker="o", linewidth=1.8)
        ax.set_yscale("log")
        
        ax.set_xlabel("nbar")
        ax.set_ylabel("|<0|1>|")
        ax.set_title("GKP logical overlap vs nbar")
        ax.grid(alpha=0.35)

        ax2 = ax.twinx()
        ax2.plot(
            nbar_array,
            cutoff_sensitivity_pct,
            marker="s",
            linestyle="--",
            linewidth=1.4,
            color="tab:red",
            alpha=0.9,
        )
        ax2.set_ylabel("Cutoff sensitivity dN%")

        fig.tight_layout()

    return nbar_array, overlaps, cutoff_sensitivity_pct





if __name__ == "__main__":
    # gkp_overlap()
    # gkp_overlap_vs_nbar()   
    # gkp_compare_orthonormalization_methods()
    gkp_plot_plus_states()