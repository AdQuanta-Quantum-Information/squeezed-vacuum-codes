"""
Test hypothesis: GKP codes show constant loss error due to Fock space truncation artifacts.

The reviewer's comment suggests:
- Finite energy GKP codes should be MORE robust to loss than dephasing
- But Figure 7 shows GKP as almost flat (constant) vs loss rate
- This might be numerical: large wavefunction extent in phase space + low Fock cutoff

Hypothesis:
- GKP states at nbar=2 with N=100 Fock states are being severely truncated
- The truncation creates a "noise floor" in the cost metric
- Adding loss noise doesn't increase cost much because truncation already dominates
- Result: looks like "constant function of error rate"

Test approach:
1. Generate GKP states with various N values (100, 200, 400, 800)
2. Measure the cost metric for loss at different γ values
3. Check if the flat line disappears with larger N
4. Check if there are visible truncation issues (tail probability)
"""

if __name__ == "__main__":
    from __init__ import add_root_to_path
    add_root_to_path()

import numpy as np
import matplotlib.pyplot as plt
from typing import cast as type_cast

from src.cost_functions import _compute_cost_given_m_r_and_noise
from src.codes_built_in_superposition import get_m_legged_states
from src.gkp import gkp_from_nbar
from src.utils.prints import ProgressBar


def analyze_gkp_truncation(
    nbar: float = 2.0,
    N_values: list[int] = [100, 200, 400, 800],
    gamma_values: list[float] = None,
    verbose: bool = True
):
    """
    Analyze how Fock space truncation affects GKP cost metrics.
    
    Parameters
    ----------
    nbar : float
        Target mean photon number for GKP codes
    N_values : list[int]
        Fock space cutoffs to test
    gamma_values : list[float]
        Loss rates to evaluate
    verbose : bool
        Print diagnostic information
    
    Returns
    -------
    dict
        Results organized by (N, gamma) pairs
    """
    
    if gamma_values is None:
        gamma_values = np.logspace(-7, -3, 10).tolist()
    
    print(f"\n{'='*80}")
    print(f"ANALYZING GKP TRUNCATION EFFECTS")
    print(f"{'='*80}")
    print(f"Target mean photon number: nbar = {nbar}")
    print(f"Fock cutoffs to test: {N_values}")
    print(f"Loss rates (γ): {[f'{g:.2e}' for g in gamma_values]}")
    print()
    
    results = {}
    tail_probabilities = {}
    
    for N in ProgressBar(N_values, prefix="Testing Fock dimension "):
        ProgressBar.newest().append_extra_str(f"N={N}")
        
        # Generate GKP states
        psi0 = gkp_from_nbar(logical_value=0, N=N, nbar_target=nbar, return_meta=False, _prog_bar=False)
        psi1 = gkp_from_nbar(logical_value=1, N=N, nbar_target=nbar, return_meta=False, _prog_bar=False)
        
        # Normalize
        psi0.unit(inplace=True)
        psi1.unit(inplace=True)
        
        # Measure tail probability (weight above N-10)
        tail_cutoff = max(0, N - 10)
        vec0 = psi0.full().flatten()
        vec1 = psi1.full().flatten()
        tail_prob_0 = np.sum(np.abs(vec0[tail_cutoff:])**2)
        tail_prob_1 = np.sum(np.abs(vec1[tail_cutoff:])**2)
        tail_prob = max(tail_prob_0, tail_prob_1)
        tail_probabilities[N] = tail_prob
        
        if verbose:
            print(f"\n  N = {N}:")
            print(f"    Tail probability (weight in last 10 Fock levels): {tail_prob:.6e}")
        
        # Compute costs at different γ values
        costs_loss = []
        costs_dephasing = []
        
        for gamma in ProgressBar(gamma_values, prefix=f"  different γ     "):
            ProgressBar.newest().append_extra_str(f"γ={gamma:.2e}")
            
            try:
                # Compute loss error cost
                cost_loss = _compute_cost_given_m_r_and_noise(
                    m=1, r=nbar, γ=gamma,
                    num_moments=N,
                    noise_type="loss",
                    noise_method="kraus-KL-style",
                    measurement="overlap01",
                    code_type="gkp",
                    use_dual_code=False
                )
                
                # Compute dephasing error cost (for comparison)
                cost_dephase = _compute_cost_given_m_r_and_noise(
                    m=1, r=nbar, γ=gamma,
                    num_moments=N,
                    noise_type="dephasing",
                    noise_method="kraus-KL-style",
                    measurement="overlap01",
                    code_type="gkp",
                    use_dual_code=False
                )
                
                costs_loss.append(cost_loss)
                costs_dephasing.append(cost_dephase)
                
            except Exception as e:
                if verbose:
                    print(f"    ERROR at γ={gamma}: {e}")
                costs_loss.append(np.nan)
                costs_dephasing.append(np.nan)
        
        results[N] = {
            "gamma_values": gamma_values,
            "costs_loss": costs_loss,
            "costs_dephasing": costs_dephasing,
            "tail_prob": tail_prob
        }
    
    return results, tail_probabilities


def visualize_results(results, tail_probabilities, save_path=None):
    """
    Plot results showing effect of Fock dimension on loss cost metric.
    """
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Loss costs vs γ for different N
    ax = axes[0]
    for N, data in sorted(results.items()):
        gamma_vals = data["gamma_values"]
        costs = data["costs_loss"]
        ax.loglog(gamma_vals, costs, marker='o', label=f'N={N} (tail={data["tail_prob"]:.2e})')
    
    ax.set_xlabel('Loss rate γ')
    ax.set_ylabel('Cost metric V_KL')
    ax.set_title('Loss Error vs Fock Cutoff (GKP Code)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Right: Dephasing costs vs γ for different N
    ax = axes[1]
    for N, data in sorted(results.items()):
        gamma_vals = data["gamma_values"]
        costs = data["costs_dephasing"]
        ax.loglog(gamma_vals, costs, marker='s', label=f'N={N}')
    
    ax.set_xlabel('Dephasing rate γ')
    ax.set_ylabel('Cost metric V_KL')
    ax.set_title('Dephasing Error vs Fock Cutoff (GKP Code)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")
    
    plt.show()
    return fig, axes


if __name__ == "__main__":
    # Run analysis
    results, tail_probs = analyze_gkp_truncation(
        nbar=2.0,
        N_values=[100, 200, 500],  # Start with these
        gamma_values=np.logspace(-7, -3, 8).tolist()
    )
    
    print(f"\n{'='*80}")
    print("SUMMARY: Tail Probabilities (weight in last 10 Fock levels)")
    print(f"{'='*80}")
    for N in sorted(tail_probs.keys()):
        print(f"N={N:4d}: tail_prob = {tail_probs[N]:.6e}  ← HIGH values indicate truncation!")
    
    print(f"\n{'='*80}")
    print("KEY OBSERVATIONS:")
    print(f"{'='*80}")
    print("1. If tail_prob is large (> 1e-3): state is being truncated significantly")
    print("2. If loss curve is flat: the cost metric is stuck at truncation noise floor")
    print("3. If dephasing curve is NOT flat: loss is different from dephasing")
    print("4. If increasing N makes loss curve less flat: confirms truncation hypothesis")
    
    # Visualize
    visualize_results(results, tail_probs, save_path="gkp_truncation_analysis.png")
