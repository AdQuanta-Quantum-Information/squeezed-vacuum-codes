#!/usr/bin/env python3
"""
Test script to demonstrate why stronger squeezing leads to worse performance.
"""

import sys
sys.path.append('.')

import numpy as np
import matplotlib.pyplot as plt
from projects.controlled_squeezing.src.cost_functions import (
    _get_forced_superposition_code_states_using_classes, kl_cost, fock_operators, 
    _kraus_dephasing_second_order, _kraus_loss_second_order
)

def analyze_squeeze_state_quality():
    """Analyze how squeeze code quality degrades with stronger squeezing."""
    
    dim = 150
    gamma = 1e-8
    
    # Test range of squeezing strengths
    r_values = np.linspace(0.1, 2.0, 20)
    
    results = {
        'r': [],
        'overlap': [],
        'norm0': [],
        'norm1': [],
        'kl_loss': [],
        'kl_dephasing': []
    }
    
    a, adag = fock_operators(dim)
    kraus_loss = _kraus_loss_second_order(gamma, a, adag)
    kraus_deph = _kraus_dephasing_second_order(gamma, a, adag)
    
    print("Analyzing squeeze code quality vs squeezing strength...")
    print("r\t\tOverlap\t\tNorm0\t\tNorm1\t\tKL_loss\t\tKL_deph")
    print("-" * 80)
    
    for r in r_values:
        try:
            # Generate squeeze code states
            psi0, psi1 = _get_forced_superposition_code_states_using_classes(
                "squeeze", amplitude=r, num_moments=dim, num_legs=2
            )
            
            # Check state quality
            overlap = abs(psi0.overlap(psi1))
            norm0 = psi0.norm()
            norm1 = psi1.norm()
            
            # Compute KL costs
            kl_loss = kl_cost(psi0, psi1, kraus_loss)
            kl_deph = kl_cost(psi0, psi1, kraus_deph)
            
            results['r'].append(r)
            results['overlap'].append(overlap)
            results['norm0'].append(norm0)
            results['norm1'].append(norm1)
            results['kl_loss'].append(kl_loss)
            results['kl_dephasing'].append(kl_deph)
            
            print(f"{r:.3f}\t\t{overlap:.6f}\t{norm0:.6f}\t{norm1:.6f}\t{kl_loss:.6f}\t{kl_deph:.6f}")
            
        except Exception as e:
            print(f"Error at r={r:.3f}: {e}")
            continue
    
    # Plot results
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    # State quality metrics
    ax1.semilogy(results['r'], results['overlap'], 'ro-', label='|⟨ψ₀|ψ₁⟩|')
    ax1.axhline(y=1e-3, color='gray', linestyle='--', alpha=0.7, label='Threshold')
    ax1.set_xlabel('Squeezing strength r')
    ax1.set_ylabel('State overlap')
    ax1.set_title('Orthogonality vs Squeezing')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(results['r'], results['norm0'], 'bo-', label='||ψ₀||')
    ax2.plot(results['r'], results['norm1'], 'go-', label='||ψ₁||')
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='Unit norm')
    ax2.set_xlabel('Squeezing strength r')
    ax2.set_ylabel('State norm')
    ax2.set_title('Normalization vs Squeezing')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # KL costs
    ax3.semilogy(results['r'], results['kl_loss'], 'ro-', label='Loss channel')
    ax3.set_xlabel('Squeezing strength r')
    ax3.set_ylabel('KL cost')
    ax3.set_title('KL Cost vs Squeezing (Loss)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    ax4.semilogy(results['r'], results['kl_dephasing'], 'bo-', label='Dephasing channel')
    ax4.set_xlabel('Squeezing strength r')
    ax4.set_ylabel('KL cost')
    ax4.set_title('KL Cost vs Squeezing (Dephasing)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('squeeze_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results

def print_physical_explanation():
    """Print explanation of the counterintuitive behavior."""
    
    explanation = """
    WHY STRONGER SQUEEZING LEADS TO WORSE PERFORMANCE:
    
    1. FINITE HILBERT SPACE TRUNCATION:
       - Squeezed states spread over many Fock number states
       - Higher squeezing → higher photon numbers → truncation errors
       - States become non-normalized and non-orthogonal
    
    2. NOISE VULNERABILITY:
       - Squeezed states occupy higher photon number regions  
       - Loss channels: more photons available to be lost
       - Dephasing: affects higher-n Fock states more severely
    
    3. OVER-SQUEEZING EFFECT:
       - There's an optimal squeezing for orthogonality
       - Beyond optimal point: diminishing returns for distinguishability
       - But increased vulnerability to noise
    
    4. PHYSICAL INTUITION:
       - Cat codes: amplitude spread in phase space  
       - Squeeze codes: variance reduction in one quadrature
       - Strong squeezing → extreme variance reduction → fragility
    
    SOLUTIONS:
    - Use larger Hilbert space dimensions
    - Find optimal squeezing parameters
    - Consider adaptive truncation schemes
    - Use error correction on top of the bosonic codes
    """
    
    print(explanation)

if __name__ == "__main__":
    print_physical_explanation()
    results = analyze_squeeze_state_quality()
    
    # Find optimal squeezing
    min_idx = np.argmin(results['kl_dephasing'])
    optimal_r = results['r'][min_idx]
    print(f"\nOptimal squeezing for dephasing: r = {optimal_r:.3f}")
    
    min_idx = np.argmin(results['kl_loss'])
    optimal_r = results['r'][min_idx]
    print(f"Optimal squeezing for loss: r = {optimal_r:.3f}")
