import numpy as np
from numpy import pi as π

from qutip import Qobj, destroy, coherent, basis, squeeze, mesolve, Options

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from typing import Literal

# ---------------------------------------------------------------------
#  1. Best‑physical times from cavity QED literature
# ---------------------------------------------------------------------
#  Reference: Reagor et al., “Reaching 10 ms single‑photon lifetimes with
#  superconducting aluminum cavities,” Appl. Phys. Lett. 102, 192604 (2013).
best_T1_phys = 10e-3        # 10 ms  — record bare 3D cavity T1
best_T2_phys = best_T1_phys # near sweet‑spot, T2 ≈ T1 for cavities

# ---------------------------------------------------------------------
#  2. Helper: simulate logical lifetime (same as earlier, copied here)
# ---------------------------------------------------------------------
def simulate_logical_lifetime(code_state: Qobj,
                               kappa: float,
                               gamma_phi: float,
                               best_T1: float,
                               best_T2: float,
                               fock_dim: int | None = None,
                               n_steps: int = 50):
    if fock_dim is None:
        fock_dim = code_state.shape[0]
    a = destroy(fock_dim)
    n_op = a.dag() @ a
    parity = _create_parith_op(fock_dim)
    c_ops = [np.sqrt(kappa) * a]
    if gamma_phi > 0:
        c_ops.append(np.sqrt(gamma_phi) * n_op)
    t_max = 2 * max(best_T1, best_T2)
    tlist = np.linspace(0.0, t_max, n_steps)
    rho0 = code_state.proj()
    
    # Store states during evolution
    opts = Options(store_states=True)
    
    result = mesolve(0 * n_op, rho0, tlist, c_ops, [parity], options=opts, progress_bar='tqdm')
    parity_t = result.expect[0]
    final_states = result.states
    fidelity_t = [np.real((code_state.dag() @ rho @ code_state).tr())
                  for rho in final_states]

    def exp_decay(t, T, A, B):
        return A * np.exp(-t / T) + B
    def fit_T(y):
        p0 = [t_max/2, y[0] - y[-1], y[-1]]
        return curve_fit(exp_decay, tlist, y, p0=p0, maxfev=4000)[0][0]

    T_par = fit_T(parity_t)
    T_fid = fit_T(fidelity_t)
    return dict(T_par=T_par, T_fid=T_fid,
                parity=parity_t, fidelity=fidelity_t, t=tlist)

# ---------------------------------------------------------------------
#  3.  m‑leg quantum error correction codes
# ---------------------------------------------------------------------
def m_leg_code_state(m: int, strength: float, num_moments: int, code_type: Literal['cat', 'squeeze'] = 'cat') -> Qobj:
    """
    Generate m-leg quantum error correction codes.
    
    Parameters:
    -----------
    m : int
        Number of legs in the code
    strength : float
        Code parameter (alpha for cat codes, r for squeezed codes)
    num_moments : int
        Fock space dimension
    code_type : str
        Type of code: 'cat' or 'squeezed'
    
    Returns:
    --------
    Qobj : The normalized m-leg code state
    """
    # Cat code: (1/√N) Σ_{k=0}^{m-1} |α e^{i2πk/m}⟩
    legs = []
    for k in range(m):
        phase = np.exp(2j * π * k / m)

        match code_type:
            case 'cat':
                leg = coherent(num_moments, strength * phase)

            case 'squeeze':
                vacuum = basis(num_moments, 0)
                squeezing_op = squeeze(num_moments, strength * phase)
                leg = squeezing_op @ vacuum

            case _:    
                raise ValueError(f"Unknown code_type: {code_type}. Use 'cat' or 'squeezed'.")   
            
        legs.append(leg)

    final_state : Qobj = sum(legs) #type: ignore
    return final_state.unit()


def _create_parith_op(fock_dim) -> Qobj:
    """Create the parity operator for a given Fock space dimension.
    
    The parity operator has eigenvalues (-1)^n for Fock state |n⟩,
    so it's just alternating 1, -1, 1, -1, ... on the diagonal.
    """
    diag_elements = [(-1)**n for n in range(fock_dim)]
    return Qobj(np.diag(diag_elements))



def compare_T1_vs_strength():
    """
    Compare logical lifetimes for different codes with comprehensive time evolution plots.
    Shows both the actual decay curves and fitted lifetime values.
    """
    # ---------------------------------------------------------------------
    #  Setup parameters
    # ---------------------------------------------------------------------
    dim = 100           # Fock cut-off
    
    # Noise rates
    kappa = 1 / best_T1_phys
    gamma_phi = 1 / best_T2_phys
    
    # Define different code configurations: (code_type, num_legs, strength, label, color)
    code_configs = [
        ('cat', 2, 2.0, '2-leg Cat (α=2.0)', 'tab:blue'),
        ('cat', 4, 2.5, '4-leg Cat (α=2.5)', 'tab:cyan'),           
        ('squeeze', 2, 1.0, '2-leg Squeeze (r=1.0)', 'tab:red'),
        ('squeeze', 4, 1.2, '4-leg Squeeze (r=1.2)', 'tab:orange')
    ]
    
    print("Computing time evolution for different code configurations...")
    
    # Helper function to compute full time evolution for a code configuration
    def compute_evolution(code_type, num_legs, strength):
        state = m_leg_code_state(num_legs, strength, dim, code_type)
        result = simulate_logical_lifetime(state, kappa, gamma_phi, 
                                         best_T1_phys, best_T2_phys, dim)
        return result
    
    # ---------------------------------------------------------------------
    #  Create comprehensive plots
    # ---------------------------------------------------------------------
    fig = plt.figure(figsize=(16, 10))
    
    # Create a 2x3 subplot layout
    ax_parity = plt.subplot(2, 3, 1)      # Parity vs time
    ax_fidelity = plt.subplot(2, 3, 2)    # Fidelity vs time  
    ax_both = plt.subplot(2, 3, 3)        # Both overlaid
    ax_T_par = plt.subplot(2, 3, 4)       # T1 parity bar chart
    ax_T_fid = plt.subplot(2, 3, 5)       # T1 fidelity bar chart
    ax_comparison = plt.subplot(2, 3, 6)  # T_par vs T_fid scatter
    
    # Store results for summary plots
    all_results = []
    T_par_values = []
    T_fid_values = []
    labels = []
    colors = []
    
    # Compute and plot time evolution for each code
    for code_type, num_legs, strength, label, color in code_configs:
        print(f"Computing {label}...")
        result = compute_evolution(code_type, num_legs, strength)
        all_results.append(result)
        
        # Extract values
        t = np.array(result['t']) * 1e3  # Convert to ms
        parity = np.array(result['parity'])
        fidelity = np.array(result['fidelity'])
        T_par = float(np.real(result['T_par'])) * 1e3  # Convert to ms
        T_fid = float(np.real(result['T_fid'])) * 1e3  # Convert to ms
        
        T_par_values.append(T_par)
        T_fid_values.append(T_fid)
        labels.append(label)
        colors.append(color)
        
        # Plot time evolution curves
        ax_parity.plot(t, parity, label=label, color=color, linewidth=2)
        ax_fidelity.plot(t, fidelity, label=label, color=color, linewidth=2)
        ax_both.plot(t, parity, label=f'{label} (parity)', color=color, linewidth=2, linestyle='-')
        ax_both.plot(t, fidelity, label=f'{label} (fidelity)', color=color, linewidth=2, linestyle='--')
        
        print(f"  T_parity = {T_par:.2f} ms, T_fidelity = {T_fid:.2f} ms")
    
    # Configure parity plot
    ax_parity.set_xlabel('Time (ms)')
    ax_parity.set_ylabel('Parity ⟨Π⟩')
    ax_parity.set_title('Parity Decay Over Time')
    ax_parity.legend(fontsize=8)
    ax_parity.grid(True, alpha=0.3)
    ax_parity.set_ylim(-1.1, 1.1)
    
    # Configure fidelity plot  
    ax_fidelity.set_xlabel('Time (ms)')
    ax_fidelity.set_ylabel('Fidelity')
    ax_fidelity.set_title('Fidelity Decay Over Time')
    ax_fidelity.legend(fontsize=8)
    ax_fidelity.grid(True, alpha=0.3)
    ax_fidelity.set_ylim(0, 1.1)
    
    # Configure combined plot
    ax_both.set_xlabel('Time (ms)')
    ax_both.set_ylabel('Expectation Value')
    ax_both.set_title('Parity & Fidelity Evolution')
    ax_both.legend(fontsize=6, ncol=2)
    ax_both.grid(True, alpha=0.3)
    ax_both.axhline(y=0.5, color='gray', linestyle=':', alpha=0.7, label='50% threshold')
    
    # Create T1 parity bar chart
    bars_par = ax_T_par.bar(range(len(T_par_values)), T_par_values, 
                           color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    ax_T_par.axhline(y=best_T1_phys * 1e3, color='gray', linestyle='--', 
                    linewidth=2, label='Physical T1')
    ax_T_par.set_xticks(range(len(labels)))
    ax_T_par.set_xticklabels([l.split('(')[0].strip() for l in labels], rotation=45, ha='right')
    ax_T_par.set_ylabel('T₁ Parity (ms)')
    ax_T_par.set_title('Fitted Parity Lifetimes')
    ax_T_par.legend()
    ax_T_par.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars_par, T_par_values):
        ax_T_par.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f'{value:.1f}', ha='center', va='bottom', fontsize=8)
    
    # Create T1 fidelity bar chart
    bars_fid = ax_T_fid.bar(range(len(T_fid_values)), T_fid_values,
                           color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    ax_T_fid.axhline(y=best_T1_phys * 1e3, color='gray', linestyle='--',
                    linewidth=2, label='Physical T1')
    ax_T_fid.set_xticks(range(len(labels)))
    ax_T_fid.set_xticklabels([l.split('(')[0].strip() for l in labels], rotation=45, ha='right')
    ax_T_fid.set_ylabel('T₁ Fidelity (ms)')
    ax_T_fid.set_title('Fitted Fidelity Lifetimes')
    ax_T_fid.legend()
    ax_T_fid.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars_fid, T_fid_values):
        ax_T_fid.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f'{value:.1f}', ha='center', va='bottom', fontsize=8)
    
    # Create T_par vs T_fid comparison scatter plot
    for i, (T_par, T_fid, label, color) in enumerate(zip(T_par_values, T_fid_values, labels, colors)):
        ax_comparison.scatter(T_par, T_fid, color=color, s=100, alpha=0.7, 
                            edgecolor='black', linewidth=1, label=label.split('(')[0].strip())
        # Add text labels next to points
        ax_comparison.annotate(f'{i+1}', (T_par, T_fid), xytext=(5, 5), 
                             textcoords='offset points', fontsize=8, fontweight='bold')
    
    # Add diagonal line (T_par = T_fid)
    min_val = min(min(T_par_values), min(T_fid_values))
    max_val = max(max(T_par_values), max(T_fid_values))
    ax_comparison.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='T_par = T_fid')
    
    ax_comparison.set_xlabel('T₁ Parity (ms)')
    ax_comparison.set_ylabel('T₁ Fidelity (ms)')
    ax_comparison.set_title('Parity vs Fidelity Lifetimes')
    ax_comparison.legend(fontsize=8)
    ax_comparison.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY:")
    print("="*60)
    for i, (label, T_par, T_fid) in enumerate(zip(labels, T_par_values, T_fid_values)):
        enhancement_par = T_par / (best_T1_phys * 1e3)
        enhancement_fid = T_fid / (best_T1_phys * 1e3)
        print(f"{i+1}. {label}")
        print(f"   T₁_parity = {T_par:.2f} ms ({enhancement_par:.1f}× enhancement)")
        print(f"   T₁_fidelity = {T_fid:.2f} ms ({enhancement_fid:.1f}× enhancement)")
        print()


def run_single_T1_test():
    # ---------------------------------------------------------------------
    #  4.  Set up two codes and run simulations
    # ---------------------------------------------------------------------
    dim       = 100         # Fock cut‑off
    alpha     = 2.0         # cat amplitude
    r         = 0.8         # squeezing strength
    m_legs    = 4           # four‑leg code

    # Build code states
    cat_state = m_leg_code_state(m_legs, alpha, dim, 'cat')
    sq_state = m_leg_code_state(m_legs, r, dim, 'squeeze')

    # Noise: choose cavity‑QED like rates (so bare T1, Tφ = best_T1_phys etc.)
    kappa      = 1 / best_T1_phys
    gamma_phi  = 1 / best_T2_phys

    cat_res = simulate_logical_lifetime(cat_state,
                                        kappa, gamma_phi,
                                        best_T1_phys, best_T2_phys,
                                        dim)
    sq_res  = simulate_logical_lifetime(sq_state,
                                        kappa, gamma_phi,
                                        best_T1_phys, best_T2_phys,
                                        dim)

    # ---------------------------------------------------------------------
    #  5.  Plot comparison of logical vs best‑physical lifetimes
    # ---------------------------------------------------------------------
    labels = ['Best T1 (phys)', 'Cat $T_{\\text{parity}}$', 'Sqz $T_{\\text{parity}}$']
    times  = [best_T1_phys,      cat_res['T_par'],           sq_res['T_par']]

    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(range(len(times)), np.array(times)*1e3)   # convert to ms
    ax.set_xticks(range(len(times)), labels, rotation=20)
    ax.set_ylabel("Lifetime (ms)")
    ax.set_title("Logical $T_1$ (parity) vs best physical $T_1$")
    plt.tight_layout()
    plt.show()


    # ---------------------------------------------------------------------
    #  Plot on a single axes both codes' fidelity over time:
    # ---------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(cat_res['t'], cat_res['fidelity'], label='Cat Code Fidelity', 
            color='tab:blue', linewidth=3)
    ax.plot(sq_res['t'], sq_res['fidelity'], label='Squeez, Code Fidelity', 
            color='tab:red', linewidth=3)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Fidelity")
    ax.set_title("Fidelity of Cat and Squeezed Codes Over Time")
    # add the T1 physical to the plot and the T1_logical:
    ax.axhline(y=0.5, color='gray', linestyle='--')
    ax.axvline(x=best_T1_phys, color='green', linestyle='--')

    # Add text annotations directly on the plot
    ax.text(0.02, 0.52, 'Fidelity = 0.5', transform=ax.transAxes, 
            color='gray', fontsize=10, verticalalignment='bottom')
    ax.text(best_T1_phys * 1.05, 0.8, 'Best T1 (phys)', 
            color='green', fontsize=10, rotation=90, verticalalignment='bottom')
    ax.text(0.02, 0.95, 'Cat Code Fidelity', transform=ax.transAxes,
            color='tab:blue', fontsize=10, fontweight='bold')
    ax.text(0.02, 0.90, 'Squeez, Code Fidelity', transform=ax.transAxes,
            color='tab:red', fontsize=10, fontweight='bold')
    
    print("Done.")




        
if __name__ == "__main__":
    # run_single_T1_test()
    
    # Run the new comparison function
    print("\nRunning T1 vs strength comparison...")
    compare_T1_vs_strength()
