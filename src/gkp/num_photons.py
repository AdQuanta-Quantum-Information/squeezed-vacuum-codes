import numpy as np
from matplotlib import pyplot as plt
import qutip as qt

from typing import Iterable, Literal

# Use project-standard path setup
try:
    from src.gkp._bootstrap import add_root_to_path
except ModuleNotFoundError:
    from _bootstrap import add_root_to_path
add_root_to_path()


from src.utils.prints import ProgressBar
from src.codes_built_in_superposition import gkp_code_state
from src.mean_photon_number import qutip_mean_photon_number
from src.gkp.fock_cutoff_recommendation import recommended_N_from_nbar



def _gkp_parameter_solver_error_vs_qutip_mean(
    target_mean_photon_numbers: Iterable[float] = np.linspace(0.1, 5.1, 21),
    N_values: Iterable[int] = (50, 100, 150),
    logical_value: int | Literal['+'] = '+',
) -> None:
    """Evaluate the physical accuracy of the GKP geometric approximation for nbar.
    
    Plots the signed error: (Qutip mean photons) - (target mean photons) 
    using the pure target_mean_photon -> Delta/kappa conversion approximation.
    Iterates over different Fock cutoff dimensions N.
    """
    targets = list(target_mean_photon_numbers)
    N_values = list(N_values)
    
    points_per_N = {N: [] for N in N_values}

    for N in ProgressBar(N_values, prefix="per N:    "):
        for target_mean_photon_number in ProgressBar(targets, prefix="per nbar:"):
            # We explicitly bypass the optimizer to test the pure scaling formula drop-off.
            parameter = target_mean_photon_number
            
            # Construct the exact QuTiP state given the analytic parameter.
            if logical_value == '+':
                state_0 = gkp_code_state(parameter, N, qubit_logical_value=0, _prog_bar=False)
                state_1 = gkp_code_state(parameter, N, qubit_logical_value=1, _prog_bar=False)
                qutip_state = (state_0 + state_1).unit()
            else:
                qutip_state = gkp_code_state(
                    parameter, N, qubit_logical_value=logical_value, _prog_bar=False
                )

            # Measure actual mean photon number and calculate discrepancy
            qutip_mean_photons = qutip_mean_photon_number(qutip_state)
            signed_error = qutip_mean_photons - target_mean_photon_number
            
            points_per_N[N].append((target_mean_photon_number, signed_error))

    plt.figure(figsize=(10, 6))
    markers = ['o', 's', '^', 'D', 'v', '<', '>']

    for i, N in enumerate(N_values):
        if not points_per_N[N]:
            continue
        x_values, y_values = zip(*points_per_N[N])
        plt.plot(
            x_values,
            y_values,
            label=f"N={N}",
            linestyle=':',
            marker=markers[i % len(markers)],
            markersize=5,
            linewidth=2.0,
            markevery=2,
        )

    plt.axhline(0.0, color='black', linestyle=':', linewidth=1.2, alpha=0.8)
    plt.xlabel("Target Mean Photon Number")
    plt.ylabel("Qutip Mean - Target Mean")
    plt.title(f"GKP Target vs Actual Mean Photon Number\nfor state |{logical_value}⟩")
    plt.legend()
    plt.grid(True, alpha=0.35)
    plt.show()

    print("Done.")

if __name__ == "__main__":
    _gkp_parameter_solver_error_vs_qutip_mean()
