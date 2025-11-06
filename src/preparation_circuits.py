
import numpy as np
sqrt2 = np.sqrt(2)
π = np.pi

from qutip import Qobj, qeye, basis, tensor, sigmax, sigmaz
from qutip import squeeze
H : Qobj = (sigmax() + sigmaz())*(1/sqrt2) 
qubit_0 = basis(2, 0)  # |0>
qubit_1 = basis(2, 1)  # |1>


import matplotlib.pyplot as plt

from dataclasses import dataclass


if __name__ == "__main__":
    from __init__ import add_root_to_path
    add_root_to_path()

from projects.controlled_squeezing.src.squeezing_direction import squeezing_direction_to_squeezing_phase
from projects.controlled_squeezing.src.measurements import _MeasureStats, measure_qubit_state
from projects.controlled_squeezing.src.visualizations import plot_light_states
from src.utils.prints import ProgressBar


NUM_MOMENTS_DEFAULT = 200


@dataclass
class ConditionalSqueezingsFactory():
    num_moments : int   
    num_qubits  : int   
    strength    : float 

    def directional_squeeze(self, θ:float) -> Qobj:
        phase = squeezing_direction_to_squeezing_phase(θ, as_exponent=True)
        squeezing_param = self.strength * phase
        return squeeze(self.num_moments, squeezing_param)  # Squeezing operator for the light mode

    def directed_conditional_squeezing(self, control_qubit_index:int, θ1:float, θ2:float) -> Qobj:
        directions = [θ1, θ2]
        I_qubit = qeye(2)

        qubit_ops : list[list[Qobj]] = [
            [
                qubit_proj if i == control_qubit_index else I_qubit
                for i in range(self.num_qubits)
            ]
            for qubit_proj in [qubit_0.proj(), qubit_1.proj()]
        ] 

        qubit_and_light_ops = [
            tensor(*qubit_ops_row, self.directional_squeeze(direction))
            for direction, qubit_ops_row in zip(directions, qubit_ops)
        ]

        op = qubit_and_light_ops[0] + qubit_and_light_ops[1]  # Sum the operators for the two qubit states

        return op 





def probabilistic_2_legged_code(
    r:float=0.05,
    num_moments:int=NUM_MOMENTS_DEFAULT
) -> list[_MeasureStats]:
    
    qubit = basis(2, 0)
    vacuum = basis(num_moments, 0)

    CS = ConditionalSqueezingsFactory(
        num_moments=num_moments, 
        num_qubits=1,
        strength=r
    )

    I = qeye(num_moments)
    HoI : Qobj = tensor(H, I)

    psi : Qobj = tensor(qubit, vacuum)


    psi = HoI @ psi
    psi = CS.directed_conditional_squeezing(0, 0, π/2) @ psi
    psi = HoI @ psi

    ## Measure the light state:
    branches = measure_qubit_state(psi, return_full_stats=True, light_at="end")
    # plot_light_states(branches)

    return branches



def get_analytical_prob_2_legged_code(r:float, l:int) -> float:
    """Get the analytical probability of measuring logical value `l` in the 2-legged code with squeezing strength `r`."""
    prob = 0.5 + (
        (-1)**l
    ) / (
        2 * np.cosh(r) * np.sqrt(np.tanh(r)**2 + 1)
    ) 

    assert 0 <= prob <= 1, f"Probability out of bounds: {prob}"
    return prob



def main_test(
    r_vals : list[float] = np.arange(0.01, 2.0, 0.01).tolist(),
    logical_values : list[int] = [0, 1]
):
    prob_lists_numerical  : list[list[float]] = [[] for _ in logical_values]
    prob_lists_analytical : list[list[float]] = [[] for _ in logical_values]


    for r in ProgressBar(r_vals, prefix="r: "):
        numerical_results = probabilistic_2_legged_code(r=r)

        for l in logical_values:
            prob_lists_numerical[l].append(numerical_results[l]["prob"])

            analytical_prob = get_analytical_prob_2_legged_code(r, l)
            prob_lists_analytical[l].append(analytical_prob)


    ## Plot comparison:
    # Styles:
    logical_colors = ["tab:blue", "tab:orange"]
    analytical_line_style = "--" 
    numerical_line_style  = "-"


    for l in logical_values:

        plt.plot(
            r_vals, 
            prob_lists_analytical[l], 
            color = logical_colors[l],
            linestyle = analytical_line_style,
            label = f"Logical {l} (analytical)"
        )

        plt.plot(
            r_vals, 
            prob_lists_numerical[l], 
            color = logical_colors[l],
            linestyle = numerical_line_style,
            label = f"Logical {l} (numerical)"
        )

    plt.xlabel("Squeezing strength r")
    plt.ylabel("Probability")
    plt.title("2-legged code: Numerical vs Analytical probabilities")
    plt.legend()
    plt.grid()

    plt.show()
    print("Done.")


if __name__ == "__main__":
    main_test()