## Math and quantum:
from typing import Literal, overload, Final


import numpy as np

if __name__ == "__main__":
    from __init__ import add_root_to_path
    add_root_to_path()

## Get stuff from the general src folder
from src.quantum.visualizations.wigner_function import plot_plain_wigner
from src.utils.visuals import matplotlib_support
from projects.controlled_squeezing.globals import Globals
from projects.controlled_squeezing.src._numerics import π, exp


@overload
def squeezing_direction_to_squeezing_phase(theta:float, as_exponent:Literal[True]=True) -> complex: ...
@overload
def squeezing_direction_to_squeezing_phase(theta:float, as_exponent:Literal[False]=False) -> float: ...
def squeezing_direction_to_squeezing_phase(theta:float, as_exponent:Literal[False, True]=True) -> complex|float:
    """
    - θ = 0 = π   => squeezing with ϕ=π
    - θ = π/2     => squeezing with ϕ=0
    - θ = π/4     => squeezing with ϕ=3*π/2
    - θ = 3*π/4   => squeezing with ϕ=π/2
    
    Everything else is somewhere in the middle according to this linear relation.
    """
    ## Normalize theta between 0 and π 
    θ = theta % (π) 
    
    φ = ( 2*θ + π ) % (2*π)  # Normalize phase between 0 and 2π
    
    if not as_exponent:
        return φ

    res = exp(1j*φ)
    # Zero-out very small real/imaginary parts

    if not Globals.PRECISE:
        res = complex(np.round(res.real, 15), np.round(res.imag, 15))
        res = complex( np.real_if_close(res) )
        
    return res
    


def _plot_squezed_vacuum(
    squeezing_strength:float = 1.7,
    n:int = 0,
    num_moments:int = 100
) -> None:
    import qutip 

    """
    Plot the Wigner function of the vacuum state.
    """
    num_moments = 50
    for theta in [0, π/4, π/2, 3*π/4, π]:
        label = f"theta={theta/π}π"
        print(label)
        fock_n = qutip.basis(num_moments, n)  # Vacuum state |0>
        squeezing_angle_phase = squeezing_direction_to_squeezing_phase(theta)
        squeezing_param : complex = squeezing_angle_phase*squeezing_strength
        squeezing = qutip.squeeze(num_moments, squeezing_param)  
        squeezed = squeezing @ fock_n  # Apply squeezing operator
        plot_plain_wigner(squeezed, title=label)
        matplotlib_support.draw_now()
    print("Done plotting squeezed vacuum states with different angles.")


def main_example():
    _plot_squezed_vacuum()
    pass

if __name__ == "__main__":
    main_example()