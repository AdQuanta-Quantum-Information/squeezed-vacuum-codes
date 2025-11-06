if __name__ == "__main__":
	from pathlib import Path
	import sys
	sys.path.append(str(Path(__file__).parents[3]))


## Get stuff from the general src folder
from src.quantum.states.bosonic.squeezed import SqueezedCoherentState, SqueezedCoherentSuperposition
from src.quantum.states.bosonic.coherent import Displacement
from src.quantum.states.bosonic.qutip_light import LightState
from src.quantum.states._base import IdentityOperator
from src.quantum.states.qudit import QuditX, QuditY, QuditZ, QuditH
from src.quantum.visualizations.wigner_function import plot_plain_wigner
from src.utils.visuals import matplotlib_support, videos

## Get stuff from this project:
from ._modified_classes import _BaseQutritAndLightState, ControlledSqueezing, ControlledDisplacement

## Math and quantum:
from src.quantum.qutip_support import qutip 
import numpy as np
from numpy import pi

## Python types:
from typing import cast, Literal, Type

from qutip import basis, tensor, qeye, squeeze, displace, measurement, rand_ket, Qobj, sigmax, sigmaz



import numpy as np

def _creating_code(
    controlled_gate:Type[ControlledSqueezing]|Type[ControlledDisplacement],
    _print:bool=True,
    strength:complex=2.0,
    starting_light_state:SqueezedCoherentSuperposition|None=None,
    use_qutip:bool=True
) -> tuple[SqueezedCoherentSuperposition, int]:
    def print_(*args, **kwargs):
        if _print:
            print(*args, **kwargs)

    if use_qutip:
        # Define qutip operators
        I = qeye(2)
        X = hadamard_transform(2)
        H = hadamard_transform()
        ControlledGate = squeeze(2, strength) if controlled_gate == ControlledSqueezing else displace(2, strength)
        
        # Create the initial state
        if starting_light_state is None:
            psi = tensor(basis(2, 0), basis(2, 0))
        else:
            psi = tensor(basis(2, 0), starting_light_state)
            
        print_("psi: ", psi)

        # Transform the qutrit from |0> to (|-1> + |1> )/sqrt(2)
        psi = tensor(X, I) * psi
        print_("psi: ", psi)
        psi = tensor(H, I) * psi
        print_("psi: ", psi)
        
        # Apply CS
        psi = ControlledGate * psi
        print_("psi: ", psi)

        # Hadamard again
        psi = tensor(H, I) * psi
        print_("psi: ", psi)

        # Measure the light state
        results = measurement.measure_povm(psi, [qeye(2), qeye(2)])
        print_("results: ", results)
        
        # Get some random result
        result = results[0]
        light = result[0]
        qutrit_value = np.random.choice([0, 1], p=[result[1], 1 - result[1]])
        
        return light, qutrit_value
    else:
        ## Ger Operators:
        I = IdentityOperator[SqueezedCoherentState]()
        X = QuditX(0, -1)
        H = QuditH(-1, 1)
        ControlledGate = controlled_gate(strength)
        M : _BaseQutritAndLightState[LightState] = _BaseQutritAndLightState.qutrit_measurement()
        
        ## Create the tensor-product of the Hilbert spaces:
        if starting_light_state is  None:
            psi = _BaseQutritAndLightState.zero_state()
        else:
            psi = _BaseQutritAndLightState.qutrit_basis(0) & starting_light_state
            psi = _BaseQutritAndLightState.cast_from_tensor_product(psi)
            
        print_("psi: ",psi)

        ## Transform the qutrit from |0> to (|-1> + |1> )/sqrt(2):
        psi = (X & I) @ psi
        print_("psi: ",psi)
        psi = (H & I) @ psi
        print_("psi: ",psi)
        
        ## Apply CS:
        psi = ControlledGate @ psi
        print_("psi: ",psi)

        ## Hadamard again:
        psi = (H & I) @ psi
        print_("psi: ",psi)

        ## Measure the light state:
        results = M.possible_results(psi)
        print_("results: ",results.__repr__())
        # Get some random result:
        result = results.sample()
        light = result.remainder
        qutrit_value = result.option.value
        light : SqueezedCoherentSuperposition = cast(SqueezedCoherentSuperposition, light)

        return light, qutrit_value

def squeezed_code(
    _print:bool=True,
    strength:float=2.0 
):
    return _creating_code(controlled_gate=ControlledSqueezing, _print=_print, strength=strength)


def cat_code(
    _print:bool=True,
    num_moments:int=100,
    strength:float=3.0,
    four_legged:bool=True 
) -> tuple[SqueezedCoherentSuperposition, int]:
    return _creating_code(controlled_gate=ControlledDisplacement, _print=_print, strength=strength)

    # for _ in range(num_tries):
    #     light, measurement = _creating_code(controlled_gate=ControlledDisplacement, _print=_print, strength=strength)
    #     if not four_legged:
    #         return light
    #     elif measurement == -1:
    #         break
    # else:
    #     raise ValueError(f"We couldn't acheive the correct 2-legged cat in {num_tries!r} tries")

    # light, measurement = _creating_code(controlled_gate=ControlledDisplacement, _print=_print, strength=1j*strength, starting_light_state=light)
    # plot_plain_wigner(light.to_qutip(num_moments=num_moments), title="Final Light State")
