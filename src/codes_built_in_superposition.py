import numpy as np
from mpmath import mp

import qutip 
from qutip import Qobj, basis, coherent, squeeze


if __name__ == "__main__":
    from __init__ import add_root_to_path
    add_root_to_path()


from src.utils.numerics import force_near_pure_complex
from src.utils.prints import ProgressBar

from src.squeezing_direction import squeezing_direction_to_squeezing_phase
from globals import Globals
from src._numerics import π, exp

from src.utils.caches import cache


from typing import TypeAlias, Literal, Generator, cast
_CodeTypes : TypeAlias = Literal["cat", "squeeze"]


if Globals.PRECISE:
    if "auto_tidyup" in qutip.settings.core:
        qutip.settings.core["auto_tidyup"] = False


def _project_ket_to_residue(psi: Qobj, N: int, ell: int) -> Qobj:
    """Keep only Fock indices n ≡ ell (mod 2N); zero everything else, then renormalize."""
    v = psi.full()  # (dim,1)
    dim = v.shape[0]
    mask = np.fromiter(((n % (2*N)) == ell for n in range(dim)), dtype=bool, count=dim)
    v[~mask, 0] = 0.0
    return Qobj(v, dims=psi.dims)


def simple_m_legged_state(
    m: int, s: float, num_moments: int, code_type: _CodeTypes, qubit_logical_value:int=0, num_qudit_values:int=2, 
    _force_normalized:bool=True,
    _prog_bar:bool=True
) -> Qobj:
    """
    Generate m-leg quantum error correction codes.

    For example:
    L-legged Cat code is..
        |0_L⟩ ∝ Σ_{j=0}^{m-1} |α e^{i2πj/m}⟩
        |1_L⟩ ∝ Σ_{j=0}^{m-1} e^{i2πj/m} |α e^{i2πj/m}⟩
        ...
        |L_L⟩ ∝ Σ_{j=0}^{m-1} e^{i2πjL/m} |α e^{i2πj/m}⟩
    
    Parameters:
    -----------
    m : int
        Number of legs in the code
    s : float
        "strength". Code parameter (α for cat codes, r for squeezed codes)
    num_moments : int
        Fock space dimension truncation (number of moments).
    code_type : str
        Type of code: 'cat' or 'squeezed'
    
    Returns:
    --------
    Qobj : The normalized m-leg code state
    """

    ## Checks:
    # assert num_qudit_values == 2, "Number of qudit values must be exactly 2. No support for qudits yet."
    assert m >= 1, "Number of legs must be at least 1."


    m_vals = range(m)
    if _prog_bar:
        m_vals = ProgressBar(m_vals, prefix=f"building |{qubit_logical_value}_L⟩  ")

    # Fix: Use the logical value directly, not scaled by m
    k = qubit_logical_value * m // num_qudit_values   # Logical state index
    
    legs = []
    for j in m_vals:

        ## phase and rotation per leg:
        phase = (2*π*j/m) * k  
        phasor = exp(1j*phase)
        if not Globals.PRECISE:
            phasor = force_near_pure_complex(phasor, threshold=1e-15)  # e^{i2πjk/m}
        
        θj = 2*π*j/m   # rotation angle for this leg
    

        ## Act according to the code type:
        match code_type:
            case 'cat':
                rotation_phase = exp(θj*1j)  # e^{i2πj/m}
                if not Globals.PRECISE:
                    rotation_phase = force_near_pure_complex(rotation_phase, threshold=1e-15)  # e^{i2πj/m}
                    
                leg = coherent(num_moments, s * rotation_phase)

            case 'squeeze':
                θj /= 2  # squeezed states are elongated such that a primitive (leg) occupies theta and theta+π in phase space
                squeeze_phasor = squeezing_direction_to_squeezing_phase(θj, as_exponent=True)
                squeezing_scaled_phasor = s * squeeze_phasor

                vacuum = basis(num_moments, 0)
                squeezing_op = squeeze(num_moments, squeezing_scaled_phasor)

                leg = squeezing_op @ vacuum

            case _:    
                raise ValueError(f"Unknown code_type: {code_type}. Use {_CodeTypes.__args__!r}.")   

        ## Add relative phase for logical-states other than |0_L⟩:
        leg *= phasor 
  
        legs.append(leg)

    ## Combine the legs into a single state:
    final_state : Qobj = sum(legs) #type: ignore


    ## Normalize the final state:
    if _force_normalized:
        try:
            final_state.unit(inplace=True)
        except ZeroDivisionError as e:
            pass  # We got a zero state, so can't normalize

    return final_state


def simple_m_legged_code(
    m: int, strength: float, num_moments: int, code_type: _CodeTypes, 
    num_digits:int=2, 
    _prog_bar:bool=True,
    _force_normalized:bool=True
) -> tuple[Qobj, ...]:
    """
    Generate m-leg quantum error correction code, with both |0⟩ and |1⟩ logical states.
    
    Parameters:
    -----------
    m : int
        Number of legs in the code
    s : float
        "strength". Code parameter (α for cat codes, r for squeezed codes)
    num_moments : int
        Fock space dimension truncation (number of moments).
    code_type : str
        Type of code: 'cat' or 'squeezed'
    num_qubit_values : int
        Number of logical qubit values (default is 2 for |0⟩ and |1⟩) AKA a qubit.
    
    Returns:
    --------
    tuple[Qobj] : The normalized m-leg code states
    """
    assert 1 <= num_digits <= m, "Need 1 ≤ num_digits ≤ m"

    qudit_values = range(num_digits)
    if _prog_bar:
        qudit_values = ProgressBar(qudit_values, prefix="gen codewords   ", expected_end=num_digits)

    states = tuple([
        simple_m_legged_state(
            m, strength, num_moments, code_type, 
            qubit_logical_value=qubit_value, 
            num_qudit_values=num_digits, 
            _prog_bar=_prog_bar,
            _force_normalized=_force_normalized
        )
        for qubit_value in qudit_values
    ])
    return states





@cache(ram=True, disk=False)
def _get_m_legged_states_before_deciding_on_basis(
    m: int, strength: float, num_moments: int, code_type: _CodeTypes,
    normalize_logical_states_before_applying_hadamard: bool
) -> tuple[Qobj, Qobj]:
    ψ0, ψ1 = simple_m_legged_code(m=m, strength=strength, num_moments=num_moments, code_type=code_type, _force_normalized=normalize_logical_states_before_applying_hadamard)
    return ψ0, ψ1

## This is a cached version of get_m_legged_states, defined below.
# This also supports choosing between standard and dual basis states.
def get_m_legged_states(
    m: int, strength: float, num_moments: int, code_type: _CodeTypes, 
    use_dual_code: bool = False, 
    normalize_logical_states_before_applying_hadamard: bool=True
) -> tuple[Qobj, Qobj]:
    """ Return the two logical states of an m-legged code.
    Cached for speed.
    """

    # Start with un-normalized states:
    ψ0, ψ1 = _get_m_legged_states_before_deciding_on_basis(
        m=m, strength=strength, num_moments=num_moments, code_type=code_type,
        normalize_logical_states_before_applying_hadamard=normalize_logical_states_before_applying_hadamard
    )

    if use_dual_code:
        # Dual basis states: |+⟩ = (|0⟩ + |1⟩)/√2 and |−⟩ = (|0⟩ - |1⟩)/√2
        ψ_plus  = (ψ0 + ψ1)/np.sqrt(2)
        ψ_minus = (ψ0 - ψ1)/np.sqrt(2)
        ψ0, ψ1 = ψ_plus, ψ_minus

    # Normalize states:
    if not normalize_logical_states_before_applying_hadamard:
        ψ0.unit(inplace=True)
        ψ1.unit(inplace=True)

    return ψ0, ψ1
