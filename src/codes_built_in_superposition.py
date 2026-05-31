import numpy as np
from mpmath import mp
from math import comb

import qutip 
from qutip import Qobj, basis, coherent, squeeze


if __name__ == "__main__":
    from __init__ import add_root_to_path
    add_root_to_path()


from src.utils.numerics import force_near_pure_complex
from src.utils.prints import ProgressBar
from src.utils.caches import cache

from src.squeezing_direction import squeezing_direction_to_squeezing_phase
from src._numerics import π, exp

from src.gkp import gkp_from_nbar, get_cached_orthonormal_gkp_states

from globals import Globals


from typing import TypeAlias, Literal, Generator, cast, Final
_CodeTypes : TypeAlias = Literal["cat", "squeeze", "binomial", "gkp"]


if Globals.PRECISE:
    if "auto_tidyup" in qutip.settings.core:
        qutip.settings.core["auto_tidyup"] = False


## whether to apply Löwdin orthogonalization to the GKP states, to make them exactly orthogonal. 
# This is needed for some applications, but not for others. Set to False if you want the "standard" GKP states.
USE_LOWDIN_ORTHOGONALIZATION_FOR_GKP : Final[bool] = True  



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
        |1_L⟩ ∝ Σ_{j=0}^{m-1} e^{-i2πj/m} |α e^{i2πj/m}⟩
        ...
        |L_L⟩ ∝ Σ_{j=0}^{m-1} e^{-i2πjL/m} |α e^{i2πj/m}⟩

    L-legged Squeezed code (Eq. 6 of paper) is..
        |k_L⟩ ∝ Σ_{j=0}^{m-1} e^{-i2πjk/m} S(r, πj/m) |vacuum⟩
    
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

    ## Special case for binomial codes:
    if code_type == "binomial":
        return binomial_code_state(
            m=m, s=s, num_moments=num_moments, qubit_logical_value=qubit_logical_value, num_qudit_values=num_qudit_values,
            _force_normalized=_force_normalized,
            _prog_bar=_prog_bar
        )
    
    if code_type == "gkp":
        assert m == 1, "If we got here with m not 1 for gkp code, this is a bug. (since it's not really an m-legged code)."
        assert num_qudit_values == 2, "Number of qudit values must be exactly 2 for gkp code. No support for qudits yet."
        return gkp_code_state(
            nbar=s, num_moments=num_moments, qubit_logical_value=qubit_logical_value,
            _force_normalized=_force_normalized,
            _prog_bar=_prog_bar
        )



    m_vals = range(m)
    if _prog_bar:
        m_vals = ProgressBar(m_vals, prefix=f"building |{qubit_logical_value}_L⟩  ")

    # Fix: Use the logical value directly, not scaled by m
    k = qubit_logical_value * m // num_qudit_values   # Logical state index
    
    legs = []
    for j in m_vals:

        ## phase and rotation per leg:
        phase = (2*π*j/m) * k  
        phasor = exp(-1j*phase)
        if not Globals.PRECISE:
            phasor = force_near_pure_complex(phasor, threshold=1e-15)  # e^{-i2πjk/m}
        
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
    if code_type != "gkp":
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


def _binomial_code_check_inputs(m, s, num_moments, qubit_logical_value, num_qudit_values, mode, p):
    """Helper to check inputs for binomial_code_state."""
    assert num_qudit_values == 2, "Number of qudit values must be exactly 2. No support for qudits yet."

    N = int(m)
    if qubit_logical_value not in (0, 1):
        raise ValueError("qubit_logical_value must be 0 or 1.")

    if mode == "K":
        K = int(round(s))
        if abs(s - K) > 1e-9:
            raise ValueError("mode='K' requires integer K (pass s as an int).")
    elif mode == "nbar":
        nbar_target = float(s)
        K = max(2, int(round(2.0 * nbar_target / N)))
    else:
        raise ValueError("mode must be 'K' or 'nbar'.")

    n_max = K * N
    if n_max >= num_moments:
        raise ValueError(f"num_moments too small: need > K*N = {n_max}, got {num_moments}.")

    offset = qubit_logical_value
    k_vals = range(offset, K + 1, 2)

    return N, K, n_max, k_vals


def binomial_code_state(
    m: int,
    s: float,
    num_moments: int,
    qubit_logical_value: int = 0,
    num_qudit_values: int = 2,
    *,
    mode: Literal["K", "nbar"] = "nbar",          # "K" or "nbar"
    p: float = 0.5,           # only used if you want biased weights
    _force_normalized: bool = True,
    _prog_bar: bool = True
) -> Qobj:
    
    N, K, n_max, k_vals = _binomial_code_check_inputs(m, s, num_moments, qubit_logical_value, num_qudit_values, mode, p)
    if _prog_bar:
        k_vals = ProgressBar(k_vals, prefix=f"building |{qubit_logical_value}_L⟩  ",
                             expected_end=len(k_vals))

    legs = []
    norm2 = 0.0  # compute sector norm numerically (works for p != 0.5 too)
    for k in k_vals:
        n = k * N
        w = comb(K, k) * (p**k) * ((1 - p)**(K - k))  # binomial pmf (up to normalization)
        amp = np.sqrt(w)
        norm2 += (amp**2)
        legs.append(amp * basis(num_moments, n))

    final_state: Qobj = sum(legs)  # type: ignore
    if _force_normalized:
        final_state = final_state / np.sqrt(norm2)   # exact for our constructed sector
    return final_state


def gkp_code_state(
    nbar: float,
    num_moments: int,
    qubit_logical_value: int = 0,
    *,
    lowdin_orthogonalize: bool = USE_LOWDIN_ORTHOGONALIZATION_FOR_GKP,  
    _force_normalized: bool = True,
    _prog_bar: bool = True
) -> Qobj:
    
    if lowdin_orthogonalize:
        ## If we want a state that is orthogonal to the other logical state, 
        # we need to first create both states, and then apply Löwdin-orthogonalization [1] to them together.
        # This function will still only output one of the states. The partner state is stored in cache and can 
        # be retrieved by calling this function again with the opposite logical value.
        # [1] Löwdin, P.O. J. Chem, Phys. 1950.
        if not _force_normalized:
            raise ValueError("If lowdin_orthogonalize is True, we must also force normalization to be True.")
        
        orthonormal_states = get_cached_orthonormal_gkp_states(
            nbar=nbar, 
            num_moments=num_moments, 
            _prog_bar=_prog_bar
        )
        return orthonormal_states[qubit_logical_value]  



    gkp_state = gkp_from_nbar(
        logical_value=qubit_logical_value,
        N=num_moments,
        nbar_target=nbar,
        return_meta=False,
        _prog_bar=_prog_bar
    )

    if _force_normalized:
        gkp_state.unit(inplace=True)

    return gkp_state


## This is a cached version of get_m_legged_states, defined below.
# This also supports choosing between standard and dual basis states.
@cache(ram=True, disk=False)
def _get_m_legged_states_before_deciding_on_basis(
    m: int, strength: float, num_moments: int, code_type: _CodeTypes,
    normalize_logical_states_before_applying_hadamard: bool
) -> tuple[Qobj, Qobj]:
    ψ0, ψ1 = simple_m_legged_code(m=m, strength=strength, num_moments=num_moments, code_type=code_type, _force_normalized=normalize_logical_states_before_applying_hadamard)
    return ψ0, ψ1


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



def _test():
    from src.visualizations import plot_light_states

    b0, b1 = get_m_legged_states(2, 20.0, 100, "binomial", use_dual_code=False)
    plot_light_states([b0, b1])
    print("Done.")

if __name__ == "__main__":
    _test()