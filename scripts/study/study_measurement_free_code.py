#%%
try:
	from scripts.study._bootstrap import add_root_to_path
except ModuleNotFoundError:
	from _bootstrap import add_root_to_path

add_root_to_path()


import numpy as np
sqrt2 = np.sqrt(2)
π = np.pi
sqrt_π = np.sqrt(π)


from typing import Callable, Literal, TypeAlias, Final, ParamSpec, TypeVar, TypedDict, Optional, overload
from enum import Enum, auto

from qutip import (
	Qobj,
    basis, tensor, qeye, 
    displace, squeeze, destroy, create, position, momentum, squeezing, num,
    sigmax, sigmay, sigmaz, sigmap, sigmam
)
from qutip.qip.operations import ry
from qutip.measurement import measurement_statistics_povm

from src.quantum.qutip_support._common import fock_str
from src.quantum.density_matrices.translations import density_matrix_to_pure_state
from src.utils import tuples, numerics

## Visuals:
from src.utils.visuals import matplotlib_support
from src.quantum.visualizations.wigner_function import plot_plain_wigner 


## For type hinting:
P = ParamSpec('P')
R = TypeVar('R')
CodeOptions : TypeAlias = Literal["squeezed", "displaced"] 
CreationMethod : TypeAlias = Literal["CNOD", "post-selection", "measurement-free"] 
C_kl_ij_Type : TypeAlias = dict[tuple[int, tuple[int, int]], complex]
_QuantumOpType = TypeVar("_QuantumOpType", Qobj, np.ndarray)


class ErrorFamily(Enum):
	PHOTON_LOSS = auto()
	PHOTON_GAIN = auto()
	DISPLACEMENT = auto()
	PHASE_ERROR = auto()

## Constants:
num_moments : Final[int] = 100


def hadamard_transform(version:Literal[1, 2]=1) -> Qobj:
	if version == 1:
		return Qobj([[1, 1], [1, -1]]) / sqrt2
	elif version == 2:
		return (sigmaz()+sigmax()) / sqrt2
H = hadamard_transform()

## look
def conditional_print(is_on:bool, *args, **kwargs) -> None:
	if not is_on:
		return
	for i, arg in enumerate(args):
		if isinstance(arg, Qobj):
			s = fock_str(arg)
			args = tuples.copy_with_replaced_val_at_index(args, i, s)

	print(*args, **kwargs)


def _op_str(op:str, l:int=20) -> str:
	# pad the string with spaces to get a string of `l` length:
	return f"{op:<{l}}"


def print_(*args, **kwargs):
	global _print
	return conditional_print(_print, *args, **kwargs)



class DeterministicParams(TypedDict):
    theta:     float  # in radians, 0 ≤ θ ≤ π
    phi:       float  # in radians, 0 ≤ φ < 2π
    beta_scale: float  # 0 < beta_scale ≤ 1  (default 1)


def _v_k(k:int, N:int) -> float:
	if k == 1:
		return -sqrt_π * np.power(2, N-1)
	elif k > 1:
		return sqrt_π * np.power(2, N-k) 
	else:
		raise ValueError(f"Invalid k={k}, must be k >= 1")
	

def _w_k(k:int, N:int) -> float:
	if k == N:
		return sqrt_π/4
	elif k < N:
		return -sqrt_π/4 * np.power(2.0, -(N-k)) 
	else:
		raise ValueError(f"Invalid k={k}, must be k >= 1")


def op_to_unitary(op: Qobj) -> Qobj:
    exponent = 1j*op
    unitary = exponent.expm()
    return unitary  



# Build Hilbert spaces & operators
I_light = qeye(num_moments)
I_qubit = qeye(2)
qubit_0 = basis(2, 0)
qubit_1 = basis(2, 1)
qubit_plus = (qubit_0 + qubit_1)/sqrt2
qubit_minus = (qubit_0 - qubit_1)/sqrt2
vacuum = basis(num_moments, 0)




def _density_matrix_to_pure_state(rho:_QuantumOpType, check:bool=True) -> tuple[_QuantumOpType, float]:
	"""
	Convert a density matrix to its corresponding pure state vector.

	Parameters:
	rho (qutip.Qobj or numpy.ndarray): Density matrix representing a pure state

	Returns:
	numpy.ndarray: Pure state vector
	"""
	# If input is a QuTiP Qobj, convert to numpy array
	if isinstance(rho, Qobj):
		original_type = Qobj
		rho_matrix = rho.full()
	else:
		original_type = np.ndarray
		rho_matrix = np.array(rho)

	# Check if the matrix is Hermitian
	if not np.allclose(rho_matrix, rho_matrix.conj().T):
		raise ValueError("Input matrix is not Hermitian")

	# Compute eigenvalues and eigenvectors
	eigenvalues, eigenvectors = np.linalg.eigh(rho_matrix)

	# Find the eigenvalue close to 1 (for pure states)
	pure_state_index = np.argmin(np.abs(eigenvalues - 1))
	dominant_eigenvalue = eigenvalues[pure_state_index]
    
    # Extract the corresponding eigenvector (pure state)
	if check:
		assert np.isclose(dominant_eigenvalue, 1, rtol=1e-6), "Dominant eigenvalue is not close to 1"
	pure_state : np.ndarray = eigenvectors[:, pure_state_index]

	# Normalize the state vector
	pure_state /= np.linalg.norm(pure_state)

	## Return the pure state in the original type
	if original_type == Qobj:
		dims = (rho.dims[0][0], 1)  #type: ignore
		state_out = Qobj(pure_state, dims=dims)
	else:
		state_out = pure_state
	return state_out, dominant_eigenvalue  #type:ignore

import numpy as np
from typing import TypedDict


class MeasureStats(TypedDict):
	"""
	Type for the light state, which is a tensor product of the light mode and a qubit state.
	"""
	state: Qobj
	prob: float
	purity: float
	qubit_measured: int


@overload
def _measure_qubit_state(psi: Qobj, return_full_stats:Literal[True]) -> list[MeasureStats]: ...
@overload
def _measure_qubit_state(psi: Qobj, return_full_stats:Literal[False]) -> list[Qobj]: ...
def _measure_qubit_state(psi: Qobj, return_full_stats:Literal[False, True]=False) -> list[MeasureStats]|list[Qobj]:
    # Measure qubit & post-select
    ops = [
		tensor(I_light, qubit_0.proj()), 
		tensor(I_light, qubit_1.proj())
	]
    collapsed_states, probabilities = measurement_statistics_povm(psi, ops)

    # Extract logical codewords
    light_states: list[Qobj] = []
    purity_values: list[float] = []
    for state in collapsed_states:
        light, purity = _density_matrix_to_pure_state(state.ptrace(0))
        light_states.append(light)
        purity_values.append(purity)
		
    if not return_full_stats:
        return light_states
	
    res : list[MeasureStats] = [
		MeasureStats(
            state=light_states,
            prob=probabilities,
            purity=purity,
			qubit_measured=i   
        ) for i, (light_states, probabilities, purity) in enumerate(zip(light_states, probabilities, purity_values))
    ]
    return res
		
		
def plot_states(states:list[MeasureStats]) -> None:
	ncols = len(states)
	fig, axes = matplotlib_support.new_figure(ncols=ncols)
	fig.set_figwidth(10)
	for i, st in enumerate(states):
		prob = st['prob']
		purity = st['purity']
		state = st['state']
		title = f"prob: {prob}\nPurity: {purity:.2f}"
		plot_plain_wigner(state, ax=axes[i], title=title)
	matplotlib_support.draw_now()


def _normalize(t:tuple[complex,...]) -> tuple[complex,...]:
	"""Normalize a tuple of complex numbers."""
	norm = np.linalg.norm(t)
	if norm == 0:
		raise ValueError("Cannot normalize a zero vector")
	return tuple(x / norm for x in t)

def measure_and_show_final_combined_state(psi):
	states = _measure_qubit_state(psi, return_full_stats=True)
	ψ0 = states[0]['state']
	ψ1 = states[1]['state']
	print_("")
	print_("|ψ0> = ", ψ0)
	print_("")
	print_("|ψ1> = ", ψ1)
	plot_states(states)

X = position(num_moments)
P = momentum(num_moments)
σ_x = sigmax()
σ_y = sigmay()
σ_z = sigmaz()


# %%  GKP process:

if "GKP Process"==False:
	_print = True
	N = 1
	# c0 = 0   
	# c1 = 1

	squeeze_db = 0  # dB
	squeezed_light = squeeze(num_moments, squeeze_db/10) @ vacuum

	psi = tensor(squeezed_light, qubit_0)
	print_(_op_str("Initial"),      "|ψ>=", psi)

	for k in range(1, N+1):

		## Displacement 
		v_k = _v_k(k, N) 
		Vk = op_to_unitary( v_k * tensor(P, σ_x) )
		psi = Vk @ psi
		print_(_op_str("displacement"),     "|ψ>=", psi)

		## Disentanglement
		w_k = _w_k(k, N)
		Wk = op_to_unitary( w_k * tensor(X, σ_y) )
		psi = Wk @ psi
		print_(_op_str("disentanglement"),   "|ψ>=", psi)
				

	states = _measure_qubit_state(psi, return_full_stats=True)
	# Get state with highest probability:
	# max_prob_state_with_stats = max(states, key=lambda x: x['prob'])
	# prob = max_prob_state_with_stats['prob']
	# purity = max_prob_state_with_stats['purity']
	# state = [max_prob_state_with_stats['state']]

	# Plot:
	plot_states(states)



#%%   2-legged Cat:
if "2-legged Cat"==False:
	_print = True
	alpha = 2.5
	coeffs = (1, 0)

	# ---------------- #


	v1 = alpha
	w1 =  -π / (4*alpha)
	c0, c1 = _normalize(coeffs)
	## Qubit superposition:
	qubit_start   = (c0.conjugate()*qubit_0 + c1.conjugate()*qubit_1).unit()

	psi = tensor(vacuum, qubit_start)
	print_(_op_str("Initial"),      "|ψ>=", psi)


	## Displacement 
	Vk = op_to_unitary( v1 * tensor(P, σ_x) )
	psi = Vk @ psi
	print_(_op_str("displacement"),     "|ψ>=", psi)

	## Disentanglement
	Wk = op_to_unitary( w1 * tensor(X, σ_y) )
	psi = Wk @ psi
	print_(_op_str("disentanglement"),   "|ψ>=", psi)
				
				
	states = _measure_qubit_state(psi, return_full_stats=True)
	# plot_states(states)
	ψ0 = states[0]['state']
	ψ1 = states[1]['state']

	pass
	print("Done.")


# =========================================
#%%   2-legged squeeze:

_print = True
r = 2.5
coeffs = (1, 0)


# ---------------- #


c0, c1 = _normalize(coeffs)
## Qubit superposition:
# qubit_start   = (c0.conjugate()*qubit_0 + c1.conjugate()*qubit_1).unit()
qubit_start = qubit_0

psi = tensor(vacuum, qubit_start)
print_(_op_str("Initial"),  "|ψ>=", psi)
squeezing_op = squeeze(num_moments, r)
squeezing_unitary

Wk = op_to_unitary( t/2 * tensor(X@X, σ_y) )
psi = Wk @ psi
print_(_op_str("disentanglement"),   "|ψ>=", psi)

Vk = op_to_unitary( s * tensor(P@P, σ_x) )
psi = Vk @ psi
print_(_op_str("displacement"),     "|ψ>=", psi)
			
Wk = op_to_unitary( t/2 * tensor(X@X, σ_y) )
psi = Wk @ psi
print_(_op_str("disentanglement"),   "|ψ>=", psi)

I_H = tensor(I_light, H)
psi = I_H @ psi
print_(_op_str("Hadamard"),   "|ψ>=", psi)

measure_and_show_final_combined_state(psi)

pass
print("Done.")




# =========================================
# %% 3-pulse squeezing echo:

_print =  True


# single-mode squeezing generator
Q  = (X*P + P*X)/2

# choose squeezing amount r and matching echo angle ϕ
t  =  2.0                   #   e^r is the stretch factor
s  =  (np.pi/4)/t           #   echo needs  r*ϕ = π/4

# three-pulse echo with Q instead of X² / P²
psi = tensor(vacuum, qubit_0)
U1 = op_to_unitary( 0.5*t * tensor(Q,  σ_y) )
U2 = op_to_unitary(     s * tensor(Q,  σ_x) )
psi = U1 @ U2 @ U1 @ psi

psi = tensor(I_light, σ_y) @ psi

measure_and_show_final_combined_state(psi)
# Now  |ψ> = |even>⊗|0₁>  +  |odd>⊗|1₁>

# 1)---   measure the ancilla **directly in the Z basis** -------------
#        (NO extra R_y rotation!)
branches = _measure_qubit_state(psi, return_full_stats=True)
# branches[0]  → qubit outcome |g〉, oscillator ket  S₊|vac〉
# branches[1]  → qubit outcome |e〉, oscillator ket  S₋|vac〉

# 2)---   build the quarter-period phase-space rotation  ---------------
#        R(π/2) = exp(−i π/2 · n̂)  with  n̂ = a†a
n_op       = num(num_moments)                       # photon-number operator
R_quarter  = op_to_unitary(-0.5 * np.pi * n_op)     # uses your helper

# 3)---   feed-forward: apply R_quarter ONLY on the |e〉 branch ---------
final_light_states = []
for idx, info in enumerate(branches):
    light = info["state"]          # oscillator ket  (Qobj)
    if idx == 1:                  # qubit outcome |e〉  →  state S₋|vac〉
        light = R_quarter @ light #   R(π/2)·S₋|0〉  =  S₊|0〉
    final_light_states.append(light.unit())   # both branches ≡ S₊|vac〉

# --- you now have ONE disentangled squeezed cat ----------------------
ψ0 = final_light_states[0]
ψ1 = final_light_states[1]

# 4)---   quick sanity check (optional) --------------------------------
print_("⟨ψ₀|ψ₁⟩ ≈ ", ψ0.overlap(ψ1) )

# 5)---   visualise ----------------------------------------------------
fig, ax = matplotlib_support.new_figure(ncols=1)
plot_plain_wigner(ψ0, ax=ax, title="deterministic squeezed cat")
matplotlib_support.draw_now()