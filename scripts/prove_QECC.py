r""" 
To demonstrate that two bosonic states in a single mode of a harmonic oscillator can serve as a quantum error correction code (QECC), one should undertake a systematic approach to establish their efficacy in detecting and correcting quantum errors. Here are the essential steps and criteria to consider:

1. **Error Model Identification**:
	 - One should identify the types of errors most likely to occur in the system. Common quantum errors include bit-flip, phase-flip, and amplitude damping errors. For continuous variables like bosonic states, displacement errors or photon loss should also be considered.
    
2. **Encoding Scheme**:
	 - It is crucial to define how quantum information is encoded into the two bosonic states. The encoding process should map logical qubit states (|0⟩ and |1⟩) into distinct bosonic states to facilitate error detection and correction.

3. **Error Detection**:
	 - One should demonstrate that the encoded states are orthogonal or nearly orthogonal, enabling error detection. This involves minimizing the overlap between the states to distinguish them even in the presence of certain errors.

4. **Error Correction**:
	 - Developing an error correction protocol is necessary to correct the identified errors. This requires designing operations that map erroneous states back to the original encoded states. For bosonic codes, operations such as displacement, squeezing, or using ancillary modes may be employed.

5. **Satisfy the Quantum Error Correction Conditions**:
	 - It is important to verify that the states satisfy the quantum error correction conditions. For a set of errors {E_i} , the conditions are:
		  - <psi_i| E_i^dagger E_j |psi_j> = delta_{ij} c_{ij} 
	 where  |psi_i>   and   |psi_j> are the encoded states, and c_{ij} are constants. This ensures that errors do not overlap between different logical states.

6. **Logical Operations**:
	 - One should demonstrate that logical operations can be performed on the encoded states without leaving the code space. This involves implementing gates that operate directly on the encoded states.

7. **Fidelity Analysis**:
	 - Analyzing the fidelity of the error correction process is essential. One should calculate the probability that the process successfully corrects errors and maintains the integrity of the quantum information.

8. **Experimental Feasibility**:
	 - Consideration of the practical implementation of the QECC is crucial. Evaluating the resources and technology required to realize the code is necessary, such as the need for high-quality oscillators, precise control over quantum operations, and robust error detection mechanisms.

By addressing these steps, one can provide a comprehensive argument that the two bosonic states can function effectively as a quantum error correction code. Emphasizing the theoretical underpinnings with rigorous mathematical proofs and, if feasible, supporting claims with simulations or experimental data that validate theoretical predictions is advisable.
"""


#%%
## In case we run this script not from the root directory, then we need to add the root path to the sys.path:
if __name__ == "__main__":
	from pathlib import Path
	import sys
	sys.path.append(str(Path(__file__).parents[3]))


import numpy as np
sqrt2 = np.sqrt(2)
pi = np.pi

from typing import Callable, Literal, TypeAlias, Final, ParamSpec, TypeVar
from functools import wraps
import itertools
from enum import Enum, auto

from qutip import basis, tensor, qeye, squeeze, displace, measurement, rand_ket, Qobj, sigmax, sigmaz, sigmap, sigmam

# Qutip ladder operator:
from qutip import destroy, create

from src.quantum.qutip_support._common import fock_str
from src.quantum.states.density_matrices.translations import density_matrix_to_pure_state
from src.utils import tuples, numerics

## Visuals:
from src.utils.visuals import matplotlib_support
from src.quantum.visualizations.wigner_function import plot_plain_wigner 
from matplotlib import pyplot as plt

## For type hinting:
P = ParamSpec('P')
R = TypeVar('R')
CodeOptions : TypeAlias = Literal["squeezed", "displaced", "fock"] 
C_kl_ij_Type : TypeAlias = dict[tuple[int, tuple[int, int]], complex]


class ErrorFamiliy(Enum):
	PHOTON_LOSS = auto()
	PHOTON_GAIN = auto()
	DISPLACEMENT = auto()
	PHASE_ERROR = auto()

## Constants:
DEFAULT_NUM_MOMENTS : Final[int] = 20



# Define the Hadamard gate manually
def hadamard_transform(version:Literal[1, 2]=1) -> Qobj:
	if version == 1:
		return Qobj([[1, 1], [1, -1]]) / sqrt2
	elif version == 2:
		return (sigmaz()+sigmax()) / sqrt2


def _the_code(
	num_moments: int = DEFAULT_NUM_MOMENTS,
	strength: float = 2.0,
	what_code: CodeOptions = "squeezed",   
	_print: bool = True,
	_plot: bool = False
) -> list[Qobj]:
	
	## look
	def print_(*args, **kwargs) -> None:
		if not _print:
			return
		for i, arg in enumerate(args):
			if isinstance(arg, Qobj):
				s = fock_str(arg)
				args = tuples.copy_with_replaced_val_at_index(args, i, s)

		print(*args, **kwargs)

	def _op_str(op:str, l:int=20) -> str:
		# pad the string with spaces to get a string of `l` length:
		return f"{op:<{l}}"

	## Prepare the initial state and operators
	I_qubit = qeye(2)
	I_light = qeye(num_moments)
	X = sigmax()
	Z = sigmaz()
	H = hadamard_transform()
	qubit_0 = basis(2, 0)
	qubit_1 = basis(2, 1)
	light_0 = basis(num_moments, 0)
	light_1 = basis(num_moments, 1)	

	match what_code:
		case "squeezed":
			controlled_gate = tensor(squeeze(num_moments, +strength), qubit_0.proj()) +  \
							  tensor(squeeze(num_moments, -strength), qubit_1.proj()) 
			psi_initial = tensor(light_0, qubit_1)
		case "displaced":
			controlled_gate = tensor(displace(num_moments, +strength), qubit_0.proj()) +  \
							  tensor(displace(num_moments, -strength), qubit_1.proj()) 
			psi_initial = tensor(light_0, qubit_0)
		case "fock":
			controlled_gate = tensor(create( num_moments), qubit_0.proj()) +  \
							  tensor(destroy(num_moments), qubit_1.proj()) 
			psi_initial = tensor(light_1, qubit_1)
		case _: 
			raise ValueError("Invalid code type")
		
	# Initial state:
	psi = psi_initial	
	print_(_op_str("Initial"), "psi: ", psi)

	# Transform the qubit from |0> to (|01> + |1> )/sqrt(2)
	psi = tensor(I_light, H) * psi
	print_(_op_str("Hadamard"), "psi: ", psi)

	# Apply CS
	psi = controlled_gate * psi
	print_(_op_str("after CS()"),"psi: ", psi)

	# Hadamard again
	psi = tensor(I_light, H) * psi
	print_(_op_str("Hadamard again"), "psi: ", psi)

	# Measure the light state
	ops = [
		tensor(I_light, qubit_0.proj()),
		tensor(I_light, qubit_1.proj())
	]

	psi.unit(inplace=True)

	collapsed_states, probabilities = measurement.measurement_statistics_povm(psi, ops)
	print_("collapsed_states: ")
	for i, (state, prob) in enumerate(zip(collapsed_states, probabilities)):
		print_(f"    {i}: prob={prob} ", state)


	code = []
	print_("")
	for i, state in enumerate(collapsed_states):
		## The subspace of the qubit is determined by the measuremnet result; unpack the light sub-space:
		state : Qobj
		light = density_matrix_to_pure_state( state.ptrace(0) )
		qubit = density_matrix_to_pure_state( state.ptrace(1) )
		print_(f"logical-state {i}:  ", light)
		code.append(light)

	if _plot:
		fig, ax = matplotlib_support.new_figure()
		axes = fig.subplots(1, 2)
		for i, state in enumerate(code):
			title=f"{i}"
			ax = axes[i]
			plot_plain_wigner(state, ax=ax, title=title)
		matplotlib_support.draw_now()

	return code


def _error_operators(error_type:ErrorFamiliy, num_moments:int) -> list[Qobj]:
	"""
	Generate a family of error operators for a given error type.
	"""
	match error_type:
		case ErrorFamiliy.PHOTON_LOSS:
			return [destroy(num_moments)]
		
		case ErrorFamiliy.PHOTON_GAIN:
			return [create(num_moments)]
		
		case ErrorFamiliy.DISPLACEMENT:
			return [
				displace(num_moments, 0.5), displace(num_moments, -0.5), 
		   		displace(num_moments, +1j*0.5), displace(num_moments, -1j*0.5)
			]
		
		case ErrorFamiliy.PHASE_ERROR:
			raise NotImplementedError("Phase error not implemented yet")
		
		case _:
			raise ValueError("Not recognized error type")


def _check_code_qecc_condition(code:list[Qobj], error_family:ErrorFamiliy, num_moments:int) -> C_kl_ij_Type:
	r""" Verify \( \langle \psi_k | E_i^\dagger E_j | \psi_l \rangle = \delta_{kl} c_{ij} \)
		where \( |\psi_i\rangle \) and \( |\psi_j\rangle \) are the encoded states, and \( c_{ij} \) are constants. 
	"""
	errors = _error_operators(error_family, num_moments=num_moments) 
	n_errors = len(errors)
	n_code = len(code)
	c_kl_ij : C_kl_ij_Type = dict()

	for i, j in itertools.product(range(n_errors), repeat=2):
		ei, ej = errors[i], errors[j]

		for k, l in itertools.product(range(n_code), repeat=2):
			bra_k, ket_l = code[k].dag(), code[l]

			product : complex = bra_k * ei.dag() * ej * ket_l
			product = numerics.force_near_pure_complex(product)
	
			## Verify ⟨ψ_k| E_i† E_j |ψ_l⟩ = δ_{kl} c_{ij}
			if k == l:
				assert not np.isclose(product, 0), f"Error correction conditions not satisfied for {error_family} error"
				c_kl_ij[(k, (i, j))] = product
			else:
				assert np.isclose(product, 0), f"Error correction conditions not satisfied for {error_family} error"

	return c_kl_ij


from qutip import coherent, ket2dm, mesolve, destroy, liouvillian
def _apply_photon_loss_lindbladian(state: Qobj, loss_rate: float, time: float) -> Qobj:
    """
    Apply photon loss using Lindbladian evolution.
    """
    a = destroy(state.shape[0])
    H = 0 * state  # No Hamiltonian
    L = liouvillian(H, [np.sqrt(loss_rate) * a])
    result = mesolve(L, state, [time])
    return result.states[-1]




""" ======== process of proving QECC ========= :"""
" =============================================== "
_WRAPPER_ATTR_NAME : Final[str] = "_proving_QECC_part"


def proving_QECC_part(part:int) -> Callable[[Callable[P, R]], Callable[P, R]]:
	def decorator(func:Callable[P, R]) -> Callable[P, R]:
		@wraps(func)
		def wrapper(*args:P.args, **kwargs:P.kwargs) -> R:
			return func(*args, **kwargs)
		setattr(wrapper, _WRAPPER_ATTR_NAME, part)
		return wrapper	
	return decorator


@proving_QECC_part(1)
def identify_error_model() -> list[ErrorFamiliy]:
	"""
	1. **Error Model Identification**:
		- One should identify the types of errors most likely to occur in the system. Common quantum errors include bit-flip, phase-flip, and amplitude damping errors. For continuous variables like bosonic states, displacement errors or photon loss should also be considered.
	"""
	errors = [
		ErrorFamiliy.PHOTON_LOSS,
		ErrorFamiliy.PHOTON_GAIN
	]
	return errors


@proving_QECC_part(2)
def define_encoding_scheme(what_code:CodeOptions, num_moments:int=DEFAULT_NUM_MOMENTS) -> list[Qobj]:
	"""
	2. **Encoding Scheme**:
		- It is crucial to define how quantum information is encoded into the two bosonic states. The encoding process should map logical qubit states (|0⟩ and |1⟩) into distinct bosonic states to facilitate error detection and correction.
	"""
	kwargs = dict(
		num_moments=num_moments,
		what_code=what_code
	)
	return _the_code(**kwargs)


@proving_QECC_part(3)
def demonstrate_error_detection(logical_0:Qobj, logical_1:Qobj):
	"""
	3. **Error Detection**:
		- One should demonstrate that the encoded states are orthogonal or nearly orthogonal, enabling error detection. This involves minimizing the overlap between the states to distinguish them even in the presence of certain errors.
	"""
	overlap = logical_0.overlap(logical_1)
	assert np.isclose(overlap, 0), f"Overlap between logical states: {overlap}"
	return overlap


@proving_QECC_part(4)
def develop_error_correction_protocol(what_code:CodeOptions, num_moments:int) -> Callable[[Qobj], Qobj]:
	"""
	4. **Error Correction**:
		- Developing an error correction protocol is necessary to correct the identified errors. 
		This requires designing operations that map erroneous states back to the original encoded states. 
		For bosonic codes, operations such as displacement, squeezing, or using ancillary modes may be employed.
	"""
	def correct_error(state):
		pass
	return correct_error

@proving_QECC_part(5)
def verify_qecc_conditions(code:list[Qobj], possible_errors:list[ErrorFamiliy], num_moments:int):
	r"""
	5. **Satisfy the Quantum Error Correction Conditions (Knill-Laflamme condition)**:
		- It is important to verify that the states satisfy the quantum error correction conditions: 
			For a set of errors \( \{E_i\} \), the conditions are:
			- \( \langle \psi_k | E_i^\dagger E_j | \psi_l \rangle = \delta_{kl} c_{ij} \)
		where \( |\psi_i\rangle \) and \( |\psi_j\rangle \) are the encoded states, and \( c_{ij} \) are constants. 
		This ensures that errors do not overlap between different logical states.
	"""
	c_ij_per_error = dict()

	for error_family in possible_errors:
		print(error_family)
		## Get errors:
		c_ij = _check_code_qecc_condition(code, error_family, num_moments)
		c_ij_per_error[error_family] = c_ij

	return c_ij_per_error

def perform_logical_operations(logical_0, logical_1):
	"""
	6. **Logical Operations**:
		- One should demonstrate that logical operations can be performed on the encoded states without leaving the code space. This involves implementing gates that operate directly on the encoded states.
	"""
	def logical_gate(state):
		# Example logical gate operation
		new_state = state  # Placeholder for actual gate logic
		return new_state
	return logical_gate

def analyze_fidelity():
	"""
	7. **Fidelity Analysis**:
		- Analyzing the fidelity of the error correction process is essential. 
		One should calculate the probability that the process successfully corrects errors and maintains the integrity of the quantum information.
	"""
	fidelity = 0.99  # Placeholder for actual fidelity calculation
	return fidelity

def evaluate_experimental_feasibility():
	"""
	8. **Experimental Feasibility**:
		- Consideration of the practical implementation of the QECC is crucial. 
		Evaluating the resources and technology required to realize the code is necessary, such as the need for 
		high-quality oscillators, precise control over quantum operations, and robust error detection mechanisms.
	"""
	feasibility = True  # Placeholder for actual feasibility evaluation
	return feasibility


#%% prove QECC:
def main():
	what_code : CodeOptions = "squeezed"
	num_moments : int = 6

	errors = identify_error_model()
	code = define_encoding_scheme(what_code=what_code, num_moments=num_moments)
	overlap = demonstrate_error_detection(*code)
	correct_error = develop_error_correction_protocol(what_code=what_code, num_moments=num_moments)
	conditions_satisfied = verify_qecc_conditions(code, errors, num_moments=num_moments)
	logical_gate = perform_logical_operations(*code)
	fidelity = analyze_fidelity()
	feasibility = evaluate_experimental_feasibility()

	print(f"Errors: {errors}")
	print(f"Overlap: {overlap}")
	print(f"Conditions Satisfied: {conditions_satisfied}")
	print(f"Fidelity: {fidelity}")
	print(f"Feasibility: {feasibility}")


if __name__ == "__main__":
	main()
# %%
