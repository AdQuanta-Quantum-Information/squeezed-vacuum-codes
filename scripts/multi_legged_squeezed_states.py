if __name__ == "__main__":
	from __init__ import import_src; import_src()

import numpy as np

from src.utils.assertions import bit
sqrt2 = np.sqrt(2)
π = np.pi
sqrt_π = np.sqrt(π)

from typing import Callable, Literal, TypeAlias, Final, ParamSpec, TypeVar, TypedDict, Optional
from enum import Enum, auto

## Qutip:
from qutip import (
	Qobj,
    basis, tensor, qeye, qzero,
    displace, squeeze, destroy, create, position, momentum,
    sigmax, sigmay, sigmaz, sigmap, sigmam,
)
from qutip.measurement import measurement_statistics_povm

## out supporting quantum modules:
from src.quantum.qutip_support._common import fock_str
from src.quantum.states.density_matrices.translations import density_matrix_to_pure_state
from src.utils import tuples, numerics
from src.quantum.states.bosonic.squeezed import exact_non_displaced_term, FockSum

## This project:
from projects.controlled_squeezing.src.squeezing_direction import squeezing_direction_to_squeezing_phase
from projects.controlled_squeezing.src.measurements import measure_qubit_state, _MeasureStats
from projects.controlled_squeezing.src.visualizations import plot_light_states
from projects.controlled_squeezing.src.analytical_expressions import fock_rep_of_squeezed_vacuum_in_direction

## Visuals:
from matplotlib import pyplot as plt
from src.utils.visuals import matplotlib_support
from src.quantum.visualizations.wigner_function import plot_plain_wigner 


## Constants:
DEFAULT_NUM_MOMENTS : Final[int] = 30


# Define the Hadamard gate manually
def qutrit_op(which:Literal['X', 'Y', 'Z', 'H'], *states:int) -> Qobj:

	assert len(states) == 2, "qutrit_op expects exactly two states."

	match which:
		case 'X':
			qubit_op = sigmax()
		case 'Y':
			qubit_op = sigmay()
		case 'Z':
			qubit_op = sigmaz()
		case 'H':
			qubit_op = hadamard_transform()
		case _:
			raise ValueError(f"Unknown qutrit operator: {which}")

	## Create the qutrit operator basis aligned with the qubit basis:
	qutrit_states = list(states)	
	qutrit_states.sort()

	## Create the qutrit operator:
	qutrit_op = np.zeros((3, 3))
	for i_qubit_index, i_qutrit_index in enumerate(qutrit_states):
		for j_qubit_index, j_qutrit_index in enumerate(qutrit_states):
			qutrit_op[i_qutrit_index, j_qutrit_index] = qubit_op[i_qubit_index, j_qubit_index]
	
	return Qobj(qutrit_op, dims=[[3], [3]], isherm=True)


def hadamard_transform() -> Qobj:
	return (sigmaz()+sigmax()) / sqrt2

## look
def conditional_print(is_on:bool, *args, **kwargs) -> None:
	if not is_on:
		return
	for i, arg in enumerate(args):
		if isinstance(arg, Qobj) or isinstance(arg, FockSum):
			s = fock_str(arg)
			args = tuples.copy_with_replaced_val_at_index(args, i, s)

	print(*args, **kwargs)


def _op_str(op:str, l:int=20) -> str:
	# pad the string with spaces to get a string of `l` length:
	return f"{op:<{l}}"




def _create_code_states_qutrit(
	num_moments: int = DEFAULT_NUM_MOMENTS,
	strength: float = 1.5,
	_print: bool = True
):
	

	def print_(*args, **kwargs):
		return conditional_print(_print, *args, **kwargs)

	# Build Hilbert spaces & operators
	I_light = qeye(num_moments)
	qutrit_0 = basis(3, 0)
	qutrit_1 = basis(3, 1)
	qutrit_2 = basis(3, 2)
	light_0 = basis(num_moments, 0)
	
	H02 = qutrit_op('H', 0, 2)  # Hadamard gate between qutrit states |0> and |2>

	def _s(direction:float) -> Qobj:
		phase = squeezing_direction_to_squeezing_phase(direction)
		squeezing_param = strength * phase
		return squeeze(num_moments, squeezing_param)  # Squeezing operator for the light mode

	conditional_squeezing = tensor(_s(0)  , qutrit_0.proj()) + \
							tensor(_s(π/2), qutrit_2.proj()) 


	
	## Initial state:
	psi = tensor(light_0, qutrit_0)
	print_(_op_str("Initial"),      "|ψ>=", psi)

	## hadamard transform:
	psi = tensor(I_light, H02) @ psi
	print_(_op_str("I⊗H"),      "|ψ>=", psi)

	## conditional squeezing:
	psi = conditional_squeezing @ psi
	print_(_op_str("CS"),      "|ψ>=", psi)


def _create_code_m_legged_code(
	m:int=2,
	num_moments: int = DEFAULT_NUM_MOMENTS,
	strength: float = 1.5,
	_print: bool = True
):
	
	def print_(*args, **kwargs):
		return conditional_print(_print, *args, **kwargs)

	# Build Hilbert spaces & operators
	I_light = qeye(num_moments)
	light_0 = basis(num_moments, 0)
	I_qubit = qeye(2)
	qubit_0 = basis(2, 0)
	qubit_1 = basis(2, 1)
	H = hadamard_transform()
	num_qubits : int = 2

	def _s(direction:float) -> Qobj:
		phase = squeezing_direction_to_squeezing_phase(direction)
		squeezing_param = strength * phase
		return squeeze(num_moments, squeezing_param)  # Squeezing operator for the light mode

	def conditional_squeezing(control:int, *directions:float) -> Qobj:
		assert len(directions) in [1, 2] 

		qubit_ops : list[list[Qobj]] = [
			[
				qubit_proj if i == control else I_qubit
				for i in range(num_qubits)
			]
			for qubit_proj in [qubit_0.proj(), qubit_1.proj()]
		] 

		if len(directions) == 2:
			qubit_and_light_ops = [
				tensor(*qubit_ops_row, _s(direction))
				for direction, qubit_ops_row in zip(directions, qubit_ops)
			]
		elif len(directions) == 1:
			direction = directions[0]
			qubit_and_light_ops = [
				tensor(*qubit_ops_row, light_op)
				for light_op, qubit_ops_row in zip([I_light, _s(direction)], qubit_ops)
			]
		else:
			raise ValueError("Invalid number of directions provided. Expected 1 or 2.")

		op = qubit_and_light_ops[0] + qubit_and_light_ops[1]  # Sum the operators for the two qubit states

		return op 


	## Initial state:
	psi = tensor(qubit_0, qubit_0, light_0)
	print_(_op_str("Initial"),      "|ψ>=", psi)

	## hadamard transform:
	psi = tensor(H, H, I_light) @ psi
	print_(_op_str("I⊗H"),      "|ψ>=", psi)

	## conditional squeezing:
	psi = conditional_squeezing(0, 0, π/2) @ psi
	print_(_op_str("CS 0→ψ"),      "|ψ>=", psi)
	## conditional squeezing:

	psi = conditional_squeezing(1, π/4) @ psi
	print_(_op_str("CS 1→ψ"),      "|ψ>=", psi)

	## hadamard transform:
	psi = tensor(H, H, I_light) @ psi
	print_(_op_str("I⊗H"),      "|ψ>=", psi)

	branches : list[_MeasureStats] = measure_qubit_state(psi, num_qubits=num_qubits, return_full_stats=True, light_at="end")

	d = plot_light_states(branches)
		
	return 



def _study_squeeze_rotations(
	num_moments: int = DEFAULT_NUM_MOMENTS,
	_print: bool = True
):
	
	def print_(*args, **kwargs):
		return conditional_print(_print, *args, **kwargs)

	# Build Hilbert spaces & operators
	I_light = qeye(num_moments)
	light_0 = basis(num_moments, 0)
	I_qubit = qeye(2)

	def s_(strength_, direction:float) -> Qobj:
		phase = squeezing_direction_to_squeezing_phase(direction)
		squeezing_param = strength_ * phase
		return squeeze(num_moments, squeezing_param)

	psi = light_0
	print_(_op_str("Initial"),      "|ψ>=", psi)

	psi = s_(1.0, 0) @ psi
	print_(_op_str("s(0)"),      "|ψ>=", psi)

	# psi_analytic = fock_rep_of_squeezed_vacuum_in_direction(10, strength, 0)
	# print(psi_analytic)


	psi_rotated = s_(0.5, -π/4) @ psi
	plot_light_states([psi, psi_rotated])

	matplotlib_support.close_all()

	print("Done.")



if __name__ == "__main__":
	_create_code_m_legged_code()
	# _study_squeeze_rotations()