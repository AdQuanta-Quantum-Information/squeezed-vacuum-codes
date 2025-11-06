from qutip.measurement import measurement_statistics_povm
from qutip import Qobj, basis, tensor, qeye

import numpy as np

from typing import Literal, overload, TypedDict, List, TypeVar

from src.utils.binary import binary_tuples_generator
from src.quantum.density_matrices.translations import density_matrix_to_pure_state




class _MeasureStats(TypedDict):
	"""
	Type for measurement statistics of light states after qubit measurement.
	"""
	state: Qobj
	prob: float
	purity: float
	qubits_value: int


@overload
def measure_qubit_state(psi: Qobj, num_qubits:int=1, return_full_stats:Literal[True]=True, light_at:Literal["end", "beginning"]="beginning") -> list[_MeasureStats]: ...
@overload
def measure_qubit_state(psi: Qobj, num_qubits:int=1, return_full_stats:Literal[False]=False, light_at:Literal["end", "beginning"]="beginning") -> list[Qobj]: ...
def measure_qubit_state(psi: Qobj, num_qubits:int=1, return_full_stats:Literal[False, True]=False, light_at:Literal["end", "beginning"]="beginning") -> list[_MeasureStats]|list[Qobj]:

	## Where the light is at:
	match light_at:
		case "beginning":
			light_index = 0
		case "end":
			# final index. like writing [-1] but also fits the qutip.ptrace method:
			light_index = num_qubits  

	light_dim = psi.dims[0][light_index]  # Dimension of the light mode
	I_light = qeye(light_dim)  # Identity operator for the light mode

    # Measure qubit & post-select
	# Create measurement operators for all 2^n_qubits computational basis states
	measurement_operators = []
	for binary_tuple in binary_tuples_generator(num_qubits):
		# Create the multi-qubit projector for this computational basis state
		qubit_projectors = [basis(2, bit).proj() for bit in binary_tuple]
		multi_qubit_projector = tensor(*qubit_projectors)
		
		# Tensor with light identity
		if light_at == "beginning":
			measurement_op = tensor(I_light, multi_qubit_projector)
		elif light_at == "end":
			measurement_op = tensor(multi_qubit_projector, I_light)
		else:
			raise ValueError(f"Invalid light_at value: {light_at}. Expected 'beginning' or 'end'.")
		
		measurement_operators.append(measurement_op)

	collapsed_states, probabilities = measurement_statistics_povm(psi, measurement_operators)

	# Extract logical codewords
	light_states: list[Qobj] = []
	purity_values: list[float] = []
	qubits_values: list[int] = []
	for qubits_value, state in enumerate(collapsed_states):
		if state is None:
			continue
		else:
			state : Qobj  # just for type hinting

		light_reduced_density_matrix = state.ptrace(light_index)

		light, purity = density_matrix_to_pure_state(light_reduced_density_matrix)
		light_states.append(light)
		purity_values.append(purity)
		qubits_values.append(qubits_value)
		
	if not return_full_stats:
		return light_states

	branches : list[_MeasureStats] = [
		_MeasureStats(
			state=light_states,
			prob=probabilities,
			purity=purity,
			qubits_value=qubit_value
		) for light_states, probabilities, purity, qubit_value in zip(light_states, probabilities, purity_values, qubits_values)
	]
	return branches
