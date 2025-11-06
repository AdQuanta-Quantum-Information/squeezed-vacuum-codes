import numpy as np
import qutip as qt
from typing import Final, overload

sqrt2 = np.sqrt(2)
π = np.pi

import sympy as sp
from sympy.physics import quantum 


if __name__ == "__main__":
	from __init__ import add_root_to_path; add_root_to_path()

## out supporting quantum modules:
from src.utils import assertions

## This project:
from src.squeezing_direction import squeezing_direction_to_squeezing_phase
from src.codes_built_in_superposition import simple_m_legged_code
from src.measurements import measure_qubit_state, _MeasureStats
from src.visualizations import plot_light_states
from src.analytical_expressions import fock_rep_of_squeezed_vacuum_in_direction
from src.rotation import rotation

## Visuals:
from matplotlib import pyplot as plt
from src.utils.visuals import matplotlib_support
from src.quantum.visualizations.wigner_function import plot_plain_wigner 


NUM_MODES : Final[int] = 100

qubit_0_proj = qt.basis(2, 0).proj()
qubit_1_proj = qt.basis(2, 1).proj()

I_light = qt.qeye(NUM_MODES)
H = qt.Qobj([[1, 1], [1, -1]]) / sqrt2


def _is_integer(n:int|float) -> bool:
	return int(round(n)) == n


def _is_even(n:int)	-> bool:
	return n % 2 == 0


def _is_power_of_2(n:int) -> bool:
	return _is_integer(np.log2(n))


def rot(θ:float) -> qt.Qobj:
	return rotation(NUM_MODES, θ)


def conditional_rotation_2_angles(θ0:float, θ1:float) -> qt.Qobj:
	return qt.tensor(qubit_0_proj, rot(θ0)) + qt.tensor(qubit_1_proj, rot(θ1))


def conditional_rotation_1_angle(θ:float, ϕ:float=0.0) -> qt.Qobj:
	phase = np.exp(1j*ϕ)
	return qt.tensor(qubit_0_proj, I_light) + phase * qt.tensor(qubit_1_proj, rot(θ))


def measure_qubit_and_get_light(ψ: qt.Qobj) -> list[qt.Qobj]:
	## Measure the qubit state
	branches = measure_qubit_state(ψ, num_qubits=1, return_full_stats=True, light_at="end")
	probabilities = [b['prob'] for b in branches]
	light_states = [b['state'] for b in branches]

	plot_light_states(branches)

	## print options:
	if False:
		print(probabilities)
		## randomly pick:
		random_index = np.random.choice(len(probabilities), p=probabilities)
		random_state = light_states[random_index]

	return light_states


def _phase_for_iteration_k(k:int, m:int) -> float:
	assert 0 <= k < m, f"Invalid k={k!r} for m={m!r}"
	return π - 2*π*(k+1)/m


def _iter_sequence(ψ_light: qt.Qobj, θ:float) -> tuple[qt.Qobj, ...]:
	## Init qubit in |0⟩
	qubit = qt.basis(2, 0)
	ψ = qt.tensor(qubit, ψ_light)

	## H⊗I --- C-Rot ---  H⊗I:
	ψ = qt.tensor(H, I_light) @ ψ
	ψ = conditional_rotation_1_angle(θ) @ ψ
	ψ = qt.tensor(H, I_light) @ ψ

	## Measure light state
	ψ_light_0, ψ_light_1 = measure_qubit_and_get_light(ψ)

	return ψ_light_0, ψ_light_1


def get_code_states_using_rotation_log2_m(m:int, r:float) -> tuple[qt.Qobj, ...]: 

	k = assertions.integer(np.log2(m), reason=f"`m` must be an integer power of 2, such that m=2^k for some k. Got {m!r}")

	vacuum_state = qt.basis(NUM_MODES, 0)
	ϕ0 = squeezing_direction_to_squeezing_phase(0)
	ψ_light = qt.squeeze(NUM_MODES, r * ϕ0) @ vacuum_state

	for i in range(k):
		θ = π * (1/2)**(i+1) 
		ψ_light, ψ_light_second_option = _iter_sequence(ψ_light, θ)

		## plot:
		plot_light_states([ψ_light, ψ_light_second_option])
		print("Plotted")

	return ψ_light, ψ_light_second_option


def _create_positive_m_legged_superposition(m:int, r:float) -> qt.Qobj:
	vacuum_state = qt.basis(NUM_MODES, 0)
	ϕ0 = squeezing_direction_to_squeezing_phase(0)
	ψ_light = qt.squeeze(NUM_MODES, r * ϕ0) @ vacuum_state
	θ = 2*π / m


	for k in range(m-1):
		## This iteration operators:
		ϕ = _phase_for_iteration_k(k, m)
		HoI = qt.tensor(H, I_light)
		C_Rot = conditional_rotation_1_angle(θ, ϕ)
		iter_op = HoI @ C_Rot @ HoI

		## State before measurement
		qubit_0 = qt.basis(2, 0)
		ψ = qt.tensor(qubit_0, ψ_light)
		ψ = iter_op @ ψ

		## Measure light state 
		ψ_light_0, ψ_light_1 = measure_qubit_and_get_light(ψ)

		## Post-select |0⟩ qubit outcome:
		ψ_light = ψ_light_0

	return ψ_light


def get_code_states_using_rotation_even(m:int, r:float) -> tuple[qt.Qobj, ...]:
	m = assertions.even(m, reason="`m` must be an even integer")
	half_m = m // 2

	ψ_light = _create_positive_m_legged_superposition(half_m, r)

	θ = π/m
	ψ_light1, ψ_light2 =_iter_sequence(ψ_light, θ)

	# plot_light_states([ψ_light1, ψ_light2])

	return ψ_light1, ψ_light2



def get_code_states_using_rotation(m:int, r:float) -> tuple[qt.Qobj, ...]:
	

	## If m=2^k
	if _is_power_of_2(m):
		return get_code_states_using_rotation_log2_m(m, r)
	
	## if m is even
	elif _is_even(m):
		return get_code_states_using_rotation_even(m, r)


	## if m is odd:
	else:
		raise ValueError("do not support odd `m`")


@overload
def Ry(θ:float) -> np.matrix: ...
@overload
def Ry(θ:sp.Symbol) -> sp.Matrix: ...
def Ry(θ:sp.Symbol|float) -> sp.Matrix|np.matrix:
	if isinstance(θ, float|int):
		c = np.cos(θ/2)
		s = np.sin(θ/2)
		return np.matrix([[c, -s], [s, c]])

	elif isinstance(θ, sp.Symbol):
		c = sp.cos(θ/2)
		s = sp.sin(θ/2)
		return sp.Matrix([[c, -s], [s, c]])
	
	raise TypeError(f"Got θ of type {type(θ)!r}")




def main_test(
	m:int = 6,
	r:float = 1.5
):


	## Get states:
	ψ0, ψ1 = get_code_states_using_rotation(m, r)

	## Compare with 
	c0, c1 = simple_m_legged_code(m, r, num_moments=NUM_MODES, code_type="squeeze")

	f1 = qt.metrics.fidelity(ψ0, c0)
	f2 = qt.metrics.fidelity(ψ1, c1)
	print(f"fidelity = {[f1, f2]}")

	print("Done.")


if __name__ == "__main__":
	main_test()