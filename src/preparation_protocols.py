import numpy as np
import qutip as qt
from qutip_qip import operations as qubit_operations
from typing import Final, TypeAlias, overload, TypedDict, Literal
from dataclasses import dataclass

sqrt2 = np.sqrt(2)
π = np.pi

import sympy as sp
from sympy.physics import quantum 


if __name__ == "__main__":
    from __init__ import add_root_to_path
    path = add_root_to_path()
	

## our supporting modules:
from src.utils import assertions

## This project:
from src.squeezing_direction import squeezing_direction_to_squeezing_phase
from src.visualizations import plot_light_states
from src.rotation import rotation
from src.measurements import measure_qubit_state, _MeasureStats



_DEFAULT_NUM_MODES : Final[int] = 500

qubit_0_proj = qt.basis(2, 0).proj()
qubit_1_proj = qt.basis(2, 1).proj()

I_light = qt.qeye(_DEFAULT_NUM_MODES)
H = qt.Qobj([[1, 1], [1, -1]]) / sqrt2


class _MetaData(TypedDict):
	probabilities: list[float]


_PreparationOutputType : TypeAlias = tuple[qt.Qobj, qt.Qobj] | tuple[qt.Qobj, qt.Qobj, _MetaData]






@dataclass
class LogicalCodewordInfo():
	a: complex = 1.0
	b: complex = 0.0


def _is_integer(n:int|float) -> bool:
	return int(round(n)) == n


def _is_even(n:int)	-> bool:
	return n % 2 == 0


def _is_power_of_2(n:int) -> bool:
	return _is_integer(np.log2(n))


def rot(θ:float, num_moments:int=_DEFAULT_NUM_MODES) -> qt.Qobj:
	return rotation(num_moments, θ)


def conditional_rotation_2_angles(θ0:float, θ1:float, num_moments:int=_DEFAULT_NUM_MODES) -> qt.Qobj:
	def _rot(θ:float) -> qt.Qobj:
		return rot(θ, num_moments=num_moments)
	return qt.tensor(qubit_0_proj, _rot(θ0)) + qt.tensor(qubit_1_proj, _rot(θ1))


def conditional_rotation_1_angle(θ:float, ϕ:float=0.0, num_moments:int=_DEFAULT_NUM_MODES) -> qt.Qobj:
	phase = np.exp(1j*ϕ)
	I_light = qt.qeye(num_moments)
	def _rot(θ:float) -> qt.Qobj:
		return rot(θ, num_moments=num_moments)
	
	return qt.tensor(qubit_0_proj, I_light) + phase * qt.tensor(qubit_1_proj, _rot(θ))


def measure_qubit_and_get_light(ψ: qt.Qobj) -> tuple[list[qt.Qobj], list[float]]:
	## Measure the qubit state
	branches = measure_qubit_state(ψ, num_qubits=1, return_full_stats=True, light_at="end")
	probabilities = [b['prob'] for b in branches]
	light_states = [b['state'] for b in branches]


	## print options:
	if False:
		plot_light_states(branches)
		
		print(probabilities)
		## randomly pick:
		random_index = np.random.choice(len(probabilities), p=probabilities)
		random_state = light_states[random_index]

	return light_states, probabilities


def _phase_for_iteration_k(k:int, m:int) -> float:
	assert 0 <= k < m, f"Invalid k={k!r} for m={m!r}"
	return π - 2*π*(k+1)/m


def qubit_rotation(logical_info:LogicalCodewordInfo) -> qt.Qobj:
	θ, φ = from_coefficients_to_angles(logical_info.a, logical_info.b)
	qubit_rot = qubit_operations.ry(-θ) @ qubit_operations.rz(φ)
	return qubit_rot


def _iter_sequence(ψ_light: qt.Qobj, θ:float, measurement_basis:LogicalCodewordInfo, num_moments:int=_DEFAULT_NUM_MODES) -> tuple[qt.Qobj, qt.Qobj, list[float]]:
	## Init qubit in |0⟩
	qubit = qt.basis(2, 0)
	ψ = qt.tensor(qubit, ψ_light)
	I_light = qt.qeye(num_moments)

	## H⊗I --- C-Rot ---  H⊗I:
	ψ = qt.tensor(H, I_light) @ ψ
	ψ = conditional_rotation_1_angle(θ, num_moments=num_moments) @ ψ
	ψ = qt.tensor(H, I_light) @ ψ

	## Qubit rotation:
	qubit_rot = qubit_rotation(measurement_basis)
	ψ = qt.tensor(qubit_rot, I_light) @ ψ

	## Measure light state
	light_states, probabilities = measure_qubit_and_get_light(ψ)
	assert len(light_states) == 2, f"Expected 2 branches from measurement, got {len(light_states)}"
	ψ_light_0, ψ_light_1 = light_states

	return ψ_light_0, ψ_light_1, probabilities 


def get_code_states_using_rotation_log2_m(m:int, r:float, logical_info:LogicalCodewordInfo, num_moments:int=_DEFAULT_NUM_MODES, with_meta:bool=False) -> tuple[qt.Qobj, qt.Qobj, _MetaData]: 

	k = assertions.integer(np.log2(m), reason=f"`m` must be an integer power of 2, such that m=2^k for some k. Got {m!r}")

	vacuum_state = qt.basis(num_moments, 0)
	ϕ0 = squeezing_direction_to_squeezing_phase(0)

	ψ_light_0 = qt.squeeze(num_moments, r * ϕ0) @ vacuum_state
	ψ_light_1 : qt.Qobj = None  # placeholder for type checking #type: ignore
	
	prob_0 = 1.0
	prob_1 = 1.0
	

	for i in range(k):
		is_final = i == k - 1


		θ = π * (1/2)**(i+1) 
		
		measurement_basis = logical_info if is_final else LogicalCodewordInfo()
		ψ_light_0, ψ_light_1, crnt_probabilities = _iter_sequence(ψ_light_0, θ, measurement_basis=measurement_basis, num_moments=num_moments)

		## Update probabilities:
		prob_0 *= crnt_probabilities[0]
		prob_1 *= crnt_probabilities[1]


		if False=="False":
			plot_light_states([ψ_light_0, ψ_light_1])
			print("Plotted")
		
	meta_data = _MetaData(probabilities=[prob_0, prob_1])
	return ψ_light_0, ψ_light_1, meta_data


def _create_positive_m_legged_superposition(m:int, r:float, num_moments:int=_DEFAULT_NUM_MODES) -> tuple[qt.Qobj, float]:
	vacuum_state = qt.basis(num_moments, 0)
	ϕ0 = squeezing_direction_to_squeezing_phase(0)
	ψ_light = qt.squeeze(num_moments, r * ϕ0) @ vacuum_state
	I_light = qt.qeye(num_moments)
	θ = 2*π / m

	probability = 1.0


	for k in range(m-1):
		## This iteration operators:
		ϕ = _phase_for_iteration_k(k, m)
		HoI = qt.tensor(H, I_light)
		C_Rot = conditional_rotation_1_angle(θ, ϕ, num_moments=num_moments)
		iter_op = HoI @ C_Rot @ HoI

		## State before measurement
		qubit_0 = qt.basis(2, 0)
		ψ = qt.tensor(qubit_0, ψ_light)
		ψ = iter_op @ ψ

		## Measure light state 
		light_states, probabilities = measure_qubit_and_get_light(ψ)
		assert len(light_states) == 2, f"Expected 2 branches from measurement, got {len(light_states)}"
		ψ_light_0, ψ_light_1 = light_states

		probability *= probabilities[0]

		## Post-select |0⟩ qubit outcome:
		ψ_light = ψ_light_0

	return ψ_light, probability


def get_code_states_using_rotation_even(m:int, r:float, logical_info:LogicalCodewordInfo, num_moments:int=_DEFAULT_NUM_MODES, with_meta:bool=False) -> tuple[qt.Qobj, qt.Qobj, _MetaData]:
	m = assertions.even(m, reason="`m` must be an even integer")
	half_m = m // 2

	ψ_light, prep_probability = _create_positive_m_legged_superposition(half_m, r, num_moments=num_moments)

	θ = π/m
	ψ_light1, ψ_light2, probabilities =_iter_sequence(ψ_light, θ, measurement_basis=logical_info, num_moments=num_moments)

	# Update probabilities:
	probabilities = [prob*prep_probability for prob in probabilities]

	# plot_light_states([ψ_light1, ψ_light2])

	meta_data = _MetaData(probabilities=probabilities)
	return ψ_light1, ψ_light2, meta_data


@overload
def get_code_states_using_rotation(m:int, r:float, logical_info:LogicalCodewordInfo, num_moments:int=_DEFAULT_NUM_MODES, with_meta:Literal[False]=False) -> tuple[qt.Qobj, qt.Qobj]: ...
@overload
def get_code_states_using_rotation(m:int, r:float, logical_info:LogicalCodewordInfo, num_moments:int=_DEFAULT_NUM_MODES, with_meta:Literal[True]=True) -> tuple[qt.Qobj, qt.Qobj, _MetaData]: ...
def get_code_states_using_rotation(
	m:int, r:float, logical_info:LogicalCodewordInfo, num_moments:int=_DEFAULT_NUM_MODES, with_meta:bool=False
) -> _PreparationOutputType:
	

	## If m=2^k
	if _is_power_of_2(m):
		ψ_light_0, ψ_light_1, meta_data = get_code_states_using_rotation_log2_m(m, r, logical_info=logical_info, num_moments=num_moments, with_meta=with_meta)
	
	## if m is even
	elif _is_even(m):
		ψ_light_0, ψ_light_1, meta_data = get_code_states_using_rotation_even(m, r, logical_info=logical_info, num_moments=num_moments, with_meta=with_meta)

	## if m is odd:
	else:
		raise ValueError("do not support odd `m`")
	
	if with_meta:
		return ψ_light_0, ψ_light_1, meta_data
	
	return ψ_light_0, ψ_light_1


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



def from_angels_to_coefficients(θ:float, φ:float) -> tuple[complex, complex]:
	if isinstance(θ, float|int):
		a = np.cos(θ/2)
		b = np.exp(1j*φ) * np.sin(θ/2)

	elif isinstance(θ, sp.Expr):
		a = sp.cos(θ/2)
		b = sp.exp(1j*φ) * sp.sin(θ/2)

	return a, b


def from_coefficients_to_angles(a:complex, b:complex) -> tuple[float, float]:
	θ = 2 * np.arccos(np.abs(a))
	φ = np.angle(b) - np.angle(a)
	φ = float(φ)
	return θ, φ	


def arbitrary_logical_state_test(
	m:int = 2,
	r:float = 1.5,
	θ:float = π,
	φ:float = 0
) -> None:
	"""
	This function tests the creation of logical state in an arbitrary state |ψ⟩ = a|0⟩ + b|1⟩

	A general single-qubit pure state is typically normalized such that |a|^2+|b|^2=1. 
	The state can be parameterized using two angles, θ (the polar angle) and φ (the azimuthal angle), using the following standard convention:
	|ψ⟩ = cos(θ/2)|0⟩ + e^{iφ} sin(θ/2)|1⟩
	"""
	a, b = from_angels_to_coefficients(θ, φ)	


def logical_codewords(m0:qt.Qobj, m1:qt.Qobj, logical_info:LogicalCodewordInfo) -> tuple[qt.Qobj, qt.Qobj]:
	""" Turn the logical |0⟩, |1⟩ states into the desired logical superposition based on the provided logical_info. """
	a = logical_info.a
	b = logical_info.b
	c0 = a * m0 + b * m1
	
	## C1 state is the orthogonal state to c0:
	c1 = -b.conjugate() * m0 + a.conjugate() * m1
	assert np.isclose(c0.overlap(c1), 0.0)

	return c0, c1


def _pretty_print_symbolic_coefficients(**kwargs):
	for name, value in kwargs.items():
		if isinstance(value, sp.Expr):
			val = sp.simplify(sp.simplify(value))
			val = sp.pretty(val)
		else:
			val = value
		print(f"{name}:\n{val}")


def compare_constructions(
	m:int = 2,
	r:float = 1.5,
	θ:float = π/4,
	φ:float = 0,
	verbose:bool = False
) -> tuple[float, float]:
	
	## Get states:
	a, b = from_angels_to_coefficients(θ, φ)
	logical_info = LogicalCodewordInfo(a=a, b=b)
	ψ0, ψ1 = get_code_states_using_rotation(m, r, logical_info=logical_info)

	## Compare with simple construction:
	m0, m1 = simple_m_legged_code(m, r, num_moments=_DEFAULT_NUM_MODES, code_type="squeeze")
	c0, c1 = logical_codewords(m0, m1, logical_info)

	f1 = float(qt.metrics.fidelity(ψ0, c0))
	f2 = float(qt.metrics.fidelity(ψ1, c1))

	if verbose:
		plot_light_states([ψ0, ψ1])
		plot_light_states([c0, c1])
		print(f"fidelity = {[f1, f2]}")
		print("Done.")

	return f1, f2


def test_construction(
	m:int = 4,
	θ:float = sp.pi/2,
	φ:float = 0,
	plus_state:bool = True
):

	## assert inputs
	if plus_state:
		assert θ==sp.pi/2
		assert φ==0.0

	## requested state coefficients:
	a, b = from_angels_to_coefficients(θ, φ)
	θ, φ = float(θ), float(φ)
	_pretty_print_symbolic_coefficients(a=a, b=b)


	## Compute:
	f1_vec = []
	f2_vec = []
	r_vec = np.linspace(0.1, 3, 31).tolist()

	for r in ProgressBar(r_vec):	
		f1, f2 = compare_constructions(m=m, r=r, θ=θ, φ=φ)
		f1_vec.append(f1)
		f2_vec.append(f2)
	


	## Plot:
	plot_kwargs = dict(
		linewidth = 4
	)
	
	if Globals.LaTeX_RENDERING:
		fontsize_label = 16
		fontsize_legend = 14
		label1 = r"$\ell=1$"
		label2 = r"$\ell=2$"
		xlabel_text = r"$r$"
		ylabel_text = r"Fidelity $|\langle \ell_L|\ell_g\rangle|^2$"
		if plus_state:
			ylabel_text = r"Fidelity $|\langle {+}_L|{+}_g\rangle|^2$"
	else:
		fontsize_label = None
		fontsize_legend = None
		label1 = "ℓ=1"
		label2 = "ℓ=2"
		xlabel_text = "r"
		ylabel_text = "Fidelity |<ℓ_L|ℓ_g>|^2"
		if plus_state:
			ylabel_text = "Fidelity |<+_L|+_g>|^2"
	
	p1 = plt.plot(r_vec, f1_vec, label=label1, **plot_kwargs)
	# p2 = plt.plot(r_vec, f2_vec, label=label2, **plot_kwargs)
	plt.xlabel(xlabel_text, fontsize=fontsize_label)
	plt.ylabel(ylabel_text, fontsize=fontsize_label)
	
	width = plt.gcf().get_figwidth()
	height = plt.gcf().get_figheight() 
	plt.gcf().set_size_inches(width, height*0.6)

	plt.tight_layout()
	# plt.legend(fontsize=fontsize_legend, loc="lower right")
	plt.show()

	print("Done.")


if __name__ == "__main__":
	# compare_constructions(verbose=True)
	test_construction()
