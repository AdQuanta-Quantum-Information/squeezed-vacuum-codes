#%%

from typing import Callable, Literal, TypeAlias, Final, ParamSpec, TypeVar, TypedDict, Optional, overload, cast
from enum import Enum, auto

import numpy as np
import mpmath as mp

from qutip import Qobj
from qutip import basis, qeye, tensor
from qutip import destroy, num
from qutip import sigmax, sigmay, sigmaz 
from qutip import squeeze


if __name__ == "__main__":
	from pathlib import Path
	import sys
	sys.path.append(str(Path(__file__).parents[3]))

# Visuals:
from src.utils import tuples

from src.squeezing_direction import squeezing_direction_to_squeezing_phase
from src.visualizations import plot_light_states
from src.measurements import measure_qubit_state, _MeasureStats



## Constants:
num_moments : Final[int] = 300
squeezing_strength : Final[float] = 2.0
_print: Final[bool] = True  # Set to False to disable printing


π = np.pi
sqrt_π = np.sqrt(π)
sqrt_2 = np.sqrt(2)


def conditional_print(is_on:bool, *args, **kwargs) -> None:
	if not is_on:
		return
	for i, arg in enumerate(args):
		if isinstance(arg, Qobj):
			s = fock_str(arg)
			args = tuples.copy_with_replaced_val_at_index(args, i, s)

	print(*args, **kwargs)

def print_(*args, **kwargs):
	global _print
	return conditional_print(_print, *args, **kwargs)


def op_to_unitary(op: Qobj) -> Qobj:
    exponent = 1j*op
    unitary = exponent.expm()
    return unitary  

def hadamard_transform(version:Literal[1, 2]=1) -> Qobj:
	if version == 1:
		return Qobj([[1, 1], [1, -1]]) / sqrt_2
	elif version == 2:
		return (σ_x + σ_z) / sqrt_2


## Build Hilbert spaces & operators
I_light = qeye(num_moments)
I_qubit = qeye(2)
σ_x = sigmax()
σ_y = sigmay()
σ_z = sigmaz()
H = hadamard_transform()
qubit_0 = basis(2, 0)
qubit_1 = basis(2, 1)
qubit_plus = (qubit_0 + qubit_1)/sqrt_2
qubit_minus = (qubit_0 - qubit_1)/sqrt_2
vacuum = basis(num_moments, 0)
a = destroy(num_moments)              # annihilation operator a
a_dag = a.dag()                   # creation operator a†



def measure_and_show_final_combined_state(psi):
    branches = measure_qubit_state(psi, return_full_stats=True)
    for info in branches:
        ψ_i = info['state']
        q = info['qubits_value']
        print_("")
        print_(f"|ψ{q}> = ", ψ_i)        
    plot_light_states(branches)

def _sigma_N_op(num_moments:int, N: int) -> Qobj:
	data = np.ones(num_moments - 2 * N)
	offsets = -2 * N
	sigma = Qobj(np.diag(data, k=offsets), dims=[[num_moments], [num_moments]])
	return sigma


def _proj(b:int) -> Qobj:
    """Projector |b><b| for a single qubit basis state b=0/1."""
    return basis(2, b).proj()


def _controlled_unitary_qubit_to_light(light_op_0: Qobj, light_op_1: Qobj, controlled_qubit_index: int, n_qubits: int):
	"""
	Controlled-op:  |0><0| ⊗ light_op_0 + |1><1| ⊗ light_op_1
	acting on 'controlled_qubit_index' (0-indexed) in an n_qubits register.
	"""
	
	sum_operator = []
	for ctrl_state, target_op in enumerate([light_op_0, light_op_1]):
		## build the list for the tensor-product operator:
		tp_operators = [qeye(2)]*n_qubits
		tp_operators[controlled_qubit_index] = _proj(ctrl_state)
		tp_operators = [target_op] + tp_operators  # Light state and then the qubits
		tp_operator = tensor(*tp_operators)
		sum_operator.append(tp_operator)
	
	return sum_operator[0] + sum_operator[1]




class PrimitiveOperatorsAndStates:
	num_moments: int
	I_light: Qobj
	I_qubit: Qobj
	σ_x: Qobj
	σ_y: Qobj
	σ_z: Qobj
	a: Qobj
	a_dag: Qobj
	H: Qobj
	qubit_0: Qobj
	qubit_1: Qobj
	qubit_plus: Qobj
	qubit_minus: Qobj
	vacuum: Qobj

	def __init__(self, num_moments: int) -> None:
		self.num_moments = num_moments
		self.I_light = qeye(self.num_moments)
		self.I_qubit = qeye(2)
		self.σ_x = sigmax()
		self.σ_y = sigmay()
		self.σ_z = sigmaz()
		self.a = destroy(self.num_moments)
		self.a_dag = self.a.dag()
		self.H = hadamard_transform()
		self.qubit_0 = basis(2, 0)
		self.qubit_1 = basis(2, 1)
		self.qubit_plus = (self.qubit_0 + self.qubit_1)/sqrt_2
		self.qubit_minus = (self.qubit_0 - self.qubit_1)/sqrt_2
		self.vacuum = basis(self.num_moments, 0)


class SqueezingCode:
	num_moments: int
	squeezing_strength: float
	num_legs: int
	_print: bool
	primitive_ops_and_states: PrimitiveOperatorsAndStates

	def __init__(self, squeezing_strength: float, num_moments: int = 100, verbose: bool = True, num_legs:int = 2) -> None:
		self.squeezing_strength : float = squeezing_strength
		self.num_moments : int = num_moments
		self.num_legs : int = num_legs
		self._print : bool =  verbose
		self.primitive_ops_and_states = PrimitiveOperatorsAndStates(num_moments)

	@property
	def num_needed_ancilla(self) -> int:
		"""
		Number of qubits needed for the code.
		For 2-leg codes, we need 1 qubit.
		For 4-leg codes, we need 2 qubits.
		"""
		if self.num_legs == 2:
			return 1
		elif self.num_legs == 4:
			return 2
		else:
			raise ValueError("Only 2- and 4-leg codes are implemented.")

	def print_(self, *args, **kwargs):
		return conditional_print(self._print, *args, **kwargs)

	def logical_Z(self) -> Qobj:
		return op_to_unitary(π/2 * num(self.num_moments))

	def _logical_X_normalization(self, k_cut_off: int = 400) -> float:
		r = self.squeezing_strength
		t, cosh_r = mp.tanh(r), mp.cosh(r)
		pref = -2 * t / cosh_r
		s = mp.mpf('0')
		for k in range(k_cut_off):
			nume = mp.sqrt(mp.factorial(4*k) * mp.factorial(4*k+2))
			den = 2**(4*k+1) * mp.factorial(2*k) * mp.factorial(2*k+1)
			s += nume/den * t**(4*k)
		return float(pref * s)

	def logical_X(self) -> Qobj:
		sigma_1 = _sigma_N_op(num_moments=self.num_moments ,N=1)
		X = sigma_1 + sigma_1.dag()
		norm = self._logical_X_normalization()
		X /= norm
		return X

	def logical_H(self) -> Qobj:
		raise NotImplementedError("Logical Hadamard not implemented yet.")

	def _get_entangled_superposition_2legs(self) -> Qobj:
		S = lambda sign_: squeeze(self.num_moments, sign_ * self.squeezing_strength)
		cs_unitary2 = tensor(S(+1), self.primitive_ops_and_states.qubit_0.proj()) + tensor(S(-1), self.primitive_ops_and_states.qubit_1.proj())
		H_qubit = tensor(self.primitive_ops_and_states.I_light, self.primitive_ops_and_states.H)
		psi = tensor(self.primitive_ops_and_states.vacuum, self.primitive_ops_and_states.qubit_0)
		psi = H_qubit @ psi
		psi = cs_unitary2 @ psi
		psi = H_qubit @ psi
		return psi

	def _get_entangled_superposition_4legs(self) -> Qobj:
		"""
		Two-qubit CS circuit that prepares the 4-leg superposition
			|+r> ± |-r> ± |+i r> ± |-i r>.
		The returned state lives in  (light)⊗(Q1)⊗(Q2) .
		"""
		r = self.squeezing_strength
		num_moments = self.num_moments
		
		primitive_ops = self.primitive_ops_and_states

		def _squeeze_at_(squeeze_angle:float) -> Qobj: 
			squeeze_param = r * squeezing_direction_to_squeezing_phase(squeeze_angle)
			op = squeeze(num_moments, squeeze_param)
			return op 


		# Controlled-squeezing unitaries
		u_cs_target_1 = _controlled_unitary_qubit_to_light(_squeeze_at_(0), _squeeze_at_(π/2), controlled_qubit_index=0, n_qubits=2)
		u_cs_target_2 = _controlled_unitary_qubit_to_light(_squeeze_at_(π/4), _squeeze_at_(3*π/4), controlled_qubit_index=1, n_qubits=2)
		controlled_squeezing_unitaries : list[Qobj] = [u_cs_target_1, u_cs_target_2]

		# basic operations:
		I_light = primitive_ops.I_light
		H = primitive_ops.H
		H12 = tensor(I_light, H, H)   # acts on both qubits

		# Startings states:
		light_vacuum = primitive_ops.vacuum

		# Initial state |0⟩_light ⊗ |00⟩_Q1Q2
		psi = tensor(light_vacuum, qubit_0, qubit_0)

		# Circuit: (H⊗H) – (U_cs on all qubits) – (H⊗H)
		psi = H12 @ psi
		for u_cs in controlled_squeezing_unitaries:
			psi = u_cs @ psi   
		psi = H12 @ psi
		return psi

    # ----------------------------------------------------------------
    #  ROUTINE CHOSEN ACCORDING TO num_legs
    # ----------------------------------------------------------------
	def get_entangled_superposition(self) -> Qobj:
		if self.num_legs == 2:
			return self._get_entangled_superposition_2legs()          # ← old one-qubit version
		elif self.num_legs == 4:
			return self._get_entangled_superposition_4legs()
		else:
			raise NotImplementedError("Only 2- and 4-leg codes are implemented.")


	def _divide_by_qubit_result(self, psi: Qobj) -> tuple[Qobj, Qobj]:
		n_qubits = self.num_needed_ancilla
		branches = measure_qubit_state(psi, return_full_stats=True, num_qubits=n_qubits)
		# plot_light_states(branches)
		
		## Accept results of qubits: 0 and n_qubits//2+1
		logical0 = None
		logical1 = None
		for info in branches:
			if info['qubits_value'] == 0:
				logical0 = info['state']
			elif info['qubits_value'] == 2**(n_qubits-1):
				logical1 = info['state']
		if logical0 is None or logical1 is None:
			raise ValueError("Failed to post-select both logical states.")
		return logical0, logical1

	def logical_states(self) -> tuple[Qobj, Qobj]:
		psi = self.get_entangled_superposition()
		psi1, psi2 = self._divide_by_qubit_result(psi)
		# psi1, psi2 = get_forced_superposition_code_states("squeeze", amplitude=self.squeezing_strength, num_moments=self.num_moments, num_legs=self.num_legs)
		assert np.isclose(psi1.overlap(psi2), 0, atol=1e-6), "Logical states are not orthogonal."
		assert np.isclose(psi1.norm(), 1, atol=1e-6), "Logical state |0> is not normalized."
		assert np.isclose(psi2.norm(), 1, atol=1e-6), "Logical state |0> is not normalized."
		return psi1, psi2

	def logical_dual_states(self) -> tuple[Qobj, Qobj]:
		logic_0, logic_1 = self.logical_states()
		logic_p = (logic_0 + logic_1) / sqrt_2
		logic_m = (logic_0 - logic_1) / sqrt_2
		assert np.isclose(logic_p.overlap(logic_m), 0, atol=1e-6), "Logical states are not orthogonal."
		assert np.isclose(logic_p.norm(), 1, atol=1e-6), "Logical state |0> is not normalized."
		assert np.isclose(logic_m.norm(), 1, atol=1e-6), "Logical state |0> is not normalized."
		return logic_p, logic_m

	def deterministic_feed_forward_generation(self, which_state: Literal[0, 1, "+", "-"] = 0) -> Qobj:
		logic_0, logic_1 = self.logical_states()
		X = self.logical_X()

		match which_state:
			case 0:
				logic_0_alternative = X @ logic_1
				logic_0_alternative.unit(inplace=True)
				assert np.isclose(logic_0_alternative.overlap(logic_0), 1, atol=1e-6), "Generated state is not the same as |0>."
				return logic_0_alternative
			
			case 1:
				logic_1_alternative = X @ logic_0
				logic_1_alternative.unit(inplace=True)
				assert np.isclose(logic_1_alternative.overlap(logic_1), 1, atol=1e-6), "Generated state is not the same as |1>."
				return logic_1_alternative
			
			case "+":
				H = self.logical_H()
				logic_p = H @ logic_0
				return logic_p
			
			case "-":
				H = self.logical_H()
				logic_m = H @ logic_1
				return logic_m
			
			case _:
				raise ValueError("Invalid state selection. Choose 0 or 1.")



def study(
	num_moments:int = 200,
	num_legs:int = 4,
) -> None:
	code = SqueezingCode(squeezing_strength=2.0,  num_legs=num_legs, num_moments=num_moments, verbose=True)
	logic_0, logic_1 = code.logical_states()

	assert np.isclose(logic_0.overlap(logic_1), 0, atol=1e-6), "Logical states are not orthogonal."
	logic_p, logic_m = code.logical_dual_states()
	plot_light_states([logic_0, logic_1, logic_p, logic_m])
	Z = code.logical_Z()
	X = code.logical_X()

	## Checks:
	z0 = logic_0.dag() @ Z @ logic_0
	z1 = logic_1.dag() @ Z @ logic_1
	z0 = cast(float, z0) 
	z1 = cast(float, z1) 
	assert np.isclose(z0, 0.0, atol=1e-6), "Logical Z is not zero on |0>."
	assert np.isclose(z1, 1.0, atol=1e-6), "Logical Z is not one on |1>."
	x_p = logic_p.dag() @ X @ logic_p
	x_m = logic_m.dag() @ X @ logic_m
	x_0_to_0 = logic_0.dag() @ X @ logic_0
	x_0_to_1 = logic_1.dag() @ X @ logic_0
	x_1_to_0 = logic_0.dag() @ X @ logic_1
	x_1_to_1 = logic_1.dag() @ X @ logic_1
	code.print_("Done.")


#%%

if __name__ == "__main__":
	study()
