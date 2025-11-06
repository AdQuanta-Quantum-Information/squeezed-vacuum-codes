from typing import TypeVar, overload, Literal, cast

from qutip import Qobj
import numpy as np

_QuantumOpType = TypeVar("_QuantumOpType", Qobj, np.ndarray)


def _density_matrix_to_pure_state(rho:_QuantumOpType, check:bool=True) -> tuple[_QuantumOpType, float]:
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


@overload
def density_matrix_to_pure_state(rho:_QuantumOpType, check:bool=True, with_purity:Literal[False]=False) -> _QuantumOpType: ...
@overload
def density_matrix_to_pure_state(rho:_QuantumOpType, check:bool=True, with_purity:Literal[True]=True) -> tuple[_QuantumOpType, float]: ...
def density_matrix_to_pure_state(rho:_QuantumOpType, check:bool=True, with_purity:Literal[True, False]=True) -> tuple[_QuantumOpType, float]|_QuantumOpType: 
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
    purity = float(dominant_eigenvalue)
    
    # Extract the corresponding eigenvector (pure state)
    assert np.isclose(dominant_eigenvalue, 1, rtol=1e-6), "Dominant eigenvalue is not close to 1"
    pure_state : np.ndarray = eigenvectors[:, pure_state_index]
    
    # Normalize the state vector
    pure_state /= np.linalg.norm(pure_state)

    ## Return the pure state in the original type
    if original_type == Qobj:
        dims = (rho.dims[0][0], 1)
        state_out = Qobj(pure_state, dims=dims)
    else:
        state_out = pure_state
    
    # Just for proper typing:
    state_out = cast(_QuantumOpType, state_out)	

    if with_purity:
        return state_out, purity
    return state_out  #type:ignore