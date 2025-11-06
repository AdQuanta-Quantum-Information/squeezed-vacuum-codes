from typing import Literal, overload, TypeAlias
import numpy as np
from scipy.linalg import sqrtm

from ...states.qubits import QubitsState, QubitsSum, StatesSum



def _parse_input_state(state:np.ndarray|QubitsSum) -> np.ndarray:
    if isinstance(state, np.ndarray):
        return state
    elif isinstance(state, StatesSum) and state.is_sum_of(QubitsState):
        return qubits_to_density_matrix(state)
    elif isinstance(state, QubitsState):
        return qubits_to_density_matrix(state.to_sum())
    else:
        raise TypeError(f"No an expected type; type(state)={type(state)!r}.")



@overload
def concurrence(state:QubitsSum, method:Literal[1, 2]=2) -> float:...
@overload
def concurrence(state:np.ndarray, method:Literal[1, 2]=2) -> float:...
def concurrence(state:np.ndarray|QubitsSum, method:Literal[1, 2]=2) -> float:
    """
    Calculate the concurrence of a two-qubit quantum state.

    Concurrence [Sam, and Wootters "Entanglement of a pair of quantum bits." PRL (1997)]
    is a measure of quantum entanglement for a pair of qubits.
    It quantifies the degree to which a quantum state exhibits entanglement,
    ranging from 0 for separable states to 1 for maximally entangled states.

    The concurrence is an entanglement monotone: a nonnegative function whose 
    value does not increase under Local Operations and Classical Communication 
    [Horodecki "Quantum entanglement" Reviews of modern physics (2009)].

    To get the concurrence we first define 

        tilde_rho = (sigma_y ⊗ sigma_y) * conj(rho) * (sigma_y ⊗ sigma_y)


    Then, the concurrence `C` for a two-qubit density matrix rho is defined as:

        C(rho) = max(0, lambda_1 - lambda_2 - lambda_3 - lambda_4)

    where `lambda_i` are the eigenvalues, in decreasing order, of the Hermitian matrix:

        R = sqrt( sqrt(rho) tilde_rho sqrt(rho) )

    or equivelantly, the square-root of the eigenvalues of the non-Hermitian matrix:  
        
        R = rho * tilde_rho

    Here, sigma_y is the Pauli Y matrix:
    
        sigma_y = [[0, -i],
                   [i,  0]]

    Parameters
    ----------
        rho : ndarray
            A 4x4 ndarray representing the density matrix of the two-qubit system.

        or

        ket : QubitSum
            A ket state. This will be transformed into a density matrix `rho` to end-up 
            with the same calculation. 

    Returns
    -------
        concurrence : float
            The concurrence of the quantum state, a value between 0 and 1.


    Example
    -------
    >>> rho = np.array([[0.5, 0, 0, 0.5],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0.5, 0, 0, 0.5]])
    >>> concurrence(rho)
    1.0

    
    Raises
    ------
        ValueError
            If the input matrix is not a valid 4x4 density matrix.
    """
    
    ## Input can be either (density) matrix or QubitSum
    rho = _parse_input_state(state)

    if rho.shape != (4, 4):
        raise ValueError("Input matrix must be a 4x4 density matrix.")

    ## Prepare needed matrices: 
    sigma_y = np.array([[0, -1j], [1j, 0]])  # Pauli Y matrix
    sigma_y_tensor = np.kron(sigma_y, sigma_y)  # (sigma_y ⊗ sigma_y) matrix
    conj_rho = np.conjugate(rho)

    ## We have two methods:
    if method == 1:
        # Prepare intermediate results:
        tilde_rho = sigma_y_tensor @ conj_rho @ sigma_y_tensor
        sqrt_rho = sqrtm(rho)

        # compute the matrix R:
        R = sqrtm( sqrt_rho @ tilde_rho @ sqrt_rho )

        # Compute the eigenvalues of R
        eigenvalues = np.linalg.eigvals(R)
        # Sort eigenvalues in decreasing order of their absolute values
        eigenvalues = np.sort(np.abs(eigenvalues))[::-1]

        # Define the lambda in decreasing order
        lambdas = eigenvalues

    elif method == 2:
        # Compute the matrix R:
        R = np.dot(np.dot(rho, sigma_y_tensor), np.dot(conj_rho, sigma_y_tensor))

        # Compute the eigenvalues of R
        eigenvalues = np.linalg.eigvals(R)
        # Sort eigenvalues in decreasing order of their absolute values
        eigenvalues = np.sort(np.abs(eigenvalues))[::-1]

        lambdas = [np.sqrt(value) for value in eigenvalues]

    else:
        raise ValueError(f"Not a possible method. Got method={method!r}.")

    ## Compute concurrence
    concurrence_value = max(0, np.sqrt(lambdas[0]) - np.sqrt(lambdas[1])
                               - np.sqrt(lambdas[2]) - np.sqrt(lambdas[3]))

    return concurrence_value



## Example usage
if __name__ == "__main__":
    rho = np.array([[0.5, 0, 0, 0.5],
                    [  0, 0, 0, 0  ],
                    [  0, 0, 0, 0  ],
                    [0.5, 0, 0, 0.5]])
    print("Concurrence bell: method-1:", concurrence(rho, method=1))
    print("Concurrence bell: method-2:", concurrence(rho, method=2))
    #
    rho = np.array([[1, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0]])
    print("Concurrence zero: method-1:", concurrence(rho, method=1))
    print("Concurrence zero: method-2:", concurrence(rho, method=2))
    #
    print("Done")