if __name__ == "__main__":
    import pathlib, sys
    main_folder = pathlib.Path(__file__).parents[4].__str__()
    if main_folder not in sys.path:
        sys.path.append(main_folder)


import numpy as np


def random_matrix(n: int) -> np.ndarray:
    return np.random.normal(size=(n, n))


def random_complex_matrix(n:int) -> np.ndarray:
    return  random_matrix(n) + 1j * random_matrix(n)


def _random_density_matrix_vers1(n: int) -> np.ndarray:
    # Generate a random complex matrix
    random_matrix_ = random_complex_matrix(n)
    
    # Create a Hermitian matrix
    hermitian_matrix = (random_matrix_ + random_matrix_.conj().T) / 2
    
    # Perform eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(hermitian_matrix)
    
    # Ensure eigenvalues are non-negative (positive semidefinite)
    eigenvalues = np.abs(eigenvalues)
    
    # Normalize eigenvalues to make the trace equal to 1
    eigenvalues /= np.sum(eigenvalues)
    
    # Reconstruct the density matrix
    density_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.conj().T
    
    return density_matrix


def _random_density_matrix_vers2(dim):
    """ This is the faster version """
    # Generate a random complex matrix
    A = random_complex_matrix(dim)
    
    # Create a positive semidefinite matrix by computing A†A
    # This ensures both hermiticity and positive semidefiniteness
    rho = np.matmul(A.conj().T, A)
    
    # Normalize to have trace = 1
    rho = rho / np.trace(rho)
    
    return rho


def _run_test(n:int):
    format = ".6f"

    t1 = perf_counter()
    dm1 = _random_density_matrix_vers1(n)
    t2 = perf_counter()
    print(f"Time for _random_density_matrix_vers1: {t2-t1:{format}} seconds")
    assertions.density_matrix(dm1)

    t1 = perf_counter()
    dm2 = _random_density_matrix_vers2(n)
    t2 = perf_counter()
    print(f"Time for _random_density_matrix_vers2: {t2-t1:{format}} seconds")
    assertions.density_matrix(dm2)



def random_density_matrix(dim:int) -> np.ndarray:
    """
    Generate a random density matrix of dimension dim x dim.
    
    A valid density matrix must be:
        1. Square
        2. Have trace = 1
        3. Be Hermitian (ρ = ρ†)
        4. Be positive semidefinite
    
    Args:
        dim (int): Dimension of the density matrix
        
    Returns:
        numpy.ndarray: Random density matrix of shape (dim, dim)
    """
    return _random_density_matrix_vers2(dim)


if __name__ == "__main__":
    # Validate the density matrices and speed
    from src.utils import assertions
    from time import perf_counter 

    _run_test(200)


    print("all good")