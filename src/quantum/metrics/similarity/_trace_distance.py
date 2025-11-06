from typing import TypeAlias
from qutip import Qobj
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse import linalg as sparse_linalg
from scipy import linalg


_AcceptableInputTypes : TypeAlias = Qobj|np.ndarray|lil_matrix

def _get_numpy_dense_data(obj:_AcceptableInputTypes) -> np.ndarray:
    if isinstance(obj, Qobj):
        return obj.full()
    elif isinstance(obj, np.ndarray):
        return obj
    elif isinstance(obj, lil_matrix):
        return obj.toarray()
    else:
        raise ValueError(f"Invalid type: {type(obj)}")
    

def _assert_type_of_objects(rho:_AcceptableInputTypes, sigma:_AcceptableInputTypes) -> tuple[_AcceptableInputTypes, _AcceptableInputTypes]:
    if type(rho) == type(sigma):
        return rho, sigma
    elif isinstance(rho, Qobj) and isinstance(sigma, np.ndarray):
        return rho, Qobj(sigma)
    elif isinstance(rho, np.ndarray) and isinstance(sigma, Qobj):
        return Qobj(rho), sigma
    elif isinstance(rho, lil_matrix) and isinstance(sigma, csr_matrix):
        return csr_matrix(rho), sigma
    elif isinstance(rho, csr_matrix) and isinstance(sigma, lil_matrix):
        return rho, csr_matrix(sigma)
    else:
        raise TypeError(f"Mismatch type between rho and sigma: {type(rho)!r} and {type(sigma)!r}")


def trace_distance(rho:_AcceptableInputTypes, sigma:_AcceptableInputTypes) -> float:
    """ Compute the trace distance between two quantum states.
    The trace distance between two quantum states:
      D(ρ, σ) = 1/2 ||ρ - σ||_1
    where ||A||_1 is the trace norm of A, which is the sum of the absolute values of the eigenvalues of A.
    The trace distance is a metric on the space of density matrices, and is always between 0 and 1.
    It is equal to 0 if and only if the two states are equal, and is equal to 1 if and only if the two states are orthogonal.

    If `D` is the trace distance between two states, and `F` is the Fidelity, then they are related by:
    ```python
    (1 - D)**2 <=  F  <=  1 - D**2
    ```
    """
    rho, sigma = _assert_type_of_objects(rho, sigma)
    # Compute the difference
    diff = rho - sigma

    # Sum the absolute values of eigenvalues and divide by 2
    if isinstance(diff, Qobj):
        eigenvalues = diff.eigenenergies()
    elif isinstance(diff, np.ndarray):
        eigenvalues = linalg.eigvalsh(diff)
    elif isinstance(diff, lil_matrix|csr_matrix):
        eigenvalues = sparse_linalg.eigsh(diff, return_eigenvectors=False, tol=1e-6)

    res = np.sum(np.abs(eigenvalues)) / 2
    return res


def _trace_distance_test(rho:_AcceptableInputTypes, sigma:_AcceptableInputTypes) -> float:
    # The trace distance between two quantum states:

    # return res

    from time import perf_counter
    from matplotlib import pyplot as plt
    from src.utils.visuals.matplotlib_support import twin_axis

    # Compute the difference
    t1 = perf_counter()

    diff_sparse = rho - sigma
    eigenvalues_sparse = sparse_linalg.eigsh(diff_sparse, return_eigenvectors=False, tol=1e-6)
    res_sparse = np.sum(np.abs(eigenvalues_sparse)) / 2

    t2 = perf_counter()

    m1 = _get_numpy_dense_data(rho)
    m2 = _get_numpy_dense_data(sigma)
    diff_dense  = m1 - m2
    eigenvalues_dense  = linalg.eigvalsh(diff_dense)
    res_dense = np.sum(np.abs(eigenvalues_dense)) / 2

    t3 = perf_counter()

    m1 = _get_numpy_dense_data(rho)
    m2 = _get_numpy_dense_data(sigma)
    res_norm = 0.5 * np.linalg.norm(m1 - m2, ord='nuc')
    assert isinstance(res_norm, np.floating)
    res_norm = float(res_norm)

    t4 = perf_counter()

    print(f"Time 1: {t2 - t1}")
    print(f"Time 2: {t3 - t2}")
    print(f"Time 3: {t4 - t3}")
    print("Results:", res_sparse, res_dense, res_norm)

    tols = []
    diffs = []
    times = []
    for tol in [0, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
        t1 = perf_counter()
        eigenvalues_sparse = sparse_linalg.eigsh(diff_sparse, return_eigenvectors=False, tol=tol, k=4)
        res_sparse = np.sum(np.abs(eigenvalues_sparse)) / 2
        t2 = perf_counter()
        diff = abs(res_dense - res_sparse)
        time = t2 - t1
        tols.append(tol)
        diffs.append(diff)
        times.append(time)
    plt.plot(tols, diffs, label="Difference")
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("Tolerance")
    ax = plt.gca()
    twin = twin_axis(ax)
    plt.plot(tols, times, label="Time", color="tab:red")
    ax.set_ylabel(f"Diff in result")
    twin.set_ylabel(f"sparse time")
    twin.set_yscale("log")
    plt.show()

    print("Done.")

    
    # Sum the absolute values of eigenvalues and divide by 2
    return None

