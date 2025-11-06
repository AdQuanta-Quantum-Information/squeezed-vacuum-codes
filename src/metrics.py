from typing import Literal, TypeAlias, overload

import numpy as np

import warnings
from contextlib import nullcontext, contextmanager
from scipy.linalg import LinAlgWarning

from qutip import Qobj, fidelity


@contextmanager
def maybe_suppress_warnings(suppress: bool = True):
    """Context manager that conditionally suppresses LinAlgWarning."""
    if suppress:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=LinAlgWarning)
            yield
    else:
        yield


FunctionType : TypeAlias = Literal["overlap", "fidelity"]


@overload
def _function(ψ1:Qobj, ψ2:Qobj, function:Literal["overlap"]) -> complex: ...
@overload
def _function(ψ1:Qobj, ψ2:Qobj, function:Literal["fidelity"]) -> float: ...
def _function(ψ1:Qobj, ψ2:Qobj, function:FunctionType) -> complex|float:
    match function:
        case "fidelity":
            return fidelity(ψ1, ψ2)
        case "overlap":
            return ψ1.overlap(ψ2)
        case _:
            raise ValueError(f"Unknown function type: {function!r}")


def compute_cross_overlap_mat(
    states1: list[Qobj],
    states2: list[Qobj],
    function: FunctionType = "overlap",
    suppress_warnings: bool = True
) -> np.ndarray:

    ## Start with an empty n1 by n2 matrix:
    n1 = len(states1)
    n2 = len(states2)
    dtype = np.float64 if function == "fidelity" else np.complex128
    mat = np.zeros((n1, n2), dtype=dtype)

    with maybe_suppress_warnings(suppress_warnings):
        for i1, ψ1 in enumerate(states1):
            for i2, ψ2 in enumerate(states2):                
                mat[i1, i2] = _function(ψ1, ψ2, function)

    # ensure all entries are real
    assert np.all(np.abs(np.imag(mat))<1e-14), f"Imaginary parts in fidelity matrix: {np.imag(mat)!r}" 
    mat = np.real(mat)

    return mat
