# ==================================================================================== #
# |                                   Imports                                        | #
# ==================================================================================== #

from typing import Any, Union, Optional, TypeVar, Iterator, Literal
import numpy as np
from qutip import Qobj

# ==================================================================================== #
# |                                  Constants                                       | #
# ==================================================================================== #
TOLERANCE = 0.00001  # Used for distance validation
IMAGINARY_TOLERANCE = 0.0001


Bit = Literal[0, 1]


# ==================================================================================== #
# |                               Inner Functions                                    | #
# ==================================================================================== #

def _assert(condition:bool, reason:Optional[str]=None, default_reason:Optional[str]=None, got:Any=None):
    # if condition passed successfully:
    if condition:
        return
    ## If error is needed:
    # Get error message
    msg = ""
    if reason is not None and isinstance(reason, str):
        msg = reason
    elif default_reason is not None and isinstance(default_reason, str):
        msg = default_reason
    if got is not None:
        msg += f" (got {got})"
    raise AssertionError(msg)


def _is_positive_semidefinite(m:np.matrix) -> bool:
    eigen_vals = np.linalg.eigvals(m)
    if np.any(np.imag(eigen_vals)>TOLERANCE):  # Must be real
        return False
    if np.any(np.real(eigen_vals)<-TOLERANCE):  # Must be positive
        return False
    return True

def _is_hermitian(m:np.matrix) -> bool:
    diff = m.H-m
    shape = m.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            if abs(diff[i,j])>TOLERANCE:
                return False
    return True  

# ==================================================================================== #
# |                              Declared Functions                                  | #
# ==================================================================================== #

def real(x:complex|float|int, /, reason:Optional[str]=None) -> float|int:
    if np.isclose(np.imag(x), 0):
        return np.real(x)
    imaginary_factor = abs(np.imag(x))/abs(np.real(x))
    _assert( imaginary_factor<IMAGINARY_TOLERANCE, reason=reason, default_reason=f"Must be real", got=x)
    return float(np.real(x))

def integer(x:float|int, /, reason:Optional[str]=None) -> int:
    _assert( round(x) == x, reason=reason, got=x)
    return int(x)

def index(x:float|int, /, reason:Optional[str]=None) -> int:
    x = integer(x, reason=reason)
    _assert( x >= 0, reason=reason, got=x)
    return x

def bit(x:float|int, /, reason:Optional[str]=None) -> Bit:
    x = integer(x, reason=reason)
    _assert( x in [0, 1], reason=reason, got=x)
    return 0 if x == 0 else 1  # Explicitly return Literal[0, 1]

def even(x:float|int, /, reason:Optional[str]=None) -> int:
    x = integer(x, reason=reason)
    _assert( float(x)/2 == int(int(x)/2), reason=reason, got=x)
    return x

def odd(x:float|int, /, reason:Optional[str]=None ) -> int:
    x = even(x-1, reason=reason) + 1
    return x

def logical(x:bool|int, /, reason:Optional[str]=None ) -> bool:
    if isinstance(x, bool):
        return x
    else:
        x = bit(x, reason=reason)  # also checks if 0 or 1
        if x == 0:
            return False
        elif x == 1:
            return True
        else:
            raise ValueError("bug: Not an expected option")

def density_matrix(m:np.ndarray|Qobj, /,*, reason:Optional[str]=None, robust_check:bool=True) -> np.ndarray:
    if isinstance(m, Qobj):
        m_ = m.full()
    elif isinstance(m, np.ndarray):
        m_ = m
    else:
        raise TypeError(f"Density Matrix must be a numpy array or a Qobj. Got {type(m)!r}")
    
    _assert( isinstance(m_, (np.matrix, np.ndarray)), reason=reason, default_reason="Must be a matrix type" )
    if not isinstance(m_, np.matrix):
        m_ = np.matrix(m_)
    _assert( len(m_.shape)==2, reason=reason, default_reason="Must be a matrix" )
    _assert( m_.shape[0]==m_.shape[1], reason=reason, default_reason="Must be a square matrix" )
    _assert( abs(np.trace(m_)-1)<TOLERANCE, reason=reason, default_reason="Density Matrix must have trace==1")
    if robust_check:
        _assert( _is_hermitian(m_), reason=reason, default_reason="Density Matrix must be Hermitian")
        _assert( _is_positive_semidefinite(m_), reason=reason, default_reason="Density Matrix must be positive semidefinite")
    return m_


def depleted_iterator(it:Iterator) -> Iterator:
    try:
        next(it)
    except:
        pass
    else:
        raise AssertionError(f"Iterator is not depleted!")
    return it