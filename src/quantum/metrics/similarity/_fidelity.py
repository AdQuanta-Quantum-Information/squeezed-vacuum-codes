from qutip import Qobj
import numpy as np
from scipy import linalg, sparse
from typing import TypeVar, Optional, Callable

from src.utils import assertions
from src.algorithms.optimization.types import OptimizationBadPoint

from ._trace_distance import trace_distance

_QuantumState = TypeVar("_QuantumState", Qobj, np.ndarray, sparse.lil_matrix)


def _common_get_trace(
    rho:_QuantumState,
    sigma:_QuantumState,
    sqrt_op:Callable[[_QuantumState], _QuantumState],
    trace_op:Callable[[_QuantumState], float],
    multiply_op:Callable[[_QuantumState, _QuantumState, _QuantumState], _QuantumState],
) -> float:
    sqrt_rho = sqrt_op(rho)
    product = multiply_op(sqrt_rho, sigma, sqrt_rho)
    prod_sqrt = sqrt_op(product)
    trace = trace_op(prod_sqrt)
    return trace


def fidelity_approx(
    rho:_QuantumState,
    sigma:_QuantumState,
) -> float:
    d = trace_distance(rho, sigma)    
    lower = (1-d)**2 
    upper = 1 - d**2 
    return (lower + upper) / 2


def fidelity(
    rho:_QuantumState, 
    sigma:_QuantumState,
    check_trace:Optional[bool]=True
) -> float:
    # The fidelity between two quantum states:
    if isinstance(rho, Qobj) and isinstance(sigma, Qobj):
        trace = _common_get_trace(
            rho, sigma, 
            sqrt_op=Qobj.sqrtm, 
            trace_op=Qobj.tr,
            multiply_op=lambda x, y, z: x * y * z
        )

    elif isinstance(rho, np.ndarray) and isinstance(sigma, np.ndarray):
        trace = _common_get_trace(
            rho, sigma, 
            sqrt_op=linalg.sqrtm, 
            trace_op=np.trace,
            multiply_op=lambda x, y, z: x @ y @ z
        )

    elif isinstance(rho, sparse.lil_matrix) and isinstance(sigma, sparse.lil_matrix):
        m1 = rho.toarray()
        m2 = sigma.toarray()
        kwargs = dict(check_trace=check_trace)
        return fidelity(m1, m2, **kwargs)


        # rho_qt = Qobj(rho)
        # sigma_qt = Qobj(sigma)
        # trace = _common_get_trace(
        #     rho_qt, sigma_qt, 
        #     sqrt_op=lambda x: Qobj.sqrtm(x), 
        #     trace_op=lambda x: Qobj.tr(x),
        #     multiply_op=lambda x, y, z: x * y * z
        # )

        # from scipy.sparse.linalg import matrix_power
        # sqrt = lambda x: matrix_power(x, 0.5)
        # sqrt_rho = sqrt(rho)
        # big_sqrt : np.ndarray = sqrt(sqrt_rho @ sigma @ sqrt_rho) 
        # trace1 = big_sqrt.trace()
        # trace2 = big_sqrt.diagonal().sum()
        
        
    else:
        raise TypeError(f"The states should be both qutip.Qobj ; numpy.ndarray or scipy.sparse.csr_matrix")
    
    if check_trace:
        try:
            trace = assertions.real(trace)
        except:
            raise OptimizationBadPoint(f"Got a non-real trace: {trace!r}")
    else:
        trace = np.real(trace)
    return trace**2
    
