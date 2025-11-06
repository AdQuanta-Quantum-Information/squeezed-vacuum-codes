from typing import Generator, Any


import numpy as np
from qutip import Qobj
from ...utils import numerics, strings


def _is_tensor_product_state(state:Qobj) -> bool:
    non_one_count = 0
    def _count_non_one(d:int):
        nonlocal non_one_count
        if d != 1:
            non_one_count += 1
    
    if not state.isket and len(state.dims) <= 1:
        return False
    
    # Check if more than a single dimension is not-1:
    for dim in state.dims:
        if isinstance(dim, list):
            for d in dim:
                _count_non_one(d)
        elif isinstance(dim, int):
            _count_non_one(dim)
        else:
            raise TypeError(f"Invalid dimension type: {dim}")

    return non_one_count > 1


def _str_without_brackets(s:str) -> str:
    return s[1:-1]


def _iterate_state(state:Qobj) -> Generator[tuple[str, Any], None, None]:
    if isinstance(state, Qobj):
        if _is_tensor_product_state(state):
            dims = state.dims
            data_tensor = state.data_as('ndarray').reshape(*dims[0])
            for indices, value in np.ndenumerate(data_tensor):
                s = _str_without_brackets(str(indices))
                yield s, value

        else:    
            for i, v in enumerate(state.data_as('ndarray')):
                yield f"{i}", v[0] 


def fock_str(state:Qobj, prefix:str="", max_terms:int=40) -> str:

    terms_counter = 0
    s = prefix
    for crnt_state, weight in _iterate_state(state):
        weight = numerics.force_near_pure_complex(weight)
        if abs(weight-0) < 1e-9:
            continue

        weight_str = strings.simplified_complex_str(weight)
        if weight_str=="1":
            weight_str = ""
        
        if np.isreal(weight) and weight_str.startswith("-"):
            if s.endswith("+ "):
                weight_str = weight_str[1:]
                s = s[:-2] + "- "

        terms_counter += 1

        s += f"{weight_str}|{crnt_state}⟩ + "

        if terms_counter >= max_terms:
            s += "..."
            break

    if s.endswith(" + "):
        s = s[:-3]

    return s


def print_fock(state:Qobj, prefix:str="", max_terms:int=40) -> None:
    print(fock_str(state, prefix, max_terms))


def empty_fock_state(dim:int) -> Qobj:
    """
    Create an empty Fock state with the given dimension.
    
    Args:
        dim (int): The dimension of the Fock state.
    
    Returns:
        Qobj: The empty Fock state.
    """
    return Qobj(np.zeros((dim, 1)), dims=[[dim], [1]], type='ket', isherm=False)