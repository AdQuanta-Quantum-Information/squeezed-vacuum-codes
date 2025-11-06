from typing import Callable
import numpy as np
from qutip import Qobj
import qutip

from functools import cache

@cache
def get_operator(input: Callable[..., Qobj]|str, N: int, to_the_power:int=1) -> Qobj:
    """ To save time, only create a, a^dagger and n operators once.
    """
    assert isinstance(N, int) and N > 0, "`N` must be a positive integer."
    assert isinstance(to_the_power, int), "`to_the_power` must be a positive integer."

    if isinstance(input, str):
        name = input
        match name:
            case "identity":
                func = lambda N_: qutip.qeye(N_)
            case "create":
                func = lambda N_: qutip.create(N_)
            case "destroy":
                func = lambda N_: qutip.destroy(N_)
            case "num": 
                func = lambda N_: qutip.num(N_)
            case "create_sqr":
                func = lambda N_: get_operator("create", N_, to_the_power=2)
            case "destroy_sqr":
                func = lambda N_: get_operator("destroy", N_, to_the_power=2)
            case _:
                raise ValueError(f"Unknown operator name '{name!r}'.")
            
    elif callable(input):
        func = input
    else: 
        raise TypeError(f"Input must be a callable or a string, not {type(input)!r}.")    

    op = func(N)
    if to_the_power == 0:
        return get_operator("identity", N)
    elif to_the_power == 1:
        return op
    elif to_the_power < 0:
        raise ValueError("`to_the_power` must be a positive integer.")
    else:
        ## Recursion:
        op_to_the_j_minus_1 = get_operator(input, N, to_the_power - 1)
        return op_to_the_j_minus_1 * op



def num(N: int) -> Qobj:
    return get_operator("num", N)


def create(N: int) -> Qobj:
    return get_operator("create", N)


def destroy(N: int) -> Qobj:
    return get_operator("destroy", N)