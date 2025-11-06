from typing import TypeAlias, TypeVar
import numpy as np

from ...states._base import StatesSum, _StateBase, KetBra


_StateType = TypeVar("_StateType", bound=_StateBase)


def overlap(psi1:StatesSum[_StateType], psi2:StatesSum[_StateType]) -> float:
    ## Checks:
    if not isinstance(psi1, StatesSum) or not isinstance(psi2, StatesSum):
        raise TypeError("The states should be both StatesSum")
    
    assert type(psi1) is type(psi2), "The states should be of the same type"
    assert psi1.ket_or_bra is KetBra.Ket
    assert psi2.ket_or_bra is KetBra.Ket


    ## complexity of O(n^2):
    total : complex = 0.0
    for ket1 in psi1:
        for ket2 in psi2:

            assert type(ket1) is type(ket2), "The states should be of the same type"                            
            bra2 = ket2.inverted()
            this_overlap : complex = bra2 | ket1

            total += this_overlap

    res = np.power(np.abs(total), 2)

    return res