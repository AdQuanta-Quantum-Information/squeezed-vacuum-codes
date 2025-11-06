from typing import Literal, overload, Final

import qutip as qt

import numpy as np
π : Final[float] = np.pi

if __name__ == "__main__":
    from __init__ import add_root_to_path
    add_root_to_path()

## Get stuff from the general src folder
from src.quantum.visualizations.wigner_function import plot_plain_wigner
from src.utils.visuals import matplotlib_support



def rotation(num_modes:int, θ:float) -> qt.Qobj:
    n = qt.num(num_modes)
    exponent : qt.Qobj = 1j * θ * n
    unitary = exponent.expm()
    return unitary


def main_example(
    num_modes : int = 100
):
    ψ = qt.basis(num_modes, 0)
    ψ = qt.squeeze(num_modes, -1.5) @ ψ
    plot_plain_wigner(ψ)

    ψ = rotation(num_modes, π/4) @ ψ
    plot_plain_wigner(ψ)
    pass


if __name__ == "__main__":
    main_example()