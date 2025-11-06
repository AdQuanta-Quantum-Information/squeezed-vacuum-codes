import numpy as np
from qutip import basis, Qobj, zero_ket



from matplotlib import pyplot as plt


if __name__ == "__main__":
    from __init__ import add_root_to_path
    add_root_to_path()


from src.visualizations import plot_light_states, plot_fock_distribution
from src.squeezing_direction import squeezing_direction_to_squeezing_phase
from src.codes_built_in_superposition import simple_m_legged_code

from src.utils.prints import ProgressBar
from src.utils.caches import cache


DEFAULT_NUM_MOMENTS = 1_000


def logical_x(m:int, num_moments:int) -> Qobj:
    """
    Create the logical X operator for a simple m-legged squeeze code.

    Args:
        m (int): Number of legs in the squeeze code.
        num_moments (int): Number of moments to consider in the Fock basis.

    Returns:
        Qobj: The logical X operator as a Qutip object.
    """
    ## Start with the zero operator
    zero = zero_ket(num_moments)
    logical_x_op = zero @ zero.dag()
    

    for n in range(num_moments - m):
        ket = basis(num_moments, n)
        bra = basis(num_moments, (n + m)).dag()
        ket_bra = ket @ bra
        logical_x_op += ket_bra

    return logical_x_op


@cache(disk=True)
def compute_diff_norm(m:int, r:float, num_moments:int) -> float:
    ψ0, ψ1 = simple_m_legged_code(m=m, strength=r, num_moments=num_moments, code_type="squeeze")
    Xm = logical_x(m=m, num_moments=num_moments)

    diff = ψ1 - Xm@ψ0
    diff_norm = diff.norm()

    return diff_norm


def test1(
    m: int = 2,
    num_moments: int = DEFAULT_NUM_MOMENTS,
    r_vals: list[float] = np.linspace(0.1, 4.0, 11).tolist()
):
    norm_vals = []
    for r in ProgressBar(r_vals, prefix="r vals:      "):
        diff_norm = compute_diff_norm(m=m, r=r, num_moments=num_moments)
        norm_vals.append(diff_norm) 

    ## Print:
    plt.plot(r_vals, norm_vals, linewidth=4.0)
    plt.ylim(0, max(norm_vals)*1.1)
    plt.xlabel("Squeezing Strength r")
    plt.ylabel("Norm of Difference || |ψ1> - X_m |ψ0> ||")
    plt.show()

    ## Stop execution until user accepts
    input("Press Enter to continue...")

    print("Test complete.")


if __name__ == "__main__":
    test1()