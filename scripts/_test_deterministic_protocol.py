import numpy as np
from qutip import basis, qeye, tensor, Qobj, metrics
from typing import Literal

π = np.pi

# ---------- Operators ----------
def shift_op(m: int) -> Qobj:
    """σ_+ (cyclic shift) on an m-level qudit."""
    data = np.zeros((m, m), dtype=complex)
    for j in range(m):
        data[(j + 1) % m, j] = 1.0
    return Qobj(data)


def Hx(x: float) -> Qobj:
    """'Partial Hadamard' on a qubit with amplitudes proportional to (x, 1-x)."""
    N = 1.0 / np.sqrt(x**2 + (1 - x)**2)
    a, b = N * x, N * (1 - x)
    return Qobj(np.array([[a, -b],
                          [b,  a]], dtype=complex))


def controlled_shift(m: int, ϕ: float = 0.0) -> Qobj:
    """Controlled-(e^{i phi} σ_+) with a qubit control and m-level target."""
    I_m = qeye(m)
    sigp = shift_op(m)
    P0 = Qobj(np.array([[1, 0], [0, 0]], dtype=complex))
    P1 = Qobj(np.array([[0, 0], [0, 1]], dtype=complex))
    return tensor(P0, I_m) + np.exp(1j * ϕ) * tensor(P1, sigp)


# ---------- Round and measurement ----------
def one_round_state(state: Qobj, H: Qobj, Cshift: Qobj, m: int) -> Qobj:
    """Apply (H ⊗ I) · Cshift · (H ⊗ I) to a state vector."""
    U_H = tensor(H, qeye(m))
    U_round = U_H * Cshift * U_H
    return U_round * state


def measure_qubit_zero(state: Qobj, m: int):
    """Project onto |0><0| ⊗ I. Return (post_state, success_prob, qudit_ket)."""
    P0 = tensor(basis(2, 0) * basis(2, 0).dag(), qeye(m))
    amp = P0 * state
    p0 = np.real(amp.dag() * amp)
    if p0 <= 0:
        raise ValueError("Zero probabilirt")
        # return None, 0.0, None
    post = amp * (1.0 / np.sqrt(p0))  # normalized |0>⊗|phi>
    # Extract |phi> on the qudit: (⟨0| ⊗ I) |ψ_post⟩
    bra0 = tensor(basis(2, 0).dag(), qeye(m))
    qudit_ket = (bra0 * post)  # already normalized
    return post, p0, qudit_ket


def derive_phases(m:int) -> list[float]:
    return [π - 2*π*r/m for r in range(1, m)]


# ---------- Protocol driver ----------
def run_protocol_with_inputs(m=3, rounds=3, x:float|Literal[False]=0.5, phis=None, verbose=True):
    """
    Start in |0>_qubit ⊗ |0>_qudit(m).
    Each round: (H(x)⊗I) · C-(e^{i φ_k} σ_+) · (H(x)⊗I), then measure qubit in |0>.
    Returns final bipartite state (if postselect=False),
            or final post-selected qudit ket and success probability product (if postselect=True).
    """
    if phis is None:
        phis = [0.0] * rounds
    assert len(phis) == rounds

    ## H operator:
    if isinstance(x, float):
        H = Hx(x)
    elif x is False:
        H = Qobj([[1, 1], [1, -1]]) / np.sqrt(2)
    else:
        raise TypeError(f"Invalid x: {x!r}")


    # Initial state
    ψ : Qobj = tensor(basis(2, 0), basis(m, 0))
    total_p = 1.0
    qudit_ket : Qobj = None  # placeholder

    for k in range(rounds):
        ϕ = phis[k]
        Cshift = controlled_shift(m, ϕ=ϕ)
        ψ = one_round_state(ψ, H, Cshift, m)

        ψ, p0, qudit_ket = measure_qubit_zero(ψ, m)
        total_p *= p0
        if verbose:
            print(f"[Round {k+1}]  p(qubit=|0⟩) = {p0:.6f}")
        if p0 == 0.0:
            if verbose:
                print("Post-selection failed.")
            return None, 0.0

    if verbose:
        print(f"Overall post-selection success probability: {total_p:.6f}")
        print("Final qutrit/qudit ket (in computational basis):")
        print(qudit_ket.full().flatten())
    return qudit_ket, total_p


def run_protocol(m:int, verbose:bool=True):
    phis = derive_phases(m)
    rounds = len(phis)
    return run_protocol_with_inputs(m, rounds, x=False, phis=phis, verbose=True)
    


def run_test():
    m = 3
    rounds = 3
    x = 0.5  # true "Hadamard-like": a=b
    phis = [4*np.pi/3, 2*np.pi/3, np.pi]
    qudit, success_prob = run_protocol_with_inputs(m=m, rounds=rounds, x=x, phis=phis, verbose=True)

    # Check fidelity with uniform superposition on m=3
    if qudit is not None:
        target = (basis(m,0) + basis(m,1) + basis(m,2)).unit()
        F = metrics.fidelity(qudit, target) 
        print(f"Fidelity with (|0>+|1>+|2>)/√3: {F:.12f}")


def run_general_protocol():
    m = 4
    qudit, success_prob = run_protocol(m=m)

    # Check fidelity with uniform superposition on m=3
    if qudit is not None:
        equal_superpostion_qudit : Qobj = sum([basis(m, j) for j in range(m)])  #type: ignore
        equal_superpostion_qudit.unit(inplace=True)
        F = metrics.fidelity(qudit, equal_superpostion_qudit) 
        equal_string = "+".join([f"|{j}⟩" for j in range(m)])
        print(f"Fidelity with ({equal_string})/√{m}: {F:.12f}")



# ---------- Example: perfect 3-level superposition after 3 rounds ----------
# For m=3, the phase choice φ1=4π/3, φ2=2π/3, φ3=π with x=1/2 yields (|0>+|1>+|2>)/√3.
if __name__ == "__main__":
    # run_test()
    run_general_protocol()