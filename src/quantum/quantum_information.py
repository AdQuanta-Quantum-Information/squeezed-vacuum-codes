import numpy as np
import qutip as qt


def gram_schmidt_orthonormal_pair(
    psi0: qt.Qobj,
    psi1: qt.Qobj,
    tol: float = 1e-12,
) -> tuple[qt.Qobj, qt.Qobj]:
    """Return an orthonormal pair spanning the same 2D subspace as (psi0, psi1)."""
    phi0 = psi0.unit()
    psi1n = psi1.unit()

    proj = phi0.overlap(psi1n)
    residual = psi1n - proj * phi0
    residual_norm = residual.norm()
    if residual_norm < tol:
        raise ValueError("Cannot orthonormalize: states are numerically collinear.")

    phi1 = residual / residual_norm
    return phi0, phi1


def lowdin_orthonormal_pair(
    psi0: qt.Qobj,
    psi1: qt.Qobj,
    tol: float = 1e-12,
) -> tuple[qt.Qobj, qt.Qobj]:
    """Return symmetric (Lowdin) orthonormalized pair spanning (psi0, psi1)."""
    p0 = psi0.unit()
    p1 = psi1.unit()

    overlap = p0.overlap(p1)
    S = np.array(
        [
            [1.0 + 0.0j, overlap],
            [np.conj(overlap), 1.0 + 0.0j],
        ],
        dtype=complex,
    )

    evals, evecs = np.linalg.eigh(S)
    if float(np.min(evals)) < tol:
        raise ValueError("Cannot orthonormalize: overlap matrix is near-singular.")

    S_inv_sqrt = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.conj().T
    phi0 = S_inv_sqrt[0, 0] * p0 + S_inv_sqrt[1, 0] * p1
    phi1 = S_inv_sqrt[0, 1] * p0 + S_inv_sqrt[1, 1] * p1
    return phi0.unit(), phi1.unit()
