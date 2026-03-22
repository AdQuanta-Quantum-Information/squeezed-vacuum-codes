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
