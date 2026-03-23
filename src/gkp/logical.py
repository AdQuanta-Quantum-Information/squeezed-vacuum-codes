import numpy as np
import qutip as qt

from src.utils.prints import ProgressBar

sqrt_pi = np.sqrt(np.pi)

__all__ = ["gkp_logical"]


def gkp_logical(
    logical_val: int,
    N: int,
    Delta: float = 0.2,
    kappa: float = 0.2,
    s_max: int | None = None,
    ampl_cutoff: float = 1e-12,
    _prog_bar: bool = True,
) -> qt.Qobj:
    """
    Approximate square-lattice GKP logical |logical_val> (logical_val=0 or 1) in truncated Fock basis.

    Conventions:
      [q,p]=i, a=(q+ip)/sqrt(2).
      Peaks at q = (2s+mu)*sqrt(pi).
      Each peak has q-stddev Delta (Var(q)=Delta^2).
      Envelope weights exp[-(kappa*q_s)^2/2].
    """
    if logical_val not in (0, 1):
        raise ValueError("mu must be 0 or 1.")

    r = np.log(1.0 / (np.sqrt(2.0) * Delta))
    vac = qt.basis(N, 0)
    peak_state = qt.squeeze(N, r) @ vac

    if s_max is None:
        mmax = np.sqrt(2.0 * np.log(1.0 / ampl_cutoff)) / (kappa * np.sqrt(np.pi))
        s_max = int(np.ceil((mmax - logical_val) / 2.0))
        s_max = max(s_max, 1)

    psi = 0 * peak_state
    if _prog_bar:
        s_values = ProgressBar(
            range(-s_max, s_max + 1),
            prefix=f"building |{logical_val}_L⟩  ",
            expected_end=s_max * 2 + 1,
        )
    else:
        s_values = range(-s_max, s_max + 1)

    for s in s_values:
        q_s = (2 * s + logical_val) * sqrt_pi
        weight = np.exp(-0.5 * (kappa * q_s) ** 2)
        alpha = q_s / np.sqrt(2.0)  # because dq = sqrt(2) * Re(alpha)
        psi += weight * (qt.displace(N, alpha) @ peak_state)

    return psi  # not normalized; caller can .unit()
