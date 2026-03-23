from typing import overload, Literal
import qutip as qt

import numpy as np
from numpy import pi as π
sqrt_pi = np.sqrt(np.pi)



if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.visualizations import plot_light_states, plot_fock_distribution
from src.utils.prints import ProgressBar



def gkp_params_from_nbar(
    nbar: float,
    ratio_kappa_over_Delta: float = 1.0,
    ampl_cutoff: float = 1e-12,
):
    """
    Map a single target mean photon number nbar to (Delta, kappa) using:
        nbar ≈ 1/(4 Delta^2) + 1/(4 kappa^2)
    plus a chosen ratio: kappa = ratio * Delta.

    Default ratio=1 => symmetric approximate GKP (Delta=kappa).

    Returns:
      Delta, kappa, r, s_max
    where r is the squeezing parameter for a q-squeezed vacuum with Var(q)=Delta^2,
    and s_max is a reasonable truncation of the peak index sum based on the envelope.
    """
    if nbar <= 0:
        raise ValueError("nbar must be > 0.")
    if ratio_kappa_over_Delta <= 0:
        raise ValueError("ratio_kappa_over_Delta must be > 0.")
    if not (0 < ampl_cutoff < 1):
        raise ValueError("ampl_cutoff must be between 0 and 1.")

    rho = float(ratio_kappa_over_Delta)

    # 1. Account for vacuum fluctuations (optional, improves low-n accuracy)
    # Energy from quadratures = nbar + 0.5
    target_energy = nbar + 0.5 
    
    # 2. Correct Formula: E = 1/(4*kappa^2) + 1/(8*Delta^2)
    # Substitute kappa = rho * Delta  =>  E = 1/(4*rho^2*Delta^2) + 1/(8*Delta^2)
    # E = (1/Delta^2) * ( 1/(4*rho^2) + 1/8 )
    
    term = (1.0 / (4.0 * rho * rho)) + (1.0 / 8.0)
    Delta = np.sqrt(term / target_energy)

    kappa = rho * Delta

    # For q-squeezed vacuum: Var(q) = Delta^2 = (1/2) e^{-2r}
    r = np.log(1.0 / (np.sqrt(2.0) * Delta))

    # Choose s_max so outermost envelope weight exp[-(kappa*q_s)^2/2] < ampl_cutoff
    # Use mu=0 worst case for envelope extent (slightly more conservative).
    # q_s = (2s+mu)*sqrt(pi)
    mmax = np.sqrt(2.0 * np.log(1.0 / ampl_cutoff)) / (kappa * np.sqrt(np.pi))
    s_max = int(np.ceil(mmax / 2.0))
    s_max = max(s_max, 1)

    return Delta, kappa, r, s_max



def gkp_logical(
    logical_val, N, Delta=0.2, kappa=0.2, s_max=None, ampl_cutoff=1e-12, _prog_bar=True
):
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
        s_values = ProgressBar(range(-s_max, s_max + 1), prefix=f"building |{logical_val}_L⟩  ", expected_end=s_max*2+1)
    else:
        s_values = range(-s_max, s_max + 1)

    for s in s_values:
        q_s = (2 * s + logical_val) * sqrt_pi
        weight = np.exp(-0.5 * (kappa * q_s) ** 2)
        alpha = q_s / np.sqrt(2.0)  # because dq = sqrt(2) * Re(alpha)
        psi += weight * (qt.displace(N, alpha) @ peak_state)

    # return psi.unit()
    return psi  # not, final state is not normalized


@overload
def gkp_from_nbar(*args, return_meta: Literal[True] = ..., **kwargs) -> tuple[qt.Qobj, dict]: ...
@overload
def gkp_from_nbar(*args, return_meta: Literal[False], **kwargs) -> qt.Qobj: ...
def gkp_from_nbar(
    logical_value: int,
    N: int,
    nbar_target: float,
    ratio_kappa_over_Delta: float = 1.0,
    ampl_cutoff: float = 1e-12,
    return_meta: bool = True,
    _prog_bar: bool = True,
) -> qt.Qobj | tuple[qt.Qobj, dict]:
    """
    Build an approximate GKP logical state using only nbar_target (plus mu, N).

    The mapping is controlled by ratio_kappa_over_Delta:
      kappa = ratio * Delta
      nbar_target ≈ 1/(4Delta^2) + 1/(4kappa^2)

    Returns:
      psi  (and optionally meta dict with Delta,kappa,r,s_max,nbar_actual,overlap_ready etc.)
    """
    Delta, kappa, r, s_max = gkp_params_from_nbar(
        nbar_target, ratio_kappa_over_Delta=ratio_kappa_over_Delta, ampl_cutoff=ampl_cutoff
    )

    psi = gkp_logical(logical_value, N, Delta=Delta, kappa=kappa, s_max=s_max, ampl_cutoff=ampl_cutoff, _prog_bar=_prog_bar)

    if not return_meta:
        return psi

    nbar_actual = qt.expect(qt.num(N), psi)

    meta = {
        "nbar_target": float(nbar_target),
        "nbar_actual": float(np.real_if_close(nbar_actual)),
        "Delta": float(Delta),
        "kappa": float(kappa),
        "r": float(r),
        "s_max": int(s_max),
        "ratio_kappa_over_Delta": float(ratio_kappa_over_Delta),
        # useful diagnostic: squeezing in dB relative to vacuum Var(q)=1/2
        "q_squeezing_dB": float(-10.0 * np.log10(2.0 * Delta * Delta)),
    }
    return psi, meta


def _gkp_test_single(
    nbar:float = 2.0,
    N:int = 100,
    verbose:bool = True,

):
    psi0, meta0 = gkp_from_nbar(logical_value=0, N=N, nbar_target=nbar)
    psi1, meta1 = gkp_from_nbar(logical_value=1, N=N, nbar_target=nbar)

    overlap = psi0.overlap(psi1)  # should be small for good GKP approx

    if verbose:
        print("N:", N)
        print("meta0:", meta0)
        print("<0|1>:", overlap)

        plot_light_states([psi0, psi1])

        print("Done.")
    
    return meta0


def _gkp_test_N():
    N_vec = np.linspace(10, 100, 3, dtype=int)
    nbar_targets = np.linspace(0.1, 10.0, 15)


    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(8, 6))
    plt.plot(nbar_targets, nbar_targets, linestyle='--', color='gray', label='Target nbar')



    ## Compute:
    for N in ProgressBar(N_vec):
        nbar_actuals = []
        for nbar_target in ProgressBar(nbar_targets):
            # print(f"Testing nbar={nbar_target:.2f}...")
            meta = _gkp_test_single(nbar=nbar_target, verbose=False, N=N)
            nbar_actuals.append(meta['nbar_actual'])

        plt.plot(nbar_targets, nbar_actuals, marker='o', label=f'{N}')
        plt.show()

    ## Plot:
    plt.xlabel('Target nbar')
    plt.ylabel('Actual nbar')
    plt.title('Actual vs Target nbar for Approximate GKP States')
    plt.grid()
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

    print("All tests done.")


def _gkp_test_nbar_recommendation():
    nbar_targets = np.linspace(0.1, 50.0, 15)
    nbar_actuals = []


    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(8, 6))
    plt.plot(nbar_targets, nbar_targets, linestyle='--', color='gray', label='Target nbar')



    ## Compute:
    for nbar_target in ProgressBar(nbar_targets):
        N = recommended_N_from_nbar(nbar_target)
        # print(f"Testing nbar={nbar_target:.2f}...")
        meta = _gkp_test_single(nbar=nbar_target, verbose=False, N=N)
        nbar_actuals.append(meta['nbar_actual'])


    ## Plot:
    plt.plot(nbar_targets, nbar_actuals, marker='o', label=f'{N}')
    plt.xlabel('Target nbar')
    plt.ylabel('Actual nbar')
    plt.title('Actual vs Target nbar for Approximate GKP States')
    plt.grid()
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

    print("All tests done.")



if __name__ == "__main__":
    # _gkp_test_single()
    # _gkp_test_N()
    _gkp_test_nbar_recommendation()
