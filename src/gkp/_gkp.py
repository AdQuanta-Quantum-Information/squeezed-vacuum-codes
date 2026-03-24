from typing import overload, Literal
import qutip as qt

import numpy as np



if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.visualizations import plot_light_states, plot_fock_distribution
from src.utils.prints import ProgressBar
from src.gkp.num_photons import lookup_gkp_params_from_nbar, estimate_gkp_params_from_nbar, GKPParams
from src.gkp.logical import gkp_logical


def _gkp_params_from_nbar(
    nbar: float,
    /, 
    ratio_kappa_over_Delta: float = 1.0,
    amp_cutoff: float = 1e-12,
    logical_value: int = 0,
    *, 
    use_estimation: bool = False,
) -> GKPParams:
    """
    Lookup (lazily cached) mapping from target nbar to (Delta, kappa, r, s_max).

    Uses a cached uniform Delta->nbar table and inverts it by interpolation.
    """
    if use_estimation:
        return estimate_gkp_params_from_nbar(
            nbar,
            ratio_kappa_over_Delta=ratio_kappa_over_Delta,
            amp_cutoff=amp_cutoff
        )
    else:
        return lookup_gkp_params_from_nbar(
            nbar,
            ratio_kappa_over_Delta=ratio_kappa_over_Delta,
            amp_cutoff=amp_cutoff,
            logical_value=logical_value,
        )



@overload
def gkp_from_nbar(*args, return_meta: Literal[True] = ..., **kwargs) -> tuple[qt.Qobj, dict]: ...
@overload
def gkp_from_nbar(*args, return_meta: Literal[False], **kwargs) -> qt.Qobj: ...
def gkp_from_nbar(
    logical_value: int,
    N: int,
    nbar_target: float,
    ratio_kappa_over_Delta: float = 1.0,
    amp_cutoff: float = 1e-12,
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
    Delta, kappa, r, s_max = _gkp_params_from_nbar(
        nbar_target,
        ratio_kappa_over_Delta=ratio_kappa_over_Delta,
        amp_cutoff=amp_cutoff,
        logical_value=logical_value
    )

    psi = gkp_logical(logical_value, N, Delta=Delta, kappa=kappa, s_max=s_max, ampl_cutoff=amp_cutoff, _prog_bar=_prog_bar)

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
    from src.gkp.fock_cutoff_recommendation import recommended_N_from_nbar


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
