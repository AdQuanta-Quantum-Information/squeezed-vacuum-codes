import numpy as np
from matplotlib import pyplot as plt
import qutip as qt

from typing import Iterable, Literal, NamedTuple

# Use project-standard path setup
try:
    from src.gkp._bootstrap import add_root_to_path
except ModuleNotFoundError:
    from _bootstrap import add_root_to_path
add_root_to_path()


from src.utils.prints import ProgressBar
from src.utils.caches import cache
from src.utils.inverse_lookup import MonotonicDirection, MonotonicInverseLookup
from src.gkp.fock_cutoff_recommendation import recommended_N_from_nbar
from src.gkp.logical import gkp_logical


# Shared Delta-grid resolution for inverse lookup interpolation.
DELTA_X_GRID: float = 0.02

# Smallest positive nbar we solve against to avoid singular behavior at exactly 0.
MIN_POSITIVE_NBAR: float = 1e-16


class GKPParams(NamedTuple):
    Delta: float
    kappa: float
    r: float
    s_max: int


def _mean_photon_number(state: qt.Qobj) -> float:
    """Compute ⟨n⟩ for a single-mode state without importing mean_photon_number (avoids cycles)."""
    dim = state.dims[0][0]
    n_op = qt.num(dim)
    val = qt.expect(n_op, state)
    return float(val)



def _gkp_parameter_solver_error_vs_qutip_mean(
    target_mean_photon_numbers: Iterable[float] = np.linspace(0.1, 5.1, 21),
    N_values: Iterable[int] = (50, 100, 150),
    logical_value: int | Literal['+'] = '+',
) -> None:
    """Evaluate the physical accuracy of the GKP geometric approximation for nbar.
    
    Plots the signed error: (Qutip mean photons) - (target mean photons) 
    using the pure target_mean_photon -> Delta/kappa conversion approximation.
    Iterates over different Fock cutoff dimensions N.
    """
    targets = list(target_mean_photon_numbers)
    N_values = list(N_values)
    
    points_per_N = {N: [] for N in N_values}

    for N in ProgressBar(N_values, prefix="per N:    "):
        for target_mean_photon_number in ProgressBar(targets, prefix="per nbar:"):
            # We explicitly bypass the optimizer to test the pure scaling formula drop-off.
            parameter = target_mean_photon_number
            
            # Construct the exact QuTiP state given the analytic parameter.
            if logical_value == '+':
                from src.codes_built_in_superposition import gkp_code_state
                state_0 = gkp_code_state(parameter, N, qubit_logical_value=0, _prog_bar=False)
                state_1 = gkp_code_state(parameter, N, qubit_logical_value=1, _prog_bar=False)
                qutip_state = (state_0 + state_1).unit()
            else:
                from src.codes_built_in_superposition import gkp_code_state
                qutip_state = gkp_code_state(
                    parameter, N, qubit_logical_value=logical_value, _prog_bar=False
                )

            # Measure actual mean photon number and calculate discrepancy
            qutip_mean_photons = _mean_photon_number(qutip_state)
            signed_error = qutip_mean_photons - target_mean_photon_number
            
            points_per_N[N].append((target_mean_photon_number, signed_error))

    plt.figure(figsize=(10, 6))
    markers = ['o', 's', '^', 'D', 'v', '<', '>']

    for i, N in enumerate(N_values):
        if not points_per_N[N]:
            continue
        x_values, y_values = zip(*points_per_N[N])
        plt.plot(
            x_values,
            y_values,
            label=f"N={N}",
            linestyle=':',
            marker=markers[i % len(markers)],
            markersize=5,
            linewidth=2.0,
            markevery=2,
        )

    plt.axhline(0.0, color='black', linestyle=':', linewidth=1.2, alpha=0.8)
    plt.xlabel("Target Mean Photon Number")
    plt.ylabel("Qutip Mean - Target Mean")
    plt.title(f"GKP Target vs Actual Mean Photon Number\nfor state |{logical_value}⟩")
    plt.legend()
    plt.grid(True, alpha=0.35)
    plt.show()

    print("Done.")



# ---------------------------------------------------------------------------
# Parameter estimation and lookup support
# ---------------------------------------------------------------------------


def _s_max_from_params(kappa: float, ampl_cutoff: float) -> int:
    mmax = np.sqrt(2.0 * np.log(1.0 / ampl_cutoff)) / (kappa * np.sqrt(np.pi))
    s_max = int(np.ceil(mmax / 2.0))
    return max(s_max, 1)


def estimate_gkp_params_from_nbar(
    nbar: float,
    ratio_kappa_over_Delta: float = 1.0,
    ampl_cutoff: float = 1e-12,
) -> GKPParams:
    """
    Fast analytic estimate: map target nbar -> (Delta, kappa, r, s_max).

    This is the previous closed-form approximation, migrated here so it can
    be reused and contrasted with the grid-backed solver below.
    """
    if nbar <= 0:
        raise ValueError("nbar must be > 0.")
    if ratio_kappa_over_Delta <= 0:
        raise ValueError("ratio_kappa_over_Delta must be > 0.")
    if not (0 < ampl_cutoff < 1):
        raise ValueError("ampl_cutoff must be between 0 and 1.")

    rho = float(ratio_kappa_over_Delta)

    target_energy = nbar + 0.5  # account for vacuum fluctuations

    # E = 1/(4*kappa^2) + 1/(8*Delta^2); kappa = rho * Delta
    term = (1.0 / (4.0 * rho * rho)) + (1.0 / 8.0)
    Delta = np.sqrt(term / target_energy)

    kappa = rho * Delta

    # For q-squeezed vacuum: Var(q) = Delta^2 = (1/2) e^{-2r}
    r = np.log(1.0 / (np.sqrt(2.0) * Delta))

    s_max = _s_max_from_params(kappa, ampl_cutoff)

    return GKPParams(float(Delta), float(kappa), float(r), int(s_max))


def _build_state_and_nbar(
    Delta: float,
    kappa: float,
    N: int,
    logical_value: int,
    ampl_cutoff: float,
) -> tuple[qt.Qobj, float]:
    s_max = _s_max_from_params(kappa, ampl_cutoff)
    state = gkp_logical(
        logical_val=logical_value,
        N=N,
        Delta=Delta,
        kappa=kappa,
        s_max=s_max,
        ampl_cutoff=ampl_cutoff,
        _prog_bar=False,
    )
    state.unit(inplace=True)
    nbar_actual = _mean_photon_number(state)
    return state, float(nbar_actual)


@cache(ram=False, disk=True)
def _nbar_only(
    Delta: float,
    kappa: float,
    N: int,
    logical_value: int,
    ampl_cutoff: float,
) -> float:
    """RAM-cached nbar evaluation without persisting the full state."""
    _, nbar_actual = _build_state_and_nbar(
        Delta=Delta,
        kappa=kappa,
        N=N,
        logical_value=logical_value,
        ampl_cutoff=ampl_cutoff,
    )
    return nbar_actual


@cache(ram=True, disk=True)
def _cached_inverse_lookup(
    *,
    ratio_kappa_over_Delta: float,
    ampl_cutoff: float,
    logical_value: int,
    N: int,
) -> MonotonicInverseLookup:
    rho = float(ratio_kappa_over_Delta)
    Delta0, _, _, _ = estimate_gkp_params_from_nbar(
        max(MIN_POSITIVE_NBAR, 0.2),
        ratio_kappa_over_Delta=rho,
        ampl_cutoff=ampl_cutoff,
    )

    class _GKPInverseLookup(MonotonicInverseLookup[float, float]):
        @classmethod
        def func(cls, x: float) -> float:
            Delta = x
            if Delta <= 0:
                raise ArithmeticError("Delta must stay positive")
            return _nbar_only(
                Delta=Delta,
                kappa=rho * Delta,
                N=N,
                logical_value=logical_value,
                ampl_cutoff=ampl_cutoff,
            )

    return _GKPInverseLookup(
        delta_x=DELTA_X_GRID,
        x0=max(float(Delta0), DELTA_X_GRID),
        direction=MonotonicDirection.INCREASING,
        initial_points=2,
    )


def lookup_gkp_params_from_nbar(
    nbar_target: float,
    *,
    ratio_kappa_over_Delta: float = 1.0,
    ampl_cutoff: float = 1e-12,
    logical_value: int = 0,
) -> GKPParams:
    """
    Inverse lookup from target nbar to (Delta, kappa, r, s_max).

    Builds (and disk-caches) a uniform Delta->nbar table, then inverts it by
    linear interpolation in nbar-space.
    """
    if nbar_target < 0:
        raise ValueError("nbar_target must be >= 0.")
    if DELTA_X_GRID <= 0:
        raise ValueError("DELTA_X_GRID must be > 0.")

    target_nbar = max(nbar_target, MIN_POSITIVE_NBAR)
    N = recommended_N_from_nbar(target_nbar)

    lookup = _cached_inverse_lookup(
        ratio_kappa_over_Delta=ratio_kappa_over_Delta,
        ampl_cutoff=ampl_cutoff,
        logical_value=logical_value,
        N=N,
    )

    Delta = lookup.x_from_y(target_nbar, clamp=True)
    kappa = float(ratio_kappa_over_Delta) * Delta
    r = np.log(1.0 / (np.sqrt(2.0) * Delta))
    s_max = _s_max_from_params(kappa, ampl_cutoff)

    return GKPParams(float(Delta), float(kappa), float(r), int(s_max))


def _test_lookup_vs_qutip(
    logical_value: int = 0,
    nbar_targets: Iterable[float] = np.linspace(0.1, 5.1, 21),
    ampl_cutoff: float = 1e-12,
) -> None:
    """Plot target nbar vs actual for lookup and analytic estimate (no execution here)."""

    targets = list(nbar_targets)
    lookup_actuals: list[float] = []
    estimate_actuals: list[float] = []

    for nbar_target in ProgressBar(targets, prefix="lookup test: "):
        N = recommended_N_from_nbar(nbar_target)

        # Lookup-based params
        Delta_l, kappa_l, _, _ = lookup_gkp_params_from_nbar(
            nbar_target,
            ratio_kappa_over_Delta=1.0,
            ampl_cutoff=ampl_cutoff,
            logical_value=logical_value,
        )
        nbar_l = _nbar_only(
            Delta=Delta_l,
            kappa=kappa_l,
            N=N,
            logical_value=logical_value,
            ampl_cutoff=ampl_cutoff,
        )
        lookup_actuals.append(nbar_l)

        # Analytic estimate path
        Delta_e, kappa_e, _, _ = estimate_gkp_params_from_nbar(
            nbar_target,
            ratio_kappa_over_Delta=1.0,
            ampl_cutoff=ampl_cutoff,
        )
        nbar_e = _nbar_only(
            Delta=Delta_e,
            kappa=kappa_e,
            N=N,
            logical_value=logical_value,
            ampl_cutoff=ampl_cutoff,
        )
        estimate_actuals.append(nbar_e)

    plt.figure(figsize=(8, 6))
    plt.plot(targets, targets, linestyle='--', color='gray', label='target nbar')
    plt.plot(targets, lookup_actuals, marker='o', label='lookup (cached grid)')
    plt.plot(targets, estimate_actuals, marker='s', label='analytic estimate')
    plt.xlabel("Target nbar")
    plt.ylabel("Actual nbar from QuTiP")
    plt.title("GKP nbar: lookup vs analytic estimate")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()

    plt.show()
    print("Test complete.") 





if __name__ == "__main__":
    _test_lookup_vs_qutip()