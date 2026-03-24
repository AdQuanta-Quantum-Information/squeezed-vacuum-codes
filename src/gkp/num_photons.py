import numpy as np
from matplotlib import pyplot as plt
import pytest
import qutip as qt

from typing import Iterable, Literal, NamedTuple, Final

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



# Smallest positive nbar we solve against to avoid singular behavior at exactly 0.
MIN_POSITIVE_NBAR: Final[float] = 1e-16

# Script-wide defaults for ratio and amplitude cutoff to keep usage consistent.
DEFAULT_RATIO_KAPPA_OVER_DELTA: Final[float] = 1.0
DEFAULT_AMP_CUTOFF: Final[float] = 1e-12

# Lookup tables properties:
X_GRID_SPACING: Final[float] = 0.01
MAX_N_FOR_LOOKUP : Final[int] = 200
MIN_N_FOR_LOOKUP : Final[int] = 50
MAX_X_FOR_LOOKUP : Final[float] = 0.70
MIN_X_FOR_LOOKUP : Final[float] = 0.05
X_POINTS_SPACING : Final[float] = 0.01

class GKPParams(NamedTuple):
    Delta: float
    kappa: float
    r: float
    s_max: int


def _compute_mean_photon_number(state: qt.Qobj) -> float:
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
            qutip_mean_photons = _compute_mean_photon_number(qutip_state)
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


def _s_max_from_params(kappa: float, amp_cutoff: float) -> int:
    mmax = np.sqrt(2.0 * np.log(1.0 / amp_cutoff)) / (kappa * np.sqrt(np.pi))
    s_max = int(np.ceil(mmax / 2.0))
    return max(s_max, 1)


def estimate_gkp_params_from_nbar(
    nbar: float,
    ratio_kappa_over_Delta: float = DEFAULT_RATIO_KAPPA_OVER_DELTA,
    amp_cutoff: float = DEFAULT_AMP_CUTOFF,
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
    if not (0 < amp_cutoff < 1):
        raise ValueError("amp_cutoff must be between 0 and 1.")

    rho = float(ratio_kappa_over_Delta)

    target_energy = nbar + 0.5  # account for vacuum fluctuations

    # E = 1/(4*kappa^2) + 1/(8*Delta^2); kappa = rho * Delta
    term = (1.0 / (4.0 * rho * rho)) + (1.0 / 8.0)
    Delta = np.sqrt(term / target_energy)

    kappa = rho * Delta

    # For q-squeezed vacuum: Var(q) = Delta^2 = (1/2) e^{-2r}
    r = np.log(1.0 / (np.sqrt(2.0) * Delta))

    s_max = _s_max_from_params(kappa, amp_cutoff)

    return GKPParams(float(Delta), float(kappa), float(r), int(s_max))


def estimate_nbar_from_delta(
    delta: float, 
    ratio_kappa_over_Delta: float = DEFAULT_RATIO_KAPPA_OVER_DELTA
) -> float:
    if delta <= 0:
        raise ValueError("delta must be > 0")
    if ratio_kappa_over_Delta <= 0:
        raise ValueError("ratio_kappa_over_Delta must be > 0")
    rho = float(ratio_kappa_over_Delta)
    term = (1.0 / (4.0 * rho * rho)) + 0.125  # 1/8
    return (term / (delta * delta)) - 0.5


def _build_state_and_nbar(
    Delta: float,
    kappa: float,
    N: int,
    logical_value: int,
    amp_cutoff: float,
    _prog_bar: bool = True
) -> tuple[qt.Qobj, float]:
    s_max = _s_max_from_params(kappa, amp_cutoff)
    state = gkp_logical(
        logical_val=logical_value,
        N=N,
        Delta=Delta,
        kappa=kappa,
        s_max=s_max,
        amp_cutoff=amp_cutoff,
        _prog_bar=_prog_bar,
    )
    state.unit(inplace=True)
    nbar_actual = _compute_mean_photon_number(state)
    return state, float(nbar_actual)


@cache(ram=False, disk=True)
def _compute_nbar_from_numeric_gkp_states(
    Delta: float,
    kappa: float,
    N: int,
    logical_value: int,
    amp_cutoff: float,
) -> float:
    """RAM-cached nbar evaluation without persisting the full state."""
    _, nbar_actual = _build_state_and_nbar(
        Delta=Delta,
        kappa=kappa,
        N=N,
        logical_value=logical_value,
        amp_cutoff=amp_cutoff,
    )
    return nbar_actual


def test_estimate_nbar_from_delta_is_inverse() -> None:
    """Ensure analytic Delta<->nbar formulas invert each other at sample points."""
    sample_nbars = np.array([0.1, 0.5, 1.0, 3.0])

    for target_nbar in sample_nbars:
        Delta, _, _, _ = estimate_gkp_params_from_nbar(
            target_nbar,
            amp_cutoff=DEFAULT_AMP_CUTOFF,
        )
        recovered_nbar = estimate_nbar_from_delta(Delta)
        assert recovered_nbar == pytest.approx(target_nbar, rel=1e-12, abs=0.0)


def _plot_numeric_nbar_vs_delta(
    delta_values: Iterable[float] = np.linspace(0.05, 0.80, 11),
    *,
    ratio_kappa_over_Delta: float = DEFAULT_RATIO_KAPPA_OVER_DELTA,
    amp_cutoff: float = DEFAULT_AMP_CUTOFF,
    logical_value: int = 1,
) -> None:
    """Sweep Delta, plot analytic and numeric nbar for quick sanity checks."""
    deltas = np.asarray(list(delta_values), dtype=float)
    if deltas.size == 0:
        raise ValueError("delta_values must be non-empty")
    if np.any(deltas <= 0):
        raise ValueError("All delta_values must be > 0")
    if ratio_kappa_over_Delta <= 0:
        raise ValueError("ratio_kappa_over_Delta must be > 0")

    analytic_nbars: list[float] = []
    numeric_nbars: list[float] = []

    for Delta in ProgressBar(deltas, prefix="Δ sweep: "):
        analytic_nbar = estimate_nbar_from_delta(Delta, ratio_kappa_over_Delta=ratio_kappa_over_Delta)
        # N = recommended_N_from_nbar(max(analytic_nbar, MIN_POSITIVE_NBAR))
        N = 100
        numeric_nbar = _compute_nbar_from_numeric_gkp_states(
            Delta=Delta,
            kappa=ratio_kappa_over_Delta * Delta,
            N=N,
            logical_value=logical_value,
            amp_cutoff=amp_cutoff,
        )
        analytic_nbars.append(analytic_nbar)
        numeric_nbars.append(numeric_nbar)

    plt.figure(figsize=(8, 5))
    plt.plot(deltas, analytic_nbars, linestyle='--', color='gray', label='analytic estimate')
    plt.plot(deltas, numeric_nbars, marker='o', label='numeric nbar (cached)')
    plt.xlabel("Delta")
    plt.ylabel("nbar")
    plt.title("nbar vs Delta sweep")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("Done.")



@cache(ram=True, disk=False)
def _get_inverse_lookup(
    *,
    ratio_kappa_over_Delta: float,
    amp_cutoff: float,
    logical_value: int
) -> MonotonicInverseLookup:
    
    def _forward_function_for_lookup_from_delta_to_nbar(x: float) -> float:
        Delta = x
        estimated_nbar = estimate_nbar_from_delta(Delta, ratio_kappa_over_Delta=ratio_kappa_over_Delta)
        if estimated_nbar <= 0:
            N = MAX_N_FOR_LOOKUP
        else:
            N = recommended_N_from_nbar(estimated_nbar)
            N = max(min(N, MAX_N_FOR_LOOKUP), MIN_N_FOR_LOOKUP)  # cap N to keep lookup table generation tractable
        
        if Delta <= 0:
            raise ArithmeticError("Delta must stay positive")
        return _compute_nbar_from_numeric_gkp_states(
            Delta=Delta,
            kappa=ratio_kappa_over_Delta * Delta,
            N=N,
            logical_value=logical_value,
            amp_cutoff=amp_cutoff,
        )
    
    x_points = np.arange(MIN_X_FOR_LOOKUP, MAX_X_FOR_LOOKUP + X_GRID_SPACING, X_GRID_SPACING)

    gkp_params_inverse_lookup_table = MonotonicInverseLookup.from_function(
        f=_forward_function_for_lookup_from_delta_to_nbar,
        x_points=x_points,
        direction=MonotonicDirection.DECREASING,  # nbar decreases as Delta increases
        progress_bar=True
    )

    return gkp_params_inverse_lookup_table



def lookup_gkp_params_from_nbar(
    nbar_target: float,
    *,
    logical_value: int = 0,
    ratio_kappa_over_Delta: float = DEFAULT_RATIO_KAPPA_OVER_DELTA,
    amp_cutoff: float = DEFAULT_AMP_CUTOFF,
) -> GKPParams:
    """
    Inverse lookup from target nbar to (Delta, kappa, r, s_max).

    Builds (and disk-caches) a uniform Delta->nbar table, then inverts it by
    linear interpolation in nbar-space.
    """
    if nbar_target < 0:
        raise ValueError("nbar_target must be >= 0.")
    if X_GRID_SPACING <= 0:
        raise ValueError("DELTA_X_GRID must be > 0.")

    target_nbar = max(nbar_target, MIN_POSITIVE_NBAR)


    ## Use inverse lookup to find the Delta that corresponds to the target nbar
    inverse_lookup = _get_inverse_lookup(
        ratio_kappa_over_Delta=ratio_kappa_over_Delta,
        amp_cutoff=amp_cutoff,
        logical_value=logical_value
    )
    Delta = inverse_lookup.x_from_y(target_nbar, clamp=True)

    ## Compute the corresponding kappa, r, and s_max from the found Delta
    kappa = float(ratio_kappa_over_Delta) * Delta
    r = np.log(1.0 / (np.sqrt(2.0) * Delta))
    s_max = _s_max_from_params(kappa, amp_cutoff)

    return GKPParams(float(Delta), float(kappa), float(r), int(s_max))


def gkp_params_from_nbar(
    nbar_target: float,
    *,
    ratio_kappa_over_Delta=DEFAULT_RATIO_KAPPA_OVER_DELTA,
    amp_cutoff=DEFAULT_AMP_CUTOFF,
    logical_value: int = 0,
    estimation_only: bool = False
) -> GKPParams:
    if estimation_only:
        return estimate_gkp_params_from_nbar(
            nbar_target,
            ratio_kappa_over_Delta=ratio_kappa_over_Delta,
            amp_cutoff=amp_cutoff,
        )
    else:
        return lookup_gkp_params_from_nbar(
            nbar_target,
            logical_value=logical_value,
            ratio_kappa_over_Delta=ratio_kappa_over_Delta,
            amp_cutoff=amp_cutoff,
        )


def _test_lookup_vs_qutip(
    logical_value: int = 1,
    nbar_targets: Iterable[float] = np.linspace(0.1, 5.1, 21),
    ratio_kappa_over_Delta: float = DEFAULT_RATIO_KAPPA_OVER_DELTA,
    amp_cutoff: float = DEFAULT_AMP_CUTOFF,
    N = 200
) -> None:
    """Plot target nbar vs actual for lookup and analytic estimate (no execution here)."""

    targets = list(nbar_targets)
    lookup_errors: list[float] = []
    estimate_errors: list[float] = []

    for nbar_target in ProgressBar(targets, prefix="lookup test: "):
        # N = recommended_N_from_nbar(nbar_target)

        # Lookup-based params
        Delta_l, kappa_l, _, _ = lookup_gkp_params_from_nbar(
            nbar_target,
            ratio_kappa_over_Delta=ratio_kappa_over_Delta,
            amp_cutoff=amp_cutoff,
            logical_value=logical_value,
        )
        nbar_l = _compute_nbar_from_numeric_gkp_states(
            Delta=Delta_l,
            kappa=kappa_l,
            N=N,
            logical_value=logical_value,
            amp_cutoff=amp_cutoff,
        )
        lookup_errors.append(nbar_l - nbar_target)

        # Analytic estimate path
        Delta_e, kappa_e, _, _ = estimate_gkp_params_from_nbar(
            nbar_target,
            ratio_kappa_over_Delta=ratio_kappa_over_Delta,
            amp_cutoff=amp_cutoff,
        )
        nbar_e = _compute_nbar_from_numeric_gkp_states(
            Delta=Delta_e,
            kappa=kappa_e,
            N=N,
            logical_value=logical_value,
            amp_cutoff=amp_cutoff,
        )
        estimate_errors.append(nbar_e - nbar_target)

    plt.figure(figsize=(10, 6))
    plt.axhline(0.0, color='gray', linestyle='--', linewidth=1.2, label='zero error')
    plt.plot(targets, lookup_errors, marker='o', label='lookup error (cached grid)')
    plt.plot(targets, estimate_errors, marker='s', label='analytic error')
    plt.xlabel("Target nbar")
    plt.ylabel("Actual - target (nbar)")
    plt.title("GKP nbar error: lookup vs analytic estimate")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()

    plt.show()
    print("Test complete.") 





if __name__ == "__main__":
    # _plot_numeric_nbar_vs_delta()
    _test_lookup_vs_qutip()
    test_estimate_nbar_from_delta_is_inverse()