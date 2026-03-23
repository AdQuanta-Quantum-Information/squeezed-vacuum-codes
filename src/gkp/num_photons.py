import numpy as np
from matplotlib import pyplot as plt
import qutip as qt

from typing import Iterable, Literal, TypedDict, NamedTuple

# Use project-standard path setup
try:
    from src.gkp._bootstrap import add_root_to_path
except ModuleNotFoundError:
    from _bootstrap import add_root_to_path
add_root_to_path()


from src.utils.prints import ProgressBar
from src.utils.caches import cache
from src.utils.searches import binary_search_callable_increasing
from src.gkp.fock_cutoff_recommendation import recommended_N_from_nbar
from src.gkp.logical import gkp_logical


# Shared grid resolution for nbar lookup/interpolation.
DELTA_NBAR_GRID: float = 0.2

# Smallest positive nbar we solve against to avoid singular behavior at exactly 0.
MIN_POSITIVE_NBAR: float = 1e-16

# Safety cap on bracket expansion iterations when solving anchors.
BRACKET_EXPAND_MAX_ITERS: int = 8


class AnchorSolution(TypedDict):
    anchor_nbar: float
    Delta: float
    kappa: float
    r: float
    s_max: int
    nbar_actual: float
    p_star: float
    N: int


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


def _solve_anchor_for_nbar(
    anchor_nbar: float,
    *,
    ratio_kappa_over_Delta: float,
    ampl_cutoff: float,
    logical_value: int,
) -> AnchorSolution:
    if anchor_nbar < 0:
        raise ValueError("anchor_nbar must be >= 0.")

    # Use a tiny positive value to avoid singular behavior when anchoring vacuum.
    target_nbar = max(anchor_nbar, MIN_POSITIVE_NBAR)

    rho = float(ratio_kappa_over_Delta)

    # Initial guess from analytic estimate; search over Delta directly
    # because nbar is monotonic increasing with Delta in practice.
    Delta_est, _, _, _ = estimate_gkp_params_from_nbar(
        target_nbar, ratio_kappa_over_Delta=rho, ampl_cutoff=ampl_cutoff
    )

    N = recommended_N_from_nbar(target_nbar)

    cache_nbar: dict[float, float] = {}

    def nbar_for_Delta(Delta: float) -> float:
        if Delta <= 0:
            raise ArithmeticError("Delta must stay positive")
        if Delta in cache_nbar:
            return cache_nbar[Delta]
        kappa = rho * Delta
        nbar_val = _nbar_only(
            Delta=Delta,
            kappa=kappa,
            N=N,
            logical_value=logical_value,
            ampl_cutoff=ampl_cutoff,
        )
        cache_nbar[Delta] = nbar_val
        return nbar_val

    # Find bracket where nbar_for_Delta(lo) <= target <= nbar_for_Delta(hi)
    lo = max(1e-12, Delta_est * 0.5)
    hi = max(lo * 2.0, Delta_est * 2.0)

    y_lo = nbar_for_Delta(lo)
    y_hi = nbar_for_Delta(hi)

    # Near-vacuum fast path: if we are at or below ~1e-15, clamp low side immediately.
    if target_nbar <= MIN_POSITIVE_NBAR * 10:
        lo = MIN_POSITIVE_NBAR
        y_lo = nbar_for_Delta(lo)
        # Refresh hi if lo moved.
        if hi <= lo:
            hi = lo * 2.0
            y_hi = nbar_for_Delta(hi)
    else:
        # Clamp threshold to avoid overly aggressive downward expansion when target is tiny.
        fast_thresh = max(target_nbar * 1.05, target_nbar + MIN_POSITIVE_NBAR)
        if y_lo > fast_thresh:
            for _ in ProgressBar.range(BRACKET_EXPAND_MAX_ITERS, prefix="lo expand: "):
                if y_lo <= target_nbar:
                    break
                lo *= 0.5
                y_lo = nbar_for_Delta(lo)
                if lo < MIN_POSITIVE_NBAR:
                    lo = MIN_POSITIVE_NBAR
                    y_lo = nbar_for_Delta(lo)
                    break

    # Only expand upward if needed.
    if y_hi < target_nbar:
        for _ in ProgressBar.range(BRACKET_EXPAND_MAX_ITERS, prefix="hi expand: "):
            if y_hi >= target_nbar:
                break
            hi *= 2.0
            y_hi = nbar_for_Delta(hi)
        else:
            raise RuntimeError("Failed to bracket target nbar after many expansions")

    # Ensure binary search bracket condition by clamping to reachable floor if needed.
    target_for_search = max(target_nbar, y_lo)

    Delta_star, _ = binary_search_callable_increasing(
        nbar_for_Delta,
        target=target_for_search,
        x_bounds=(lo, hi),
        progress_bar=False,
        x_tol=1e-5,
        y_tol=1e-4,
        max_iters=200,
    )

    kappa_star = rho * Delta_star
    p_star = 1.0 / Delta_star
    _, nbar_star = _build_state_and_nbar(
        Delta=Delta_star,
        kappa=kappa_star,
        N=N,
        logical_value=logical_value,
        ampl_cutoff=ampl_cutoff,
    )
    r_star = np.log(1.0 / (np.sqrt(2.0) * Delta_star))
    s_max_star = _s_max_from_params(kappa_star, ampl_cutoff)

    return AnchorSolution(
        anchor_nbar=float(anchor_nbar),
        Delta=float(Delta_star),
        kappa=float(kappa_star),
        r=float(r_star),
        s_max=int(s_max_star),
        nbar_actual=float(nbar_star),
        p_star=float(p_star),
        N=int(N),
    )


@cache(ram=True, disk=True)
def _cached_anchor(
    anchor_nbar: float,
    *,
    ratio_kappa_over_Delta: float,
    ampl_cutoff: float,
    logical_value: int,
) -> AnchorSolution:
    return _solve_anchor_for_nbar(
        anchor_nbar,
        ratio_kappa_over_Delta=ratio_kappa_over_Delta,
        ampl_cutoff=ampl_cutoff,
        logical_value=logical_value,
    )


def lookup_gkp_params_from_nbar(
    nbar_target: float,
    *,
    ratio_kappa_over_Delta: float = 1.0,
    ampl_cutoff: float = 1e-12,
    logical_value: int = 0,
) -> GKPParams:
    """
    Lazy lookup (with interpolation) from target nbar to (Delta, kappa, r, s_max).

    - Stores anchor points on a regular grid of spacing DELTA_NBAR_GRID using disk cache.
    - Interpolates linearly between neighboring anchors when available.
    - Computes a new anchor only when a needed grid point is missing.
    """
    if nbar_target < 0:
        raise ValueError("nbar_target must be >= 0.")
    if DELTA_NBAR_GRID <= 0:
        raise ValueError("DELTA_NBAR_GRID must be > 0.")
    
    def _get_sol(anchor: float) -> AnchorSolution:
        return _cached_anchor(
            anchor,
            ratio_kappa_over_Delta=ratio_kappa_over_Delta,
            ampl_cutoff=ampl_cutoff,
            logical_value=logical_value,
        )

    # Snap to grid anchors
    lower_anchor = max(0.0, DELTA_NBAR_GRID * np.floor(nbar_target / DELTA_NBAR_GRID))
    upper_anchor = lower_anchor + DELTA_NBAR_GRID

    if upper_anchor < lower_anchor:
        raise RuntimeError("Upper anchor is less than lower anchor, which should never happen.")
    
    if lower_anchor == 0.0:
        lower_anchor = MIN_POSITIVE_NBAR

    lower_sol = _get_sol(lower_anchor)
    upper_sol = _get_sol(upper_anchor)

    t = (nbar_target - lower_anchor) / (upper_anchor - lower_anchor)
    Delta = (1.0 - t) * lower_sol["Delta"] + t * upper_sol["Delta"]
    kappa = (1.0 - t) * lower_sol["kappa"] + t * upper_sol["kappa"]
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