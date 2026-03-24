from typing import Sequence, TypedDict

import matplotlib.pyplot as plt
import numpy as np
import qutip as qt

try:
    from scripts.study._bootstrap import add_root_to_path
except ModuleNotFoundError:
    from _bootstrap import add_root_to_path

add_root_to_path()

from src.codes_built_in_superposition import simple_m_legged_code, gkp_code_state
from src.gkp import recommended_N_from_nbar
from src.quantum.visualizations.orthonormalization_comparison import (
    compare_orthonormalization_methods_for_states,
)

from src.mean_photon_number import LOWER_THRESHOLD_FOR_1_LOGICAL_STATE_PER_CODE
from src.utils.prints import ProgressBar


class _Case(TypedDict):
    code_type: str
    m: int
    strength: float
    num_moments: int


def _build_case_states(case: _Case) -> tuple[qt.Qobj, qt.Qobj]:
    psi0, psi1 = simple_m_legged_code(
        m=case["m"],
        strength=case["strength"],
        num_moments=case["num_moments"],
        code_type=case["code_type"],
        _prog_bar=False,
    )
    return psi0.unit(), psi1.unit()


class GKPOverlapResult(TypedDict):
    nbar: float
    num_moments: int
    infidelity_logical0: float
    infidelity_logical1: float


def plot_gkp_lowdin_infidelities(
    nbar_values: Sequence[float] = np.linspace(0.5, 10.0, 27),
    *,
    num_moments: int | None = 300,
    use_recommended_cutoff: bool = False,
    _plot: bool = True,
    _prog_bar: bool = True,
) -> list[GKPOverlapResult]:
    """Plot infidelity between physical and Lowdin-orthonormalized GKP states.

    The infidelity is 1 - |<physical|lowdin>| for each logical state. When
    use_recommended_cutoff=True, the per-point Fock cutoff is picked by
    recommended_N_from_nbar; otherwise, a fixed num_moments is used.
    """

    infidelities0: list[float] = []
    infidelities1: list[float] = []
    results: list[GKPOverlapResult] = []

    nbar_list = [float(x) for x in nbar_values]
    if _prog_bar:
        nbar_iter = ProgressBar(nbar_list, prefix="nbar values: ")
    else:
        nbar_iter = nbar_list

    for nbar in nbar_iter:
        if _prog_bar:
            ProgressBar.newest().append_extra_str(f"nbar={nbar:.3f}")

        if use_recommended_cutoff or num_moments is None:
            N = recommended_N_from_nbar(nbar)
        else:
            N = num_moments

        def _get_gkp(logical_value:int, orthonormalize:bool) -> qt.Qobj:
            return gkp_code_state(
                nbar=nbar, num_moments=N, _prog_bar=_prog_bar,
                qubit_logical_value=logical_value,
                lowdin_orthogonalize=orthonormalize
            )
        
        original0 = _get_gkp(0, False)
        if nbar > LOWER_THRESHOLD_FOR_1_LOGICAL_STATE_PER_CODE["gkp"](1):
            original1 = _get_gkp(1, False)
        else:
            original1 = None

        lowdin0 = _get_gkp(0, True)
        lowdin1 = _get_gkp(1, True) 

        fid0 = float(qt.fidelity(original0, lowdin0))
        if original1 is not None:
            fid1 = float(qt.fidelity(original1, lowdin1))
        else:
            fid1 = float('nan')

        infidelities0.append(1.0 - fid0)
        infidelities1.append(1.0 - fid1 if not np.isnan(fid1) else float('nan'))

        results.append(
            {
                "nbar": nbar,
                "num_moments": int(N),
                "infidelity_logical0": 1.0 - fid0,
                "infidelity_logical1": 1.0 - fid1 if not np.isnan(fid1) else float('nan'),
            }
        )

    if _plot:
        plot_kwargs = dict(
            marker = "o", 
            linewidth = 2
        )
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        ax.plot(nbar_list, infidelities0, label="|0⟩", **plot_kwargs)
        ax.plot(nbar_list, infidelities1, label="|1⟩", **plot_kwargs)
        ax.set_xlabel("Target nbar")
        ax.set_ylabel("Infidelity 1 - |<physical|lowdin>|")
        ax.set_yscale("log")    
        ax.set_title("GKP infidelity vs Lowdin orthonormalization")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(title="Logical state")
        fig.tight_layout()
        plt.show()

    return results


def test_orthonormalization(
    plot: bool = True,
) -> list[dict[str, object]]:
    """Test GS-vs-Lowdin orthonormalization on different code families."""
    cases: list[_Case] = [
        {"code_type": "gkp",      "m": 1, "strength": 2.0, "num_moments": 100},
        {"code_type": "squeeze",  "m": 2, "strength": 2.0, "num_moments": 100},
        {"code_type": "squeeze",  "m": 4, "strength": 1.5, "num_moments": 100},
        {"code_type": "cat",      "m": 2, "strength": 2.0, "num_moments": 100},
        {"code_type": "cat",      "m": 4, "strength": 2.0, "num_moments": 100},
        {"code_type": "binomial", "m": 2, "strength": 2.0, "num_moments": 100},
        {"code_type": "binomial", "m": 4, "strength": 2.0, "num_moments": 100},
    ]

    summaries: list[dict[str, object]] = []

    for i, case in ProgressBar(enumerate(cases)):
        psi0, psi1 = _build_case_states(case)

        result = compare_orthonormalization_methods_for_states(
            psi0,
            psi1,
            state_labels=("|0>", "|1>"),
            figure_title=(
                f"{case['code_type']} code comparison (m={case['m']}, "
                f"strength={case['strength']:.2f}, N={case['num_moments']})\n"
                "physical overlap={physical_overlap_abs:.3e}"
            ),
            verbose=False,
            plot=plot,
        )

        gs_ov = float(result["gram_schmidt"]["orthonormal_overlap_abs"])  # type: ignore[index]
        low_ov = float(result["lowdin"]["orthonormal_overlap_abs"])  # type: ignore[index]
        phys_ov = float(result["physical_overlap_abs"])  # type: ignore[index]

        summary = {
            **case,
            "physical_overlap_abs": phys_ov,
            "gs_orthonormal_overlap_abs": gs_ov,
            "lowdin_orthonormal_overlap_abs": low_ov,
        }
        summaries.append(summary)

        print(
            f"[{case['code_type']:<8} m={case['m']}] "
            f"physical={phys_ov:.3e}, gs={gs_ov:.3e}, lowdin={low_ov:.3e}"
        )

    return summaries


if __name__ == "__main__":
    # test_orthonormalization(plot=True)
    plot_gkp_lowdin_infidelities()
