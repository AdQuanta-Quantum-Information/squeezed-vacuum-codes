from typing import TypedDict

import qutip as qt

try:
    from scripts.study._bootstrap import add_root_to_path
except ModuleNotFoundError:
    from _bootstrap import add_root_to_path

add_root_to_path()

from src.codes_built_in_superposition import simple_m_legged_code
from src.quantum.visualizations.orthonormalization_comparison import (
    compare_orthonormalization_methods_for_states,
)
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


def test_orthonormalization_other_codes(
    plot: bool = True,
) -> list[dict[str, object]]:
    """Test GS-vs-Lowdin orthonormalization on non-GKP code families."""
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
    test_orthonormalization_other_codes(plot=True)
