from typing import TypedDict

import numpy as np
import matplotlib.pyplot as plt
import qutip as qt
from globals import Globals

from src.quantum.quantum_information import (
    gram_schmidt_orthonormal_pair,
    lowdin_orthonormal_pair,
)
from src.quantum.visualizations.wigner_function import plot_plain_wigner


class OthonormalizationComparisonResult(TypedDict):
    physical_overlap_abs: float
    gram_schmidt: dict
    lowdin: dict
    cross_basis_overlap_abs: dict


def _overlap_line(left: str, right: str, value: float, use_latex: bool) -> str:
    if use_latex:
        return r"$|\langle {left} | {right} \rangle|={value:.4f}$".format(
            left=left,
            right=right,
            value=value,
        )
    return f"|<{left}|{right}>|={value:.4f}"


def compare_orthonormalization_methods_for_states(
    psi0: qt.Qobj,
    psi1: qt.Qobj,
    state_labels: tuple[str, str] = ("|0>", "|1>"),
    figure_title: str | None = None,
    verbose: bool = True,
    plot: bool = True,
) -> OthonormalizationComparisonResult:
    """Compare GS and Lowdin orthonormalization for two states and optionally plot.

    The function assumes the states are appropriate for Wigner plotting when
    plot=True.
    """
    p0 = psi0.unit()
    p1 = psi1.unit()

    physical_overlap_abs = float(np.abs(p0.overlap(p1)))

    phi0_gs, phi1_gs = gram_schmidt_orthonormal_pair(p0, p1)
    phi0_low, phi1_low = lowdin_orthonormal_pair(p0, p1)

    gs_overlap_abs = float(np.abs(phi0_gs.overlap(phi1_gs)))
    low_overlap_abs = float(np.abs(phi0_low.overlap(phi1_low)))

    ov_p0_gs0 = float(np.abs(p0.overlap(phi0_gs)))
    ov_p0_low0 = float(np.abs(p0.overlap(phi0_low)))
    ov_p1_gs1 = float(np.abs(p1.overlap(phi1_gs)))
    ov_p1_low1 = float(np.abs(p1.overlap(phi1_low)))

    gs0_low0 = float(np.abs(phi0_gs.overlap(phi0_low)))
    gs1_low1 = float(np.abs(phi1_gs.overlap(phi1_low)))

    results: OthonormalizationComparisonResult = {
        "physical_overlap_abs": physical_overlap_abs,
        "gram_schmidt": {
            "orthonormal_overlap_abs": gs_overlap_abs,
            "overlap_to_physical": {
                "state0": ov_p0_gs0,
                "state1": ov_p1_gs1,
            },
        },
        "lowdin": {
            "orthonormal_overlap_abs": low_overlap_abs,
            "overlap_to_physical": {
                "state0": ov_p0_low0,
                "state1": ov_p1_low1,
            },
        },
        "cross_basis_overlap_abs": {
            "gs0_low0": gs0_low0,
            "gs1_low1": gs1_low1,
        },
    }

    if verbose:
        print(f"Physical |<0|1>| = {physical_overlap_abs:.6g}")
        print(
            "GS: "
            f"|<gs0|gs1>|={gs_overlap_abs:.3e}, "
            f"|<physical0|gs0>|={ov_p0_gs0:.6f}, "
            f"|<physical1|gs1>|={ov_p1_gs1:.6f}"
        )
        print(
            "Lowdin: "
            f"|<low0|low1>|={low_overlap_abs:.3e}, "
            f"|<physical0|low0>|={ov_p0_low0:.6f}, "
            f"|<physical1|low1>|={ov_p1_low1:.6f}"
        )
        print(
            "GS-vs-Lowdin basis overlap: "
            f"|<gs0|low0>|={gs0_low0:.6f}, "
            f"|<gs1|low1>|={gs1_low1:.6f}"
        )

    if plot:
        use_latex = Globals.LaTeX_RENDERING
        fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)

        label0, label1 = state_labels

        if use_latex:
            name_phys0 = r"\psi^{\mathrm{phys}}_0"
            name_phys1 = r"\psi^{\mathrm{phys}}_1"
            name_gs0 = r"\psi^{\mathrm{gs}}_0"
            name_gs1 = r"\psi^{\mathrm{gs}}_1"
            name_low0 = r"\psi^{\mathrm{low}}_0"
            name_low1 = r"\psi^{\mathrm{low}}_1"
            line_phys0 = _overlap_line(name_phys0, name_phys0, 1.0, use_latex)
            line_gs0 = _overlap_line(name_phys0, name_gs0, ov_p0_gs0, use_latex)
            line_low0 = _overlap_line(name_phys0, name_low0, ov_p0_low0, use_latex)
            line_phys1 = _overlap_line(name_phys1, name_phys1, 1.0, use_latex)
            line_gs1 = _overlap_line(name_phys1, name_gs1, ov_p1_gs1, use_latex)
            line_low1 = _overlap_line(name_phys1, name_low1, ov_p1_low1, use_latex)
        else:
            line_phys0 = _overlap_line("physical0", "physical0", 1.0, use_latex)
            line_gs0 = _overlap_line("physical0", "gs0", ov_p0_gs0, use_latex)
            line_low0 = _overlap_line("physical0", "low0", ov_p0_low0, use_latex)
            line_phys1 = _overlap_line("physical1", "physical1", 1.0, use_latex)
            line_gs1 = _overlap_line("physical1", "gs1", ov_p1_gs1, use_latex)
            line_low1 = _overlap_line("physical1", "low1", ov_p1_low1, use_latex)

        plot_plain_wigner(
            p0,
            ax=axes[0, 0],
            with_colorbar=False,
            title=f"{label0} physical\n{line_phys0}",
        )
        plot_plain_wigner(
            phi0_gs,
            ax=axes[0, 1],
            with_colorbar=False,
            title=f"{label0} GS\n{line_gs0}",
        )
        plot_plain_wigner(
            phi0_low,
            ax=axes[0, 2],
            with_colorbar=False,
            title=f"{label0} Lowdin\n{line_low0}",
        )

        plot_plain_wigner(
            p1,
            ax=axes[1, 0],
            with_colorbar=False,
            title=f"{label1} physical\n{line_phys1}",
        )
        plot_plain_wigner(
            phi1_gs,
            ax=axes[1, 1],
            with_colorbar=False,
            title=f"{label1} GS\n{line_gs1}",
        )
        plot_plain_wigner(
            phi1_low,
            ax=axes[1, 2],
            with_colorbar=False,
            title=f"{label1} Lowdin\n{line_low1}",
        )

        if figure_title is None:
            if use_latex:
                figure_title = (
                    "Basis comparison\\n"
                    + r"$|\langle \psi^{\mathrm{phys}}_0 | \psi^{\mathrm{phys}}_1 \rangle|={:.3e}$".format(
                        physical_overlap_abs
                    )
                )
            else:
                figure_title = (
                    "Basis comparison\\n"
                    f"|<physical0|physical1>|={physical_overlap_abs:.3e}"
                )
        else:
            figure_title = figure_title.format(physical_overlap_abs=physical_overlap_abs)
        fig.suptitle(figure_title)
        plt.show()

    return results
