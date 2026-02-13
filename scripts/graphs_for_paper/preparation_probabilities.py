import numpy as np

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.path import Path
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms


from typing import Literal, Callable, TypeAlias
from typing import cast as type_cast


if __name__ == "__main__":
    from __init__ import add_root_to_path
    path = add_root_to_path()


from src.utils.visuals.matplotlib_support import save_figure, draw_now
from src.utils.visuals.colors import color_shades, _RgbFloatTuple
from src.utils.prints import ProgressBar
from src.utils.strings import format_float_for_as_str


from src.codes_built_in_superposition import _CodeTypes
from src.cost_functions import compute_cost_on_logical_codewords
from src.cost_functions import MeasurementTypeLiteral, NoiseOptionLiteral, BosonicNoiseType, CostPerNoiseDict, CostPerLegsPerNoiseDict, CostPerLogicalBasis, LogicalBasisName

from globals import Globals
from _visual_helper import (
    _NumMomentsFuncType,
    latex_toggled_str as _latex_toggled_str,
    extend_axis_without_grid as _extend_axis_without_grid,
    adjust_label_positions_to_avoid_overlap as _adjust_label_positions_to_avoid_overlap,
    add_legend_background_box as _add_legend_background_box,
    place_header_labels_above_legend as _place_header_labels_above_legend,
)


# ---------------------------------------------------------------------------
#  Axis setup — single panel (no noise split)
# ---------------------------------------------------------------------------

def _axis_setup(
    grid: Literal["on", "off", "weak"] = "weak",
    enlarge_text: bool = True,
    fig_dpi: int | None = None,
    fig_size: tuple[float, float] = (7, 5.5),
) -> tuple[Figure, Axes]:
    """Create a single-panel figure for preparation-probability plots."""

    if fig_dpi is None:
        fig = plt.figure(figsize=fig_size)
    else:
        fig = plt.figure(figsize=fig_size, dpi=fig_dpi)

    ax: Axes = plt.subplot(1, 1, 1)

    match grid:
        case "on":
            ax.grid(True, which='both')
        case "off":
            pass
        case "weak":
            ax.grid(True, which='major', alpha=0.5)

    if enlarge_text:
        ax.xaxis.set_tick_params(labelsize=12)
        ax.yaxis.set_tick_params(labelsize=12)
        ax.set_xlabel(ax.get_xlabel(), fontsize=16)
        ax.set_ylabel(ax.get_ylabel(), fontsize=16)
        ax.set_title(ax.get_title(), fontsize=16)

    return fig, ax


# ---------------------------------------------------------------------------
#  Legend — table style adapted for (m, logical-state) layout
# ---------------------------------------------------------------------------

def _add_preparation_legend(
    fig: Figure,
    ax: Axes,
    legend_colors: dict[int, _RgbFloatTuple],
    linewidth: float = 3,
    fontsize: int = 12,
    title_row_x_shifts: list[float] | None = [-20, -20, -20],
    title_row_y_shift: float = -11.0,
    bg_padding_left: float = -65.0,
    bg_padding_right: float = +1.0,
    bg_padding_top: float = 0.0,
    bg_padding_bottom: float = -10.0,
    legend_x_offset: float = 0.23,
    legend_y_offset: float = +0.85,
) -> None:
    r"""
    Add a table-style legend for preparation probabilities.

    Columns: ``m`` values.  Two rows: solid  = |0_L\rangle,  dashed = |1_L\rangle.
    """

    all_m = sorted(legend_colors.keys())

    # Table dimensions: 2 data-rows (|0_L⟩, |1_L⟩) + 1 header row
    # Columns: 1 for row-label + len(all_m) for each m value
    ncol = len(all_m) + 1   # row-label column + one col per m
    nrow = 3                 # header row + |0_L⟩ row + |1_L⟩ row

    # ----- build row-major table -----
    table_handles: list[list[Line2D]] = []
    table_labels: list[list[str]] = []

    # Header row: empty + invisible placeholders for m-values
    header_handles = [Line2D([], [], linestyle="none")] + [
        Line2D([], [], linestyle="none") for _ in all_m
    ]
    header_labels = [""] + ["" for _ in all_m]
    table_handles.append(header_handles)
    table_labels.append(header_labels)

    # |0_L⟩ row  (solid)
    row0_handles: list[Line2D] = [Line2D([], [], linestyle="none")]
    row0_labels: list[str] = [_latex_toggled_str(r"$|0_L\rangle$", "|0_L⟩")]
    for m in all_m:
        row0_handles.append(
            Line2D([0, 1], [0, 0], color=legend_colors[m], linewidth=linewidth, linestyle="-")
        )
        row0_labels.append("")
    table_handles.append(row0_handles)
    table_labels.append(row0_labels)

    # |1_L⟩ row  (dashed)
    row1_handles: list[Line2D] = [Line2D([], [], linestyle="none")]
    row1_labels: list[str] = [_latex_toggled_str(r"$|1_L\rangle$", "|1_L⟩")]
    for m in all_m:
        row1_handles.append(
            Line2D([0, 1], [0, 0], color=legend_colors[m], linewidth=linewidth, linestyle="--")
        )
        row1_labels.append("")
    table_handles.append(row1_handles)
    table_labels.append(row1_labels)

    # ----- flatten to column-major for matplotlib legend -----
    handles: list[Line2D] = []
    labels: list[str] = []
    for c in range(ncol):
        for r in range(nrow):
            handles.append(table_handles[r][c])
            labels.append(table_labels[r][c])

    loc = "lower center"
    bbox_to_anchor = (0.5 + legend_x_offset, -0.08 + legend_y_offset)

    legend = fig.legend(
        handles, labels,
        ncol=ncol,
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        fontsize=fontsize + 2,
        columnspacing=0.7,
        labelspacing=0.2,
        handletextpad=0.5,
        borderaxespad=0.5,
        frameon=False,
    )

    # Make header row invisible (keeps table spacing)
    header_indices = [c * nrow for c in range(ncol)]
    for idx in header_indices:
        legend.get_texts()[idx].set_alpha(0)

    # Add visible header labels ("m=2", "m=4", …) above their columns
    header_labels_for_m = [f"m={m}" for m in all_m]
    header_indices_for_m = [header_indices[i + 1] for i in range(len(all_m))]  # +1 to skip row-label column
    _place_header_labels_above_legend(
        fig, legend,
        header_indices=header_indices_for_m,
        header_labels=header_labels_for_m,
        nrow=nrow,
        title_row_x_shifts=title_row_x_shifts,
        title_row_y_shift=title_row_y_shift,
        fontsize=fontsize + 2,
    )

    # Grey background box
    _add_legend_background_box(
        fig, legend,
        title_row_y_shift=title_row_y_shift,
        bg_padding_left=bg_padding_left,
        bg_padding_right=bg_padding_right,
        bg_padding_top=bg_padding_top,
        bg_padding_bottom=bg_padding_bottom,
    )

    return legend


# ---------------------------------------------------------------------------
#  Inline-label helpers  (copied & simplified from numerics1)
# ---------------------------------------------------------------------------

def _get_text_pos(
    final_graph_point: tuple[float, float],
    m: int,
    x_scale: Literal['linear', 'log'] = 'linear',
    x_dif: float = 1.0,
) -> tuple[float, float]:
    if x_scale == 'log':
        x = final_graph_point[0] * 1.6
    else:
        x = final_graph_point[0] + x_dif * 0.4
    y = final_graph_point[1]
    return x, y


# _extend_axis_without_grid  →  imported from _visual_helper
# _adjust_label_positions_to_avoid_overlap  →  imported from _visual_helper


# ---------------------------------------------------------------------------
#  Main plotting function — preparation probabilities
# ---------------------------------------------------------------------------

def _plot_preparation_results(
    per_code_results: CostPerLegsPerNoiseDict,
    x_vec: list[float],
    x_vec_name: Literal["num_photons", "r"] = "num_photons",
    basis: LogicalBasisName = "main",
    # -- visual knobs --
    grid: Literal["on", "off", "weak"] = "weak",
    fig_dpi: int = 500,
    figure_name_prefix: str = "",
    figure_name_extra: str = "",
    N: int | _NumMomentsFuncType | None = None,
    label_style: Literal["inline", "inline-adjusted", "legend"] = "inline-adjusted",
    text_font_size: int = 16,
    text_legend_on_plot_font_size: int = 14,
    _adjust_ticks_font: bool = True,
    x_scale: Literal['linear', 'log'] = 'linear',
    y_scale: Literal['linear', 'log'] = 'linear',
    figure_title: str = "",
    _fig_realtive_height_factor: float = 0.7,
):
    """
    Plot preparation probabilities for squeeze-vacuum codes.

    Each m-value produces **two** curves:
      • solid  — probability of successfully preparing |0_L⟩
      • dashed — probability of successfully preparing |1_L⟩

    Colors are a gradient of blue (light → dark as m grows).
    """

    _linewidth = 3

    ## ========= Set up figure =========:
    fig, ax = _axis_setup(grid=grid)

    match x_vec_name:
        case "num_photons":
            xlabel = "mean number " + _latex_toggled_str(r'$\bar{n}$', 'n')
        case "r":
            xlabel = r'squeezing strength $r$'

    ylabel = _latex_toggled_str(r'$P_{\mathrm{prep}}$', 'P_prep')

    ax.set_xlabel(xlabel, fontsize=text_font_size)
    ax.set_ylabel(ylabel, fontsize=text_font_size)

    if x_scale == 'log':
        ax.set_xscale('log')
    if y_scale == 'log':
        ax.set_yscale('log')

    ## ========= Colours =========:
    all_m = sorted(per_code_results.keys())
    num_m = len(all_m)
    colors = color_shades("blue", num_m + 1)[:-1]   # skip the darkest shade

    legend_colors: dict[int, _RgbFloatTuple] = {}
    label_data_list: list[tuple[tuple[float, float], str, dict]] = []

    ## ========= Plot each m =========:
    for j, m in enumerate(all_m):
        costs_per_noise = per_code_results[m]
        # Pick any noise key — with No-noise they are identical
        noise_key = list(costs_per_noise.keys())[0]
        cost_vec: list[CostPerLogicalBasis] = costs_per_noise[noise_key]

        color = colors[j]
        legend_colors[m] = color

        # Unpack (prob_0, prob_1) tuples:
        y_vec_0 = [c[basis][0] for c in cost_vec]   # |0_L⟩  probability
        y_vec_1 = [c[basis][1] for c in cost_vec]   # |1_L⟩  probability

        # Solid line for |0_L⟩
        ax.plot(x_vec, y_vec_0, color=color, linestyle='-',  linewidth=_linewidth)
        # Dashed line for |1_L⟩
        ax.plot(x_vec, y_vec_1, color=color, linestyle='--', linewidth=_linewidth)

        ## Inline labels (if requested)
        if label_style in ["inline", "inline-adjusted"]:
            x_dif = x_vec[1] - x_vec[0] if len(x_vec) > 1 else 1.0
            for y_vec, state_label in [(y_vec_0, "0"), (y_vec_1, "1")]:
                final_pt = (x_vec[-1], y_vec[-1])
                text_pos = _get_text_pos(final_pt, m, x_scale=x_scale, x_dif=x_dif)

                if Globals.LaTeX_RENDERING:
                    text = r"$|%s_L\rangle, m{=}%s$" % (state_label, m)
                else:
                    text = f"m={m}, |{state_label}>"
                text_kwargs = dict(fontsize=text_legend_on_plot_font_size, ha='left', va='center')
                if label_style == "inline":
                    ax.text(*text_pos, text, **text_kwargs)
                else:
                    label_data_list.append((text_pos, text, text_kwargs))

    # Adjusted inline labels
    if label_style == "inline-adjusted" and label_data_list:
        adjusted = _adjust_label_positions_to_avoid_overlap(
            ax, label_data_list, x_scale=x_scale, y_scale=y_scale,
            min_y_distance_factor = 0.06
        )
        for (_, text, kwargs), pos in zip(label_data_list, adjusted):
            ax.text(*pos, text, **kwargs)

    ## Change figure height (after tight_layout) to make room for legend if needed:
    width = fig.get_figwidth()
    height = fig.get_figheight() 
    fig.set_size_inches(width, height*_fig_realtive_height_factor)

    ## ========= Legend / layout =========:
    if label_style in ["inline", "inline-adjusted"]:
        _extend_axis_without_grid(ax, extension_factor=0.12)
        legend = None
        plt.tight_layout()
    elif label_style == "legend":
        legend = _add_preparation_legend(
            fig, ax, legend_colors, linewidth=_linewidth,
        )
        plt.tight_layout(rect=(0, 0.12, 1, 1))
    else:
        raise ValueError(f"Unknown label_style: {label_style!r}")
    

    if figure_title:
        fig.suptitle(figure_title, fontsize=text_font_size)

    if _adjust_ticks_font:
        for axis in [ax.xaxis, ax.yaxis]:
            axis.set_tick_params(labelsize=text_font_size)

    plt.show()
    print("Plotted.")

    file_name = (
        figure_name_prefix
        + "probability"
        + (f" - N={N}" if isinstance(N, (int, float)) else "")
        + (f" - {figure_name_extra}" if figure_name_extra else "")
    )
    save_figure(plt.gcf(), file_name, dpi=fig_dpi, transparent=True, extensions=['pdf', 'png', 'svg'])
    print("Saved.")

    return fig, ax, legend


# ---------------------------------------------------------------------------
#  Top-level entry point
# ---------------------------------------------------------------------------

def plot_preparation_probability_for_squeezed_codes(
    num_moments: int = 300,
    photon_num_vec = [float(n) for n in np.linspace(0.0, 7.5, 151)],
    num_code_states: int = 3,
    basis: LogicalBasisName = "main",
) -> None:

    ## ========= Inputs =========:
    photon_num_vec = [num for num in photon_num_vec if num<=6.0]
    γ = 0
    measurement: MeasurementTypeLiteral = "probability"
    noise_method: NoiseOptionLiteral = "No-noise"
    x_vec_name = "num_photons"
    print(f"photon_num_vec = {photon_num_vec}")

    ## ========= Compute =========:
    results = compute_cost_on_logical_codewords(
        fixed_param_name="γ",
        fixed_value=γ,
        x_name="mean_n",
        x_vec=photon_num_vec,
        num_moments=num_moments,
        num_code_states=num_code_states,
        code="squeeze",
        measurement=measurement,
        noise_method=noise_method,
    )

    ## ========= Plot =========:
    fig, ax, legend = _plot_preparation_results(
        results,
        x_vec=photon_num_vec,
        x_vec_name=x_vec_name,
        basis=basis,
        figure_name_prefix="x-is-nbar - ",
        figure_name_extra="squeeze-code",
        N=num_moments,
        x_scale='linear',
    )

    draw_now()
    input("Press Enter to close the plots and end the program...")



if __name__ == "__main__":
    plot_preparation_probability_for_squeezed_codes()