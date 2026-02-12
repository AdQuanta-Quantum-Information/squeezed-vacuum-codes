"""
Shared visual helpers used by numerics1.py, preparation_probabilities.py,
and other graph scripts in this folder.
"""

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

from __init__ import add_root_to_path
add_root_to_path()

from src.utils.visuals.colors import color_shades, _RgbFloatTuple
from globals import Globals


# ---------------------------------------------------------------------------
#  LaTeX rendering setup  (call once at module level in each script)
# ---------------------------------------------------------------------------

def setup_latex_rendering() -> None:
    """Apply LaTeX rcParams if Globals.LaTeX_RENDERING is enabled."""
    if Globals.LaTeX_RENDERING:
        plt.rcParams['text.usetex'] = True
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Computer Modern Serif']
        plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

setup_latex_rendering()


# ---------------------------------------------------------------------------
#  Common type aliases
# ---------------------------------------------------------------------------

_NumMomentsFuncType: TypeAlias = Callable[[float], int]


# ---------------------------------------------------------------------------
#  Small helpers
# ---------------------------------------------------------------------------

def latex_toggled_str(latex_str: str, plain_str: str) -> str:
    """Return *latex_str* if LaTeX rendering is enabled, else *plain_str*."""
    if Globals.LaTeX_RENDERING:
        return latex_str
    else:
        return plain_str


# ---------------------------------------------------------------------------
#  Axis / grid utilities
# ---------------------------------------------------------------------------

def extend_axis_without_grid(ax: Axes, extension_factor: float = 0.2) -> None:
    """
    Extend the x-axis limits while keeping the grid at its original extent.

    Parameters
    ----------
    ax : Axes
        The matplotlib axes to modify.
    extension_factor : float, optional
        Factor by which to extend the x-axis.  For log scale, this
        multiplies the upper limit in log-space.  For linear scale,
        this adds ``extension_factor * range``.
    """
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    original_xlim = xlim[1]

    is_log_scale = ax.get_xscale() == 'log'

    if is_log_scale:
        log_xlim = np.log10(xlim[1])
        log_range = np.log10(xlim[1]) - np.log10(xlim[0])
        new_xlim = 10 ** (log_xlim + extension_factor * log_range)
    else:
        x_range = xlim[1] - xlim[0]
        new_xlim = xlim[1] + extension_factor * x_range

    ax.set_xlim(xlim[0], new_xlim)

    # Clip grid lines at the original x-limit
    clip_path = Path([
        [xlim[0], ylim[0]],
        [original_xlim, ylim[0]],
        [original_xlim, ylim[1]],
        [xlim[0], ylim[1]],
        [xlim[0], ylim[0]],
    ])
    clip_patch = mpatches.PathPatch(clip_path, transform=ax.transData, visible=False)

    for line in ax.get_xgridlines():
        line.set_clip_path(clip_patch)
        line.set_clip_on(True)
    for line in ax.get_ygridlines():
        line.set_clip_path(clip_patch)
        line.set_clip_on(True)


# ---------------------------------------------------------------------------
#  Label-overlap avoidance
# ---------------------------------------------------------------------------

def adjust_label_positions_to_avoid_overlap(
    ax: Axes,
    label_data: list[tuple[tuple[float, float], str, dict]],
    min_y_distance_factor: float = 1.5,
    x_scale: Literal['linear', 'log'] = 'log',
) -> list[tuple[float, float]]:
    """
    Adjust label positions to avoid overlaps on a log-scale y-axis.

    Parameters
    ----------
    ax : Axes
        The matplotlib axes containing the labels.
    label_data : list[tuple[tuple[float, float], str, dict]]
        List of ``(position, text, kwargs)`` for each label.
    min_y_distance_factor : float, optional
        Minimum distance between labels as a factor of the smaller y-value.
    x_scale : Literal['linear', 'log'], optional
        Scale of the x-axis.

    Returns
    -------
    adjusted_positions : list[tuple[float, float]]
        List of adjusted ``(x, y)`` positions for each label.
    """
    if not label_data:
        return []

    sorted_indices = sorted(range(len(label_data)), key=lambda i: label_data[i][0][1])
    sorted_labels = [label_data[i] for i in sorted_indices]
    adjusted_positions = [pos for pos, _, _ in sorted_labels]

    for i in range(1, len(adjusted_positions)):
        prev_x, prev_y = adjusted_positions[i - 1]
        curr_x, curr_y = adjusted_positions[i]
        if prev_y > 0 and curr_y > 0:
            log_prev_y = np.log10(prev_y)
            log_curr_y = np.log10(curr_y)
            min_log_distance = np.log10(min_y_distance_factor)
            if log_curr_y - log_prev_y < min_log_distance:
                adjusted_positions[i] = (curr_x, 10 ** (log_prev_y + min_log_distance))

    final_positions: list[tuple[float, float] | None] = [None] * len(label_data)
    for i, orig_idx in enumerate(sorted_indices):
        final_positions[orig_idx] = adjusted_positions[i]
    return final_positions  # type: ignore


# ---------------------------------------------------------------------------
#  Grey background box for legends
# ---------------------------------------------------------------------------

def add_legend_background_box(
    fig: Figure,
    legend,
    title_row_y_shift: float = -11.0,
    bg_padding_left: float = -65.0,
    bg_padding_right: float = +1.0,
    bg_padding_top: float = 0.0,
    bg_padding_bottom: float = -10.0,
    extra_right_x: float | None = None,
) -> None:
    """
    Draw a rounded grey background rectangle behind a legend.

    Parameters
    ----------
    fig : Figure
    legend : matplotlib Legend object
    title_row_y_shift, bg_padding_* : float
        Padding values (in display units / points).
    extra_right_x : float | None
        If given (in display coords), extend background to at least this x.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend_bbox = legend.get_window_extent(renderer)

    left_x = legend_bbox.x0 - bg_padding_left
    right_x = legend_bbox.x1 + bg_padding_right
    if extra_right_x is not None:
        right_x = max(right_x, extra_right_x + bg_padding_right)
    top_y = legend_bbox.y1 + (title_row_y_shift * fig.dpi / 72.0) + bg_padding_top
    bottom_y = legend_bbox.y0 - bg_padding_bottom

    left_fig, bottom_fig = fig.transFigure.inverted().transform((left_x, bottom_y))
    right_fig, top_fig = fig.transFigure.inverted().transform((right_x, top_y))

    from matplotlib.patches import FancyBboxPatch
    bg_rect = FancyBboxPatch(
        (left_fig, bottom_fig),
        right_fig - left_fig,
        top_fig - bottom_fig,
        boxstyle="round,pad=0.02",
        transform=fig.transFigure,
        facecolor='lightgrey',
        alpha=0.15,
        edgecolor='darkgrey',
        linewidth=1.8,
        zorder=-1,
    )
    fig.add_artist(bg_rect)


# ---------------------------------------------------------------------------
#  Place header labels above a table-style legend
# ---------------------------------------------------------------------------

def place_header_labels_above_legend(
    fig: Figure,
    legend,
    header_indices: list[int],
    header_labels: list[str],
    nrow: int,
    title_row_x_shifts: list[float] | None = None,
    title_row_y_shift: float = -11.0,
    fontsize: int = 14,
) -> None:
    """
    Add bold header text above a table-style matplotlib legend.

    Parameters
    ----------
    fig : Figure
    legend : matplotlib Legend
    header_indices : list[int]
        Indices (into the legend entries, column-major) of the header
        placeholder entries, **excluding** the row-label column.
    header_labels : list[str]
        Text for each header.
    nrow : int
        Number of rows in the legend table (used only as documentation;
        positions come from header_indices).
    title_row_x_shifts : list[float] | None
        Per-header horizontal shift in points.
    title_row_y_shift : float
        Vertical offset in points above the legend top.
    fontsize : int
        Font size for headers.
    """
    if title_row_x_shifts is None:
        title_row_x_shifts = [0.0] * len(header_labels)
    elif len(title_row_x_shifts) < len(header_labels):
        title_row_x_shifts = list(title_row_x_shifts) + [0.0] * (len(header_labels) - len(title_row_x_shifts))

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend_bbox = legend.get_window_extent(renderer)
    y_display = legend_bbox.y1 + (title_row_y_shift * fig.dpi / 72.0)
    legend_texts = legend.get_texts()

    for i, label in enumerate(header_labels):
        idx = header_indices[i]
        header_text = legend_texts[idx]
        header_bbox = header_text.get_window_extent(renderer)
        x_display = (header_bbox.x0 + header_bbox.x1) / 2.0
        x_display += title_row_x_shifts[i] * fig.dpi / 72.0

        x_fig, y_fig = fig.transFigure.inverted().transform((x_display, y_display))

        fig.text(
            x_fig, y_fig, label,
            ha='center', va='center',
            fontsize=fontsize, fontweight='bold',
            transform=fig.transFigure,
        )
