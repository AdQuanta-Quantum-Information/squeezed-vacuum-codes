# for type hints:
from typing import Optional, Sequence, Literal, NamedTuple

# For saving plots:
from pathlib import Path
import os
import time

## Import matplotlib:
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

import numpy as np

from .mpl_types import *
from .mpl_backends import _detect_backend


# Use our other utils 
from ... import strings, assertions, arguments, types, prints, lists
from ....code_paths import outputs

## select the proper backend for rendering figures: 
try:
    mpl_backend = _detect_backend()
    mpl.use(mpl_backend)

except ImportError as e:
    import warnings
    warnings.warn(str(e))

plt.ion()




SupportedFigExtension = Literal["png", "jpg", "jpeg", "tiff", "bmp", "svg", "pdf", "eps"]


class InsetAxesBounds(NamedTuple):
    """Axes-relative inset bounds: (x0, y0, width, height)."""
    x0: float
    y0: float
    width: float
    height: float






def _toggle_latex_state(value:bool) -> None:
    plt.rcParams['text.usetex'] = value

def turn_latex_on() -> None:
    _toggle_latex_state(True)

def turn_latex_off() -> None:
    _toggle_latex_state(False)


def _assert_path_exists_or_create(path:Path) -> None:
    if not path.is_dir():
        os.mkdir(str(path.resolve()))

def get_saved_figures_folder()->Path:
    figures_folder = outputs / "figures"
    _assert_path_exists_or_create(outputs)
    _assert_path_exists_or_create(figures_folder)
    return figures_folder


def save_figure(
    fig:Optional[Figure]=None, 
    file_name:Optional[str]=None, 
    extensions:list[SupportedFigExtension]=["png", "svg"],
    dpi=300, 
    transparent=True
) -> None:
    # Figure:
    if fig is None:
        fig = plt.gcf()
    # Title:
    if file_name is None:
        file_name = strings.time_stamp()
    # Figures folder:
    folder = get_saved_figures_folder()
    # Full path:
    fullpath = folder.joinpath(file_name)
    fullpath_str = str(fullpath.resolve())
    # Add extensions
    for ext in extensions:
        if fullpath_str.endswith(ext):            
            fullpath_str_with_ext = fullpath_str
        else:
            fullpath_str_with_ext = fullpath_str+"."+ext
        # Save:
        fig.savefig(fullpath_str_with_ext, dpi=dpi, transparent=transparent)
    return 


def save_all_figures(extensions:list[str]=["png"]) -> None:
    time_stamp = strings.time_stamp()
    for i in plt.get_fignums():
        fig = plt.figure(i)
        name = time_stamp + f"_{i}"
        save_figure(fig, file_name=name, extensions=extensions)


def new_figure(nrows:int=1, ncols:int=1) -> tuple[Figure, Sequence[Axes]]:
    fig, (axes) = plt.subplots(nrows=nrows, ncols=ncols) 
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    return fig, axes


def close_all():
    plt.close('all')


def draw_now(sleep_time: float = 0.01) -> None:
    time.sleep(sleep_time)
    plt.pause(sleep_time)
    time.sleep(sleep_time)


def no_y_axis_offset(ax:Axes) -> None:
    ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))


def twin_axis(axis:Axes) -> Axes:
    twin = axis.twinx()
    for ax, color in zip([axis, twin], ["tab:blue", "tab:red"], strict=True):
        ax.set_ylabel(ax.get_ylabel(), color=color)
        ax.tick_params(axis='y', labelcolor=color)
    plt.sca(twin)
    return twin


def clean_figure_memory(fig:Figure) -> None:
    """ Cleans the memory of a figure by removing all its axes and artists.
    """
    for ax in fig.axes:
        ax.cla()
    fig.clf()
    ## Delete the figure's window:
    plt.close(fig)


def add_magnification_glass_inset(
    ax: Axes,
    x_scale: Literal["linear", "log"] = "log",
    y_scale: Literal["linear", "log"] = "log",
    x_window: tuple[float, float] | None = None,
    y_window: tuple[float, float] | None = None,
    x_window_rel_start: float = 0.86,
    y_padding_ratio: float = 0.18,
    inset_axes_bounds: InsetAxesBounds | tuple[float, float, float, float] = InsetAxesBounds(0.56, 0.05, 0.40, 0.38),
    border_color: str = "red",
    border_linewidth: float = 1.8,
    connector_linewidth: float = 1.3,
    connector_loc1: int = 2,
    connector_loc2: int = 4,
    hide_inset_ticks: bool = True,
    inset_tick_fontsize: int = 8,
) -> Axes | None:
    """Create a zoomed inset (magnification-glass style) for the tail behavior of plotted lines."""
    lines = [
        line
        for line in ax.get_lines()
        if np.asarray(line.get_xdata()).size > 0 and np.asarray(line.get_ydata()).size > 0
    ]
    if not lines:
        return None

    x_min_all = min(float(np.min(line.get_xdata())) for line in lines)
    x_max_all = max(float(np.max(line.get_xdata())) for line in lines)

    if x_window is None:
        if x_scale == "log" and x_min_all > 0:
            log_min = float(np.log10(x_min_all))
            log_max = float(np.log10(x_max_all))
            x_start = 10 ** (log_min + x_window_rel_start * (log_max - log_min))
        else:
            x_start = x_min_all + x_window_rel_start * (x_max_all - x_min_all)
        x_window = (x_start, x_max_all)

    if x_window[0] >= x_window[1]:
        return None

    if y_window is None:
        y_candidates: list[float] = []
        for line in lines:
            x_data = np.asarray(line.get_xdata(), dtype=float)
            y_data = np.asarray(line.get_ydata(), dtype=float)
            mask = (x_data >= x_window[0]) & (x_data <= x_window[1])
            if np.any(mask):
                y_candidates.extend(y_data[mask].tolist())

        if not y_candidates:
            return None

        y_min = float(np.min(y_candidates))
        y_max = float(np.max(y_candidates))
        if y_scale == "log" and y_min > 0:
            log_y_min = float(np.log10(y_min))
            log_y_max = float(np.log10(y_max))
            log_span = max(log_y_max - log_y_min, 1e-6)
            y_window = (
                10 ** (log_y_min - y_padding_ratio * log_span),
                10 ** (log_y_max + y_padding_ratio * log_span),
            )
        else:
            y_span = max(y_max - y_min, 1e-12)
            y_window = (
                y_min - y_padding_ratio * y_span,
                y_max + y_padding_ratio * y_span,
            )

    if y_window[0] >= y_window[1]:
        return None

    bounds = InsetAxesBounds(*inset_axes_bounds)
    inset_ax = ax.inset_axes((bounds.x0, bounds.y0, bounds.width, bounds.height))

    for line in lines:
        inset_ax.plot(
            line.get_xdata(),
            line.get_ydata(),
            color=line.get_color(),
            linestyle=line.get_linestyle(),
            linewidth=line.get_linewidth(),
            alpha=line.get_alpha(),
        )

    inset_ax.set_xscale(x_scale)
    inset_ax.set_yscale(y_scale)
    inset_ax.set_xlim(*x_window)
    inset_ax.set_ylim(*y_window)

    if hide_inset_ticks:
        inset_ax.set_xticks([])
        inset_ax.set_yticks([])
    else:
        inset_ax.tick_params(axis="both", labelsize=inset_tick_fontsize)

    for spine in inset_ax.spines.values():
        spine.set_edgecolor(border_color)
        spine.set_linewidth(border_linewidth)

    mark_inset(
        ax,
        inset_ax,
        loc1=connector_loc1,
        loc2=connector_loc2,
        fc="none",
        ec=border_color,
        lw=connector_linewidth,
    )
    return inset_ax