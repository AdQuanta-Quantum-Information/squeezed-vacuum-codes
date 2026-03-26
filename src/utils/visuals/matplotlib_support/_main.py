# for type hints:
from typing import Optional, Sequence, Literal, NamedTuple
from dataclasses import dataclass

# For saving plots:
from pathlib import Path
import os
import time

## Import matplotlib:
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mticker
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
PlotBoxBoundsUnits = Literal["data", "axes-window-fraction"]


@dataclass(frozen=True)
class PlotBoxBounds:
    """Rectangle bounds stored as (x0, y0, width, height).

    Constructor accepts either:
    - (x0, y0, width, height), or
    - (x0, y0, x1=..., y1=...), or
    - mixed by axis (x0 with width/x1 and y0 with height/y1).
    """

    x0: float
    y0: float
    width: float
    height: float

    def __init__(
        self,
        x0: float,
        y0: float,
        width: float | None = None,
        height: float | None = None,
        *,
        x1: float | None = None,
        y1: float | None = None,
    ) -> None:
        if width is None and x1 is None:
            raise TypeError("Provide either 'width' or 'x1'.")
        if height is None and y1 is None:
            raise TypeError("Provide either 'height' or 'y1'.")

        computed_width = width if width is not None else (x1 - x0)  # type: ignore[operator]
        computed_height = height if height is not None else (y1 - y0)  # type: ignore[operator]

        if computed_width <= 0:
            raise ValueError("Bounds width must be positive.")
        if computed_height <= 0:
            raise ValueError("Bounds height must be positive.")

        if width is not None and x1 is not None and abs((x0 + width) - x1) > 1e-12:
            raise ValueError("Inconsistent x-axis bounds: width != x1 - x0.")
        if height is not None and y1 is not None and abs((y0 + height) - y1) > 1e-12:
            raise ValueError("Inconsistent y-axis bounds: height != y1 - y0.")

        object.__setattr__(self, "x0", x0)
        object.__setattr__(self, "y0", y0)
        object.__setattr__(self, "width", computed_width)
        object.__setattr__(self, "height", computed_height)

    @property
    def x1(self) -> float:
        return self.x0 + self.width

    @property
    def y1(self) -> float:
        return self.y0 + self.height






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
    magnified_data_bounds: PlotBoxBounds,
    inset_axes_bounds: PlotBoxBounds,
    magnified_data_bounds_units: PlotBoxBoundsUnits = "data",
    inset_axes_bounds_units: PlotBoxBoundsUnits = "data",
    use_readable_ticks: bool = False,
    readable_tick_count: int = 3,
    x_window_rel_start: float = 0.86,
    y_padding_ratio: float = 0.18,
    border_color: str = "red",
    border_linewidth: float = 1.8,
    connector_linewidth: float = 1.3,
    connector_loc1: int = 2,
    connector_loc2: int = 4,
    auto_avoid_diagonal_connectors: bool = True,
    hide_inset_ticks: bool = True,
    inset_tick_fontsize: int = 8,
) -> Axes | None:
    """Create a zoomed inset (magnification-glass style).

    Bounds can be interpreted either as:
    - "data": original axis data units
    - "axes-window-fraction": fractions in the current visible axis window
      (0 means window minimum, 1 means window maximum on each axis).

        When ``auto_avoid_diagonal_connectors`` is enabled, connector corner
        indices are adjusted automatically in overlap cases to prefer edge-aligned
        connectors over long diagonals crossing the inset interior.
    """

    def _axis_value_to_scaled(value: float, scale: str, axis_name: str) -> float:
        if scale == "log":
            if value <= 0:
                raise ValueError(f"{axis_name}-axis log scale requires positive values, got {value}.")
            return float(np.log10(value))
        return value

    def _scaled_to_axis_value(value: float, scale: str) -> float:
        if scale == "log":
            return float(10 ** value)
        return value

    def _window_fraction_to_data(
        frac: float,
        lim0: float,
        lim1: float,
        scale: str,
        axis_name: str,
    ) -> float:
        scaled0 = _axis_value_to_scaled(lim0, scale, axis_name)
        scaled1 = _axis_value_to_scaled(lim1, scale, axis_name)
        return _scaled_to_axis_value(scaled0 + frac * (scaled1 - scaled0), scale)

    def _data_to_window_fraction(
        value: float,
        lim0: float,
        lim1: float,
        scale: str,
        axis_name: str,
    ) -> float:
        scaled0 = _axis_value_to_scaled(lim0, scale, axis_name)
        scaled1 = _axis_value_to_scaled(lim1, scale, axis_name)
        span = scaled1 - scaled0
        if abs(span) < 1e-15:
            raise ValueError(f"Cannot map {axis_name}-axis bounds: axis window span is zero.")
        scaled_value = _axis_value_to_scaled(value, scale, axis_name)
        return (scaled_value - scaled0) / span

    def _to_data_bounds(bounds: PlotBoxBounds, units: PlotBoxBoundsUnits) -> PlotBoxBounds:
        x_lim = ax.get_xlim()
        y_lim = ax.get_ylim()
        if units == "data":
            return bounds
        if units == "axes-window-fraction":
            return PlotBoxBounds(
                x0=_window_fraction_to_data(bounds.x0, x_lim[0], x_lim[1], x_scale, "x"),
                y0=_window_fraction_to_data(bounds.y0, y_lim[0], y_lim[1], y_scale, "y"),
                x1=_window_fraction_to_data(bounds.x1, x_lim[0], x_lim[1], x_scale, "x"),
                y1=_window_fraction_to_data(bounds.y1, y_lim[0], y_lim[1], y_scale, "y"),
            )
        raise ValueError(f"Unknown bounds units: {units!r}")

    def _to_inset_axes_fraction_bounds(bounds: PlotBoxBounds, units: PlotBoxBoundsUnits) -> PlotBoxBounds:
        if units == "axes-window-fraction":
            return bounds

        data_bounds = _to_data_bounds(bounds, units)
        x_lim = ax.get_xlim()
        y_lim = ax.get_ylim()
        return PlotBoxBounds(
            x0=_data_to_window_fraction(data_bounds.x0, x_lim[0], x_lim[1], x_scale, "x"),
            y0=_data_to_window_fraction(data_bounds.y0, y_lim[0], y_lim[1], y_scale, "y"),
            x1=_data_to_window_fraction(data_bounds.x1, x_lim[0], x_lim[1], x_scale, "x"),
            y1=_data_to_window_fraction(data_bounds.y1, y_lim[0], y_lim[1], y_scale, "y"),
        )

    x_scale = ax.get_xscale()
    y_scale = ax.get_yscale()

    magnified_bounds_data = _to_data_bounds(magnified_data_bounds, magnified_data_bounds_units)
    magnified_bounds_fraction = _to_inset_axes_fraction_bounds(magnified_bounds_data, "data")
    inset_bounds_fraction = _to_inset_axes_fraction_bounds(inset_axes_bounds, inset_axes_bounds_units)

    x_window: tuple[float, float] | None = None
    y_window: tuple[float, float] | None = None
    if magnified_bounds_data is not None:
        x_window = (
            magnified_bounds_data.x0,
            magnified_bounds_data.x0 + magnified_bounds_data.width,
        )
        y_window = (
            magnified_bounds_data.y0,
            magnified_bounds_data.y0 + magnified_bounds_data.height,
        )

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

    bounds = inset_bounds_fraction
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
        inset_ax.tick_params(
            axis="both",
            which="major",
            labelsize=inset_tick_fontsize,
            length=4,
            width=1.0,
            colors="black",
        )
        inset_ax.tick_params(axis="both", which="minor", length=2, width=0.8, colors="black")

        if use_readable_ticks:
            tick_count = max(2, readable_tick_count)

            def _interior_ticks(v0: float, v1: float, scale: str) -> list[float]:
                lo, hi = (v0, v1) if v0 <= v1 else (v1, v0)
                if scale == "log":
                    if lo <= 0:
                        return []
                    log_lo = float(np.log10(lo))
                    log_hi = float(np.log10(hi))
                    fractions = np.linspace(0.15, 0.85, tick_count)
                    return [float(10 ** (log_lo + frac * (log_hi - log_lo))) for frac in fractions]
                span = hi - lo
                fractions = np.linspace(0.15, 0.85, tick_count)
                return [float(lo + frac * span) for frac in fractions]

            def _short_num_label(v: float) -> str:
                return f"{v:.2g}"

            x0, x1 = inset_ax.get_xlim()
            y0, y1 = inset_ax.get_ylim()
            x_ticks = _interior_ticks(x0, x1, inset_ax.get_xscale())
            y_ticks = _interior_ticks(y0, y1, inset_ax.get_yscale())

            if x_ticks:
                inset_ax.xaxis.set_major_locator(mticker.FixedLocator(x_ticks))
                inset_ax.set_xticklabels([_short_num_label(v) for v in x_ticks])
            if y_ticks:
                inset_ax.yaxis.set_major_locator(mticker.FixedLocator(y_ticks))
                inset_ax.set_yticklabels([_short_num_label(v) for v in y_ticks])

            inset_ax.xaxis.set_minor_locator(mticker.NullLocator())
            inset_ax.yaxis.set_minor_locator(mticker.NullLocator())

    for spine in inset_ax.spines.values():
        spine.set_edgecolor(border_color)
        spine.set_linewidth(border_linewidth)

    mark_loc1 = connector_loc1
    mark_loc2 = connector_loc2

    if auto_avoid_diagonal_connectors:
        roi = magnified_bounds_fraction
        inset = inset_bounds_fraction

        x_overlap = min(roi.x1, inset.x1) > max(roi.x0, inset.x0)
        y_overlap = min(roi.y1, inset.y1) > max(roi.y0, inset.y0)

        # Corner indices follow Matplotlib convention:
        # 1=upper-right, 2=upper-left, 3=lower-left, 4=lower-right.
        if x_overlap and not y_overlap:
            if roi.y1 <= inset.y0:
                mark_loc1, mark_loc2 = 3, 4
            elif roi.y0 >= inset.y1:
                mark_loc1, mark_loc2 = 1, 2
        elif y_overlap and not x_overlap:
            if roi.x1 <= inset.x0:
                mark_loc1, mark_loc2 = 2, 3
            elif roi.x0 >= inset.x1:
                mark_loc1, mark_loc2 = 1, 4

    mark_inset(
        ax,
        inset_ax,
        loc1=mark_loc1,
        loc2=mark_loc2,
        fc="none",
        ec=border_color,
        lw=connector_linewidth,
    )
    return inset_ax