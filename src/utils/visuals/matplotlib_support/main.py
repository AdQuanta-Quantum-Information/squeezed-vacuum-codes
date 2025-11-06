# for type hints:
from typing import Optional, Sequence, Literal

# For saving plots:
from pathlib import Path
import os
import time

## Import matplotlib:
import matplotlib.pyplot as plt
import matplotlib as mpl

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


def draw_now():
    sleep_time: float = 0.01
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