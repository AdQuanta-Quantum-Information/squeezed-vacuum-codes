from .mpl_types import *

import matplotlib.pyplot as plt

# Everyone needs numpy:
from typing import TypeAlias


def _get_crnt_axis_if_not_given(ax:Axes|None) -> Axes:
    if ax is None:
        ax = plt.gca()
    return ax


_LegendMap : TypeAlias = dict[Line2D, str]


def _add_legend(ax: Axes, legend_map: _LegendMap, **kwargs) -> Legend:
    legend = ax.legend(handles=legend_map.keys(), labels=legend_map.values(), **kwargs)
    ax.add_artist(legend)
    return legend


def _legend_map_copy_with_chosen_attributes(map: _LegendMap, get:list[str]=["marker", "linestyle", "color"]) -> _LegendMap:    
    ## Assume not get:
    map_copy = {}
    for line, label in map.items():

        marker = line.get_marker() if "marker" in get else 'None'
        linestyle = line.get_linestyle() if "linestyle" in get else '-'
        color = line.get_color() if "color" in get else 'gray'

        new_line = Line2D([0], [0], marker=marker, linestyle=linestyle, color=color)
        map_copy[new_line] = label
    return map_copy

def _force_legends_to_different_locations(ax: Axes, legends: list[Legend]) -> None:
    possible_locations = ['upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left', 'center right', 'lower center', 'upper center', 'center'] 
    for i, legend in enumerate(legends):
        location = possible_locations[i % len(possible_locations)]
        legend.set_bbox_to_anchor(None)  # Reset any previous bbox settings
        legend.set_loc(location)
        ax.add_artist(legend)
    

def add_split_legend(
    ax: Axes | None = None, 
    colors: _LegendMap | None = None, 
    markers: _LegendMap | None = None,
    linestyles: _LegendMap | None = None,
    markers_and_linestyles: _LegendMap | None = None,
) -> dict[str, Legend]:
    """ This function gets axes with many plots that are separated by different aspect of style, 
    and instead of a single big legend, it creates multiple legends depending on the style of the plots that were provided.
    """
    ax = _get_crnt_axis_if_not_given(ax)
    res = {}
    
    if colors:
        # If the legend is about color, should not contain different styles: 
        map_copy = _legend_map_copy_with_chosen_attributes(colors, get=["color"])
        legend = _add_legend(ax, map_copy, title="Colors")
        res["colors"] = legend
    
    if markers:
        # If the legend is about markers, the examples brought in should contain no color:
        map_copy = _legend_map_copy_with_chosen_attributes(markers, get=["marker"])
        legend = _add_legend(ax, map_copy, title="Markers:")
        res["markers"] = legend

    if linestyles:
        legend = _add_legend(ax, linestyles, title="Line Styles")
        res["linestyles"] = legend
    
    if markers_and_linestyles:
        # If the legend is about markers, the examples brought in should contain no color:
        map_copy = _legend_map_copy_with_chosen_attributes(markers_and_linestyles, get=["marker", "linestyle"])
        legend = _add_legend(ax, map_copy, title="Markers and Line Styles")
        res["markers_and_linestyles"] = legend

    ## Force different legends to be in different locations:
    _force_legends_to_different_locations(ax, [l for l in res.values()])

    return res
