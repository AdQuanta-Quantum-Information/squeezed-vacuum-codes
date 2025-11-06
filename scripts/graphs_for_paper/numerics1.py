import numpy as np

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.path import Path
import matplotlib.patches as mpatches


from typing import Literal 
from typing import cast as type_cast

from soupsieve import match 

if __name__ == "__main__":
    from __init__ import add_project_to_path, add_root_to_path
    path = add_root_to_path()

from src.utils.visuals.matplotlib_support import save_figure, draw_now
from src.utils.visuals.colors import color_shades, _RgbFloatTuple
from src.utils.prints import ProgressBar

from projects.controlled_squeezing.src.cost_functions import compute_cost_on_logical_codewords
from projects.controlled_squeezing.src.cost_functions import CodeTypeLiteral, MeasurementTypeLiteral, NoiseOptionLiteral, BosonicNoiseType, CostPerNoiseDict, CostPerLegsPerNoiseDict

from projects.controlled_squeezing.globals import Globals

if Globals.LaTeX_RENDERING:
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Computer Modern Serif']
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'


def _add_outside_legend(
    fig, legend_colors, 
    linewidth=4, 
    legend_layout: Literal["2-columns", "2-rows"] = "2-columns",
    vertical_plots: bool = False,
    show_box: bool = False
):
    """
    Add a table-style legend to the figure.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to attach the legend to.
    legend_colors : dict
        Nested dict mapping {"squeeze": {m: color}, "cat": {m: color}}.
        Should contain the colors actually used in the plots.
    linewidth : float, optional
        Line width for legend handles (default = 4).
    legend_layout : str, optional
        Layout style: "2-columns" or "2-rows" (default = "2-columns").
        - "2-columns": squeeze and cat as column headers
        - "2-rows": squeeze and cat as row headers, m values as columns
    vertical_plots : bool, optional
        Whether plots are arranged vertically (default = False).
    show_box : bool, optional
        Whether to show a box/frame around the legend (default = False).
    
    Returns
    -------
    legend : matplotlib.legend.Legend
        The created legend object.
    """
    # Collect code values:
    codes = list(legend_colors.keys())
    # Collect all unique m values
    all_m = sorted(set().union(*(legend_colors[code].keys() for code in codes)))

    handles: list[Line2D] = []
    labels: list[str] = []
    title_indices = []

    if legend_layout == "2-columns":
        # Current layout: squeeze and cat as columns
        counter = 0
        for code in codes:
            # Create a handle with zero width for the title
            title_handle = Line2D([], [], linestyle="none", marker="", markersize=0)
            handles += [title_handle]
            labels  += [code]
            title_indices.append(counter)
            counter += 1
            for m in all_m:
                if m in legend_colors[code]:
                    handles.append(Line2D([0], [0], color=legend_colors[code][m], linewidth=linewidth))
                    labels.append(f"{m}")
                    counter += 1
        
        ncol = 2
        
    elif legend_layout == "2-rows":
        # New layout: m values as columns, squeeze and cat as rows
            
            # Add entries for each m value
        counter = 0
        for m in [-1]+all_m:
            for code in codes:

                if m == -1:
                    # Add title for this row
                    title_handle = Line2D([], [], linestyle="none", marker="", markersize=0)
                    handles += [title_handle]
                    labels += [code]
                    title_indices.append(counter)

                elif m in legend_colors[code]:
                    handles.append(Line2D([0], [0], color=legend_colors[code][m], linewidth=linewidth))
                    labels.append(f"{m}")
                else:
                    # Add empty entry to maintain alignment
                    handles.append(Line2D([], [], linestyle="none", alpha=0))
                    labels.append("")

                counter += 1
        
        ncol = len(all_m) + 1  # +1 for the title column
        
    else:
        raise ValueError(f"Unknown layout: {legend_layout}. Use '2-columns' or '2-rows'.")

    if vertical_plots:
        loc="lower center"
        bbox_to_anchor=(0.6, -0.0)
    else:
        loc="lower center"
        bbox_to_anchor=(0.5, -0.0)


    legend = fig.legend(
        handles, labels,
        ncol=ncol,
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        fontsize=9,
        columnspacing=0.7,
        handletextpad=1.1,
        borderaxespad=1.2,
        frameon=show_box
    )

    # Format header titles
    for title_i in title_indices: 
        text : Text = legend.get_texts()[title_i]
        legend_handle : Line2D = legend.legend_handles[title_i]

        text.set_fontweight("bold")
        text.set_horizontalalignment("left")
        legend_handle.set_visible(False)

    return legend


def _ylabel_for_measurement(measurement: MeasurementTypeLiteral, noise_method: NoiseOptionLiteral, dual_code:bool, noise:Literal["loss", "dephasing"]) -> str:
    """Return the y-axis label for the given measurement type."""
    L0, L1 = [r'+\!', r'-\!'] if dual_code else [r'0', r'1']
    sL0, sL1 = [r'%s_{L}' % (L) for L in [L0, L1]]
    match noise_method:
        case "simulated":
            prefix = ""
            mid = r'\mathcal{N}_{\gamma}'
        case "kraus-KL-style":
            prefix = r'\sum_{a,b} \;'
            mid = r'K_{a}^{\dagger}(\gamma) K_{b}(\gamma)'
        case "kraus-channel":
            prefix = ""
            mid = r'\mathcal{N}_{\gamma}'
        case _:
            raise ValueError(f"Unknown noise method: {noise_method!r}")
        
    if measurement in ["fidelity00", "overlap00"]:
        prefix = "1 - " + prefix


    match measurement:
        case "KL":
            main_str = r'C_{\rm KL}(|%s\rangle , |%s\rangle)'% (sL0, sL1)
            return "$"+main_str+"$"
        case "overlap01":
            main_str =  r'{|\; \langle %s |' % (sL0) + mid + r'| %s \rangle \; |}' % (sL1)
        case "overlap00":
            main_str =  r'1 \  - \  {|\; \langle %s |' % (sL0) + mid + r'| %s \rangle \; |}' % (sL1)
        case "fidelity01":
            main_str = r'{F(\rho_{%s}, \mathcal{N}_{\gamma}( \rho_{%s} ))}' % (sL0, sL1)
        case "fidelity00":
            main_str = r'{F(\rho_{%s}, \mathcal{N}_{\gamma}( \rho_{%s} ))}' % (sL0, sL0)
        case _:
            raise ValueError(f"Unknown measurement type: {measurement!r}")

    full_str = r"$%s%s$" % (prefix, main_str)

    ## Special cases:
    if measurement == "overlap01":
        full_str = r'$V_{KL}(\mathcal{N}^{\,%s}(\gamma))_{%s,%s}$' % (noise, L0, L1)

    return full_str


def _axis_setup(
    grid : Literal["on", "off", "weak"] = "weak",
    measurement: MeasurementTypeLiteral = "KL",
    noise_method : NoiseOptionLiteral = "kraus",
    enlarge_text: bool = True,
    vertical_plots: bool = False,
    fig_dpi: int|None = None,
    _connected_plots:bool = False
) -> tuple[
    Figure,
    dict[BosonicNoiseType, Axes ]
]:
    
    if vertical_plots:
        nrows, ncols = 2, 1
        if _connected_plots:
            fig_size = (5,8)
        else:
            fig_size = (5,10)
    else:
        if _connected_plots:
            raise ValueError("Connected plots only supported for vertical arrangement.")
        nrows, ncols = 1, 2
        fig_size = (10,4)

    ## Plotting setup
    if fig_dpi is None:
        fig = plt.figure(figsize=fig_size)
    else:
        fig = plt.figure(figsize=fig_size, dpi=fig_dpi)


    ax_loss = plt.subplot(nrows,ncols,1)
    ax_dephase = plt.subplot(nrows,ncols,2)
    #
    ax_loss.set_title('Loss error')
    # (b) Pure-dephasing
    ax_dephase.set_title('Dephasing error')

    match grid:
        case "on":
            ax_loss.grid(True, which='both')
            ax_dephase.grid(True, which='both')    
        case "off":
            pass
        case "weak":
            ax_loss.grid(True, which='major', alpha=0.5)
            ax_dephase.grid(True, which='major', alpha=0.5)

    if enlarge_text:
        for ax in [ax_loss, ax_dephase]:
            ax.xaxis.set_tick_params(labelsize=12)
            ax.yaxis.set_tick_params(labelsize=12)
            ax.set_xlabel(ax.get_xlabel(), fontsize=16)
            ax.set_ylabel(ax.get_ylabel(), fontsize=16)
            ax.set_title(ax.get_title(), fontsize=16)

    axes : dict[BosonicNoiseType, Axes] = dict(
        loss = ax_loss,
        dephasing = ax_dephase
    ) #type: ignore

    return fig, axes


def _get_text_pos(final_graph_point: tuple[float, float], code: CodeTypeLiteral, m: int, noise_type: BosonicNoiseType) -> tuple[float, float]:
    x = final_graph_point[0]*1.6 
    y = final_graph_point[1]

    if noise_type == "photon_loss":
        if code == "squeeze" and m == 2:
            y *= 1.5
        elif code == "cat" and m == 4:
            y *= 0.6
    return x, y


def _extend_axis_without_grid(ax: Axes, extension_factor: float = 0.2) -> None:
    """
    Extend the x-axis limits while keeping the grid at its original extent.
    
    Parameters
    ----------
    ax : Axes
        The matplotlib axes to modify.
    extension_factor : float, optional
        Factor by which to extend the x-axis. For log scale, this multiplies the upper limit.
        For linear scale, this adds extension_factor * range.
    """
    
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    original_xlim = xlim[1]
    
    # Check if x-axis is log scale
    is_log_scale = ax.get_xscale() == 'log'
    
    if is_log_scale:
        # For log scale: extend in log space
        # Convert to log space, extend linearly, then convert back
        log_xlim = np.log10(xlim[1])
        log_range = np.log10(xlim[1]) - np.log10(xlim[0])
        new_log_xlim = log_xlim + extension_factor * log_range
        new_xlim = 10 ** new_log_xlim
    else:
        # For linear scale: add extension_factor * range
        x_range = xlim[1] - xlim[0]
        new_xlim = xlim[1] + extension_factor * x_range
    
    # Extend the visible axis range first
    ax.set_xlim(xlim[0], new_xlim)
    
    # Create a transform that clips at the original x limit
    # This works by creating a bbox in data coordinates
    
    # Define the clipping path as a rectangle
    clip_path = Path([
        [xlim[0], ylim[0]],
        [original_xlim, ylim[0]],
        [original_xlim, ylim[1]],
        [xlim[0], ylim[1]],
        [xlim[0], ylim[0]]
    ])
    
    clip_patch = mpatches.PathPatch(clip_path, transform=ax.transData, visible=False)
    
    # Apply clipping to all grid lines
    for line in ax.get_xgridlines():
        line.set_clip_path(clip_patch)
        line.set_clip_on(True)
    
    for line in ax.get_ygridlines():
        line.set_clip_path(clip_patch)
        line.set_clip_on(True)
        


def _plot_results(
    # Mandatory inputs:
    per_code_results: dict[CodeTypeLiteral, CostPerLegsPerNoiseDict],
    x_vec_name: Literal["γ", "gamma", "r"],
    x_vec: list[float],
    measurement: MeasurementTypeLiteral,
    noise_method : NoiseOptionLiteral,
    # defaults for plotting:
    grid: Literal["on", "off", "weak"] = "weak",
    fig_dpi: int = 500,
    vertical_plots: bool = True,
    figure_name_extra: str = "",
    N: int|None = None,
    _connected_plots:bool = True,
    _text_on_plots:bool = True,
    text_font_size:int = 16,
    text_legend_on_plot_font_size:int = 12, # only used if _text_on_plots is True
    _adjust_ticks_font:bool = True
):
    """ Plot the results from compute_cost_on_logical_codewords(). """

    ## Simplify x_vec_name:
    if x_vec_name in ["γ", "gamma"]:
        x_vec_name = "γ"
    elif x_vec_name == "r":
        pass
    else:
        raise ValueError(f"Unknown x_vec_name: {x_vec_name!r}")
    
    ## ========= Extract info =========:
    codes = list(per_code_results.keys())

    ## ========= Constants =========:
    _linewidth = 3
    # _colors =  ["tab_blue", "tab_red"]
    _colors =  ["blue", "red"]

    ## ========= Plot =========:
    fig, axes = _axis_setup(
        grid, measurement, noise_method, 
        vertical_plots=vertical_plots,
        _connected_plots=_connected_plots
    )
    match x_vec_name:
        case "γ":
            xlabel = r'$\gamma$ noise rate' 
        case "r":
            xlabel = r'$r (\alpha)$ squeezing (displacement) strength'
    # Loss plot:
    y_label = _ylabel_for_measurement(measurement, noise_method, dual_code=False, noise="loss")
    axes["loss"].set_xlabel(xlabel , fontsize=text_font_size)
    axes["loss"].set_ylabel(y_label, fontsize=text_font_size)

    # Dephasing plot:
    y_label = _ylabel_for_measurement(measurement, noise_method, dual_code=True, noise="dephasing")
    axes["dephasing"].set_xlabel(xlabel , fontsize=text_font_size)
    axes["dephasing"].set_ylabel(y_label, fontsize=text_font_size)

    for ax in axes.values():
        if x_vec_name == "γ":
            ax.set_xscale('log')
        ax.set_yscale('log')

    if _connected_plots:
        ax = axes["loss"]
        ## Remove x labels and x ticks-labels:
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax.set_xlabel("")   
        for ax in axes.values():
            # put title on the right side, as vertical text:
            # title_str = ax.get_title()
            ax.set_title("")
            # ax.text(1.02, 0.5, title_str, transform=ax.transAxes, rotation=-90, va='center', fontsize=16)


    legend_colors: dict[str, dict[int, _RgbFloatTuple]] = {code: {} for code in codes}

    plot_style = dict(
        linewidth = _linewidth
    )

    for code, base_color in zip(codes, _colors, strict=True):
        code = type_cast(CodeTypeLiteral, code)
        results = per_code_results[code]
        num_m = len(results)
        colors = color_shades(base_color, num_m+1)[:-1]  # Skip the darkest color

        for noise_type, ax in axes.items():
            noise_type = type_cast(BosonicNoiseType, noise_type)

            for j, (m, costs) in enumerate(results.items()):
                y_vec = costs[noise_type]

                color = colors[j]
                legend_colors[code][m] = color

                label = f"{m}"
                ax.plot(x_vec, y_vec, label=label, color=color, **plot_style)  #type: ignore

                ## add label next to final point:
                if _text_on_plots:
                    x_dif = x_vec[1] - x_vec[0]
                    final_graph_point = (x_vec[-1], y_vec[-1])
                    text_pos = _get_text_pos(final_graph_point, code, m, noise_type)
                    text = r"$\textbf{%s}$ $\mathbf{%s}$" % (code, m)
                    ax.text(*text_pos, text, fontsize=text_legend_on_plot_font_size, ha='left', va='center')

    if _text_on_plots:
        for ax in axes.values():
            _extend_axis_without_grid(ax, extension_factor=0.25)

    if _text_on_plots:
        legend = None
    else:
        legend = _add_outside_legend(fig, legend_colors, linewidth=_linewidth, legend_layout="2-rows", vertical_plots=vertical_plots)

    plt.tight_layout(rect=(0, 0.05, 1, 1))  # Reserve space at bottom for legend

    if _adjust_ticks_font:
        for ax in axes.values():
            for axis in [ax.xaxis, ax.yaxis]:
                axis.set_tick_params(labelsize=text_font_size)

    if _connected_plots:
        plt.subplots_adjust(hspace=0.001)

    plt.show()
    print("Plotted.")

    file_name = ""\
        + measurement  \
        + f" - {noise_method}" \
        + (f" - N={N}" if N is not None else "") \
        + (f" - {figure_name_extra}" if figure_name_extra else "") 
    
    save_figure(plt.gcf(), file_name, dpi=fig_dpi, transparent=True, extensions=['pdf', 'png'])
    print("Saved.")

    return fig, axes, legend


def plot_full_codewords_numeric_figure_x_is_gamma(
    num_moments : int = 500,
    num_gammas:int = 5,
    num_code_states:int = 2,
    measurement: MeasurementTypeLiteral = "overlap01",  # "KL", "overlap01", "overlap00"
    noise_method : NoiseOptionLiteral = "kraus-KL-style",  # "simulated", "kraus-KL-style", "kraus-channel"
    mean_photon_number : float = 2.0
) -> None:

    ## ========= Inputs =========:
    x_vec_name = "γ"

    γ_vec = np.logspace(-16, -5, num_gammas).tolist()

    ## ========= Compute =========:
    per_code_results : dict[CodeTypeLiteral, CostPerLegsPerNoiseDict] = dict()

    for code in ProgressBar(["squeeze", "cat"], prefix="different code  "):
        ProgressBar.newest().append_extra_str(f"{code!r}")
        code = type_cast(CodeTypeLiteral, code)

        results = compute_cost_on_logical_codewords(
            fixed_value=mean_photon_number,
            fixed_param_name="mean_number",
            x_name="γ",
            x_vec=γ_vec,
            num_moments=num_moments,
            num_code_states=num_code_states,
            code=code,
            measurement=measurement,
            noise_method=noise_method
        )
        per_code_results[code] = results 

    ## ========= Plot =========:
    fig, axes, legend = _plot_results(
        per_code_results, 
        x_vec_name=x_vec_name,
        x_vec=γ_vec,
        measurement=measurement,
        noise_method=noise_method,
        N=num_moments,
    )

    ## Wait for user to close:
    draw_now()
    input("Press Enter to close the plots and end the program...")



def plot_full_codewords_numeric_figure_x_is_r(
    num_moments : int = 200,
    num_photon_num: int = 21,
    max_photon_num: float = 10.0,
    num_code_states:int = 3,
    measurement: MeasurementTypeLiteral = "overlap01",  # "KL", "overlap01", "overlap00", "fidelity01", "fidelity00"
    noise_method : NoiseOptionLiteral = "kraus-KL-style",  # "simulated", "kraus" "kraus-channel"
    γ = 1e-6
) -> None:
    
    ## ========= Inputs =========:
    x_vec_name = "r"
    photon_num_vec = np.linspace(1e-3, max_photon_num, num_photon_num).tolist()
    # r_vec = r_vec[:-4]


    ## ========= Compute =========:
    per_code_results : dict[CodeTypeLiteral, CostPerLegsPerNoiseDict] = dict()

    for code in ProgressBar(["squeeze", "cat"], prefix="different code  "):
        code = type_cast(CodeTypeLiteral, code)
        ProgressBar.newest().append_extra_str(f"code={code}")

        results = compute_cost_on_logical_codewords(
            fixed_param_name="γ",
            fixed_value=γ,
            x_name="mean_number",
            x_vec=photon_num_vec,
            num_moments=num_moments,
            num_code_states=num_code_states,
            code=code,
            measurement=measurement,
            noise_method=noise_method
        )
        per_code_results[code] = results 

    ## ========= Plot =========:
    fig, axes, legend = _plot_results(
        per_code_results, 
        x_vec_name=x_vec_name,
        x_vec=photon_num_vec,
        measurement=measurement,
        noise_method=noise_method,
        figure_name_extra=f"r-changing - γ={γ}",
        N=num_moments,
    )



if __name__ == "__main__":
    plot_full_codewords_numeric_figure_x_is_gamma()
    # plot_full_codewords_numeric_figure_x_is_r()