import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

import itertools

from qutip import Qobj, identity, destroy, create, basis, qeye, qzero, mesolve



if __name__ == "__main__":
    from __init__ import add_project_to_path, add_root_to_path
    # add_project_to_path()
    path = add_root_to_path()

from src.utils.visuals.matplotlib_support import save_figure, draw_now, clean_figure_memory
from src.utils.visuals.videos import VideoRecorder
from src.utils.prints import ProgressBar

from src.codes_built_in_superposition import simple_m_legged_code
from src.visualizations import plot_light_states
from src.noise import noise_simulation


def _plus_or_minus_str(i:int) -> str:
    match i:
        case 0:
            return "+"
        case 1:
            return "-"
        case _:
            raise ValueError(f"Invalid index {i}, expected 0 or 1.")


def _get_code_states(num_legs:int, r:float, num_moments:int) -> tuple[Qobj, Qobj]:
    ψ0, ψ1 = simple_m_legged_code(m=num_legs, strength=r, num_moments=num_moments, code_type="squeeze", num_digits=2)
    return ψ0, ψ1


def _single_plot_states(ψ0:Qobj, ψ1:Qobj, r:float, t:float, high_resolution:bool) -> Figure:
    if high_resolution:
        num_points = 1000
    else:
        num_points = 30

    dict = plot_light_states([ψ0, ψ1], _draw_now=False, num_points=num_points)

    for i in [0, 1]:
        state_str = f"|{i}⟩ ∝ S(+r)|0⟩ "+_plus_or_minus_str(i)+" S(-r)|0⟩"
        dict["axes"][i].set_title(state_str, fontsize=14)
    dict["axes"][1].set_ylabel("")
    dict['fig'].suptitle(f"r={r:0.3f}, t={t:0.3f}", fontsize=16)
    return dict['fig']


def record_video(
    r_values:list[float] = np.linspace(0.0, 2.0,  51).tolist(),
    t_values:list[float] = np.linspace(0.0, 3.0, 101).tolist(),
    num_legs:int = 2,
    num_moments:int = 30,
    γ_photon_loss:float = 1.0,
    high_resolution:bool = True
):
    ## Helpers:
    def _plot_per_r(r:float) -> Figure:
        ψ0, ψ1 = _get_code_states(num_legs, r, num_moments)
        return _single_plot_states(ψ0, ψ1, r, 0.0, high_resolution=high_resolution)
    
    # Initiate video recording object:
    video_recorder = VideoRecorder(fps=30)
    dpi = 500 if high_resolution else 10
    def _capture_frame(fig, duration_in_seconds:float|None=None) -> None:
        video_recorder.capture(fig, clear_figure_after=True, duration_in_seconds=duration_in_seconds, dpi=dpi)

    ## First frame should last longer:
    fig = _plot_per_r(r_values[0])
    _capture_frame(fig, duration_in_seconds=1)
    
    ## Increasing r:
    for r in ProgressBar(r_values[1:], prefix="Recording video.. "):
        fig = _plot_per_r(r)
        _capture_frame(fig)

    ## Decoherence time:
    # Simulate decoherence:
    ψ0, ψ1 = _get_code_states(num_legs, r_values[-1], num_moments)
    noised_states = [
        noise_simulation(ψi, γ_photon_loss, times=t_values)
        for ψi in [ψ0, ψ1]
    ]
    # Iterate couples of states for each time t
    for t, (ψ0_t, ψ1_t) in ProgressBar(enumerate(zip(*noised_states))):
        fig = _single_plot_states(ψ0_t, ψ1_t, r_values[-1], t, high_resolution=high_resolution)
        _capture_frame(fig)
    
    video_recorder.write_video()

    print("All done.")


if __name__ == "__main__":
    # single_plot(r=0.5, num_legs=2, num_moments=100)
    record_video()