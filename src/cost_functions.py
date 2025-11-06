#!/usr/bin/env python3
"""
Recreate KL‐cost graphs (Fig. 3 in Bashmakova et al. 2025) for arbitrary logical codewords.
Requires: matplotlib
"""

#%%
import numpy as np
from numpy.typing import NDArray

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import functools  
import itertools

import warnings
warnings.simplefilter('error', RuntimeWarning)  #TODO remove

from typing import Literal, Callable, Iterable, TypeAlias, Final, Any
from typing import cast as type_cast

import warnings

import qutip
from qutip import Qobj



if __name__ == "__main__":
    from __init__ import add_project_to_path, add_root_to_path
    # add_project_to_path()
    path = add_root_to_path()

from src.utils.visuals.matplotlib_support import save_figure, draw_now
from src.utils.prints import ProgressBar
from src.utils.files import saveload
from src.utils.maths import factorial, sqrt_factorial, power_computed_in_log_space
from src.utils.caches import cache

from projects.controlled_squeezing.src.squeezing_code import SqueezingCode
from projects.controlled_squeezing.src.visualizations import plot_light_states, plot_fock_distribution
from projects.controlled_squeezing.src.codes_built_in_superposition import simple_m_legged_code, simple_m_legged_state, _CodeTypes
from projects.controlled_squeezing.src.noise import noise_simulation, BosonicNoiseType, test_effect_of_time_resolution
from projects.controlled_squeezing.src.metrics import compute_cross_overlap_mat
from projects.controlled_squeezing.src.bosonic_operators import get_operator
from projects.controlled_squeezing.globals import Globals
from projects.controlled_squeezing.src.mean_photon_number import find_parameter_for_target_mean_photon_number
from projects.controlled_squeezing.src.kraus_maps import kraus_operators_series, kraus_operator_j, _check_kraus_series_completeness, _assert_correct_kraus_ops, _derive_num_kraus_operators
from projects.controlled_squeezing.src.kraus_maps import KRAUS_COST_THRESHOLD, KRAUS_TOO_SMALL_STREAK_SIZE, KrausTruncationError




if Globals.PRECISE:
    if "auto_tidyup" in qutip.settings.core:         #type: ignore
        qutip.settings.core["auto_tidyup"] = False   #type: ignore


## Types:
MeasurementTypeLiteral : TypeAlias = Literal["KL", "overlap01", "overlap00", "fidelity01", "fidelity00"]
CodeTypeLiteral : TypeAlias = Literal["cat", "squeeze"]
NoiseOptionLiteral : TypeAlias = Literal["simulated", "kraus-KL-style", "kraus-channel"]
VariablesNameLiteral : TypeAlias = Literal["r", "γ", "mean_number"]
CostPerNoiseDict : TypeAlias = dict[BosonicNoiseType, list[float]]
CostPerLegsPerNoiseDict : TypeAlias = dict[int, CostPerNoiseDict]


## Constants:
PROG_BAR_SIGNIFICANT_DIGITS : Final[int] = 6
NORMALIZE_LOGICAL_STATES_BEFORE_APPLYING_HADAMARD : Final[bool] = True


def _cache_function():
    if Globals.CACHE_ON_DISK:
        return cache(ram=False, disk=True)
    else:
        return cache(ram=True, disk=False)


def fock_operators(dimension:int):
    """
    Build the annihilation operator a and creation operator a†
    in a Hilbert space truncated to 'dimension' Fock states.
    """
    a = get_operator("destroy", dimension)
    a_dag = get_operator("create", dimension)
    return a, a_dag


r_in_proposal_db = 4   #[dB]
r_in_proposal_strength = r_in_proposal_db / (20 * np.log10(np.e))


def compute_cost_from_overlap_matrix(f: np.ndarray, measurement: MeasurementTypeLiteral) -> float:
    match measurement:
        case "KL":
            return abs(f[0,0] - f[1,1])**2 + abs(f[0,1])**2 + abs(f[1,0])**2
        case "overlap00":
            return abs(1 - f[0, 0])
        case "overlap01":
            return abs(f[0, 1])
        case "fidelity01":
            return abs(f[0, 1])**2  # fidelity is the square of the overlap
        case "fidelity00":
            return 1 - abs(f[0, 0])**2  # fidelity is the square of the overlap
        case _:
            raise ValueError(f"Unknown measurement type: {measurement!r}")


def f_ij_ab_kraus_overlap(ψi:Qobj, ψj:Qobj, ka:Qobj, kb:Qobj) -> complex:
    ka_dag_kb_ψj = ka.dag() @ kb @ ψj
    return ψi.overlap(ka_dag_kb_ψj)


def overlap_matrix(ρ_vec_1: list[Qobj], ρ_vec_2: list[Qobj]) -> np.ndarray:
    f = np.zeros((2,2), dtype=complex)

    for (i, ρi), (j, ρj) in itertools.product(enumerate(ρ_vec_1), enumerate(ρ_vec_2)):
        f[i, j] = ρi.overlap(ρj)

    return f


class _ConsecutiveTrue:
    def __init__(self,  streak_len:int):
        self.streak_len = streak_len
        self._count = 0

    def __call__(self, value:bool|np.bool_) -> bool:
        if value:
            self._count += 1
        else:
            self._count = 0
        return self._count >= self.streak_len
    
    def reset(self) -> None:
        self._count = 0

    def __repr__(self) -> str:
        return f"_ConsecutiveSmall(streak_len={self.streak_len}, current_count={self._count})"
    
    
def _check_operators(_num_kraus_ops:int, _kraus_j:Callable[[int], Qobj]) -> int:
    _expected_ops = []
    for j in ProgressBar.range(_num_kraus_ops, prefix="checking kraus"):
        try:
            kj = _kraus_j(j)  
        except KrausTruncationError:
            _num_kraus_ops = j   # adjust number of ops
            break
        _expected_ops.append(kj)

    if Globals.DEBUG:
        _check_kraus_series_completeness(_expected_ops)

    return _num_kraus_ops


@_cache_function()
def kraus_map_overlap_matrices(
    m:int, r:float, γ:float, 
    N: int,
    code_type: CodeTypeLiteral,
    noise_type: BosonicNoiseType, 
    noise_method: NoiseOptionLiteral
) -> NDArray[np.object_]:  # a matrix of overlap matrices

    # Helper function wrapper already taking everything except j:
    def _kraus_j(j:int) -> Qobj:
        return kraus_operator_j(noise_type, N, γ, j)
    
    ψ0, ψ1 = _get_m_legged_states(m=m, strength=r, num_moments=N, code_type=code_type, noise_type=noise_type)
    if "False" == False:
        _plot_code(ψ0, ψ1)

    _num_kraus_ops = _derive_num_kraus_operators(noise_type, N, γ)
    _num_kraus_ops = _check_operators(_num_kraus_ops, _kraus_j)   


    prog_bar = ProgressBar(_num_kraus_ops**2, prefix="compute overlap ")

    small_values_streak_a = _ConsecutiveTrue(streak_len=KRAUS_TOO_SMALL_STREAK_SIZE)
    small_values_streak_b = _ConsecutiveTrue(streak_len=KRAUS_TOO_SMALL_STREAK_SIZE)

    ## init these variables to suppress linter warning about existence:
    too_small : bool = False 
    b : int = 0  
    _highest_used_index = -1

    # init an `_num_kraus_ops×_num_kraus_ops` array filled with nans:
    overlap_matrices : NDArray[np.object_] = np.full((_num_kraus_ops, _num_kraus_ops), np.nan, dtype=object)

    for a in range(_num_kraus_ops):
        ka = _kraus_j(a)

        for b in range(_num_kraus_ops):
            kb = _kraus_j(b)

            prog_bar.next()
            too_small = False

            ## Actual overlap matrix computation:       <-----
            f = np.zeros((2,2), dtype=complex)
            for (i, ψi), (j, ψj) in itertools.product(enumerate([ψ0, ψ1]), repeat=2):
                f[i, j] = f_ij_ab_kraus_overlap(ψi, ψj, ka, kb)

            ## When to stop results that are way too small:
            too_small_ = np.any(np.isnan(f)) or np.linalg.norm(f, ord='fro') < KRAUS_COST_THRESHOLD 
            too_small = type_cast(bool, too_small_)

            if small_values_streak_b(too_small):  # meaning that kb is too small
                small_values_streak_b.reset()

                # Progress bar skip:
                _count_remainder = _num_kraus_ops - b - 1
                if _count_remainder > 0:
                    prog_bar.next(_count_remainder)

                break

            overlap_matrices[a,b] = f
            if a == b:
                _highest_used_index = max(_highest_used_index, a)


        if too_small and b <= 2*KRAUS_TOO_SMALL_STREAK_SIZE and a <= b  :  # meaning that also ka itself is too small
            if small_values_streak_a(True):
                small_values_streak_a.reset()
                break

    prog_bar.clear()

    ## Completeness check:
    if Globals.DEBUG:
        _used_ops = [_kraus_j(j) for j in range(_highest_used_index+1)]
        _check_kraus_series_completeness(_used_ops)

        if False == "False":
            used_norms = [Kj.norm() for Kj in ProgressBar(_used_ops)]
            plt.plot(used_norms, linewidth=6)
            axis = plt.gca()
            axis.set_yscale("log")
            plt.xlabel("Kraus Operator Index")
            plt.ylabel("Kraus Operator Norm")
            draw_now()

    return overlap_matrices


def kraus_representation_channel(ρ: Qobj, kraus_ops: Iterable[Qobj]) -> Qobj:
    if Globals.DEBUG:
        _assert_correct_kraus_ops(list(kraus_ops), _check_completeness_cond=True)

    elements : list[Qobj] = []
    for k in ProgressBar(kraus_ops, prefix="apply kraus     "):
        elem = k @ ρ @ k.dag()
        if np.any(np.isnan(elem.full())):
            continue
        elements.append(elem)

    res = sum(elements)
    res = type_cast(Qobj, res)
    return res


def _plot_code(ψ0:Qobj, ψ1:Qobj) -> None:
    plot_fock_distribution(ψ0, title=r"$|\psi_0\rangle$")
    plot_fock_distribution(ψ1, title=r"$|\psi_1\rangle$")
    plot_light_states([ψ0, ψ1])
    draw_now()


@cache(ram=True, disk=False)
def _get_m_legged_states_before_deciding_on_basis(m: int, strength: float, num_moments: int, code_type: _CodeTypes) -> tuple[Qobj, Qobj]:
    ψ0, ψ1 = simple_m_legged_code(m=m, strength=strength, num_moments=num_moments, code_type=code_type, _force_normalized=NORMALIZE_LOGICAL_STATES_BEFORE_APPLYING_HADAMARD)
    return ψ0, ψ1


def _get_m_legged_states(m: int, strength: float, num_moments: int, code_type: _CodeTypes, noise_type: BosonicNoiseType) -> tuple[Qobj, Qobj]:
    """ Return the two logical states of an m-legged code.
    Cached for speed.
    """
    ## Get logical states or dual basis states:
    if noise_type == "dephasing":
        use_dual_code = True
    elif noise_type == "loss":
        use_dual_code = False
    else:
        raise ValueError(f"Unknown noise type: {noise_type!r}")

    # Start with un-normalized states:
    ψ0, ψ1 = _get_m_legged_states_before_deciding_on_basis(m=m, strength=strength, num_moments=num_moments, code_type=code_type)

    if use_dual_code:
        # Dual basis states: |+⟩ = (|0⟩ + |1⟩)/√2 and |−⟩ = (|0⟩ - |1⟩)/√2
        ψ_plus  = (ψ0 + ψ1)/np.sqrt(2)
        ψ_minus = (ψ0 - ψ1)/np.sqrt(2)
        ψ0, ψ1 = ψ_plus, ψ_minus

    # Normalize states:
    if not NORMALIZE_LOGICAL_STATES_BEFORE_APPLYING_HADAMARD:
        ψ0.unit(inplace=True)
        ψ1.unit(inplace=True)

    return ψ0, ψ1


def _get_noised_state(ρ_in: Qobj, γ: float, **kwargs) -> Qobj:
    N : int = int(ρ_in.dims[0][0])
    noise_type : BosonicNoiseType = kwargs["noise_type"] 
    noise_method : NoiseOptionLiteral = kwargs["noise_method"]
    mesolve_time_res : int = kwargs["mesolve_time_res"]

    match noise_method:
        case "kraus-channel":
            kraus_ops = kraus_operators_series(noise_type, N, γ)
            ρ_out = kraus_representation_channel(ρ_in, kraus_ops)
        case "simulated":
            ρ_out = noise_simulation(
                ρ_in, noise_type, rate=γ, 
                with_progress_bar=True, 
                time_res=mesolve_time_res, 
                output_intermediate_states=False
            )

    return ρ_out  #type: ignore


@_cache_function()
def _compute_cost_given_m_r_and_noise(
    m:int, r:float, γ:float, 
    num_moments: int,
    noise_type: BosonicNoiseType, 
    noise_method: NoiseOptionLiteral, 
    measurement: MeasurementTypeLiteral,
    code_type: CodeTypeLiteral,
    **kwargs
) -> float:
           
    match noise_method:
        case "kraus-KL-style":
            overlap_matrices = kraus_map_overlap_matrices(m, r, γ, num_moments, code_type, noise_type, noise_method)
            costs_matrix = _costs_matrix_from_overlap_matrices(overlap_matrices, measurement="overlap01")
            cost = float(np.nansum(costs_matrix))  # Sum matrix while ignoring NaNs
            if False == "False":
                _plot_costs_matrix(costs_matrix)

        case "simulated" | "kraus-channel":
            ψ0, ψ1 = _get_m_legged_states(m=m, strength=r, num_moments=num_moments, code_type=code_type, noise_type=noise_type)
            noise_kwargs = dict(
                noise_type = noise_type,
                noise_method = noise_method,
                mesolve_time_res = kwargs.get("mesolve_time_res", None)
            )
            ρ_ins = [ψ0.proj(), ψ1.proj()]  
            ρ_outs = [_get_noised_state(ρ_in, γ, **noise_kwargs) for ρ_in in ProgressBar(ρ_ins, prefix="different ρ     ")]
            f = overlap_matrix(ρ_ins, ρ_outs)
            cost = compute_cost_from_overlap_matrix(f, measurement)
        
        case _:
            raise ValueError(f"Unknown noise method: {noise_method!r}")

    return cost


def _get_parameters_from_fixed_and_x(
    fixed_param_name: VariablesNameLiteral,
    fixed_value: float,
    x_name: VariablesNameLiteral,
    x: float,
    code: CodeTypeLiteral,
    m: int
) -> tuple[float, float]:
            
    r = None
    γ = None

    def _assign_param(name:str, value:float) -> None:
        nonlocal r, γ
        match name:
            case "r":
                r = value
            case "γ":
                γ = value
            case "mean_number":
                r = find_parameter_for_target_mean_photon_number(
                    code_type=code,
                    m=m,
                    logical_value=0,
                    target_mean_photon_number=value
                )
            case _:
                raise ValueError(f"Unknown fixed_param_name: {fixed_param_name!r}")
            
    _assign_param(fixed_param_name, fixed_value)
    _assign_param(x_name, x)

    assert r is not None, "Both r and γ must be assigned." 
    assert γ is not None, "Both r and γ must be assigned."

    return r, γ  #type: ignore


def _costs_matrix_from_overlap_matrices(overlap_matrices:NDArray[np.object_], measurement:MeasurementTypeLiteral) -> NDArray[np.float_]:
    num_ops = overlap_matrices.shape[0]
    overlap_matrices_iter = np.nditer(overlap_matrices, flags=["refs_ok", "multi_index"])
    costs_matrix : NDArray[np.float_] = np.full((num_ops, num_ops), np.nan, dtype=float)

    for _overlap_matrix in ProgressBar(overlap_matrices_iter, prefix="compute cost    "):
        a, b = overlap_matrices_iter.multi_index
    
        _overlap_matrix = type_cast(np.ndarray, _overlap_matrix) # for type checker
        overlap_matrix = _overlap_matrix.item() 

        if not isinstance(overlap_matrix, np.ndarray) and np.isnan(overlap_matrix).any():
            continue

        cost = compute_cost_from_overlap_matrix(overlap_matrix, measurement=measurement) 
        costs_matrix[a, b] = cost

    return costs_matrix


def compute_cost_on_logical_codewords(
    fixed_param_name: VariablesNameLiteral,
    fixed_value: float,
    x_name: VariablesNameLiteral,
    x_vec:list[float],
    num_moments : int = 500,
    mesolve_time_res: int = 1501,
    num_code_states:int = 3,
    code: CodeTypeLiteral = "squeeze",  # "squeeze", "cat"
    measurement: MeasurementTypeLiteral = "overlap01",  # "KL", "worst_fidelity", "average_fidelity", "coherence_survival", "overlap01", "overlap00"
    noise_method : NoiseOptionLiteral = "kraus-KL-style"  # "simulated", "kraus"
) -> CostPerLegsPerNoiseDict:
    
    num_legs_vec = np.arange(2, 2+num_code_states*2, 2) 
    costs_for_all_m: CostPerLegsPerNoiseDict = dict()

    ## ========= Compute stuff =========:
    for m in ProgressBar(num_legs_vec, prefix="different m     "):
        ProgressBar.newest().append_extra_str(f"m={m:.{PROG_BAR_SIGNIFICANT_DIGITS}g}")
        costs_for_all_noises : CostPerNoiseDict = dict()

        for noise_type in ProgressBar(['dephasing', 'loss'], prefix="different noise "):
            ProgressBar.newest().append_extra_str(f"noise={noise_type}")
            noise_type = type_cast(BosonicNoiseType, noise_type)

            cost_vec = []
            for x in ProgressBar(x_vec, prefix=f"different {x_name}     "):      
                ProgressBar.newest().append_extra_str(f"{x_name}={x:.{PROG_BAR_SIGNIFICANT_DIGITS}g}")

                r, γ = _get_parameters_from_fixed_and_x(
                    fixed_param_name=fixed_param_name,
                    fixed_value=fixed_value,
                    x_name=x_name,
                    x=x,
                    code=code,
                    m=m
                )

                ## Compute cost (this call is cached):
                cost = _compute_cost_given_m_r_and_noise(
                    m, r, γ, 
                    num_moments=num_moments,
                    noise_type=noise_type, 
                    noise_method=noise_method,
                    measurement=measurement,
                    code_type=code,
                    mesolve_time_res=mesolve_time_res
                )
                cost_vec.append(cost)

            costs_for_all_noises[noise_type] = cost_vec

        costs_for_all_m[m] = costs_for_all_noises

    saveload.save(costs_for_all_m, "kl_costs - num_legs - "+measurement)

    return costs_for_all_m


def _plot_costs_matrix(costs_matrix:NDArray[np.float_]) -> None:
    plt.figure(figsize=(6,4))
    
    # Mask NaN values
    masked_costs = np.ma.masked_invalid(costs_matrix)
    
    # Get non-zero, non-NaN values for better range estimation
    valid_costs = costs_matrix[~np.isnan(costs_matrix)]
    non_zero_costs = valid_costs[valid_costs > 0]
    
    if len(non_zero_costs) > 0:
        vmin = max(non_zero_costs.min(), 1e-20)  # Floor at 1e-20
        vmax = min(non_zero_costs.max(), 1.0)     # Cap at 1.0
    else:
        vmin, vmax = 1e-10, 1e-1  # Fallback to original values
    
    # Create the plot with masked array
    im = plt.imshow(masked_costs, cmap='viridis', norm=mcolors.LogNorm(vmin=vmin, vmax=vmax))
    
    # Set color for NaN values 
    im.cmap.set_bad(color='white', alpha=1.0)
    
    # Create colorbar with proxy artist for NaN
    cbar = plt.colorbar(im, label="Cost contribution")
    
    # Add text annotation for NaN on the colorbar
    cbar.ax.text(0.0, -0.1, 'NaN', transform=cbar.ax.transAxes, 
                 va='center', ha='left', fontsize=12, 
                 bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', linewidth=0.5))
    
    plt.title(f"Cost contributions from Kraus operators")
    plt.xlabel("Kraus Operator Index")
    plt.ylabel("Kraus Operator Index")
    draw_now()


def _sanity_check_overlap_with_no_noise(
    m:int = 2,
    r_vals:list[float] = np.linspace(1e-5, 2.5, 21).tolist(),
    num_moments:int = 400
):
    
    plt.figure(figsize=(6,4))
    overlaps = {False: [], True: []}
    
    for r in ProgressBar(r_vals, prefix="different r     "):
        ## Get codewords:
        ψ0, ψ1 = simple_m_legged_code(m=m, strength=r, num_moments=num_moments, code_type="squeeze", _force_normalized=False)

        for dual_basis in [False, True]:
            
            if dual_basis:
                ψ_plus  = ψ0 + ψ1
                ψ_minus = ψ0 - ψ1
                ψ0, ψ1 = ψ_plus, ψ_minus
            ψ0.unit(inplace=True)
            ψ1.unit(inplace=True)

            if False == "False":
                _plot_code(ψ0, ψ1)

            ## Overlap:
            overlap = ψ0.overlap(ψ1)
            assert np.isclose(np.imag(overlap), 0.0, atol=1e-10), f"Overlap is not real! Got {overlap!r} for r={r:.6g}, m={m}"
            overlaps[dual_basis].append(np.abs(overlap))

    for dual_basis, y_vec in overlaps.items():
        label = "dual basis" if dual_basis else "logical basis"
        plt.plot(r_vals, y_vec, marker='o', label=label)

    ## Finalize plot:
    plt.xlabel("r") 
    plt.ylabel(r"$\langle \psi_0 | \psi_1 \rangle$")
    plt.title(f"Overlap between logical codewords vs r (m={m})")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    draw_now()
    print("Done.")
    


def _sanity_check_2_no_noise_overlap(
    m:int = 4,
    r = 2.3,   # for mean-n=2
    num_moments:int = 200,
    γ = 1e-16
) -> None:

    ψ0, ψ1 = _get_m_legged_states(m, r, num_moments, code_type="squeeze", noise_type="loss")
    ψ_plus, ψ_minus = _get_m_legged_states(m, r, num_moments, code_type="squeeze", noise_type="dephasing")
    # _plot_code(ψ0, ψ1)
    # _plot_code(ψ_plus, ψ_minus)

    overlap = ψ_plus.overlap(ψ_minus)
    assert np.isclose(np.imag(overlap), 0.0, atol=1e-10), f"Overlap is not real! Got {overlap!r} for r={r:.6g}, m={m}"
    abs_overlap = np.abs(overlap)


    ## Now investigate the kraus_map cost:
    overlap_matrices = kraus_map_overlap_matrices(m, r, γ, num_moments, code_type="squeeze", noise_type="dephasing", noise_method=None)
    costs_matrix = _costs_matrix_from_overlap_matrices(overlap_matrices, measurement="overlap01")
    cost = np.nansum(costs_matrix)

    ## plot the cost matrix:
    _plot_costs_matrix(costs_matrix)

    print(f"Direct overlap: {abs_overlap:.6g}")
    print(f"Kraus-map overlap01 cost: {cost:.6g}")

    print("Done.")


def _plotting_kraus_op_norms(
    num_moments:int = 50,
    γ_vec : list[float] = [1e-16, 1e-10, 1e-7, 1e-5, 1e-4],
    _check_kraus_ops: bool = True
) -> None:

    fig, ax = plt.subplots(figsize=(6,4))
    ax.set_yscale("log")
    ax.set_xlabel("Kraus Operator Index", fontsize=16)
    ax.set_ylabel("Operator Norm", fontsize=16)
    ax.grid(True)


    for γ in γ_vec:
        used_kraus_ops : list[Qobj] = kraus_operators_series(noise_type="dephasing", N=num_moments, γ=γ, _check=_check_kraus_ops)

        norms = []
        for Kj in ProgressBar(used_kraus_ops, prefix="used kraus     "):
            norm = Kj.norm() 
            if np.any(np.isnan(norm)):
                break
            if norm == 0:
                pass
            norms.append(norm)

        x_final = len(used_kraus_ops)
        x_vec = np.arange(len(norms))
        line = ax.plot(x_vec, norms, linewidth=6, label=f"γ={γ:.1e}")
        color = line[0].get_color()
        # Vertical line at x_final:
        ax.axvline(x=x_final, linestyle='--', color=color)
        draw_now()

    ax.legend(fontsize=12)
    print("Done.")


if __name__ == "__main__":
    # _sanity_check_overlap_with_no_noise()
    _sanity_check_2_no_noise_overlap()
    # _plotting_kraus_op_norms()
