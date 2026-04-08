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

from typing import Literal, Callable, Iterable, TypeAlias, Final, Any, TypedDict
from typing import cast as type_cast

import warnings

import qutip
from qutip import Qobj



if __name__ == "__main__":
    from __init__ import  add_root_to_path
    path = add_root_to_path()

 

from src.preparation_protocols import get_code_states_using_rotation, LogicalCodewordInfo
from src.utils.visuals.matplotlib_support import save_figure, draw_now
from src.utils.prints import ProgressBar
from src.utils.files import saveload
from src.utils.maths import factorial, sqrt_factorial, power_computed_in_log_space
from src.utils.caches import cache

from src.visualizations import plot_light_states, plot_fock_distribution
from src.codes_built_in_superposition import simple_m_legged_code, get_m_legged_states, _CodeTypes
from src.noise import noise_simulation, BosonicNoiseType, test_effect_of_time_resolution
from src.metrics import compute_cross_overlap_mat
from src.bosonic_operators import get_operator
from globals import Globals
from src.mean_photon_number import find_parameter_for_target_mean_photon_number
from src.kraus_maps import kraus_operators_series, kraus_operator_j, _check_kraus_series_completeness, _assert_correct_kraus_ops, _derive_num_kraus_operators
from src.kraus_maps import KRAUS_COST_THRESHOLD, KRAUS_TOO_SMALL_STREAK_SIZE, KrausTruncationError




if Globals.PRECISE:
    if "auto_tidyup" in qutip.settings.core:         #type: ignore
        qutip.settings.core["auto_tidyup"] = False   #type: ignore


## Types:

LogicalBasisName : TypeAlias = Literal["main", "dual"]
class CostPerLogicalBasis(TypedDict):
    main : float
    dual : float

MeasurementTypeLiteral : TypeAlias = Literal["KL", "overlap01", "overlap00", "fidelity01", "fidelity00", "probability"]
NoiseOptionLiteral : TypeAlias = Literal["simulated", "kraus-KL-style", "kraus-channel", "No-noise"]
VariablesNameLiteral : TypeAlias = Literal["r", "γ", "mean_n"]
CostPerNoiseDict : TypeAlias = dict[BosonicNoiseType, list[CostPerLogicalBasis]]
CostPerLegsPerNoiseDict : TypeAlias = dict[int, CostPerNoiseDict]


class _SpecificBasisOptionType(TypedDict):
    loss: LogicalBasisName
    dephasing: LogicalBasisName


## Constants:
PROG_BAR_SIGNIFICANT_DIGITS : Final[int] = 6
NORMALIZE_LOGICAL_STATES_BEFORE_APPLYING_HADAMARD : Final[bool] = True
TARGET_STATE_FOR_MEAN_NUMBER_CALCULATION : Final[Literal[0, 1, '+']] = 0


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
    code_type: _CodeTypes,
    noise_type: BosonicNoiseType, 
    use_dual_code: bool = False,
) -> NDArray[np.object_]:  # a matrix of overlap matrices


    # Helper function wrapper already taking everything except j:
    def _kraus_j(j:int) -> Qobj:
        return kraus_operator_j(noise_type, N, γ, j)
    
    ψ0, ψ1 = get_m_legged_states(m=m, strength=r, num_moments=N, code_type=code_type, use_dual_code=use_dual_code)
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
    _values_inserted :bool = False

    for a in range(_num_kraus_ops):
        k_a = _kraus_j(a)

        for b in range(_num_kraus_ops):
            k_b = _kraus_j(b)

            prog_bar.next()
            too_small = False

            ## Actual overlap matrix computation:       <-----
            f = np.zeros((2,2), dtype=complex)
            for (i, ψi), (j, ψj) in itertools.product(enumerate([ψ0, ψ1]), repeat=2):
                f[i, j] = f_ij_ab_kraus_overlap(ψi, ψj, k_a, k_b)

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
            _values_inserted = True

            if a == b:
                _highest_used_index = max(_highest_used_index, a)


        if too_small and b <= 2*KRAUS_TOO_SMALL_STREAK_SIZE and a <= b  :  # meaning that also k_a itself is too small
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

    
    if overlap_matrices.size == 0: # check if the array is empty
        raise ValueError("Overlap matrices array is empty! This should never happen. Check the kraus operators and the cost threshold.")
    # Check if the input even had any inserted values (i.e. not all nans):
    if not _values_inserted:
        raise ValueError("All overlap matrices are NaN! This should never happen. Check the kraus operators and the cost threshold.")

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


@cache(ram=True, disk=False)
def derive_preparation_probabilities(m:int, r:float, num_moments:int, use_dual_code:bool) -> tuple[float, float]:
    if use_dual_code:
        one_over_sqrt2 = 1/np.sqrt(2)
        logical_info = LogicalCodewordInfo(a=one_over_sqrt2, b=one_over_sqrt2)
    else:
        logical_info = LogicalCodewordInfo(a=1.0, b=0.0)


    ψ0, ψ1, meta_data = get_code_states_using_rotation(
        m=m, r=r, logical_info=logical_info, with_meta=True, num_moments=num_moments
    )
    
    probabilities = meta_data["probabilities"]
    return probabilities[0], probabilities[1]



@_cache_function()
def _compute_cost_given_m_r_and_noise(
    m:int, r:float, γ:float, 
    num_moments: int,
    noise_type: BosonicNoiseType, 
    noise_method: NoiseOptionLiteral, 
    measurement: MeasurementTypeLiteral,
    code_type: _CodeTypes,
    use_dual_code: bool,
    **kwargs
) -> float:
               

    match noise_method:
        case "kraus-KL-style":
            overlap_matrices = kraus_map_overlap_matrices(m, r, γ, num_moments, code_type, noise_type, use_dual_code)
            costs_matrix = _costs_matrix_from_overlap_matrices(overlap_matrices, measurement="overlap01")
            cost = float(np.nansum(costs_matrix))  # Sum matrix while ignoring NaNs
            if False == "False":
                _plot_costs_matrix(costs_matrix)

        case "simulated" | "kraus-channel":      
            ψ0, ψ1 = get_m_legged_states(
                m=m, strength=r, num_moments=num_moments, code_type=code_type, 
                use_dual_code=use_dual_code,
                normalize_logical_states_before_applying_hadamard=NORMALIZE_LOGICAL_STATES_BEFORE_APPLYING_HADAMARD
            )
            
            noise_kwargs = dict(
                noise_type = noise_type,
                noise_method = noise_method,
                mesolve_time_res = kwargs.get("mesolve_time_res", None)
            )
            ρ_ins = [ψ0.proj(), ψ1.proj()]  
            ρ_outs = [_get_noised_state(ρ_in, γ, **noise_kwargs) for ρ_in in ProgressBar(ρ_ins, prefix="different ρ     ")]
            f = overlap_matrix(ρ_ins, ρ_outs)
            cost = compute_cost_from_overlap_matrix(f, measurement)
                
        case "No-noise":
            assert measurement=="probability", f"No noise option only implemented for 'probability' measurement, got {measurement!r}"
            assert γ == 0, f"γ must be 0 for no-noise option, got {γ!r}"
            assert code_type == "squeeze", f"No noise option only implemented for 'squeeze' code type, got {code_type!r}"
            cost = derive_preparation_probabilities(m, r, num_moments, use_dual_code=use_dual_code)

        case _:
            raise ValueError(f"Unknown noise method: {noise_method!r}")
            

    return cost


def _get_parameters_from_fixed_and_x(
    fixed_param_name: VariablesNameLiteral,
    fixed_value: float,
    x_name: VariablesNameLiteral,
    x: float,
    code: _CodeTypes,
    m: int,
    num_moments: int|Callable[[float], int]
) -> tuple[float, float, int]:
            
    r = None
    γ = None

    def _assign_param(name:str, value:float) -> None:
        nonlocal r, γ
        match name:
            case "r":
                r = value
            case "γ":
                γ = value
            case "mean_n":
                r = find_parameter_for_target_mean_photon_number(
                    code_type=code,
                    m=m,
                    logical_value=TARGET_STATE_FOR_MEAN_NUMBER_CALCULATION,
                    target_mean_photon_number=value
                )
            case _:
                raise ValueError(f"Unknown fixed_param_name: {fixed_param_name!r}")
            
    _assign_param(fixed_param_name, fixed_value)
    _assign_param(x_name, x)

    assert r is not None, "Both r and γ must be assigned." 
    assert γ is not None, "Both r and γ must be assigned."

    if callable(num_moments):
        _num_moment = num_moments(x)
    else:
        _num_moment = num_moments

    return r, γ, _num_moment  #type: ignore


def _costs_matrix_from_overlap_matrices(overlap_matrices:NDArray[np.object_], measurement:MeasurementTypeLiteral) -> NDArray[np.float64]:
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


def _basis_str_from_use_dual_code_bool(use_dual_code: bool) -> LogicalBasisName: 
    return "dual" if use_dual_code else "main"


def _skip_on_mismatched_basis(specific_bases: _SpecificBasisOptionType, noise_type: BosonicNoiseType, use_dual_code: bool) -> bool:
    """Return True if the basis is not in the specific_bases list for the given noise_type."""

    ## Checks:
    assert isinstance(specific_bases, dict|None), f"specific_bases must be a dict, got {type(specific_bases)}"
    if specific_bases is None:
        return False
    
    if noise_type not in specific_bases:
        raise KeyError(f"Noise type {noise_type!r} not found in specific_bases keys: {list(specific_bases.keys())}")
    
    ## Logic:
    crnt_basis = _basis_str_from_use_dual_code_bool(use_dual_code)
    allowed_basis = specific_bases[noise_type]
    if crnt_basis == allowed_basis:
        return False
    else:
        return True


def compute_cost_on_logical_codewords(
    fixed_param_name: VariablesNameLiteral,
    fixed_value: float,
    x_name: VariablesNameLiteral,
    x_vec:list[float],
    num_moments : int|Callable[[float], int] = 500,
    mesolve_time_res: int = 1501,
    num_code_states:int = 3,
    code: _CodeTypes = "squeeze",  # "squeeze", "cat"
    measurement: MeasurementTypeLiteral = "overlap01",  # "KL", "worst_fidelity", "average_fidelity", "coherence_survival", "overlap01", "overlap00"
    noise_method : NoiseOptionLiteral = "kraus-KL-style",  # "simulated", "kraus"
    specific_bases: _SpecificBasisOptionType | None = None
) -> CostPerLegsPerNoiseDict:
    
    if code == "gkp":
        num_legs_vec : list[int] = [1]
    else:
        num_legs_vec : list[int] = np.arange(2, 2+num_code_states*2, 2).tolist()  # even numbers from 2 to 2*num_code_states


    costs_for_all_m: CostPerLegsPerNoiseDict = dict()

    ## ========= Compute stuff =========:
    for m in ProgressBar(num_legs_vec, prefix="different m     "):
        ProgressBar.newest().append_extra_str(f"m={m:.{PROG_BAR_SIGNIFICANT_DIGITS}g}")
        costs_for_all_noises : CostPerNoiseDict = dict()

        for noise_type in ProgressBar(['dephasing', 'loss'], prefix="different noise "):
            ProgressBar.newest().append_extra_str(f"noise={noise_type}")
            noise_type = type_cast(BosonicNoiseType, noise_type)

            cost_vec : list[CostPerLogicalBasis] = []
            for x in ProgressBar(x_vec, prefix=f"different {x_name:6}"):      
                ProgressBar.newest().append_extra_str(f"{x_name}={x:.{PROG_BAR_SIGNIFICANT_DIGITS}g}")

                r, γ, _num_moment = _get_parameters_from_fixed_and_x(
                    fixed_param_name=fixed_param_name,
                    fixed_value=fixed_value,
                    x_name=x_name,
                    x=x,
                    code=code,
                    m=m,
                    num_moments=num_moments
                )

                ## Compute cost (this call is cached):
                costs : CostPerLogicalBasis = {}  #type: ignore
                for use_dual_code in ProgressBar([False, True], prefix=f"logical-basis   "):      
                    basis = _basis_str_from_use_dual_code_bool(use_dual_code)
                    ProgressBar.newest().append_extra_str(f"basis: {basis!r}")

                    if (specific_bases is not None) and _skip_on_mismatched_basis(specific_bases, noise_type, use_dual_code):
                        cost = np.nan    
                    
                    else:
                        cost = _compute_cost_given_m_r_and_noise(
                            m, r, γ, 
                            num_moments=_num_moment,
                            noise_type=noise_type, 
                            noise_method=noise_method,
                            measurement=measurement,
                            code_type=code,
                            use_dual_code=use_dual_code,
                            mesolve_time_res=mesolve_time_res
                        )

                    costs["dual" if use_dual_code else "main"] = cost

                cost_vec.append(costs)

            costs_for_all_noises[noise_type] = cost_vec

        costs_for_all_m[m] = costs_for_all_noises

    saveload.save(costs_for_all_m, "kl_costs - num_legs - "+measurement)

    return costs_for_all_m


def _plot_costs_matrix(costs_matrix:NDArray[np.float64]) -> None:
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

    ψ0, ψ1 = get_m_legged_states(m, r, num_moments, code_type="squeeze", noise_type="loss", normalize_logical_states_before_applying_hadamard=NORMALIZE_LOGICAL_STATES_BEFORE_APPLYING_HADAMARD)
    ψ_plus, ψ_minus = get_m_legged_states(m, r, num_moments, code_type="squeeze", noise_type="dephasing", normalize_logical_states_before_applying_hadamard=NORMALIZE_LOGICAL_STATES_BEFORE_APPLYING_HADAMARD)
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
