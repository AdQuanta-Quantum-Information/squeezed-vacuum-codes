from typing import TypeAlias, Literal, overload
import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import gc  # For garbage collection
import warnings

## For tests:
import time
import queue

# Import memory monitoring utilities
try:
    from .memory_monitor import check_memory_requirements, memory_monitor, get_memory_usage_gb
    MEMORY_MONITORING_AVAILABLE = True
except ImportError:
    # Fallback if memory_monitor is not available
    MEMORY_MONITORING_AVAILABLE = False
    def memory_monitor(_print:bool=False):
        return lambda func: func  # No-op decorator


BosonicNoiseType: TypeAlias = Literal['loss', 'dephasing']

DEFAULT_TIME_RES = 501


def robust_noise_simulation(
    ψ_in: qt.Qobj,
    noise_family: BosonicNoiseType,
    time_res: int = 1001,
    rate: float = 0.1,
    max_retries: int = 3,
    **kwargs
) -> list[qt.Qobj] | qt.Qobj:
    """
    Wrapper around noise_simulation that retries on memory failures.

    This helps handle intermittent memory allocation failures that can
    occur due to memory fragmentation, especially with large Hilbert spaces.

    Parameters:
    -----------
    max_retries : int
        Number of times to retry on MemoryError (default: 3)
    **kwargs
        Additional arguments passed to noise_simulation
    """

    for attempt in range(max_retries + 1):
        try:
            # Force garbage collection before each attempt
            gc.collect()
            
            if attempt > 0:
                print(f"Retry attempt {attempt}/{max_retries} for simulation...")
                # Wait a moment for system memory to stabilize
                import time
                time.sleep(1)
            
            return noise_simulation(
                ψ_in, 
                noise_family, 
                time_res=time_res, 
                rate=rate,
                _force_garbage_collection=True,
                **kwargs
            )

        except MemoryError as e:
            if attempt < max_retries:
                print(f"MemoryError on attempt {attempt + 1}, retrying... ({e})")
                # Aggressive cleanup
                gc.collect()
                continue
            else:
                # Final attempt failed
                print(f"All {max_retries + 1} attempts failed due to memory issues.")
                raise MemoryError(
                    f"Simulation failed after {max_retries + 1} attempts. "
                    f"Consider reducing parameters or freeing system memory."
                ) from e

    # This should never be reached due to the loop structure, but for type safety
    raise RuntimeError("Unexpected end of retry loop")


@overload
def noise_simulation(
    ψ_in:qt.Qobj,
    noise_family:BosonicNoiseType,
    rate: float,
    time_res:int|None=DEFAULT_TIME_RES,
    with_progress_bar : bool = True,
    *,
    output_intermediate_states: Literal[True] = True,
    _check_memory: bool = True,
    _force_garbage_collection: bool = False
) -> list[qt.Qobj]:...
@overload
def noise_simulation(
    ψ_in:qt.Qobj,
    noise_family:BosonicNoiseType,
    rate: float,
    time_res:int|None=DEFAULT_TIME_RES,
    with_progress_bar : bool = True,
    *,
    output_intermediate_states: Literal[False],
    _check_memory: bool = True,
    _force_garbage_collection: bool = False
) -> qt.Qobj: ...
def noise_simulation(
    ψ_in:qt.Qobj,
    noise_family:BosonicNoiseType,
    rate: float,
    time_res:int|None=DEFAULT_TIME_RES,
    with_progress_bar : bool = True,
    *,
    output_intermediate_states: bool = True,  # ← Add default here!
    _check_memory: bool = False,
    _force_garbage_collection: bool = False
) -> list[qt.Qobj]|qt.Qobj:

    ## Renaming and arranging:
    γ = rate
    if time_res is None:
        time_res = DEFAULT_TIME_RES
    times : list[float] = np.linspace(0.0, 1.0, time_res).tolist()

    ## If ψ_initial is a ket, form ρ₀ = |ψ⟩⟨ψ|
    if ψ_in.isoper:
        ρ_in = ψ_in  
    else:
        assert ψ_in.isket, "ψ_initial must be a ket or an operator."
        ρ_in = ψ_in.proj()

    ## Prepare operators:
    # Determine Hilbert-space dimension
    dims = ρ_in.dims[0]
    if isinstance(dims, list):
        N = dims[0]
    else:
        N = dims
    
    # Ensure N is an integer
    if not isinstance(N, int):
        raise TypeError(f"Could not determine Hilbert space dimension. Got: {type(N)}")
    
    # Check memory requirements before proceeding
    if _check_memory and MEMORY_MONITORING_AVAILABLE:
        report = check_memory_requirements(N, time_res, _print=False)
        if not report.is_feasible:
            raise MemoryError("Insufficient memory for simulation. Consider reducing Hilbert space dimension or time resolution."+report.msg)
    
    
    a = qt.destroy(N)
    n = a.dag() @ a

    match noise_family:
        case 'loss'|'photon_loss':
            c_ops = [np.sqrt(γ) * a]
        case 'dephasing':
            c_ops = [np.sqrt(γ) * n]
        case _:
            raise ValueError(f"Unsupported noise family: {noise_family!r}")
        
    ## options:
    options = dict(
        store_states=output_intermediate_states,
        store_final_state=True,
        progress_bar='tqdm' if with_progress_bar else None,
    )
    
    ## Run the simulation:
    H = qt.qzero(N)
    
    # Force one more garbage collection right before the memory-intensive operation
    if _force_garbage_collection:
        gc.collect()

    try:    
        results = qt.mesolve(H, ρ_in, times, c_ops=c_ops, options=options)
        
    except MemoryError as e:
        # Clean up and provide helpful error message
        gc.collect()
        raise MemoryError(
            f"Memory allocation failed during simulation with N={N}, time_res={time_res} "
            f"(estimated {report.required_mb:.2f} GB required). "
            f"Try: 1) Reduce time_res, 2) Use smaller N, 3) Run gc.collect() before simulation, "
            f"4) Close other applications to free memory."
        ) from e
    except Exception as e:
        # Clean up on any other error
        gc.collect()
        raise RuntimeError(f"Error during simulation: {e}") from e


    if _force_garbage_collection:
        gc.collect()

    if with_progress_bar:
        LineUp = '\033[1A'   # Move cursor up one line
        LineClear = '\x1b[2K'   # Clear the entire line        
        print(LineUp, end=LineClear)

    ## Unpack results:
    if output_intermediate_states:
        return results.states
    else:    
        final_state : qt.Qobj = results.final_state  #type: ignore
        return final_state


def cat_growth_decay(
    alpha_max: float = 2.0,
    t_pump: float = 10.0,
    t_decay: float = 15.0,
    kappa: float = 0.25,
    N: int = 60,
    num_frames: int = 200,
    xlimit: float = 3.0,
    outfile: str = "cat_growth_decay.gif",
    fps: int = 15
) -> FuncAnimation:
    """
    Generate and save an animated Wigner-function GIF illustrating:
      1) Growth of an even Schrödinger-cat state up to |alpha_max|
      2) Decoherence under photon loss at rate kappa

    Parameters:
    -----------
    alpha_max : float
        Target coherent amplitude for the cat state.
    t_pump : float
        Duration of the driving stage.
    t_decay : float
        Duration of the dissipation stage.
    kappa : float
        Photon-loss rate.
    N : int
        Dimension of the Fock basis.
    num_frames : int
        Total frames in the animation.
    xlimit : float
        Phase-space extent for the Wigner grid (±xlimit).
    outfile : str
        Filename for the output GIF.
    fps : int
        Frames per second for the saved GIF.

    Returns:
    --------
    anim : matplotlib.animation.FuncAnimation
        The animation object (also saved to `outfile`).
    """
    # 1) System operators
    a    = qt.destroy(N)
    adag = a.dag()

    # 2) Drive strength so α(t_pump)=alpha_max
    Omega = alpha_max / t_pump

    # 3) Time grid
    tlist = np.linspace(0, t_pump + t_decay, num_frames)

    # 4) Time‐dependent Hamiltonian: iΩ(t)(adag−a)
    def Ω_t(t, _):
        return Omega if t <= t_pump else 0.0
    H = [[1j * (adag - a), Ω_t]]

    # 5) Time‐dependent collapse: loss only after t_pump
    def κ_t(t, _):
        return 0.0 if t <= t_pump else kappa
    c_ops = [[a, κ_t]]

    # 6) Initial state: vacuum
    rho0 = qt.fock_dm(N, 0)

    # 7) Solve master equation
    result = qt.mesolve(H, rho0, tlist, c_ops=c_ops)

    # 8) Compute Wigner functions
    xvec = np.linspace(-xlimit, xlimit, 200)
    X, Y = np.meshgrid(xvec, xvec)
    wigners = [qt.wigner(rho, xvec, xvec) for rho in result.states]

    # 9) Set up animation
    fig, ax = plt.subplots(figsize=(4,4))
    def update(i):
        ax.clear()
        cf = ax.contourf(X, Y, wigners[i], 100)
        ax.set_title(f"t = {tlist[i]:.2f}")
        ax.set_xlabel("Re α")
        ax.set_ylabel("Im α")
        return cf.collections

    anim = FuncAnimation(fig, update, frames=len(wigners), interval=100)
    writer = PillowWriter(fps=fps)
    anim.save(outfile, writer=writer)
    plt.close(fig)

    print(f"Saved animation to {outfile}")
    return anim



def test_effect_of_time_resolution(
    ψ_in: qt.Qobj,
    rate: float = 1e-1,
    min_time_res: int =    500,
    max_time_res: int = 10_000,
    num_time_res: int = 50,
    noise_family: BosonicNoiseType = 'dephasing',
    _print: bool = True,
    _num_states_to_check: int = 5,
    _fidelity_threshold: float = 1 - 1e-3
) -> None:
    """
    Test how time resolution affects the final state of a noise simulation.
    We run the simulation for increasing time resolutions and see at what point we converge.
    convergence is determined by analyzing the fidelity between the 10 last states.

    This is useful to understand how much time resolution is needed for a given noise model
    and Hilbert space dimension.

    Parameters:
    -----------
    N : int
        Hilbert space dimension
    max_time_res : int
        Maximum time resolution (number of time steps)
    noise_family : BosonicNoiseType
        Type of noise to simulate
    rate : float
        Rate parameter for the noise
    _print : bool
        Whether to print progress messages

    Returns:
    --------
    None
    """
    ## Check inputs:
    assert max_time_res > min_time_res, "max_time_res must be greater than min_time_res"


    ## Basics:
    q = queue.Queue(maxsize=_num_states_to_check)
    time_res_vec = [ int(val) for val in  np.linspace(min_time_res, max_time_res, num_time_res)]
    kwargs = dict(
        keep_intermediate_states=False,
        with_progress_bar=False,
        _check_memory=False,
        _force_garbage_collection=False
    )

    max_time_str = len(str(max_time_res))

    def _get_mean_fidelity(states: list[qt.Qobj]) -> float:
        ## Avoid racing condition betwwen queue increase and check by
        # sleeping for a moment
        time.sleep(0.001)
        n = len(states)

        fidelities = [qt.fidelity(states[-i-2], states[-1]) for i in range(n-1)]
        return float(np.mean(fidelities))

    def _check_convergence(mean_fid:float, num_states:int):

        if num_states < _num_states_to_check: 
            return False

        qsize = q.qsize()  
        assert num_states == _num_states_to_check == qsize, f"Queue size mismatch: expected {_num_states_to_check}, got {qsize}"
        
        return mean_fid > _fidelity_threshold

    

    for time_res in time_res_vec:
            
        try:
            ψ_out = noise_simulation(ψ_in, noise_family, time_res=time_res, rate=rate, **kwargs)
        except MemoryError as e:
            print(f"MemoryError at time_res={time_res}: {e}")
            continue


            
        # Store the final state in the queue
        time.sleep(0.001)  # Small delay to avoid racing condition
        try:
            q.put_nowait(ψ_out)
        except queue.Full:
            q.get_nowait()  # Remove the oldest state if queue is full
            q.put_nowait(ψ_out)

        last_states = list(q.queue)
        fid = _get_mean_fidelity(last_states)



        if _print:
            time_str = f"{time_res:>{max_time_str}}"
            print(f"Running simulation with time_res = {time_str}; fidelity = {fid!r}")

        if _check_convergence(fid, len(last_states) ):
            if _print:
                print(f"Convergence reached at time_res = {time_str} with fidelity = {fid!r}")
            break
    else:

        if _print:
            print("No convergence reached within the specified time resolutions. crnt fidelity:", fid)





if __name__ == "__main__":
    N = 200
    ψ_in = qt.squeeze(N, 1.5) @ qt.fock(N, 2)  # Squeezed vacuum state
    test_effect_of_time_resolution(ψ_in)