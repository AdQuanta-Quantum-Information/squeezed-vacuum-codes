"""
Memory monitoring utilities for quantum simulations.
"""

import psutil
import numpy as np
from typing import Optional, Callable, TypeVar, ParamSpec
import functools
import warnings
from dataclasses import dataclass, field

# Type variables for proper generic typing
P = ParamSpec('P')
R = TypeVar('R')


def get_available_memory_gb() -> float:
    """Get available system memory in GB."""
    return psutil.virtual_memory().available / (1024**3)


def get_memory_usage_gb() -> float:
    """Get current process memory usage in GB."""
    process = psutil.Process()
    return process.memory_info().rss / (1024**3)


def estimate_qobj_memory_mb(hilber_space_dim: int, num_states: int = 1) -> float:
    """
    Estimate memory usage for quantum objects.
    
    Parameters:
    -----------
    N : int
        Hilbert space dimension
    num_states : int
        Number of quantum states to store
        
    Returns:
    --------
    float
        Estimated memory in MB
    """
    # Each complex number: 16 bytes (8 for real + 8 for imaginary)
    # Density matrix: N x N complex numbers
    # Plus overhead for QuTip objects
    matrix_bytes = hilber_space_dim * hilber_space_dim * 16
    overhead_factor = 1.5  # QuTip overhead
    total_bytes = matrix_bytes * overhead_factor * num_states
    return total_bytes / (1024**2)

@dataclass(slots=True)
class MemoryReport:
    required_mb: float = field(default=None)
    available_mb: float = field(default=None)
    is_feasible: bool = field(default=None)
    msg: str = field(default=None)


def check_memory_requirements(
    N: int, num_states: int = 1000, 
    safety_factor: float = 0.9,
    _print: bool = True
) -> MemoryReport:
    """
    Check if there's enough memory for a quantum simulation.
    
    Parameters:
    -----------
    N : int
        Hilbert space dimension
    num_states : int
        Number of time steps (states to store)
    safety_factor : float
        Fraction of available memory to use (default: 80%)
        
    Returns:
    --------
    bool
        True if simulation is feasible, False otherwise
    """
    report = MemoryReport()
    report.required_mb = estimate_qobj_memory_mb(N, num_states)
    available_gb = get_available_memory_gb()
    available_mb = available_gb * 1024 * safety_factor
    report.available_mb = available_mb
    report.is_feasible = report.required_mb < report.available_mb
    
    msg = ""
    if report.is_feasible:
        msg += f"Memory check PASSED:"
        msg += f"  Required:  {report.required_mb:.1f} MB"
        msg += f"  Available: {report.available_mb:.1f} MB"

    else:
        msg += f"Memory check FAILED:"
        msg += f"  Required:  {report.required_mb:.1f} MB"
        msg += f"  Available: {report.available_mb:.1f} MB (with {safety_factor*100:.0f}% safety margin)"
        msg += f"  Hilbert dimension: {N}"
        msg += f"  Number of states: {num_states}"

    if _print:
        print(msg)
            
    report.msg = msg
    return report
    



def memory_monitor(_print: bool = True) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Decorator to monitor (and print) memory usage of a function.
    """
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)

        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            # Memory before
            mem_before = get_memory_usage_gb()
            if _print:
                print(f"Memory before {func.__name__}: {mem_before:.2f} GB")
            
            try:
                result = func(*args, **kwargs)
                
            except MemoryError as e:
                mem_error = get_memory_usage_gb()
                if _print:
                    print(f"MemoryError in {func.__name__} at {mem_error:.2f} GB")
                raise e
            
            finally:
                # Memory after
                mem_after = get_memory_usage_gb()
                mem_delta = mem_after - mem_before
                if _print:
                    print(f"Memory after {func.__name__}: {mem_after:.2f} GB (Δ={mem_delta:+.2f} GB)")
                
            return result
        return wrapper
    return decorator


def suggest_optimization(N: int, num_states: int) -> None:
    """
    Suggest optimizations for large quantum simulations.
    """
    required_mb = estimate_qobj_memory_mb(N, num_states)
    available_gb = get_available_memory_gb()
    
    print("\n=== Optimization Suggestions ===")
    
    if required_mb > available_gb * 1024:
        print("❌ Simulation not feasible with current parameters")
        
        # Suggest reducing Hilbert space
        max_N = int(np.sqrt(available_gb * 1024 * 1024 * 16 / num_states / 1.5))
        print(f"💡 Try reducing Hilbert space dimension to N ≤ {max_N}")
        
        # Suggest reducing time resolution
        max_states = int(available_gb * 1024 * 1024 * 16 / (N * N * 1.5))
        print(f"💡 Or reduce time resolution to ≤ {max_states} steps")
        
        # Suggest chunking
        chunk_size = max(1, max_states // 10)
        print(f"💡 Consider processing in chunks of {chunk_size} time steps")
        
    else:
        print("✅ Simulation should be feasible")
        
        # Suggest memory-efficient options
        if required_mb > 100:  # > 100MB
            print("💡 Consider using sparse matrices if applicable")
            print("💡 Use store_states=False if you don't need all intermediate states")
            print("💡 Process results incrementally rather than storing all states")


if __name__ == "__main__":
    # Example usage
    N = 250  # The dimension that caused your error
    num_states = 1001
    
    print("=== Memory Analysis ===")
    print(f"System memory: {psutil.virtual_memory().total / (1024**3):.1f} GB total")
    print(f"Available memory: {get_available_memory_gb():.1f} GB")
    print(f"Current usage: {get_memory_usage_gb():.2f} GB")
    print()
    
    # Check if simulation is feasible
    feasible = check_memory_requirements(N, num_states)
    
    if not feasible:
        suggest_optimization(N, num_states)
