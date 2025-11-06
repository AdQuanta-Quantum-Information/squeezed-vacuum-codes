"""
Simple memory estimation utilities that don't require psutil.
"""

import numpy as np
from typing import Optional
import warnings


def estimate_qobj_memory_mb(N: int, num_states: int = 1) -> float:
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
    matrix_bytes = N * N * 16
    overhead_factor = 1.5  # QuTip overhead
    total_bytes = matrix_bytes * overhead_factor * num_states
    return total_bytes / (1024**2)


def check_memory_requirements_simple(N: int, num_states: int = 1000, 
                                    max_memory_gb: float = 4.0) -> bool:
    """
    Simple memory check without system introspection.
    
    Parameters:
    -----------
    N : int
        Hilbert space dimension
    num_states : int
        Number of time steps (states to store)
    max_memory_gb : float
        Maximum memory to allow (default: 4GB)
        
    Returns:
    --------
    bool
        True if simulation should be feasible, False otherwise
    """
    required_mb = estimate_qobj_memory_mb(N, num_states)
    max_memory_mb = max_memory_gb * 1024
    
    if required_mb > max_memory_mb:
        print(f"Memory check FAILED:")
        print(f"  Required: {required_mb:.1f} MB")
        print(f"  Maximum allowed: {max_memory_mb:.1f} MB")
        print(f"  Hilbert dimension: {N}")
        print(f"  Number of states: {num_states}")
        return False
    else:
        print(f"Memory check PASSED:")
        print(f"  Required: {required_mb:.1f} MB")
        print(f"  Maximum allowed: {max_memory_mb:.1f} MB")
        return True


def suggest_optimization_simple(N: int, num_states: int, max_memory_gb: float = 4.0) -> None:
    """
    Suggest optimizations for large quantum simulations.
    """
    required_mb = estimate_qobj_memory_mb(N, num_states)
    max_memory_mb = max_memory_gb * 1024
    
    print("\n=== Optimization Suggestions ===")
    
    if required_mb > max_memory_mb:
        print("❌ Simulation not feasible with current parameters")
        
        # Suggest reducing Hilbert space
        max_N = int(np.sqrt(max_memory_mb * 1024 * 16 / num_states / 1.5))
        print(f"💡 Try reducing Hilbert space dimension to N ≤ {max_N}")
        
        # Suggest reducing time resolution
        max_states = int(max_memory_mb * 1024 * 16 / (N * N * 1.5))
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
    # Example usage - Test the problem case
    N = 250  # The dimension that caused your error
    num_states = 1001
    
    print("=== Simple Memory Analysis ===")
    print(f"Testing N={N}, time_steps={num_states}")
    print(f"Estimated memory: {estimate_qobj_memory_mb(N, num_states):.1f} MB")
    print()
    
    # Check if simulation is feasible
    feasible = check_memory_requirements_simple(N, num_states, max_memory_gb=8.0)
    
    if not feasible:
        suggest_optimization_simple(N, num_states, max_memory_gb=8.0)
