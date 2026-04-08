#!/usr/bin/env python3
"""
Test script to verify the safe factorial functions work correctly.
"""
import numpy as np

# Use project-standard path setup
try:
    from tests._bootstrap import add_root_to_path
except ModuleNotFoundError:
    from _bootstrap import add_root_to_path
add_root_to_path()

from src.utils import maths

def test_safe_functions():
    print("Testing safe factorial functions...")
    
    # Test small numbers (should match old implementation)
    print("\n=== Testing small numbers ===")
    for n in [0, 1, 5, 10]:
        old_sqrt_fact = maths.sqrt_factorial(n)
        log_sqrt_fact = maths.log_sqrt_factorial(n)
        new_sqrt_fact = np.exp(log_sqrt_fact)
        
        print(f"n={n}: old={old_sqrt_fact:.6f}, new={new_sqrt_fact:.6f}, diff={abs(old_sqrt_fact - new_sqrt_fact):.2e}")
    
    # Test large numbers (where old would overflow)
    print("\n=== Testing large numbers ===")
    for n in [50, 100, 200, 500]:
        try:
            old_sqrt_fact = maths.sqrt_factorial(n)
            if np.isinf(old_sqrt_fact):
                print(f"n={n}: old=inf, log_sqrt_factorial={maths.log_sqrt_factorial(n):.2f}")
            else:
                print(f"n={n}: old={old_sqrt_fact:.2e}, log_sqrt_factorial={maths.log_sqrt_factorial(n):.2f}")
        except OverflowError:
            print(f"n={n}: old=OverflowError, log_sqrt_factorial={maths.log_sqrt_factorial(n):.2f}")
    
    # Test ratio function
    print("\n=== Testing safe ratio function ===")
    # Test case: sqrt(100!) / sqrt(50!) should equal sqrt(100!/50!)
    n1, n2 = 100, 50
    safe_ratio = maths.safe_sqrt_factorial_ratio(n1, n2)
    print(f"sqrt({n1}!) / sqrt({n2}!) = {safe_ratio:.6e}")
    
    # Test coherent coefficient function
    print("\n=== Testing safe coherent coefficient ===")
    alpha = 2.0 + 1j
    for n in [0, 5, 10, 50, 100]:
        coef = maths.safe_coherent_coefficient(alpha, n, normalize=True)
        print(f"Coherent coef (α={alpha}, n={n}): {coef:.6e}")

if __name__ == "__main__":
    test_safe_functions()
