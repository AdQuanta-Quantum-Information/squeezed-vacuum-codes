#!/usr/bin/env python3
"""
Test the correctness of the second-order Kraus operators for the loss channel.
"""

import numpy as np
from qutip import destroy, create, qeye

def test_kraus_completeness():
    """Test if the Kraus operators satisfy the completeness condition."""
    dim = 10
    gamma = 0.1
    
    a = destroy(dim)
    adag = create(dim)
    n_op = adag @ a
    I = qeye(dim)
    
    # Current implementation
    K0 = (I - 0.5 * gamma * n_op + gamma**2 * (0.125 * n_op @ n_op - 0.25 * n_op))
    K1 = (gamma**0.5) * a
    K2 = (gamma / 2.0**0.5) * (a @ a)
    
    # Check completeness: sum(K_i^† K_i) = I
    sum_kraus = K0.dag() @ K0 + K1.dag() @ K1 + K2.dag() @ K2
    
    print('Testing Kraus operator completeness:')
    print('Difference from identity (max element):')
    max_diff = np.max(np.abs((sum_kraus - I).full()))
    print(f'{max_diff:.2e}')
    
    # Check if this is acceptable (should be O(gamma^3))
    print(f'Expected O(gamma^3) = {gamma**3:.2e}')
    print(f'Ratio: {max_diff / gamma**3:.2f}')
    
    return max_diff

def analyze_k0_structure():
    """Analyze the structure of K0 operator."""
    dim = 10
    gamma = 0.1
    
    a = destroy(dim)
    adag = create(dim)
    n_op = adag @ a
    I = qeye(dim)
    
    # Current K0
    K0 = (I - 0.5 * gamma * n_op + gamma**2 * (0.125 * n_op @ n_op - 0.25 * n_op))
    
    # What it should be for second-order loss channel
    # The loss channel evolution operator is exp(-gamma * n / 2)
    # Second-order expansion: 1 - gamma*n/2 + gamma^2*n^2/8 - gamma^2*n/4 + O(gamma^3)
    # Simplifying: 1 - gamma*n/2 + gamma^2*(n^2/8 - n/4) + O(gamma^3)
    # = 1 - gamma*n/2 + gamma^2*n*(n-2)/8 + O(gamma^3)
    
    print('\nAnalyzing K0 structure:')
    print('Current K0 diagonal elements:')
    k0_diag = np.real(np.diag(K0.full()))
    print('n=0:', k0_diag[0])
    print('n=1:', k0_diag[1])
    print('n=2:', k0_diag[2])
    print('n=3:', k0_diag[3])
    
    # Expected values from exact second-order expansion of exp(-gamma*n/2)
    expected = []
    for n in range(4):
        val = 1 - gamma*n/2 + gamma**2*n**2/8
        expected.append(val)
    
    print('\nExpected from exp(-gamma*n/2) expansion:')
    for i, val in enumerate(expected):
        print(f'n={i}: {val:.6f}')
    
    print('\nDifferences:')
    for i in range(4):
        diff = k0_diag[i] - expected[i]
        print(f'n={i}: {diff:.2e}')

def check_second_order_coefficients():
    """Check if the gamma^2 coefficients are correct."""
    print('\n=== Checking second-order coefficients ===')
    
    # For the loss channel, the exact evolution is:
    # rho -> sum_k K_k rho K_k^†
    # where the superoperator is exp(-gamma * (n + 1/2))
    # But for pure loss, we have exp(-gamma * n)
    
    # The second-order expansion of exp(-gamma * n) is:
    # 1 - gamma*n + gamma^2*n^2/2 - gamma^3*n^3/6 + ...
    
    # But the current implementation seems to use:
    # K0 = 1 - gamma*n/2 + gamma^2*(n^2/8 - n/4)
    #    = 1 - gamma*n/2 + gamma^2*n*(n-2)/8
    
    print('Current gamma^2 coefficient for n^2 term: 1/8 = 0.125')
    print('Current gamma^2 coefficient for n term: -1/4 = -0.25')
    print('Combined: gamma^2 * (n^2/8 - n/4) = gamma^2 * n*(n-2)/8')
    
    # This looks like it might be coming from a different expansion
    # Let's check what happens if we expand (1 - gamma/2 * n)^2
    print('\nIf we expand (1 - gamma*n/2)^2:')
    print('= 1 - gamma*n + gamma^2*n^2/4')
    print('So the correction would be: -gamma^2*n^2/4 to get the right second order')
    
    # The current implementation has: gamma^2*(n^2/8 - n/4)
    # = gamma^2*n^2/8 - gamma^2*n/4
    
    print('\nCurrent implementation contributes:')
    print('gamma^2*n^2/8 - gamma^2*n/4')
    print('This does not match the expected -gamma^2*n^2/4 correction')

if __name__ == "__main__":
    test_kraus_completeness()
    analyze_k0_structure()
    check_second_order_coefficients()
