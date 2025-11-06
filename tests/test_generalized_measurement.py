"""
Test script for the generalized qubit measurement function.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[3]))

from qutip import basis, tensor, qeye
import numpy as np

from projects.controlled_squeezing.src.squeezing_code import measure_qubit_state

def test_single_qubit_measurement():
    """Test measurement with 1 qubit (original case)."""
    print("Testing 1-qubit measurement:")
    
    # Create a simple state: vacuum ⊗ |+⟩ = vacuum ⊗ (|0⟩ + |1⟩)/√2
    num_moments = 10
    vacuum = basis(num_moments, 0)
    qubit_0 = basis(2, 0)
    qubit_1 = basis(2, 1)
    qubit_plus = (qubit_0 + qubit_1) / np.sqrt(2)
    
    test_state = tensor(vacuum, qubit_plus)
    
    # Measure with 1 qubit
    results = measure_qubit_state(test_state, num_qubits=1, return_full_stats=True)
    
    print(f"Number of measurement outcomes: {len(results)}")
    for i, result in enumerate(results):
        print(f"  Outcome {i}: qubit_measured={result['qubit_measured']}, prob={result['prob']:.3f}")
    
    assert len(results) == 2, f"Expected 2 outcomes, got {len(results)}"
    assert abs(results[0]['prob'] - 0.5) < 1e-6, f"Expected prob ~0.5, got {results[0]['prob']}"
    assert abs(results[1]['prob'] - 0.5) < 1e-6, f"Expected prob ~0.5, got {results[1]['prob']}"
    print("✓ 1-qubit test passed!")


def test_two_qubit_measurement():
    """Test measurement with 2 qubits."""
    print("\nTesting 2-qubit measurement:")
    
    # Create a state: vacuum ⊗ |++⟩ = vacuum ⊗ (|00⟩ + |01⟩ + |10⟩ + |11⟩)/2
    num_moments = 10
    vacuum = basis(num_moments, 0)
    qubit_0 = basis(2, 0)
    qubit_1 = basis(2, 1)
    qubit_plus = (qubit_0 + qubit_1) / np.sqrt(2)
    
    test_state = tensor(vacuum, qubit_plus, qubit_plus)
    
    # Measure with 2 qubits
    results = measure_qubit_state(test_state, num_qubits=2, return_full_stats=True)
    
    print(f"Number of measurement outcomes: {len(results)}")
    for i, result in enumerate(results):
        print(f"  Outcome {i}: qubit_measured={result['qubit_measured']}, prob={result['prob']:.3f}")
    
    assert len(results) == 4, f"Expected 4 outcomes, got {len(results)}"
    for result in results:
        assert abs(result['prob'] - 0.25) < 1e-6, f"Expected prob ~0.25, got {result['prob']}"
    print("✓ 2-qubit test passed!")


def test_three_qubit_measurement():
    """Test measurement with 3 qubits."""
    print("\nTesting 3-qubit measurement:")
    
    # Create a state: vacuum ⊗ |000⟩ (definite outcome)
    num_moments = 10
    vacuum = basis(num_moments, 0)
    qubit_0 = basis(2, 0)
    
    test_state = tensor(vacuum, qubit_0, qubit_0, qubit_0)
    
    # Measure with 3 qubits
    results = measure_qubit_state(test_state, num_qubits=3, return_full_stats=True)
    
    print(f"Number of measurement outcomes: {len(results)}")
    for i, result in enumerate(results):
        if result['prob'] > 1e-10:  # Only show non-zero probabilities
            print(f"  Outcome {i}: qubit_measured={result['qubit_measured']}, prob={result['prob']:.3f}")
    
    # Should have only one outcome with probability 1
    non_zero_results = [r for r in results if r['prob'] > 1e-10]
    assert len(non_zero_results) == 1, f"Expected 1 non-zero outcome, got {len(non_zero_results)}"
    assert non_zero_results[0]['qubit_measured'] == 0, f"Expected outcome 0 (|000⟩), got {non_zero_results[0]['qubit_measured']}"
    assert abs(non_zero_results[0]['prob'] - 1.0) < 1e-6, f"Expected prob ~1.0, got {non_zero_results[0]['prob']}"
    print("✓ 3-qubit test passed!")


if __name__ == "__main__":
    test_single_qubit_measurement()
    test_two_qubit_measurement()
    test_three_qubit_measurement()
    print("\n🎉 All tests passed! The generalized measurement function works correctly.")
