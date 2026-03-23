"""
Test to verify:
1. Current behavior: solving for |0> state (logical_value=0)
2. New behavior: solving for |+> state (logical_value=2)
   where |+> = 1/sqrt(2)(|0> + |1>)
"""

import numpy as np
from typing import Literal, Union

# Use project-standard path setup
try:
    from tests._bootstrap import add_root_to_path
except ModuleNotFoundError:
    from _bootstrap import add_root_to_path
add_root_to_path()

from src.mean_photon_number import (
    get_mean_photon_number,
    find_parameter_for_target_mean_photon_number
)

def test_verification():
    """Verify that |0> state still works as before."""
    print("=" * 70)
    print("TEST 1: VERIFICATION - Current behavior for |0> state (logical_value=0)")
    print("=" * 70)
    
    target_mean_photons = 2.5
    m = 2
    
    for code_type in ['cat', 'squeeze', 'binomial']:
        print(f"\nCode Type: {code_type}")
        
        # Find parameter for |0> state
        parameter_0 = find_parameter_for_target_mean_photon_number(
            code_type=code_type,
            m=m,
            logical_value=0,  # |0> state
            target_mean_photon_number=target_mean_photons,
        )
        
        # Verify
        computed_mean_photons_0 = get_mean_photon_number(
            code_type=code_type,
            m=m,
            logical_value=0,
            parameter=parameter_0,
            analytic_substitution=False
        )
        
        assert np.isclose(computed_mean_photons_0, target_mean_photons, atol=1e-6), \
            f"Failed: {computed_mean_photons_0} != {target_mean_photons}"
        
        print(f"  |0> state:  Parameter={parameter_0:.6f} => Mean Photons={computed_mean_photons_0:.6f}")
        print(f"  Target was: {target_mean_photons:.6f}")
        print(f"  ✓ Match")


def test_superposition_state():
    """Test the new |+> superposition state."""
    print("\n" + "=" * 70)
    print("TEST 2: NEW FEATURE - Superposition state |+> (logical_value='+')")
    print("=" * 70)
    print("|+> = 1/sqrt(2)(|0> + |1>)")
    
    m = 2
    parameter = 1.5  # Use a fixed parameter to compare all states
    
    print(f"\nUsing fixed parameter = {parameter}")
    print(f"Code Type: m = {m}\n")
    
    for code_type in ['cat', 'squeeze', 'binomial']:
        print(f"Code Type: {code_type}")
        
        # Get mean photon numbers for each state
        mean_0 = get_mean_photon_number(
            code_type=code_type,
            m=m,
            logical_value=0,
            parameter=parameter,
            analytic_substitution=False
        )
        
        mean_1 = get_mean_photon_number(
            code_type=code_type,
            m=m,
            logical_value=1,
            parameter=parameter,
            analytic_substitution=False
        )
        
        mean_plus = get_mean_photon_number(
            code_type=code_type,
            m=m,
            logical_value='+',  # |+> superposition
            parameter=parameter,
            analytic_substitution=False
        )
        
        # Verify that |+> = 0.5*|0> + 0.5*|1>
        expected_plus = 0.5 * mean_0 + 0.5 * mean_1
        
        assert np.isclose(mean_plus, expected_plus, atol=1e-10), \
            f"Failed: {mean_plus} != {expected_plus}"
        
        print(f"  <n>_|0> = {mean_0:.8f}")
        print(f"  <n>_|1> = {mean_1:.8f}")
        print(f"  <n>_|+> = {mean_plus:.8f}")
        print(f"  Expected: 0.5*<n>_|0> + 0.5*<n>_|1> = {expected_plus:.8f}")
        print(f"  ✓ Match")
        print()


def test_inverse_problem():
    """Test finding parameter for target mean photon number for |+> state."""
    print("=" * 70)
    print("TEST 3: INVERSE PROBLEM - Find parameter for |+> state")
    print("=" * 70)
    
    target_mean_photons = 2.0
    m = 2
    
    for code_type in ['cat', 'squeeze', 'binomial']:
        print(f"\nCode Type: {code_type}")
        print(f"Target mean photon number: {target_mean_photons}")
        
        # For |0> state
        parameter_0 = find_parameter_for_target_mean_photon_number(
            code_type=code_type,
            m=m,
            logical_value=0,
            target_mean_photon_number=target_mean_photons,
        )
        
        verify_0 = get_mean_photon_number(
            code_type=code_type,
            m=m,
            logical_value=0,
            parameter=parameter_0,
            analytic_substitution=False
        )
        
        # For |+> state
        parameter_plus = find_parameter_for_target_mean_photon_number(
            code_type=code_type,
            m=m,
            logical_value='+',  # |+> superposition
            target_mean_photon_number=target_mean_photons,
        )
        
        verify_plus = get_mean_photon_number(
            code_type=code_type,
            m=m,
            logical_value='+',  # |+> superposition
            parameter=parameter_plus,
            analytic_substitution=False
        )
        
        assert np.isclose(verify_plus, target_mean_photons, atol=1e-6), \
            f"Failed for |+>: {verify_plus} != {target_mean_photons}"
        
        print(f"  |0> state:  Parameter={parameter_0:.6f} => <n>={verify_0:.6f}")
        print(f"  |+> state:  Parameter={parameter_plus:.6f} => <n>={verify_plus:.6f}")
        print(f"  ✓ Both match target {target_mean_photons}")


def test_gkp_photon_numbers():
    """Test GKP code photon numbers across logical states by computing from actual states."""
    print("\n" + "=" * 70)
    print("TEST 4: GKP CODE - Mean photon numbers for different logical states")
    print("=" * 70)
    
    import qutip as qt
    from src.gkp import gkp_from_nbar, recommended_N_from_nbar
    
    nbar_target = 2.0  # mean photon number parameter
    
    print(f"\nGKP Code with target mean photon number nbar = {nbar_target}")
    
    # Choose appropriate Fock truncation
    N = recommended_N_from_nbar(nbar_target)
    print(f"Fock space truncation: N = {N}")
    
    # Build actual GKP states
    print("\nBuilding GKP logical states...")
    psi_0, meta_0 = gkp_from_nbar(logical_value=0, N=N, nbar_target=nbar_target, return_meta=True, _prog_bar=False)
    psi_1, meta_1 = gkp_from_nbar(logical_value=1, N=N, nbar_target=nbar_target, return_meta=True, _prog_bar=False)
    
    # Normalize states
    psi_0 = psi_0.unit()
    psi_1 = psi_1.unit()
    
    # Compute mean photon numbers from actual states
    n_op = qt.num(N)  # photon number operator
    mean_0 = qt.expect(n_op, psi_0)
    mean_1 = qt.expect(n_op, psi_1)
    
    # Compute superposition state |+> = 1/sqrt(2)(|0> + |1>)
    psi_plus = (psi_0 + psi_1).unit()
    mean_plus = qt.expect(n_op, psi_plus)
    
    print(f"\n  <n>_|0> = {mean_0:.8f}")
    print(f"  <n>_|1> = {mean_1:.8f}")
    print(f"  <n>_|+> = {mean_plus:.8f}")
    
    # Check for differences
    diff_01 = abs(mean_1 - mean_0)
    print(f"\n  Difference |<n>_|1> - <n>_|0>| = {diff_01:.2e}")
    
    # Check if superposition matches average
    expected_plus = 0.5 * mean_0 + 0.5 * mean_1
    diff_plus = abs(mean_plus - expected_plus)
    print(f"  Difference |<n>_|+> - 0.5*(<n>_|0> + <n>_|1>)| = {diff_plus:.2e}")
    
    print("\n  FINDINGS:")
    if diff_01 < 1e-6:
        print("  ✓ GKP logical states have IDENTICAL mean photon numbers (within tolerance)")
        print("    => GKP encoding is photon-number-independent (as expected)")
    else:
        print(f"  ⚠ DIFFERENCE DETECTED: |0> and |1> differ by {diff_01:.2e}")
        print(f"    => GKP states have different photon number distributions")
        print(f"    => May indicate even/odd well structure affects mean photon number")
    
    if diff_plus < 1e-6:
        print("  ✓ Superposition state correctly averages |0> and |1>")
    else:
        print(f"  ⚠ Superposition deviation: {diff_plus:.2e}")


def analyze_and_recommend():
    """Analyze results and provide recommendations based on all test data."""
    print("\n" + "=" * 70)
    print("ANALYSIS & RECOMMENDATIONS")
    print("=" * 70)
    
    print("""
For Cat and Squeeze Codes:
  ✓ Superposition state |+> correctly computes as average of |0> and |1>
  
For GKP Code:
  The test will reveal whether the simplified model (identical photon numbers)
  is correct or whether GKP states have state-dependent photon distributions.
  
  If differences are found, next steps should include:
  1. Theoretical analysis of GKP lattice structure
  2. Investigation of how even/odd well support affects mean photon number
  3. Development of state-dependent GKP photon number model
  4. Empirical verification with GKP state preparations
    """)


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("TESTING MEAN PHOTON NUMBER FOR SUPERPOSITION STATES")
    print("=" * 70)
    
    test_verification()
    test_superposition_state()
    test_inverse_problem()
    test_gkp_photon_numbers()
    analyze_and_recommend()
    
    print("\n" + "=" * 70)
    print("✓ All tests completed successfully!")
    print("=" * 70)
