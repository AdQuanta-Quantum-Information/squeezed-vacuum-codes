# GKP Physics Knowledge Base

## For Agents: What We Know Empirically

### GKP Mean Photon Numbers (nbar=2.0, N=24)
```
|0⟩ logical state: ⟨n⟩ ≈ 1.872
|1⟩ logical state: ⟨n⟩ ≈ 2.251
|+⟩ superposition: ⟨n⟩ ≈ 1.555

Key: NOT equal. NOT averaging. This is real physics.
```

### GKP Non-Orthogonality
```
Finite-energy GKP states are NOT orthogonal:
|⟨0_L|1_L⟩| ≈ 0.1297 (significant overlap)

This causes the "loss floor" phenomenon in Fig. 7.
```

### Why |0⟩ ≠ |1⟩ in Mean Photons
- Lattice wells offset: |0⟩ at q=2πn, |1⟩ at q=π+2πn
- Finite envelope (Delta, kappa) interacts differently
- q_s = (2s + logical_value)√π creates asymmetry in phase space
- Result: Different average displacement → different ⟨n⟩

### Fix for Benchmarking Issue (Fig. 7)
Current problem: Loss curve is flat ~0.13 instead of behaving like dephasing
Root cause: Using non-orthogonal GKP states with overlap metric
Solution options (ranked):
1. **Gram-Schmidt orthogonalize before metrics**
2. **Subtract zero-noise floor**: ΔV(γ) = V(γ) - V(0)**
3. **Use full state computation** (slower but exact)

### What's NOT Implemented Yet
- State-dependent GKP photon numbers (currently all return nbar)
- GKP superposition state photon numbers (raises NotImplementedError)
- Orthogonalization in `get_m_legged_states()` (only applies pre-Hadamard normalization)

## For If You Need To Extend

### Adding State-Dependent GKP Photons
Current: `mean_photon_number_for_gkp_codeword()` ignores logical_value
To fix:
```python
# Empirical correction factors from nbar=2 calibration
if logical_value == 0:
    return nbar * 0.936  # scaled factor
elif logical_value == 1:
    return nbar * 1.126
else:  # '+' 
    # Need full state computation or empirical table
    raise NotImplementedError()
```

### Adding GKP Superposition Photons
Options:
- **Easy**: Create lookup table from `gkp_from_nbar()` sweeps
- **Hard**: Compute from actual superposition interference
- **Recommended**: Full state computation (Option C from analysis)

## Recent Analysis Insights
- Reviewer concern about Fig. 7 is valid but diagnosed wrong
- Problem is metrics/orthogonality, not numerical cutoff
- GKP IS more robust to loss than dephasing (physically correct)
- Current implementation just doesn't reflect this in benchmarking
