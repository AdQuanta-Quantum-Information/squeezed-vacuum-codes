# Agent Quick Reference

## Environment
- **OS**: Windows (PowerShell, not bash)
- **Python**: 3.11.9 in venv at `.venv/`
- **Python path**: `c:/Users/NGBig/git_repos/squeezed-vacuum-codes/.venv/Scripts/python.exe`
- **Slow imports**: gkp.py → visualizations.py → measurements.py → qiskit (takes 15+ sec)
- **Fast imports**: Direct submodules ok (gkp_params_from_nbar, gkp_logical, recommended_N_from_nbar)

## Key Module APIs (Actually Used)
- **QuTiP inner product**: `(psi_0.dag() @ psi_1)[0,0]` returns complex scalar
- **QuTiP photon operator**: `qt.num(N)` then `qt.expect(n_op, psi)` for ⟨n⟩
- **State normalization**: `psi.unit()` (in-place with `inplace=True`)
- **Ket construction**: `qt.basis(N, n)` for |n⟩ state
- **GKP building**: 
  - `gkp_params_from_nbar(nbar)` → (Delta, kappa, r, s_max)
  - `gkp_logical(logical_val, N, Delta, kappa, s_max)` → unnormalized Qobj
  - Then `.unit()` to normalize

## Code Structure
- **Main physics**: `src/mean_photon_number.py`
- **GKP states**: `src/gkp.py`
- **Code classes**: `src/codes_built_in_superposition.py` 
- **Tests**: `tests/test_superposition_state.py`
- **Types**: Use `int | Literal['+']` (PEP 604), not Union

## Known Quirks
- GKP |0> and |1> states have **different mean photon numbers** (not equal)
  - |0>: ~1.87 photons, |1>: ~2.25 photons at nbar=2
  - This is real lattice asymmetry, not a bug
- GKP superposition |+> doesn't average linearly (quantum interference)
- Binomial code is linear: `r = 2*target_mean/m` (direct solution, not binary search)
- `mean_photon_number_for_gkp_codeword()` currently raises NotImplementedError for '+' 

## Windows PowerShell Gotchas
- `head` doesn't exist → use `Select-Object -First N`
- No `<<EOF` heredocs → use `create_file` tool instead
- Pipe with proper quoting: `"string" | command`
- Reserved: `<`, `>`, `|` require context

## Testing Ground Truth
Run: `c:/Users/NGBig/git_repos/squeezed-vacuum-codes/.venv/Scripts/python.exe tests/test_superposition_state.py`
- TEST 1-3 always pass (cat, squeeze, binomial superposition)
- TEST 4 shows actual GKP lattice behavior (empirical from real states)

## If You're Stuck
1. Check error location (line 1 = shell error, not code)
2. Test imports separately before using them
3. For QuTiP: check examples in `codes_built_in_superposition.py`
4. GKP states take ~5-10 sec to build (N=24, s_max varies)
5. Ask: is this a shell issue, import issue, or API issue?
