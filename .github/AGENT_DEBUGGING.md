# Agent Debugging Patterns

## Common Issues & Fixes

### Import Timeout
**Symptom**: Script starts then hangs for 30+ seconds
**Cause**: Slow dependency chain (qiskit loads)
**Fix**: Use minimal imports or run in background with `isBackground=true`

### "Invalid operand types" with QuTiP
**Symptom**: `TypeError: invalid operand types` in expect() or @ operations
**Root causes**:
- Using wrong operator type (dag() returns adjoint, not operable with expect)
- Using wrong state shape (row vs column)
**Fix**: Use `(op @ state)[0,0]` for kets, `qt.expect(op, state)` for operators only

### PowerShell Parsing Errors at Line 1
**Cause**: Shell interpreted Python code as PowerShell syntax
**Fix**: Always use full path to python.exe, proper quoting on pipes

### Qiskit Import Failure
**Symptom**: `KeyboardInterrupt` during qiskit import chain
**Cause**: Qiskit loads slow and can get stuck with heavy imports
**Fix**: Import only `src.gkp` submodules, not full `src.gkp`

### GKP States Don't Match Test Expectations
**Check first**: Are you normalizing states? (.unit())
**Check second**: Are photon numbers supposed to match? (They don't for |0> vs |1>)
**Verify**: Run test_superposition_state.py TEST 4 to see ground truth

## Validation Checklist Before Running Tests

```python
# Minimal validation template
import qutip as qt
import numpy as np

# 1. QuTiP works
psi = qt.basis(5, 0)
assert psi.norm() == 1.0, "QuTiP basic state failed"

# 2. Inner products work
psi2 = qt.basis(5, 1)  
inner = (psi.dag() @ psi2)[0,0]
assert inner == 0, "Orthogonal basis should give 0"

# 3. Imports are available
try:
    from src.gkp import gkp_params_from_nbar
    from src.mean_photon_number import get_mean_photon_number
    print("✓ All imports ok")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    exit(1)

print("✓ Environment valid")
```

## Type System Reminders
- Logical values: `int | Literal['+']` (not Union)
- Mean photon return: always float
- Code types: `Literal['cat', 'squeeze', 'binomial', 'gkp']`
- States: qt.Qobj (QuTiP quantum object)

## Performance Notes
- N=24 is typical Fock truncation for nbar=2
- GKP state construction: ~1-2 sec per state at N=24
- Binary search for parameter finding: 5-15 iterations typical
- Binomial code: instant (direct formula)
