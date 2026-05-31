"""Verify Fock-space support of |ψ_k⟩ matches the paper.

Paper claim (Eq. 6 / S(23)):  |ψ_k⟩ has nonzero amplitudes only at
    n = 2(ℓm + k)  for ℓ = 0, 1, 2, …
i.e. n ≡ 2k  (mod 2m).

The sign-convention (e^{-i2πjk/m} instead of +i) is what
makes this true for general k. We test several (m, k) pairs.
"""

try:
    from scripts.study._bootstrap import add_root_to_path
except ModuleNotFoundError:
    from _bootstrap import add_root_to_path

add_root_to_path()

from typing import NamedTuple

import numpy as np
from qutip import Qobj
from src.codes_built_in_superposition import simple_m_legged_state, _CodeTypes
from src.analytical_expressions import (
    squeezed_superposition_state,
    analytic_fock_representation_iterate_fock_nums_and_coeffs,
    _check_qutip_and_analytical_states_are_equal,
)
from src.mean_photon_number import (
    qutip_mean_photon_number,
    mean_photon_number_for_squeezed_codeword,
    _analytic_mean_photon_number_for_squeezed_k_state,
    _analytic_mean_photon_number_for_cat_k_state,
    _numerical_exact_summation_mean_photon_number_for_cat_codeword,
    r_symbol,
    alpha_symbol,
    DEFAULT_L_CUT_OFF,
)
from src.quantum.qutip_support._common import print_fock


class _FockSupportParams(NamedTuple):
    modulus: int
    residue: int
    ns_preview: list[int]


class _SupportCheckResult(NamedTuple):
    nonzero_ns: list[int]
    violations: list[int]
    expected_residue: int


# ── Global Parameters ───────────────────────────────────────────────────────
THRESHOLD   = 1e-10  # probability below which a component is considered zero


def get_fock_support_params(
    m: int, k: int, code_type: _CodeTypes, preview_terms: int = 5
) -> _FockSupportParams:
    """Return the expected Fock-support modulus, residue, and a short preview list.

    squeeze:  n = 2(ℓm + k)  →  modulus=2m, residue=2k mod 2m
    cat:      n = ℓm + k     →  modulus=m,  residue=k mod m
    """
    if code_type == "squeeze":
        modulus = 2 * m
        residue = (2 * k) % modulus
        ns_preview = [2 * (l * m + k) for l in range(preview_terms)]
    else:  # cat
        modulus = m
        residue = k % modulus
        ns_preview = [l * m + k for l in range(preview_terms)]
    return _FockSupportParams(modulus, residue, ns_preview)


def check_support(
    state: Qobj, params: _FockSupportParams
) -> _SupportCheckResult:
    """Verify every nonzero Fock component of *state* satisfies params.residue (mod params.modulus)."""
    probs = np.abs(state.full()[:, 0]) ** 2
    nonzero_ns = [n for n, p in enumerate(probs) if p > THRESHOLD]
    violations  = [n for n in nonzero_ns if n % params.modulus != params.residue]

    return _SupportCheckResult(nonzero_ns, violations, params.residue)


def test_fock_support(
    code_type: _CodeTypes = "squeeze",
    m: int = 8,
    k: int = 3,
    s: float = 1.0,
    num_moments: int = 100,
    analytical_moments: int = 20,  # Fock terms in the analytical sum (squeeze only)
    verbose: bool = True,
) -> bool:
    """Combined test: numerical Fock support, analytical derivation, and mean photon number.

    For squeeze codes, all three checks run. For other code types only the
    numerical Fock support is verified (analytical derivation not applicable).

    Returns True if all checks pass, False if any check fails.
    """
    passed = True
    vprint = print if verbose else lambda *args, **kwargs: None

    params = get_fock_support_params(m, k, code_type)

    vprint(f"\n{'='*60}")
    vprint(f"m={m}, k={k}, code={code_type}   →   expect n ≡ {params.residue}  (mod {params.modulus})")
    vprint(f"  Predicts support at n = {', '.join(str(n) for n in params.ns_preview)}, …")

    # Build the numerical state once; for squeeze, ensure it covers all analytical terms too.
    if code_type == "squeeze":
        max_analytical_fock = 2 * ((analytical_moments - 1) * m + k)
        qutip_cutoff = max(num_moments, max_analytical_fock + 1)
    else:
        qutip_cutoff = num_moments

    state = simple_m_legged_state(
        m=m, s=s, num_moments=qutip_cutoff,
        code_type=code_type,
        qubit_logical_value=k, num_qudit_values=m,
        _prog_bar=False,
    )

    vprint("  Fock representation (leading terms):")
    vprint_fock = (lambda *a, **kw: print_fock(*a, **kw)) if verbose else lambda *a, **kw: None
    vprint_fock(state, prefix="    |ψ_k⟩ = ", max_terms=8)

    # ── 1. Numerical Fock support ───────────────────────────────────────────
    result = check_support(state, params)
    vprint(f"  Non-zero Fock indices: {result.nonzero_ns[:10]} …")
    if result.violations:
        vprint(f"  FAIL ✗  [numerical support] unexpected indices: {result.violations}")
        passed = False
    else:
        vprint(f"  PASS ✓  [numerical support] all {len(result.nonzero_ns)} nonzero terms satisfy n≡{result.expected_residue} (mod {params.modulus})")

    if code_type == "squeeze":
        # ── 2. Analytical Fock support ─────────────────────────────────────
        vprint(f"\n  Building analytical state ({analytical_moments} Fock terms) …")
        analytical = squeezed_superposition_state(m_val=m, r_val=s, k_val=k, num_moments=analytical_moments)
        analytical_fock_ns = [fock for fock, _ in analytic_fock_representation_iterate_fock_nums_and_coeffs(analytical)]
        analytical_violations = [n for n in analytical_fock_ns if n % params.modulus != params.residue]
        if analytical_violations:
            vprint(f"  FAIL ✗  [analytical support] unexpected indices: {analytical_violations}")
            passed = False
        else:
            vprint(f"  PASS ✓  [analytical support] all {len(analytical_fock_ns)} terms satisfy n≡{params.residue} (mod {params.modulus})")

        # ── 3. Analytical vs numerical coefficient comparison ──────────────
        vprint("  Comparing analytical vs numerical Fock coefficients …")
        try:
            _check_qutip_and_analytical_states_are_equal(analytical, state)
            vprint(f"  PASS ✓  [coeff match] analytical and numerical states agree")
        except Exception as e:
            vprint(f"  FAIL ✗  [coeff match] {e}")
            passed = False

    # ── 4. Mean photon number ─────────────────────────────────────────────
    def _check_nbar(label: str, nbar: float) -> bool:
        nonlocal passed
        rel_err = abs(qutip_nbar - nbar) / max(abs(nbar), 1e-15)
        ok = rel_err <= 1e-3
        if not ok:
            passed = False
        status = "PASS ✓" if ok else "FAIL ✗"
        vprint(f"  {status}  [nbar {label}] qutip={qutip_nbar:.6f} ≈ analytic={nbar:.6f} (rel_err={rel_err:.2e})")
        return ok

    if code_type in ("squeeze", "cat"):
        qutip_nbar = qutip_mean_photon_number(state)

        if code_type == "squeeze":
            sympy_nbar = float(
                _analytic_mean_photon_number_for_squeezed_k_state(m, k, L_threshold=DEFAULT_L_CUT_OFF)
                .subs({r_symbol: s}).evalf().doit()
            )
            numerical_nbar = mean_photon_number_for_squeezed_codeword(m, s, logical_value=0, _k=k, analytic_substitution=False)
            _check_nbar("sympy formula", sympy_nbar)
            _check_nbar("numerical summation", numerical_nbar)

        elif code_type == "cat":
            import sympy as _sp
            sympy_nbar = float(_sp.re(
                _analytic_mean_photon_number_for_cat_k_state(m, k)
                .subs({alpha_symbol: s}).evalf().doit()
            ))  #type: ignore
            numerical_nbar = _numerical_exact_summation_mean_photon_number_for_cat_codeword(m, s, k)
            _check_nbar("sympy formula", sympy_nbar)
            _check_nbar("numerical summation", numerical_nbar)

    vprint(f"\n{'='*60}")
    return passed


def main() -> None:
    test_fock_support()

if __name__ == "__main__":
    main()
