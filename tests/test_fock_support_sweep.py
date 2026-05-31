"""Sweep test: verify Fock-support and mean-photon-number checks pass
for many (m, k) combinations on both squeeze and cat code types.
"""
try:
    from tests._bootstrap import add_root_to_path
except ModuleNotFoundError:
    from _bootstrap import add_root_to_path

add_root_to_path()

import pytest
from scripts.study.test_fock_support import test_fock_support as _run_fock_support_check


# ── Parameter sets ──────────────────────────────────────────────────────────

_CASES = (
    # (code_type, m, k, s, num_moments)
    *[("squeeze", m, k, 1.0, 100) for m in [2, 4, 6, 8] for k in range(m)],
    *[("cat",     m, k, 1.5, 150) for m in [2, 3, 4, 6] for k in range(m)],
)


# ── Combined sweep ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("code_type,m,k,s,num_moments", _CASES)
def test_fock_support_sweep(code_type: str, m: int, k: int, s: float, num_moments: int) -> None:
    passed = _run_fock_support_check(
        code_type=code_type,  # type: ignore[arg-type]
        m=m, k=k, s=s,
        num_moments=num_moments,
        analytical_moments=15,
        verbose=False,
    )
    assert passed, f"Fock-support checks failed for {code_type} m={m}, k={k}"


if __name__ == "__main__":
    for code_type, m, k, s, num_moments in _CASES:
        test_fock_support_sweep(code_type, m, k, s, num_moments)
