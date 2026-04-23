# _agents_dump/_cover_slabs_draft.py
"""Draft floating-slabs cover — low resolution for fast iteration."""
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parents[1]))
    from scripts import add_root_to_path
    add_root_to_path()

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from PIL import Image
from scipy.ndimage import gaussian_filter, binary_dilation
from pathlib import Path

import qutip
from src.codes_built_in_superposition import simple_m_legged_code

# ── Constants ──────────────────────────────────────────────────────────────────
STRENGTH    = 1.5
NUM_MOMENTS = 300
CODE_TYPE   = "squeeze"
BG_COLOR_HEX = "#0d1b2a"
BG_COLOR_RGB = (13, 27, 42)        # BG_COLOR_HEX as uint8 tuple
COLORLIMS   = (-0.20, 0.23)

# Draft resolution — change to 400 / 1024 for final
WIGNER_POINTS = 80
SLAB_SIZE     = 256

OUTPUT_DIR  = Path(__file__).parent

# Per-state: (alpha_max, slab_scale, cx_frac, cy_frac)
LAYOUT = {
    1: (3.0, 0.28, 0.22, 0.22),
    2: (4.5, 0.38, 0.36, 0.36),
    4: (5.5, 0.50, 0.52, 0.52),
    8: (7.0, 0.65, 0.70, 0.68),
}


def _compute_states() -> dict[int, qutip.Qobj]:
    """Generate the four code states. Returns {m: state}."""
    print("Computing m=1 |0_L> ...")
    s1 = simple_m_legged_code(m=1, strength=STRENGTH, num_moments=NUM_MOMENTS,
                               code_type=CODE_TYPE, num_digits=1)[0]
    print("Computing m=2 |0_L> ...")
    s2 = simple_m_legged_code(m=2, strength=STRENGTH, num_moments=NUM_MOMENTS,
                               code_type=CODE_TYPE)[0]
    print("Computing m=4 |0_L> ...")
    s4 = simple_m_legged_code(m=4, strength=STRENGTH, num_moments=NUM_MOMENTS,
                               code_type=CODE_TYPE)[0]
    print("Computing m=8 |1_L> ...")
    s8 = simple_m_legged_code(m=8, strength=STRENGTH, num_moments=NUM_MOMENTS,
                               code_type=CODE_TYPE)[1]   # index 1 = |1_L>
    return {1: s1, 2: s2, 4: s4, 8: s8}


def _compute_wigner(state: qutip.Qobj, alpha_max: float) -> np.ndarray:
    """Return W (2D ndarray, shape WIGNER_POINTS×WIGNER_POINTS)."""
    if qutip.isket(state):
        rho = qutip.ket2dm(state)
    else:
        rho = state
    xvec = np.linspace(-alpha_max, alpha_max, WIGNER_POINTS)
    return qutip.wigner(rho, xvec, xvec, method='clenshaw')


if __name__ == "__main__":
    states = _compute_states()
    for m, (alpha_max, *_) in LAYOUT.items():
        W = _compute_wigner(states[m], alpha_max)
        print(f"  m={m}: W shape={W.shape}, min={W.min():.3f}, max={W.max():.3f}")
    print("Task 1 OK")
