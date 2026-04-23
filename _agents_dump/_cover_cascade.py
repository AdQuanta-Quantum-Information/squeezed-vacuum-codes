# _agents_dump/_cover_cascade.py
"""Draft cover image script — low resolution for fast iteration."""
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parents[1]))
    from scripts import add_root_to_path
    add_root_to_path()

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers '3d' projection
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path

import qutip

from src.codes_built_in_superposition import simple_m_legged_code

# ── Constants ──────────────────────────────────────────────────────────────────
STRENGTH    = 1.5
NUM_MOMENTS = 300
CODE_TYPE   = "squeeze"
BG_COLOR    = "#0d1b2a"
COLORLIMS   = (-0.20, 0.23)

# Draft resolution — change to 200 for final render
NUM_POINTS = 80

OUTPUT_DIR = Path(__file__).parent

# Per-state spatial range and depth parameters
SURFACE_PARAMS = {
    #  m : (alpha_max, y_offset, xy_scale, alpha_opacity)
    1:  (3.5, 18, 0.35, 0.55),
    2:  (4.5, 10, 0.55, 0.70),
    4:  (5.5,  4, 0.75, 0.85),
    8:  (7.0,  0, 1.00, 1.00),
}


def _compute_states() -> dict[int, qutip.Qobj]:
    """Generate the four code states. Returns {m: state}."""
    print("Computing m=1 |0_L> ...")
    # num_digits=1 is required for m=1 (function asserts: 1 <= num_digits <= m)
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
                               code_type=CODE_TYPE)[1]  # index 1 = |1_L>

    return {1: s1, 2: s2, 4: s4, 8: s8}


def compute_wigner(
    state: qutip.Qobj,
    alpha_max: float,
    num_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (X, Y, W) numpy arrays for a 3D surface plot.

    Notes:
    - qutip.wigner returns W (2D ndarray) directly for method='clenshaw'.
    - X, Y are built via meshgrid over the same xvec.
    - g=sqrt(2) default is used (standard Wigner normalization, matches paper).
    """
    if qutip.isket(state):
        rho = qutip.ket2dm(state)
    else:
        rho = state
    xvec = np.linspace(-alpha_max, alpha_max, num_points)
    W = qutip.wigner(rho, xvec, xvec, method='clenshaw')
    X, Y = np.meshgrid(xvec, xvec)
    return X, Y, W
