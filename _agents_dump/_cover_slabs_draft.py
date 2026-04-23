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


def _build_colormap() -> tuple[LinearSegmentedColormap, TwoSlopeNorm]:
    """Build plasma_dark colormap: cyan → BG → orange → yellow."""
    bg = tuple(c / 255.0 for c in BG_COLOR_RGB)   # normalised RGB
    cmap = LinearSegmentedColormap.from_list(
        'plasma_dark',
        ['#00e5ff', (*bg, 1.0), '#ff6600', '#ffe066'],
        N=512,
    )
    norm = TwoSlopeNorm(vmin=COLORLIMS[0], vcenter=0.0, vmax=COLORLIMS[1])
    return cmap, norm


def _render_wigner_rgba(W: np.ndarray, cmap: LinearSegmentedColormap,
                        norm: TwoSlopeNorm) -> Image.Image:
    """
    Render W → uint8 PIL RGBA image of size SLAB_SIZE×SLAB_SIZE.
    Pipeline: ScalarMappable → RGBA float → uint8 PIL → bicubic upscale.
    """
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    rgba_f = sm.to_rgba(W)                            # shape (N, N, 4), float [0,1]
    rgba_u8 = (rgba_f * 255).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(rgba_u8, mode='RGBA')
    return img.resize((SLAB_SIZE, SLAB_SIZE), Image.BICUBIC)


if __name__ == "__main__":
    states = _compute_states()
    cmap, norm = _build_colormap()
    for m, (alpha_max, *_) in LAYOUT.items():
        W = _compute_wigner(states[m], alpha_max)
        img = _render_wigner_rgba(W, cmap, norm)
        out = OUTPUT_DIR / f"_slab_test_m{m}.png"
        img.save(out)
        print(f"  m={m}: saved {out}")
    print("Task 2 OK")
