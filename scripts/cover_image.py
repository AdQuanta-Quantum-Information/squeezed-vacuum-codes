# scripts/cover_image.py
"""Final cover image script — 300 dpi TIFF for journal submission."""
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
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from pathlib import Path

import qutip

from src.codes_built_in_superposition import simple_m_legged_code

# ── Constants ──────────────────────────────────────────────────────────────────
STRENGTH    = 1.5
NUM_MOMENTS = 300
CODE_TYPE   = "squeeze"
BG_COLOR    = "#0d1b2a"
COLORLIMS   = (-0.20, 0.23)

# Final resolution
NUM_POINTS = 200

_ROOT        = Path(__file__).parents[1]
OUTPUT_TIFF  = _ROOT / "_agents_outputs" / "cover_final.tiff"
PREVIEW_PNG  = _ROOT / "_agents_dump"    / "cover_cascade_preview_final.png"

# Per-state spatial range and depth parameters
SURFACE_PARAMS = {
    #  m : (alpha_max, y_offset, xy_scale, alpha_opacity)
    1:  (2.8, 17, 0.38, 0.85),
    2:  (4.0, 11, 0.58, 0.90),
    4:  (5.5,  5, 0.78, 0.95),
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


def make_cover(num_points: int = NUM_POINTS, output_path: Path | None = None) -> None:
    """Render the depth-cascade cover and save to output_path."""
    if output_path is None:
        output_path = OUTPUT_TIFF

    states = _compute_states()

    # ── Wigner data ───────────────────────────────────────────────────────────
    wigner_data: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for m, state in states.items():
        alpha_max = SURFACE_PARAMS[m][0]
        print(f"Computing Wigner m={m} ...")
        wigner_data[m] = compute_wigner(state, alpha_max, num_points)

    # ── Figure setup ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(8.5, 11), facecolor=BG_COLOR)
    # Expand axes rect beyond [0,0,1,1] so the 3D content fills the canvas
    ax: Axes3D = fig.add_axes([-0.18, -0.28, 1.36, 1.36], projection='3d')
    ax.set_facecolor(BG_COLOR)
    try:
        ax.computed_zorder = False
    except AttributeError:
        pass  # older matplotlib — back-to-front order handles z-sorting

    # Disable all axes decoration
    ax.set_axis_off()
    for axis3d in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis3d.pane.fill = False
        axis3d.pane.set_edgecolor('none')
        axis3d.line.set_color('none')
    ax.grid(False)

    ax.view_init(elev=28, azim=-62)

    # ── Colormap / norm ───────────────────────────────────────────────────────
    # Custom colormap: blue → BG_COLOR (navy) → red
    # Zero maps to the background colour, so the "floor" is invisible.
    cmap = LinearSegmentedColormap.from_list(
        'bwr_bg',
        ['#1a55cc', BG_COLOR, '#cc2211'],
        N=512,
    )
    norm = TwoSlopeNorm(vmin=COLORLIMS[0], vcenter=0.0, vmax=COLORLIMS[1])

    # ── Surfaces — back-to-front order (m=1 first, m=8 last) ─────────────────
    all_W: list[np.ndarray] = []
    for m in [1, 2, 4, 8]:
        X, Y, W = wigner_data[m]
        _, y_offset, xy_scale, alpha_opacity = SURFACE_PARAMS[m]
        all_W.append(W)
        ax.plot_surface(
            X * xy_scale,
            Y * xy_scale + y_offset,
            W,
            rcount=100, ccount=100,
            facecolors=cmap(norm(W)),
            linewidth=0,
            antialiased=False,
            shade=False,
            alpha=alpha_opacity,
        )

    # ── Z / XY limits ─────────────────────────────────────────────────────────
    z_min = min(W.min() for W in all_W)
    z_max = max(W.max() for W in all_W)
    ax.set_zlim(z_min * 1.05, z_max * 1.3)
    ax.set_xlim(-7, 7)
    ax.set_ylim(-7, 20)

    # ── Save TIFF (300 dpi) ───────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, facecolor=BG_COLOR)
    print(f"Saved TIFF: {output_path.resolve()}")

    # ── Also save a 150 dpi PNG preview ──────────────────────────────────────
    PREVIEW_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PREVIEW_PNG, dpi=150, facecolor=BG_COLOR)
    print(f"Saved PNG:  {PREVIEW_PNG.resolve()}")

    plt.close(fig)


if __name__ == "__main__":
    make_cover()
