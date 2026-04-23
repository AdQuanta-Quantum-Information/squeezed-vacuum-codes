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


def make_cover(num_points: int = NUM_POINTS, output_path: Path | None = None) -> None:
    """Render the depth-cascade cover and save PNG/TIFF."""
    if output_path is None:
        output_path = OUTPUT_DIR / "cover_cascade_preview.png"

    states = _compute_states()

    # ── Wigner data ───────────────────────────────────────────────────────────
    wigner_data: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for m, state in states.items():
        alpha_max = SURFACE_PARAMS[m][0]
        print(f"Computing Wigner m={m} ...")
        wigner_data[m] = compute_wigner(state, alpha_max, num_points)

    # ── Figure setup ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(8.5, 11), facecolor=BG_COLOR)
    ax: Axes3D = fig.add_axes([0, 0, 1, 1], projection='3d')
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

    ax.view_init(elev=28, azim=-55)

    # ── Colormap / norm ───────────────────────────────────────────────────────
    cmap = matplotlib.colormaps['bwr']
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
            rstride=2, cstride=2,
            facecolors=cmap(norm(W)),
            linewidth=0,
            antialiased=True,
            shade=False,   # shade=True is incompatible with facecolors
            alpha=alpha_opacity,
        )

    # ── Z limits ──────────────────────────────────────────────────────────────
    z_min = min(W.min() for W in all_W)
    z_max = max(W.max() for W in all_W)
    ax.set_zlim(z_min * 1.1, z_max * 1.5)

    # ── Save ──────────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dpi = 150 if output_path.suffix == '.png' else 300
    fig.savefig(output_path, dpi=dpi, facecolor=BG_COLOR)
    plt.close(fig)
    print(f"Saved: {output_path.resolve()}")


if __name__ == "__main__":
    make_cover()
