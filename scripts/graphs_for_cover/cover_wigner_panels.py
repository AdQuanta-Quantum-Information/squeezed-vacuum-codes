"""Generate 4 standalone high-res Wigner function PNGs — no axes, labels, or grid.

Colormap: blue (negative) → white (zero, transparent) → red (positive).
Alpha:    |W|^ALPHA_GAMMA / max(|W|) — zero is fully transparent.
"""
if __name__ == "__main__":
    from __init__ import add_root_to_path
    add_root_to_path()

from typing import Final

import numpy as np
import matplotlib
import matplotlib.cm
from matplotlib.colors import TwoSlopeNorm
from PIL import Image

from src import code_paths
from src.codes_built_in_superposition import simple_m_legged_code
from src.quantum.visualizations.wigner_function import compute_wigner_values
from src.utils.prints import ProgressBar

# ── Constants ──────────────────────────────────────────────────────────────────
STRENGTH    = 1.5
NUM_MOMENTS = 500
CODE_TYPE   = "squeeze"
COLORLIMS    = (-0.20, 0.23)
ALPHA_GAMMA  = 0.6   # <1 makes faint structure more visible (softer falloff to zero)

# Fixed Wigner grid resolution requested by user.
WIGNER_POINTS = 4000
OUTPUT_PX     = 4000   # Final image size in pixels (square)

DPI : Final[int] = 600  # Dots per inch for PNG/TIFF metadata; does not affect actual pixel dimensions.

# Extend phase-space range beyond the state's natural scale so no features
# are clipped at the canvas boundary.  1.0 = no extension.
ALPHA_EXTEND: float = 2.5

OUTPUT_DIR = code_paths.outputs / "wigner_panels"

# Per-state: (codeword_index, num_digits, alpha_max, label)
# alpha_max is the natural scale of the state; actual plot range = alpha_max * ALPHA_EXTEND
STATES = {
    1: (0, 1, 3.0, "m1_code0L"),
    2: (0, 2, 4.5, "m2_code0L"),
    4: (0, 2, 5.5, "m4_code0L"),
    8: (1, 2, 7.0, "m8_code1L"),
}


def _compute_and_render(
    m: int,
    alpha_extend: float = ALPHA_EXTEND,
) -> tuple[Image.Image, np.ndarray, np.ndarray, np.ndarray]:
    """Compute state and Wigner matrix, then return RGBA image + raw data."""
    codeword_idx, num_digits, alpha_max, _ = STATES[m]
    plot_range = alpha_max * alpha_extend

    print(f"  Computing m={m} state ...")
    state = simple_m_legged_code(
        m=m, strength=STRENGTH, num_moments=NUM_MOMENTS,
        code_type=CODE_TYPE, num_digits=num_digits,
    )[codeword_idx]

    print(f"  Computing m={m} Wigner ({WIGNER_POINTS}×{WIGNER_POINTS}), "
          f"range ±{plot_range:.1f} (extend={alpha_extend}×) ...")
    W, xvec, yvec = compute_wigner_values(
        state=state,
        alpha_max=plot_range,
        num_points=WIGNER_POINTS,
        method="clenshaw",
    )

    # ── RGB: symmetric blue-white-red ──
    abs_max = max(abs(COLORLIMS[0]), abs(COLORLIMS[1]))
    norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0.0, vmax=abs_max)
    cmap = matplotlib.colormaps['bwr']
    sm   = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    rgb_f = sm.to_rgba(W)[..., :3]       # (N, N, 3) float, drop colormap alpha

    # ── Alpha: magnitude-based transparency ──
    W_abs  = np.abs(W)
    alpha  = (W_abs / W_abs.max()) ** ALPHA_GAMMA   # (N, N) in [0, 1]

    rgba_f  = np.concatenate([rgb_f, alpha[..., np.newaxis]], axis=-1)  # (N, N, 4)
    rgba_u8 = (rgba_f * 255).clip(0, 255).astype(np.uint8)

    img = Image.fromarray(rgba_u8, mode='RGBA')
    img = img.resize((OUTPUT_PX, OUTPUT_PX), Image.BICUBIC)
    return img, W, xvec, yvec


def main(alpha_extend: float = ALPHA_EXTEND) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR.resolve()}")
    print(f"Phase-space range extension: {alpha_extend}×\n")

    for m in ProgressBar([1, 2, 4, 8], prefix="panels: "):
        _, _, _, label = STATES[m]
        img, W, xvec, yvec = _compute_and_render(m, alpha_extend=alpha_extend)
        out_base = OUTPUT_DIR / f"wigner_{label}"

        image_formats = {
            ".png":  {"dpi": (DPI, DPI)},
            ".tiff": {"dpi": (DPI, DPI), "compression": "tiff_lzw"},
        }
        for suffix, kwargs in image_formats.items():
            img.save(out_base.with_suffix(suffix), **kwargs)

        npz_path = out_base.with_suffix(".npz")
        np.savez_compressed(
            npz_path,
            W=W,
            xvec=xvec,
            yvec=yvec,
            alpha_extend=np.array(alpha_extend, dtype=np.float64),
            wigner_points=np.array(WIGNER_POINTS, dtype=np.int64),
        )

        saved_names = ", ".join(out_base.with_suffix(s).name for s in [*image_formats, ".npz"])
        print(f"  Saved: {saved_names} ({img.width}×{img.height} px image)\n")

    print("All 4 panels saved.")


if __name__ == "__main__":
    main()
