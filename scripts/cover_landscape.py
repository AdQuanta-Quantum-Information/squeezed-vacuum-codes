# scripts/cover_landscape.py
"""Landscape white-background cover — 300 dpi TIFF for journal submission.

Same floating-slabs approach as cover_image_slabs.py, adapted for:
  • Landscape canvas (3300 × 2550 px at 300 dpi, A-letter rotated)
  • White background
  • Blue → white → red diverging colormap
  • Direct alpha-compositing for edge glow and peak bloom (screen blend
    is a no-op on white, so it is replaced with _composite_over)
"""
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
BG_COLOR_RGB = (255, 255, 255)    # white
COLORLIMS    = (-0.20, 0.23)

WIGNER_POINTS = 400
SLAB_SIZE     = 1024

_ROOT        = Path(__file__).parents[1]
OUTPUT_TIFF  = _ROOT / "_agents_outputs" / "cover_landscape.tiff"
PREVIEW_PNG  = _ROOT / "_agents_dump"    / "cover_landscape_preview.png"

# Per-state: (alpha_max, slab_scale, cx_frac, cy_frac)
# Diagonal staircase across the landscape width: m=1 top-left → m=8 bottom-right.
# slab_scale is relative to canvas_w (3300), so values are ~0.77× the portrait ones.
LAYOUT = {
    1: (3.0, 0.22, 0.14, 0.20),
    2: (4.5, 0.30, 0.34, 0.35),
    4: (5.5, 0.39, 0.55, 0.52),
    8: (7.0, 0.50, 0.76, 0.68),
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
    """Blue → white → red diverging colormap; zero maps to white (background)."""
    cmap = LinearSegmentedColormap.from_list(
        'bwr_white',
        ['#1a55cc', '#ffffff', '#cc2211'],
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
    rgba_f = sm.to_rgba(W)                          # shape (N, N, 4), float [0,1]
    rgba_u8 = (rgba_f * 255).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(rgba_u8, mode='RGBA')
    return img.resize((SLAB_SIZE, SLAB_SIZE), Image.BICUBIC)


def _perspective_warp(img: Image.Image) -> Image.Image:
    """
    Apply PIL QUAD warp to simulate 3D tilt (slab viewed from ~30° above).
    Returns RGBA image of same size with trapezoid slab, transparent outside.
    Uses Image.QUAD: data=(x0,y0, x1,y1, x2,y2, x3,y3) are SOURCE coords
    for (top-left, bottom-left, bottom-right, top-right) destination corners.
    """
    W, H = img.size    # both == SLAB_SIZE
    tilt_top = int(0.18 * H)
    pad      = int(0.10 * W)
    quad_data = (
        pad,   tilt_top,   # top-left source
        0,     H - 1,      # bottom-left source
        W - 1, H - 1,      # bottom-right source
        W-pad, tilt_top,   # top-right source
    )
    return img.transform((W, H), Image.QUAD, quad_data, resample=Image.BICUBIC)


def _slab_to_float32(img: Image.Image) -> np.ndarray:
    """Convert uint8 RGBA PIL Image to float32 numpy array [0, 1], shape (H, W, 4)."""
    return np.array(img, dtype=np.float32) / 255.0


def _make_canvas(width: int, height: int) -> np.ndarray:
    """Return float32 RGBA canvas (H, W, 4) filled with white, alpha=1."""
    canvas = np.ones((height, width, 4), dtype=np.float32)
    return canvas


def _composite_over(canvas: np.ndarray, layer: np.ndarray) -> np.ndarray:
    """
    Porter-Duff OVER: composite layer (H,W,4) over canvas (H,W,4).
    Both are float32 [0,1]. Returns updated canvas (same shape).
    """
    alpha = layer[..., 3:4]
    canvas[..., :3] = layer[..., :3] * alpha + canvas[..., :3] * (1.0 - alpha)
    canvas[..., 3]  = alpha[..., 0] + canvas[..., 3] * (1.0 - alpha[..., 0])
    return canvas


def _apply_drop_shadow(
    slab_f32: np.ndarray,
    canvas: np.ndarray,
    ox: int, oy: int,
    shift_x: int = 8, shift_y: int = 15,
    sigma: float = 18.0, strength: float = 0.35,
) -> np.ndarray:
    """Paint a soft drop shadow (dark) under the slab onto the white canvas."""
    H, W = canvas.shape[:2]
    shadow = np.zeros((H, W, 4), dtype=np.float32)
    alpha_mask = slab_f32[..., 3]
    sy = oy + shift_y
    sx = ox + shift_x
    sh, sw = slab_f32.shape[:2]
    dy0 = max(0, sy);  dy1 = min(H, sy + sh)
    dx0 = max(0, sx);  dx1 = min(W, sx + sw)
    sy0 = dy0 - sy;    sy1 = sy0 + (dy1 - dy0)
    sx0 = dx0 - sx;    sx1 = sx0 + (dx1 - dx0)
    shadow[dy0:dy1, dx0:dx1, 3] = alpha_mask[sy0:sy1, sx0:sx1]
    shadow[..., 3] = gaussian_filter(shadow[..., 3], sigma=sigma)
    shadow[..., 3] *= strength
    # shadow RGB stays 0,0,0 — dark shadow over white
    return _composite_over(canvas, shadow)


def _apply_edge_glow(
    slab_f32: np.ndarray,
    canvas: np.ndarray,
    ox: int, oy: int,
    sigma: float = 8.0, iterations: int = 6,
    color_hex: str = '#1976d2',
    alpha_scale: float = 0.50,
) -> np.ndarray:
    """
    Composite a soft colored border ring around the slab onto the canvas.

    On a white background screen-blend is a no-op, so we use _composite_over
    directly with alpha_scale controlling the ring opacity.
    """
    H, W = canvas.shape[:2]
    r = int(color_hex[1:3], 16) / 255.0
    g = int(color_hex[3:5], 16) / 255.0
    b = int(color_hex[5:7], 16) / 255.0

    alpha_mask = slab_f32[..., 3] > 0.05
    ring_mask  = binary_dilation(alpha_mask, iterations=iterations) & ~alpha_mask
    ring_f32   = np.zeros_like(slab_f32)
    ring_f32[ring_mask, 0] = r
    ring_f32[ring_mask, 1] = g
    ring_f32[ring_mask, 2] = b
    ring_f32[ring_mask, 3] = 1.0

    for c in range(4):
        ring_f32[..., c] = gaussian_filter(ring_f32[..., c], sigma=sigma)

    ring_f32[..., 3] *= alpha_scale

    glow_layer = np.zeros((H, W, 4), dtype=np.float32)
    sh, sw = slab_f32.shape[:2]
    dy0 = max(0, oy);  dy1 = min(H, oy + sh)
    dx0 = max(0, ox);  dx1 = min(W, ox + sw)
    ly0 = dy0 - oy;    ly1 = ly0 + (dy1 - dy0)
    lx0 = dx0 - ox;    lx1 = lx0 + (dx1 - dx0)
    glow_layer[dy0:dy1, dx0:dx1] = ring_f32[ly0:ly1, lx0:lx1]

    return _composite_over(canvas, glow_layer)


def _apply_peak_bloom(
    slab_f32: np.ndarray, W_data: np.ndarray,
    canvas: np.ndarray,
    ox: int, oy: int,
    sigma: float = 30.0, threshold: float = 0.6,
    color_hex: str = '#cc2211',
    alpha_scale: float = 0.20,
) -> np.ndarray:
    """
    Composite a diffuse warm tint over positive Wigner peaks.

    On a white background screen-blend is a no-op, so we use _composite_over
    directly with alpha_scale controlling the bloom opacity.
    """
    H, W_c = canvas.shape[:2]
    r = int(color_hex[1:3], 16) / 255.0
    g = int(color_hex[3:5], 16) / 255.0
    b = int(color_hex[5:7], 16) / 255.0

    W_max = W_data.max()
    peak_mask = (W_data > threshold * W_max).astype(np.float32)
    peak_img = Image.fromarray((peak_mask * 255).astype(np.uint8)).resize(
        (slab_f32.shape[1], slab_f32.shape[0]), Image.BICUBIC)
    peak_f = np.array(peak_img, dtype=np.float32) / 255.0

    bloom = np.zeros_like(slab_f32)
    bloom[..., 0] = peak_f * r
    bloom[..., 1] = peak_f * g
    bloom[..., 2] = peak_f * b
    bloom[..., 3] = peak_f

    for c in range(4):
        bloom[..., c] = gaussian_filter(bloom[..., c], sigma=sigma)

    bloom[..., 3] *= alpha_scale

    bloom_layer = np.zeros((H, W_c, 4), dtype=np.float32)
    sh, sw = slab_f32.shape[:2]
    dy0 = max(0, oy);  dy1 = min(H, oy + sh)
    dx0 = max(0, ox);  dx1 = min(W_c, ox + sw)
    ly0 = dy0 - oy;    ly1 = ly0 + (dy1 - dy0)
    lx0 = dx0 - ox;    lx1 = lx0 + (dx1 - dx0)
    bloom_layer[dy0:dy1, dx0:dx1] = bloom[ly0:ly1, lx0:lx1]

    return _composite_over(canvas, bloom_layer)


def _composite_slab(
    slab_f32: np.ndarray, W_data: np.ndarray,
    canvas: np.ndarray,
    cx_frac: float, cy_frac: float,
) -> np.ndarray:
    """Place one slab onto canvas at fractional centre position."""
    H, W_c = canvas.shape[:2]
    sh, sw = slab_f32.shape[:2]
    ox = int(cx_frac * W_c) - sw // 2
    oy = int(cy_frac * H)  - sh // 2
    # 1. Drop shadow (behind slab)
    canvas = _apply_drop_shadow(slab_f32, canvas, ox, oy)
    # 2. Slab itself
    slab_layer = np.zeros((H, W_c, 4), dtype=np.float32)
    dy0 = max(0, oy);  dy1 = min(H, oy + sh)
    dx0 = max(0, ox);  dx1 = min(W_c, ox + sw)
    ly0 = dy0 - oy;    ly1 = ly0 + (dy1 - dy0)
    lx0 = dx0 - ox;    lx1 = lx0 + (dx1 - dx0)
    slab_layer[dy0:dy1, dx0:dx1] = slab_f32[ly0:ly1, lx0:lx1]
    canvas = _composite_over(canvas, slab_layer)
    # 3. Edge glow (composited over slab border)
    canvas = _apply_edge_glow(slab_f32, canvas, ox, oy)
    # 4. Peak bloom (subtle warm tint over positive peaks)
    canvas = _apply_peak_bloom(slab_f32, W_data, canvas, ox, oy)
    return canvas


def make_cover(
    canvas_w: int = 3300, canvas_h: int = 2550,
    output_tiff: Path | None = None,
    output_png:  Path | None = None,
) -> None:
    """Render the landscape white-background cover and save TIFF + PNG preview."""
    if output_tiff is None:
        output_tiff = OUTPUT_TIFF
    if output_png is None:
        output_png = PREVIEW_PNG

    states = _compute_states()
    cmap, norm_cm = _build_colormap()
    canvas = _make_canvas(canvas_w, canvas_h)

    for m in [1, 2, 4, 8]:
        alpha_max, slab_scale, cx, cy = LAYOUT[m]
        print(f"Rendering m={m} ...")
        W = _compute_wigner(states[m], alpha_max)
        raw_img  = _render_wigner_rgba(W, cmap, norm_cm)
        warped   = _perspective_warp(raw_img)
        # Scale slab to slab_scale * canvas_width on longest dimension
        target_px = int(slab_scale * canvas_w)
        bw, bh = warped.size
        scale = target_px / max(bw, bh)
        warped = warped.resize((int(bw * scale), int(bh * scale)), Image.BICUBIC)
        slab_f32 = _slab_to_float32(warped)
        canvas = _composite_slab(slab_f32, W, canvas, cx, cy)

    # Convert float32 → uint8 → PIL → save
    output_u8  = (canvas * 255).clip(0, 255).astype(np.uint8)
    output_img = Image.fromarray(output_u8, mode='RGBA').convert('RGB')

    output_tiff.parent.mkdir(parents=True, exist_ok=True)
    output_img.save(output_tiff, dpi=(300, 300))
    print(f"Saved TIFF: {output_tiff.resolve()}")

    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_img.save(output_png, dpi=(150, 150))
    print(f"Saved PNG:  {output_png.resolve()}")


if __name__ == "__main__":
    make_cover()
