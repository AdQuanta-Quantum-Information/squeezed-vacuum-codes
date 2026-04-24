# Journal Cover — Image Composition Instructions
**For use with a dedicated image generation / compositing tool**

---

## Input Files

All four source images are in `_agents_outputs/wigner_panels/`:

| File | Physics | Description |
|---|---|---|
| `wigner_m1_code0L.png` | m=1, \|0_L⟩ | Single-legged code: smooth squeezed Gaussian stripe |
| `wigner_m2_code0L.png` | m=2, \|0_L⟩ | Two-legged code: two symmetric lobes |
| `wigner_m4_code0L.png` | m=4, \|0_L⟩ | Four-legged code: 4-fold rotationally symmetric pattern |
| `wigner_m8_code1L.png` | m=8, \|1_L⟩ | Eight-legged code (logical 1): concentric ring interference structure — the most complex and visually striking |

Each image is **2000 × 2000 pixels**, **RGBA (transparent background)**, 300 dpi. The colormap is:
- **Pure blue** (`#0000ff`) → most negative Wigner values
- **White** (`#ffffff`) → intermediate values (approaching zero)
- **Pure red** (`#ff0000`) → most positive values
- **Fully transparent** (alpha = 0) → exactly zero

Near-zero pixels fade to transparent via a gamma-corrected alpha channel (`alpha = |W|^0.6`). This means the "empty" phase space floor is invisible — only the quantum features are opaque. Place the panels over any background color; the Wigner structure will float against it naturally.

**Suggested background**: deep navy `#0d1b2a` or black `#000000` work well. Against navy, blue lobes partially blend into the background (which is a nice effect); against black, all features are equally prominent.

---

## ⚠️ Required Output Specification (Journal Mandated)

> The journal explicitly requires: **US Letter or A4 size**, **portrait orientation**, **300 dpi minimum**, in **TIFF, PSD, JPEG, or PDF format**. These are non-negotiable submission requirements.

| Property | **Required Value** |
|---|---|
| Orientation | **Portrait (vertical)** — not landscape |
| Canvas size | **8.5 × 11 inches (US Letter)** — or A4 (8.27 × 11.69 in) |
| Resolution | **300 dpi minimum** → exactly **2550 × 3300 px** at 300 dpi for US Letter |
| Pixel dimensions | **Width: 2550 px — Height: 3300 px** (height > width, portrait) |
| Aspect ratio | **17 : 22** (approx. 0.773 wide-to-tall) |
| File format | **TIFF** (preferred) — also accepted: PSD, JPEG, PDF |
| Color mode | RGB (not CMYK, not grayscale) |
| Background color | `#0d1b2a` (deep navy / near-black blue) |

**Do not export at any other size or orientation.** The journal production team will use this file directly for print.

---

## Composition Concept: "Floating Crystal Slabs"

Four Wigner function panels are arranged as **glowing crystal slabs** hovering in phase space — each panel tilted in 3D perspective, receding into the background from lower-right (foreground) to upper-left (background). The m=8 slab dominates the foreground; the m=1 slab is small and distant.

---

## Per-Panel 3D Transform

Each panel should be perspective-warped into a **trapezoid** shape to simulate viewing a flat horizontal slab from slightly above and to the right:

- **Top edge:** narrower and higher than the bottom edge
- **Bottom edge:** full width, at the bottom of the panel's bounding box
- **Approximate warp:** top edge is ~80% the width of the bottom edge, and raised ~18% of the panel height

**Tilt direction:** slight lean to the right (right edge slightly lower than left edge, ~5–8°).

**3D viewing angle:** ~25–30° elevation above horizontal, looking slightly from the left (azimuth ~–60° from straight-on).

---

## Layout

Arrange the four slabs on a **diagonal staircase** from upper-left (background) to lower-right (foreground):

| Panel | Position (canvas %) | Scale (% of canvas width) | Depth / Layer |
|---|---|---|---|
| m=1 | Center at (22%, 22%) | 28% of canvas width | Back (layer 1) |
| m=2 | Center at (36%, 36%) | 38% of canvas width | (layer 2) |
| m=4 | Center at (52%, 52%) | 50% of canvas width | (layer 3) |
| m=8 | Center at (70%, 68%) | 65% of canvas width | Front (layer 4) |

Positions are measured from top-left corner. Render back-to-front (m=1 first, m=8 last so it occludes earlier layers).

**Leave the top ~15% of the canvas relatively clear** — this is where the journal masthead and title will be placed by the publisher.

---

## Visual Effects (per slab)

Apply these effects to each slab, scaled proportionally to the slab's size:

### 1. Drop Shadow (behind each slab)
- Color: pure black (`#000000`)
- Offset: 8 px right, 15 px down (scaled to slab size)
- Blur radius: 18–22 px (Gaussian)
- Opacity: 55–65%
- Apply **underneath** the slab, before compositing the slab itself

### 2. Edge Glow (around slab perimeter)
- Color: light cyan-blue (`#4fc3f7`)
- Width: 4–6 px border ring outside the slab trapezoid
- Blur: Gaussian σ ≈ 8–12 px (the glow should spread outward ~20 px)
- Blend mode: **Screen**
- The effect makes each slab look like a glowing crystal with an aura

### 3. Peak Bloom (over the brightest Wigner peaks)
- Identifies the brightest orange/yellow regions in the Wigner image
- Applies a **diffuse orange halo** (`#ff6600`) over those regions
- Blur radius: 30–50 px (very soft, wide halo)
- Blend mode: **Screen**
- Creates a "burning" glow over the quantum interference peaks

### 4. Depth-of-field suggestion (optional)
- Background slabs (m=1, m=2) can have a very slight motion/depth blur (~2–3 px) to enhance the depth illusion
- Foreground slabs (m=4, m=8) should be crisp and sharp

---

## Color Grading (optional, for overall atmosphere)

After compositing all four slabs:
- Slight **cold blue** color cast in the shadows (deepen the navy)
- Slight **warm orange** color cast in the highlights (intensify the peaks)
- Optional: subtle **radial vignette** darkening the corners (keeping focus on the slab cascade)

---

## What to Avoid

- **No axes, tick marks, labels, colorbars, or mathematical notation** anywhere on the image
- **No white or grey backgrounds** — the navy must be preserved
- **Do not over-saturate** — the Wigner images already have vivid colors; light-handed treatment is better
- **Do not obscure the quantum structure** — the concentric ring pattern in m=8 and the 4-fold symmetry in m=4 should remain clearly visible

---

## Physics Notes (for contextual accuracy)

- The **negative Wigner values** (cyan regions) are a signature of non-classical quantum states. They should remain visible and distinct — they are scientifically important.
- The **m=8 state shown is |1_L⟩** (logical one codeword), which has more complex interference structure than |0_L⟩ — this is intentional.
- The cascade from simple (m=1 Gaussian stripe) to complex (m=8 interference mandala) is the physics story of the paper: increasing m gives richer, more protected code states.

---

## Suggested Caption (≤ 300 characters, for journal submission)

> "Wigner phase-space functions of squeezed-vacuum bosonic code states shown as floating crystal slabs. From background to foreground: m = 1, 2, 4, 8-legged codes. The m = 8 logical |1⟩ state dominates, illustrating the tunable complexity of this code family."

(261 characters ✓)
