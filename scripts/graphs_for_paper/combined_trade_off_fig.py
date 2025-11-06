import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Sequence, Tuple, List, Dict, Optional


# -------------------------------------------------------------
# Sub‑figure (a) – Loss‑limited logical error‑rate heat‑map
# -------------------------------------------------------------

def plot_loss_map(
    ax: plt.Axes,
    r_vals: Sequence[float],
    N_vals: Sequence[int],
    loss_rates: np.ndarray,
    photon_numbers: Optional[np.ndarray] = None,
    cmap: str = "viridis",
) -> None:
    """Draw panel (a): logical error‑rate under photon‑loss.

    Parameters
    ----------
    ax : matplotlib Axes
        Target axes provided by the layout grid.
    r_vals, N_vals
        1‑D arrays defining the squeezing strengths and leg counts.
    loss_rates : 2‑D ndarray (len(N) × len(r))
        Pre‑computed logical error rates under pure photon loss.
    photon_numbers : 2‑D ndarray, optional
        Mean‑photon‑number landscape for drawing resource contours.
    cmap : str
        Colormap for the heat‑map (defaults to "viridis").
    """

    # Heat‑map of log‑error rate (more readable than raw probabilities)
    im = ax.imshow(
        np.log10(loss_rates),
        origin="lower",
        aspect="auto",
        extent=[r_vals[0], r_vals[-1], N_vals[0], N_vals[-1]],
        cmap=cmap,
        interpolation="nearest",
    )
    ax.set_xlabel("Squeezing $r$")
    ax.set_ylabel("Number of legs $N$")
    ax.set_title("(a) Loss‑limited logical error rate")
    ax.set_yticks(N_vals)

    # Colour‑bar (shared 10‑log scale)
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("$\\log_{10}(P_\\mathrm{L})$")

    # Optional resource contours
    if photon_numbers is not None:
        cs = ax.contour(
            r_vals,
            N_vals,
            photon_numbers,
            colors="white",
            linestyles="--",
            linewidths=0.8,
        )
        ax.clabel(cs, fmt="⟨n⟩=%.1f", fontsize=7, inline=1)


# -------------------------------------------------------------
# Sub‑figure (b) – Dephasing‑limited logical error‑rate heat‑map
# -------------------------------------------------------------

def plot_deph_map(
    ax: plt.Axes,
    r_vals: Sequence[float],
    N_vals: Sequence[int],
    deph_rates: np.ndarray,
    cmap: str = "viridis",
) -> None:
    """Draw panel (b): logical error‑rate under pure dephasing."""

    im = ax.imshow(
        np.log10(deph_rates),
        origin="lower",
        aspect="auto",
        extent=[r_vals[0], r_vals[-1], N_vals[0], N_vals[-1]],
        cmap=cmap,
        interpolation="nearest",
    )
    ax.set_xlabel("Squeezing $r$")
    ax.set_title("(b) Dephasing‑limited logical error rate")
    ax.set_yticks([])  # remove to avoid repetition; shared y on panel (a)

    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("$\\log_{10}(P_\\mathrm{ϕ})$")


# -------------------------------------------------------------
# Sub‑figure (c) – Error‑bias / QEC‑gain map
# -------------------------------------------------------------

def plot_bias_map(
    ax: plt.Axes,
    r_vals: Sequence[float],
    N_vals: Sequence[int],
    loss_rates: np.ndarray,
    deph_rates: np.ndarray,
    target_bias: float = 1.0,
    cmap: str = "coolwarm",
) -> None:
    """Draw panel (c): ratio of dephasing to loss logical error rates.

    The colour scale is centred on *target_bias* so that neutral bias appears
    white, loss‑dominated blue, and dephasing‑dominated red (using a
    diverging cmap).
    """

    bias = np.divide(deph_rates, loss_rates, where=loss_rates > 0)
    im = ax.imshow(
        np.log10(bias),
        origin="lower",
        aspect="auto",
        extent=[r_vals[0], r_vals[-1], N_vals[0], N_vals[-1]],
        cmap=cmap,
        vmin=-2,  # tweak limits as you like
        vmax=2,
        interpolation="nearest",
    )
    ax.set_xlabel("Squeezing $r$")
    ax.set_ylabel("Number of legs $N$")
    ax.set_title("(c) Error‑bias $β = Γ_ϕ/Γ_L$")
    ax.set_yticks(N_vals)

    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("$\\log_{10}(β)$")


# -------------------------------------------------------------
# Sub‑figure (d) – Representative Wigner thumbnails
# -------------------------------------------------------------

def plot_wigner_insets(
    ax: plt.Axes,
    thumbnails: List[Dict[str, np.ndarray]],
    n_cols: int = 3,
    extent: Tuple[float, float, float, float] = (-5, 5, -5, 5),
) -> None:
    """Draw panel (d): a scatter of Wigner thumbnails on a blank canvas.

    Parameters
    ----------
    ax : matplotlib Axes
        Host axes that will be cleared and replaced with insets.
    thumbnails : list of dicts
        Each dict must provide keys:
          * ``'W'``   – 2‑D array of Wigner values (square grid)
          * ``'r'``   – squeezing value (float)
          * ``'N'``   – number of legs (int)
          * ``'label'`` – string label (e.g. "A")
    n_cols : int, optional
        Number of thumbnail columns; rows are inferred.
    extent : 4‑tuple, optional
        Coordinate extent (xmin, xmax, ymin, ymax) assumed for all Wigners.
    """

    ax.axis("off")  # hide the parent axes; it only hosts the title
    ax.set_title("(d) Representative Wigner slices")

    # Compute a simple grid of inset axes
    n = len(thumbnails)
    n_rows = int(np.ceil(n / n_cols))
    thumb_w = 1.0 / n_cols
    thumb_h = 1.0 / n_rows

    for k, thumb in enumerate(thumbnails):
        row = k // n_cols
        col = k % n_cols
        left = col * thumb_w + 0.02
        bottom = 1.0 - (row + 1) * thumb_h + 0.02
        width = thumb_w - 0.04
        height = thumb_h - 0.04

        ax_in = ax.inset_axes([left, bottom, width, height])
        ax_in.imshow(
            thumb["W"],
            origin="lower",
            extent=extent,
            cmap="RdBu_r",
            interpolation="bicubic",
        )
        ax_in.set_xticks([])
        ax_in.set_yticks([])
        ax_in.set_title(thumb.get("label", ""), fontsize=8, pad=1)


# -------------------------------------------------------------
# Top‑level convenience wrapper
# -------------------------------------------------------------

def plot_squeezing_code_figure(
    r_vals: Sequence[float],
    N_vals: Sequence[int],
    loss_rates: np.ndarray,
    deph_rates: np.ndarray,
    photon_numbers: Optional[np.ndarray] = None,
    thumbnails: Optional[List[Dict[str, np.ndarray]]] = None,
    figsize: Tuple[int, int] = (12, 8),
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes, plt.Axes, plt.Axes]]:
    """Create the full 4‑panel figure.

    Returns
    -------
    fig : matplotlib Figure
    axes : tuple
        (ax_loss, ax_deph, ax_bias, ax_wigner)
    """

    # Layout
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 2, figure=fig, hspace=0.25, wspace=0.15)

    ax_loss = fig.add_subplot(gs[0, 0])
    ax_deph = fig.add_subplot(gs[0, 1], sharey=ax_loss)
    ax_bias = fig.add_subplot(gs[1, 0])
    ax_wigner = fig.add_subplot(gs[1, 1])

    # Panel (a)
    plot_loss_map(ax_loss, r_vals, N_vals, loss_rates, photon_numbers)

    # Panel (b)
    plot_deph_map(ax_deph, r_vals, N_vals, deph_rates)

    # Panel (c)
    plot_bias_map(ax_bias, r_vals, N_vals, loss_rates, deph_rates)

    # Panel (d)
    if thumbnails is None:
        ax_wigner.text(
            0.5,
            0.5,
            "[add thumbnails]",
            ha="center",
            va="center",
            fontsize=10,
            color="gray",
        )
        ax_wigner.axis("off")
    else:
        plot_wigner_insets(ax_wigner, thumbnails)

    return fig, (ax_loss, ax_deph, ax_bias, ax_wigner)


# -------------------------------------------------------------
# Example stub (delete or adapt in your notebook/script)
# -------------------------------------------------------------
if __name__ == "__main__":
    # Dummy demo data just so the script can be run stand‑alone.
    r_vals = np.linspace(0.0, 1.5, 51)
    N_vals = np.arange(2, 10, 2)

    R, N = np.meshgrid(r_vals, N_vals)
    loss_rates = 1e-3 * np.exp(-(N - 1) / 2) * np.exp(-R)  # fake scaling
    deph_rates = 1e-3 * np.exp(R) * (1 + N / 6)
    photon_numbers = 2 * np.cosh(R)  # very crude proxy

    # Fake Wigner thumbnails (Gaussians)
    W = np.exp(-0.5 * (np.linspace(-5, 5, 128) ** 2))
    W = np.outer(W, W)
    thumbs = [
        {"W": W, "r": 0.3, "N": 2, "label": "A"},
        {"W": W, "r": 0.8, "N": 4, "label": "B"},
        {"W": W, "r": 1.2, "N": 6, "label": "C"},
        {"W": W, "r": 1.5, "N": 8, "label": "D"},
    ]

    fig, _ = plot_squeezing_code_figure(
        r_vals, N_vals, loss_rates, deph_rates, photon_numbers, thumbs
    )
    fig.suptitle("Demo - family of squeezed‑cat codes", fontsize=14)
    plt.show()
