"""Visualization helpers: consistent RGB stretch + the figures used to show
before/after reconstruction quality."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from . import sites as config

QUALITY_CMAP = ListedColormap(["none", "#e5383b", "#f2a413", "#2274a5", "#a56cc1", "black"])
QUALITY_LABELS = ["clear", "thick cloud", "thin cloud", "shadow", "haze", "nodata"]


def rgb_stretch(refl3, gain=0.4, gamma=1.4):
    """refl3: (3 or 4, H, W) in band order blue,green,red,[nir]. Returns
    (H,W,3) uint8-ready float in [0,1] using bands [red,green,blue]."""
    rgb = refl3[[config.RED, config.GREEN, config.BLUE]]
    rgb = np.clip(rgb / gain, 0, 1)
    rgb = np.power(rgb, 1 / gamma)
    return rgb.transpose(1, 2, 0)


def month_label(m):
    return m[:7]


def plot_month_grid(arr_TCHW, months, title, path, **stretch_kwargs):
    n = len(months)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    for i, m in enumerate(months):
        ax = axes.flat[i]
        ax.imshow(rgb_stretch(arr_TCHW[i], **stretch_kwargs))
        ax.set_title(month_label(m))
        ax.axis("off")
    for i in range(n, nrows * ncols):
        axes.flat[i].axis("off")
    fig.suptitle(title, y=1.0, fontsize=14)
    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_before_after(observed, composite, months, path, quality=None, **stretch_kwargs):
    """One row per month: observed | quality overlay | reconstructed composite."""
    n = len(months)
    ncols = 3
    fig, axes = plt.subplots(n, ncols, figsize=(4 * ncols, 4 * n))
    for i, m in enumerate(months):
        ax0, ax1, ax2 = axes[i]
        rgb_obs = rgb_stretch(observed[i], **stretch_kwargs)
        ax0.imshow(rgb_obs)
        ax0.set_title(f"{month_label(m)} — observed" if i == 0 else month_label(m))
        ax0.axis("off")

        ax1.imshow(rgb_obs)
        if quality is not None:
            overlay = np.ma.masked_where(quality[i] == 0, quality[i])
            ax1.imshow(overlay, cmap=QUALITY_CMAP, vmin=0, vmax=5, alpha=0.65)
        ax1.set_title("cloud/shadow/haze mask" if i == 0 else "")
        ax1.axis("off")

        ax2.imshow(rgb_stretch(composite[i], **stretch_kwargs))
        ax2.set_title("reconstructed composite" if i == 0 else "")
        ax2.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_pixel_phenology(refl, weight, recon_curve, composite, months, pixel_rc, path,
                          band=config.NIR, band_name="NIR"):
    """Time series for one pixel: raw observations colored by confidence,
    fitted harmonic curve, and the final composite value."""
    r, c = pixel_rc
    t = np.arange(len(months))
    y_obs = refl[:, band, r, c]
    w = weight[:, r, c]
    y_curve = recon_curve[:, band, r, c]
    y_comp = composite[:, band, r, c]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(t, y_curve, "-", color="gray", lw=2, label="phenology curve (fit)")
    ax.plot(t, y_comp, "o-", color="#2274a5", lw=1, ms=4, label="final composite")
    sc = ax.scatter(t, y_obs, c=w, cmap="RdYlGn", vmin=0, vmax=1, s=70,
                     edgecolor="k", zorder=5, label="raw observation (color=confidence)")
    ax.set_xticks(t)
    ax.set_xticklabels([month_label(m) for m in months], rotation=45, ha="right")
    ax.set_ylabel(f"{band_name} reflectance")
    ax.set_title(f"Pixel (row={r}, col={c}) phenology reconstruction")
    ax.legend(loc="best", fontsize=9)
    plt.colorbar(sc, ax=ax, label="observation confidence", fraction=0.05)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close(fig)


def plot_confidence_map(confidence, path, title="Phenology-fit confidence (per-pixel data sufficiency)"):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(confidence, cmap="viridis", vmin=confidence.min(), vmax=1.0)
    ax.set_title(title)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close(fig)