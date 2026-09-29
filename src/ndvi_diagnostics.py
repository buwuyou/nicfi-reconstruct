"""
Shared NDVI diagnostic-plot helpers: a tight monthly RGB zoom (for the
single year with the most confidently-clear observations at a pixel) paired
with that pixel's raw/masked/reconstructed NDVI across all years. Used by
scripts/amazon/04_deciduous_ndvi.py (amplitude-selected candidates) and
05_random_forest_pixels.py (unbiased random sample) -- kept here rather
than duplicated because the plot itself is site-agnostic; only pixel
*selection* differs between the two scripts.
"""
import numpy as np
import matplotlib.pyplot as plt

from . import sites, visualize

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
COLOR_RAW = "#999999"     # de-emphasized: this is the noisy "before"
COLOR_MASKED = "#e69f00"  # amber markers at confidently-clear months only
COLOR_RECON = "#0072b2"   # bold: the reconstructed answer


def ndvi(arr):
    nir, red = arr[:, sites.NIR], arr[:, sites.RED]
    return (nir - red) / (nir + red + 1e-6)


def best_year(weight_final, rc, n_years):
    """Index (0-based, into `years`) of the year with the most
    confidence-weighted clear coverage at this pixel."""
    r, c = rc
    w = weight_final[:, r, c].reshape(n_years, 12)
    return int(np.argmax(w.sum(axis=1)))


def crop_with_marker(ax, refl_month, rc, half, border_color=None):
    """Tight RGB zoom around `rc` (observed reflectance for one month),
    exact pixel marked. `border_color`, if given, flags confidence
    (e.g. green = confidently clear, gray = not) at a glance.
    """
    r, c = rc
    H, W = refl_month.shape[1:]
    y0, y1 = max(0, r - half), min(H, r + half)
    x0, x1 = max(0, c - half), min(W, c + half)
    crop = refl_month[:, y0:y1, x0:x1]
    ax.imshow(visualize.rgb_stretch(crop, gain=0.13, gamma=1.3), interpolation="nearest")
    ax.scatter([c - x0], [r - y0], s=70, marker="+", color="red", linewidth=1.8, zorder=5)
    ax.scatter([c - x0], [r - y0], s=170, facecolor="none", edgecolor="red", linewidth=1.1, zorder=5)
    ax.set_xticks([])
    ax.set_yticks([])
    if border_color is not None:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(border_color)
            spine.set_linewidth(2.5)
    else:
        ax.axis("off")


def plot_pixel_ndvi_rgb(title, rc, years, ndvi_raw, ndvi_masked, ndvi_recon, weight_final, refl,
                         out_path, zoom_half=15, caption_extra=""):
    """One figure: top = all 12 months of RGB zoom for the best-observed
    year; bottom = monthly NDVI (raw/masked/reconstructed), one panel per
    year in `years`, with the best-observed year outlined.
    """
    r, c = rc
    n_years = len(years)
    by = best_year(weight_final, rc, n_years)
    by_year = years[by]

    fig = plt.figure(figsize=(20, 12))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.5, 1], hspace=0.5, top=0.86, bottom=0.06)
    rgb_gs = outer[0].subgridspec(2, 6, wspace=0.08, hspace=0.3)
    ndvi_gs = outer[1].subgridspec(1, n_years, wspace=0.12)

    w_months = weight_final[by * 12:(by + 1) * 12, r, c]
    for m in range(12):
        ax = fig.add_subplot(rgb_gs[m // 6, m % 6])
        border = "#2ca02c" if w_months[m] >= 0.7 else "#999999"
        crop_with_marker(ax, refl[by * 12 + m], rc, zoom_half, border_color=border)
        ax.set_title(MONTH_NAMES[m], fontsize=10)

    months_x = np.arange(1, 13)
    y_raw = ndvi_raw[:, r, c].reshape(n_years, 12)
    y_masked = ndvi_masked[:, r, c].reshape(n_years, 12)
    y_recon = ndvi_recon[:, r, c].reshape(n_years, 12)

    ndvi_axes = [fig.add_subplot(ndvi_gs[0, i]) for i in range(n_years)]
    for i, year in enumerate(years):
        ax = ndvi_axes[i]
        ax.plot(months_x, y_raw[i], "-", color=COLOR_RAW, lw=1.3, alpha=0.8,
                label="raw (observed)" if i == 0 else None, zorder=2)
        ax.plot(months_x, y_recon[i], "-", color=COLOR_RECON, lw=2.2,
                label="reconstructed" if i == 0 else None, zorder=3)
        ax.scatter(months_x, y_masked[i], marker="o", s=34, facecolor=COLOR_MASKED,
                   edgecolor="k", linewidth=0.5, zorder=4,
                   label="masked (clear obs. only)" if i == 0 else None)
        ax.set_title(str(year), fontsize=11, fontweight="bold" if year == by_year else "normal")
        ax.set_xticks([1, 4, 7, 10])
        ax.set_xlim(0.5, 12.5)
        ax.grid(alpha=0.25, lw=0.5)
        if i == 0:
            ax.set_ylabel("NDVI")
        if year == by_year:
            for spine in ax.spines.values():
                spine.set_edgecolor("#2ca02c")
                spine.set_linewidth(2)
    for a in ndvi_axes[1:]:
        a.sharey(ndvi_axes[0])
    ndvi_axes[0].legend(loc="lower left", fontsize=8, framealpha=0.9)

    fig.text(0.5, 0.945,
              f"Top: {by_year} monthly RGB, zoomed (most confidently-clear observations at this "
              f"pixel) -- green border = confidently clear, gray = not.", ha="center", fontsize=11)
    fig.text(0.5, 0.925,
              f"Bottom: monthly NDVI, all {n_years} years ({by_year}, outlined in green, is the "
              f"year shown above).{caption_extra}", ha="center", fontsize=11)
    fig.suptitle(f"{title} — pixel (row={r}, col={c})", y=0.975, fontsize=15)
    plt.savefig(out_path, dpi=130)
    plt.close(fig)