"""
NDVI phenology figures for the Amazon site's deciduous-forest-mapping use
case: for a handful of forest pixels spanning a range of seasonal NDVI
amplitude (evergreen-like/stable through high-amplitude/deciduous-candidate),
plot raw vs. cloud-masked vs. reconstructed monthly NDVI, one panel per
calendar year, so the effect of reconstruction on the seasonal signal itself
-- not just visual cloud removal -- is directly inspectable.

Pixel selection (data-driven, not hand-picked coordinates):
  1. "Forest" = mean NDVI (over confidently-clear months only) > 0.6,
     which separates dense-canopy forest from the tile's lower-NDVI
     non-forest areas (open water, wetland/campina, bare ground) --
     picked from the actual NDVI distribution (median 0.66, dense cluster
     above 0.6; see docs/amazon.md).
  2. Within forest, "seasonal amplitude" = mean annual (max-min) of the
     *reconstructed* NDVI (gap-free, so amplitude isn't itself an artifact
     of missing data), averaged over the 5 years.
  3. Four pixels are picked at amplitude percentiles 10/50/90/98 among
     forest pixels with strong data support (>=45/60 confidently-clear
     months) -- from stable/evergreen-like through the most pronounced
     seasonal swings this tile's forest shows. This is a *proxy* for
     deciduousness (a real deciduous signal should show up as elevated
     amplitude with a within-year dip-and-recover shape), not a
     species-level classification -- the figures are diagnostic, not a map.

Run: python scripts/amazon/04_deciduous_ndvi.py
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import sites, io_utils, compose, visualize

site = sites.get_site("amazon")
YEARS = list(range(2021, 2026))
ZOOM_HALF = 40  # px each side -> 80x80 crop, ~380m x 380m at 4.78m/px

# Fixed categorical colors, one per series, used identically in every panel.
COLOR_RAW = "#999999"     # de-emphasized: this is the noisy "before"
COLOR_MASKED = "#e69f00"  # amber markers at confidently-clear months only
COLOR_RECON = "#0072b2"   # bold: the reconstructed answer


def ndvi(arr):
    nir, red = arr[:, sites.NIR], arr[:, sites.RED]
    return (nir - red) / (nir + red + 1e-6)


def select_pixels(ndvi_raw, ndvi_recon, weight_final):
    conf = weight_final >= 0.7
    n_conf = conf.sum(axis=0)
    ndvi_conf_masked = np.where(conf, ndvi_raw, np.nan)
    mean_ndvi_conf = np.nanmean(ndvi_conf_masked, axis=0)

    forest = (mean_ndvi_conf > 0.6) & (n_conf >= 20)

    T, H, W = ndvi_recon.shape
    ndvi_by_year = ndvi_recon.reshape(len(YEARS), 12, H, W)
    amp = (ndvi_by_year.max(axis=1) - ndvi_by_year.min(axis=1)).mean(axis=0)  # (H,W)

    reliable = forest & (n_conf >= 45)
    ys, xs = np.where(reliable)
    amp_rel = amp[reliable]

    picks = {}
    for name, p in [("evergreen_like", 10), ("typical", 50),
                    ("elevated_amplitude", 90), ("deciduous_candidate", 98)]:
        target = np.percentile(amp_rel, p)
        idx = np.argmin(np.abs(amp_rel - target))
        picks[name] = (int(ys[idx]), int(xs[idx]))
    return picks, mean_ndvi_conf, amp, forest


def crop_with_marker(ax, refl_month, rc, half=ZOOM_HALF):
    """Show an 80x80px RGB zoom around `rc` (observed reflectance for one
    month), with the exact pixel marked -- ground-truths what the NDVI
    curve below is actually looking at, and whether January of that year
    happened to be clear or cloudy there.
    """
    r, c = rc
    H, W = refl_month.shape[1:]
    y0, y1 = max(0, r - half), min(H, r + half)
    x0, x1 = max(0, c - half), min(W, c + half)
    crop = refl_month[:, y0:y1, x0:x1]
    ax.imshow(visualize.rgb_stretch(crop, gain=0.13, gamma=1.3))
    ax.scatter([c - x0], [r - y0], s=90, marker="+", color="red", linewidth=2, zorder=5)
    ax.scatter([c - x0], [r - y0], s=220, facecolor="none", edgecolor="red", linewidth=1.2, zorder=5)
    ax.axis("off")


def plot_pixel(name, rc, ndvi_raw, ndvi_masked, ndvi_recon, weight_final, refl, out_path):
    r, c = rc
    fig, axes = plt.subplots(2, len(YEARS), figsize=(4 * len(YEARS), 6.6),
                              gridspec_kw={"height_ratios": [1, 1.3]})
    months_x = np.arange(1, 13)

    y_raw = ndvi_raw[:, r, c].reshape(len(YEARS), 12)
    y_masked = ndvi_masked[:, r, c].reshape(len(YEARS), 12)
    y_recon = ndvi_recon[:, r, c].reshape(len(YEARS), 12)

    ndvi_axes = axes[1]
    for i, year in enumerate(YEARS):
        # top row: RGB zoom, January of this year, pixel marked
        jan_idx = i * 12  # month 0 of that year = January
        crop_with_marker(axes[0, i], refl[jan_idx], rc)
        axes[0, i].set_title(f"{year}-01 RGB (zoom)", fontsize=10)

        # bottom row: monthly NDVI for this year
        ax = ndvi_axes[i]
        ax.plot(months_x, y_raw[i], "-", color=COLOR_RAW, lw=1.3, alpha=0.8,
                label="raw (observed)" if i == 0 else None, zorder=2)
        ax.plot(months_x, y_recon[i], "-", color=COLOR_RECON, lw=2.2,
                label="reconstructed" if i == 0 else None, zorder=3)
        ax.scatter(months_x, y_masked[i], marker="o", s=34, facecolor=COLOR_MASKED,
                   edgecolor="k", linewidth=0.5, zorder=4,
                   label="masked (clear obs. only)" if i == 0 else None)
        ax.set_title(str(year), fontsize=11)
        ax.set_xticks([1, 4, 7, 10])
        ax.set_xlim(0.5, 12.5)
        ax.grid(alpha=0.25, lw=0.5)
        if i == 0:
            ax.set_ylabel("NDVI")
    for a in ndvi_axes[1:]:
        a.sharey(ndvi_axes[0])

    ndvi_axes[0].legend(loc="lower left", fontsize=8, framealpha=0.9)
    fig.suptitle(f"{name.replace('_', ' ')} — pixel (row={r}, col={c})", y=1.02, fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    stack = io_utils.read_aoi_stack(site)
    refl = io_utils.to_reflectance(stack.data)

    phen = np.load(site.cache_dir / "phenology_recon.npz")
    weight_final = phen["weight_final"]
    refined = np.load(site.cache_dir / "refined.npz")["refined"]
    composite, _alpha = compose.compose(refl, weight_final, refined)

    ndvi_raw = ndvi(refl)
    ndvi_recon = ndvi(composite)
    ndvi_masked = np.where(weight_final >= 0.7, ndvi_raw, np.nan)

    print("Selecting forest pixels across NDVI seasonal-amplitude percentiles...")
    picks, mean_ndvi_conf, amp, forest = select_pixels(ndvi_raw, ndvi_recon, weight_final)
    for name, rc in picks.items():
        print(f"  {name}: pixel {rc}, amplitude={amp[rc]:.4f}")

    for i, (name, rc) in enumerate(picks.items(), start=10):
        out_path = site.fig_dir / f"{i}_ndvi_{name}.png"
        plot_pixel(name, rc, ndvi_raw, ndvi_masked, ndvi_recon, weight_final, refl, out_path)
        print(f"  saved {out_path}")

    # ---- overview: where do these pixels sit relative to the whole tile? ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    im0 = axes[0].imshow(np.clip(mean_ndvi_conf, 0, 1), cmap="YlGn", vmin=0.2, vmax=0.9)
    axes[0].set_title("Mean NDVI (confidently-clear obs.)")
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(np.where(forest, amp, np.nan), cmap="inferno", vmin=0, vmax=0.15)
    axes[1].set_title("Seasonal NDVI amplitude (forest only)")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    markers = ["o", "s", "^", "D"]
    for (name, (r, c)), m in zip(picks.items(), markers):
        for ax in axes:
            ax.scatter([c], [r], s=140, marker=m, facecolor="none", edgecolor="red", linewidth=2)
        axes[1].annotate(name, (c, r), color="white", fontsize=8, ha="left", va="bottom",
                          xytext=(5, 5), textcoords="offset points")
    plt.tight_layout()
    plt.savefig(site.fig_dir / "09_ndvi_pixel_locations.png", dpi=130)
    plt.close(fig)

    print("Done.")


if __name__ == "__main__":
    main()