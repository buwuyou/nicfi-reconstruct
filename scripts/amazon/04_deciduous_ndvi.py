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

Each pixel's figure shows: (a) all 12 months of RGB for the single year
with the most confidently-clear observations at that pixel -- not just
January across years, since the point is to see the canopy actually change
month to month -- tightly zoomed (30x30px, ~143m) rather than a broad
neighborhood, since a real deciduous signal is expected to be rare and
spatially small within dense forest, not a wide patch; and (b) the NDVI
time series for all 5 years as before, with the year shown in (a) marked.

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
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
ZOOM_HALF = 15  # px each side -> 30x30 crop, ~143m x 143m at 4.78m/px -- tight enough to
                # resolve individual/small clusters of pixels, since a real deciduous
                # signature is expected to be rare and spatially small within dense forest,
                # not a broad neighborhood (see docs/amazon.md).

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


def crop_with_marker(ax, refl_month, rc, half=ZOOM_HALF, border_color=None):
    """Show a tight RGB zoom around `rc` (observed reflectance for one
    month), with the exact pixel marked -- ground-truths what the NDVI
    curve is actually looking at, and whether this month happened to be
    clear or cloudy there. `border_color`, if given, flags confidence
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


def best_year(weight_final, rc):
    """Year (of YEARS) with the most confidence-weighted clear coverage at
    this pixel -- the year the monthly RGB grid should actually show, so
    it isn't dominated by cloud."""
    r, c = rc
    w = weight_final[:, r, c].reshape(len(YEARS), 12)
    totals = w.sum(axis=1)
    return int(np.argmax(totals))


def plot_pixel(name, rc, ndvi_raw, ndvi_masked, ndvi_recon, weight_final, refl, out_path):
    r, c = rc
    by = best_year(weight_final, rc)
    by_year = YEARS[by]

    fig = plt.figure(figsize=(20, 12))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.5, 1], hspace=0.5, top=0.86, bottom=0.06)
    rgb_gs = outer[0].subgridspec(2, 6, wspace=0.08, hspace=0.3)
    ndvi_gs = outer[1].subgridspec(1, len(YEARS), wspace=0.12)

    # ---- top block: all 12 months of the single best-observed year, tightly zoomed ----
    w_months = weight_final[by * 12:(by + 1) * 12, r, c]
    for m in range(12):
        ax = fig.add_subplot(rgb_gs[m // 6, m % 6])
        border = "#2ca02c" if w_months[m] >= 0.7 else "#999999"
        crop_with_marker(ax, refl[by * 12 + m], rc, border_color=border)
        ax.set_title(MONTH_NAMES[m], fontsize=10)

    # ---- bottom block: NDVI, one panel per year, as before ----
    months_x = np.arange(1, 13)
    y_raw = ndvi_raw[:, r, c].reshape(len(YEARS), 12)
    y_masked = ndvi_masked[:, r, c].reshape(len(YEARS), 12)
    y_recon = ndvi_recon[:, r, c].reshape(len(YEARS), 12)

    ndvi_axes = [fig.add_subplot(ndvi_gs[0, i]) for i in range(len(YEARS))]
    for i, year in enumerate(YEARS):
        ax = ndvi_axes[i]
        ax.plot(months_x, y_raw[i], "-", color=COLOR_RAW, lw=1.3, alpha=0.8,
                label="raw (observed)" if i == 0 else None, zorder=2)
        ax.plot(months_x, y_recon[i], "-", color=COLOR_RECON, lw=2.2,
                label="reconstructed" if i == 0 else None, zorder=3)
        ax.scatter(months_x, y_masked[i], marker="o", s=34, facecolor=COLOR_MASKED,
                   edgecolor="k", linewidth=0.5, zorder=4,
                   label="masked (clear obs. only)" if i == 0 else None)
        title_weight = "bold" if year == by_year else "normal"
        ax.set_title(f"{year}" + (" *" if year == by_year else ""), fontsize=11,
                     fontweight=title_weight)
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
              f"pixel among the 5 years) -- green border = confidently clear, gray = not.",
              ha="center", fontsize=11)
    fig.text(0.5, 0.925,
              f"Bottom: monthly NDVI, all 5 years ({by_year}, outlined in green, is the year "
              f"shown above).", ha="center", fontsize=11)
    fig.suptitle(f"{name.replace('_', ' ')} — pixel (row={r}, col={c})", y=0.975, fontsize=15)
    plt.savefig(out_path, dpi=130)
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