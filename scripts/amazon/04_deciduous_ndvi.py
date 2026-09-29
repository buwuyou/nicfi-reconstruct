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

IMPORTANT CAVEAT (see docs/amazon.md): this amplitude-based selection turns
out to be confounded with proximity to the forest/non-forest boundary --
the two highest-amplitude picks both sit at or near that boundary, and the
boundary shape is static across all 12 months, which looks more like a
land-cover edge than deciduous leaf-drop. See
scripts/amazon/05_random_forest_pixels.py for an unbiased check that
doesn't select by amplitude at all.

Each pixel's figure shows: (a) all 12 months of RGB for the single year
with the most confidently-clear observations at that pixel -- not just
January across years, since the point is to see the canopy actually change
month to month -- tightly zoomed (30x30px, ~143m) rather than a broad
neighborhood, since a real deciduous signal is expected to be rare and
spatially small within dense forest; and (b) the NDVI time series for all 5
years, with the year shown in (a) marked.

Run: python scripts/amazon/04_deciduous_ndvi.py
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import sites, io_utils, compose
from src.ndvi_diagnostics import ndvi, plot_pixel_ndvi_rgb

site = sites.get_site("amazon")
YEARS = list(range(2021, 2026))


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
        plot_pixel_ndvi_rgb(name.replace("_", " "), rc, YEARS, ndvi_raw, ndvi_masked, ndvi_recon,
                             weight_final, refl, out_path)
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