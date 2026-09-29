"""
Unbiased complement to 04_deciduous_ndvi.py: instead of selecting pixels by
NDVI seasonal amplitude (which turned out to be confounded with proximity
to the forest/non-forest boundary -- see docs/amazon.md), randomly sample
10 pixels from *dense forest interior* and plot the same monthly-RGB +
5-year-NDVI figure for each, with no cherry-picking at all.

"Dense forest interior" here specifically means *not near an edge*: plain
mean-NDVI > 0.6 (as in 04_deciduous_ndvi.py) still allows a pixel one step
from the tan/beige boundary that biased the amplitude-based picks. This
script instead requires the *entire* zoom-crop neighborhood (30x30px, the
same window the RGB figure will show) to be forest, via binary erosion --
i.e. a pixel only qualifies if there is no non-forest pixel anywhere within
that neighborhood. That directly rules out the edge-proximity confound
found earlier, by construction rather than by inspection.

Run: python scripts/amazon/05_random_forest_pixels.py
"""
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_erosion
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import sites, io_utils, compose
from src.ndvi_diagnostics import ndvi, plot_pixel_ndvi_rgb

site = sites.get_site("amazon")
YEARS = list(range(2021, 2026))
ZOOM_HALF = 15
N_SAMPLES = 10
SEED = 0


def select_random_interior_forest_pixels(ndvi_raw, weight_final, n=N_SAMPLES, seed=SEED):
    conf = weight_final >= 0.7
    n_conf = conf.sum(axis=0)
    ndvi_conf_masked = np.where(conf, ndvi_raw, np.nan)
    mean_ndvi_conf = np.nanmean(ndvi_conf_masked, axis=0)

    forest = (mean_ndvi_conf > 0.6) & (n_conf >= 20)

    # require the whole zoom-crop neighborhood to be forest -- rules out
    # the edge-proximity confound by construction, not just by threshold
    win = 2 * ZOOM_HALF + 1
    interior_forest = binary_erosion(forest, structure=np.ones((win, win)), border_value=0)

    reliable = interior_forest & (n_conf >= 45)
    ys, xs = np.where(reliable)
    print(f"  candidate pool: {reliable.sum()} px (interior forest + >=45/60 confidently-clear "
          f"months), out of {forest.sum()} plain-forest px and {ndvi_raw.shape[1]*ndvi_raw.shape[2]} total")

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ys), size=n, replace=False)
    picks = {f"random_{i+1:02d}": (int(ys[j]), int(xs[j])) for i, j in enumerate(idx)}
    return picks, mean_ndvi_conf, forest, interior_forest


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

    print(f"Randomly sampling {N_SAMPLES} interior-forest pixels (seed={SEED}, no amplitude used)...")
    picks, mean_ndvi_conf, forest, interior_forest = select_random_interior_forest_pixels(
        ndvi_raw, weight_final)
    for name, rc in picks.items():
        print(f"  {name}: pixel {rc}")

    for i, (name, rc) in enumerate(picks.items(), start=14):
        out_path = site.fig_dir / f"{i}_ndvi_{name}.png"
        plot_pixel_ndvi_rgb(name.replace("_", " "), rc, YEARS, ndvi_raw, ndvi_masked, ndvi_recon,
                             weight_final, refl, out_path, zoom_half=ZOOM_HALF,
                             caption_extra=" Randomly sampled, interior forest, no amplitude selection.")
        print(f"  saved {out_path}")

    # ---- overview: where did the random sample land? ----
    fig, ax = plt.subplots(figsize=(8, 7.5))
    im = ax.imshow(np.clip(mean_ndvi_conf, 0, 1), cmap="YlGn", vmin=0.2, vmax=0.9)
    ax.set_title(f"Mean NDVI, with {N_SAMPLES} random interior-forest samples")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)
    for name, (r, c) in picks.items():
        ax.scatter([c], [r], s=90, marker="o", facecolor="none", edgecolor="red", linewidth=1.8)
        ax.annotate(name.replace("random_", "#"), (c, r), color="red", fontsize=8,
                     ha="left", va="bottom", xytext=(4, 4), textcoords="offset points")
    plt.tight_layout()
    plt.savefig(site.fig_dir / "24_random_pixel_locations.png", dpi=130)
    plt.close(fig)

    print("Done.")


if __name__ == "__main__":
    main()