"""
Step 4: annual composite, computed two ways for comparison --
  (a) naive: standard percentile from raw confidently-clear observations only
      (the common approach, and its uneven-sample-count problem)
  (b) robust: percentile from the reconstructed monthly stack, weighted so
      every pixel gets a full, comparable sample depth (src/annual_composite.py)
plus a band-consistent medoid version of (b).

Uses the improved (v2/v3) mask + reconstruction from 03_improve_masking.py.

Run: python scripts/vietnam/04_annual_composite.py
"""
import sys
from pathlib import Path

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import annual_composite as ac
from src import compose, sites, io_utils, visualize

site = sites.get_site("vietnam")


def main():
    d = np.load(site.cache_dir / "aoi_stack.npz")
    data, months = d["data"], list(d["months"])
    transform = rasterio.Affine.from_gdal(*d["transform"])
    crs = str(d["crs"])
    refl = io_utils.to_reflectance(data)

    # NOTE: use the actual continuous confidence weight saved by
    # 03_improve_masking.py (robust_mask.continuous_confidence), not a
    # recomputation via cloud_mask.quality_weight -- that function only
    # knows about the discrete categorical classes and would silently
    # discard the continuous severity scoring + categorical-cap fix,
    # reintroducing the stale weighting this pipeline moved away from.
    dv2 = np.load(site.cache_dir / "aoi_stack_v2.npz")
    weight_v3 = dv2["weight_v3"]

    refined = np.load(site.cache_dir / "refined_v3.npz")["refined"]
    composite, alpha_used = compose.compose(refl, weight_v3, refined)

    print("Computing naive annual median (standard approach)...")
    naive_med, n_valid = ac.naive_composite(refl, weight_v3, percentile=50)
    print(f"  n_valid (confidently-clear months per pixel): min={n_valid.min():.0f} "
          f"p5={np.percentile(n_valid,5):.1f} median={np.median(n_valid):.0f} max={n_valid.max():.0f}")

    print("Computing reconstruction-informed robust annual median...")
    robust_med = ac.robust_composite(composite, alpha_used, percentile=50, recon_trust=0.35)

    print("Computing band-consistent medoid annual composite...")
    medoid, best_t = ac.medoid_composite(composite, alpha_used, target=robust_med, recon_trust=0.35)
    from collections import Counter
    print("  medoid month-of-year selection frequency:", dict(Counter(best_t.ravel().tolist())))

    # ---- export ----
    out_dir = site.annual_dir
    for name, arr in [("naive_median", np.nan_to_num(naive_med, nan=0.0)),
                       ("robust_median", robust_med),
                       ("medoid", medoid)]:
        out = (arr * sites.REFLECTANCE_SCALE).clip(0, 32000).astype(np.int16)
        io_utils.save_geotiff(out_dir / f"nicfi_2025_annual_{name}_{site.tile_id}.tif",
                               out, transform, crs)

    # ---- figures ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    axes[0].imshow(visualize.rgb_stretch(np.nan_to_num(naive_med, nan=0.0)))
    axes[0].set_title("Naive annual median\n(raw clear obs only)")
    axes[0].axis("off")
    axes[1].imshow(visualize.rgb_stretch(robust_med))
    axes[1].set_title("Robust annual median\n(reconstruction-weighted, n=12 everywhere)")
    axes[1].axis("off")
    axes[2].imshow(visualize.rgb_stretch(medoid))
    axes[2].set_title("Medoid annual composite\n(band-consistent, single real month per px)")
    axes[2].axis("off")
    im = axes[3].imshow(n_valid, cmap="viridis", vmin=0, vmax=12)
    axes[3].set_title("n confidently-clear months\nused by the naive composite")
    axes[3].axis("off")
    plt.colorbar(im, ax=axes[3], fraction=0.046)
    plt.tight_layout()
    plt.savefig(site.fig_dir / "10_annual_composite_compare.png", dpi=130)

    # zoomed panel on a low-n_valid region (where the naive approach is weakest)
    y0, x0, s = 0, 0, 700
    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    axes[0].imshow(visualize.rgb_stretch(np.nan_to_num(naive_med, nan=0.0)[:, y0:y0+s, x0:x0+s]))
    axes[0].set_title("Naive median (zoom, low-n_valid corner)")
    axes[0].axis("off")
    axes[1].imshow(visualize.rgb_stretch(robust_med[:, y0:y0+s, x0:x0+s]))
    axes[1].set_title("Robust median (zoom)")
    axes[1].axis("off")
    im = axes[2].imshow(n_valid[y0:y0+s, x0:x0+s], cmap="viridis", vmin=0, vmax=12)
    axes[2].set_title("n_valid (zoom)")
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046)
    plt.tight_layout()
    plt.savefig(site.fig_dir / "11_annual_composite_zoom.png", dpi=130)

    print(f"Done. GeoTIFFs -> {out_dir}, figures -> {site.fig_dir}")


if __name__ == "__main__":
    main()