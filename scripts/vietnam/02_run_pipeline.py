"""
Step 2: run the full reconstruction pipeline on the cached AOI stack
(produced by 01_prepare_data.py) and generate all deliverables:
  - reconstructed monthly GeoTIFFs (outputs/vietnam/reconstructed/)
  - before/after + phenology-curve + confidence figures (outputs/vietnam/figures/)

Assumes outputs/vietnam/cache/aoi_stack.npz, phenology_recon.npz, and
refined.npz already exist (from 01_prepare_data.py and the refiner
training/phenology steps).

Run: python scripts/vietnam/02_run_pipeline.py
"""
import sys
from pathlib import Path

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import sites, cloud_mask, compose, io_utils, visualize

site = sites.get_site("vietnam")


def main():
    d = np.load(site.cache_dir / "aoi_stack.npz")
    data, quality, months = d["data"], d["quality"], list(d["months"])
    transform = rasterio.Affine.from_gdal(*d["transform"])
    crs = str(d["crs"])

    refl = io_utils.to_reflectance(data)
    weight = cloud_mask.quality_weight(quality)

    phen = np.load(site.cache_dir / "phenology_recon.npz")
    recon_curve, confidence = phen["recon"], phen["conf"]

    refined = np.load(site.cache_dir / "refined.npz")["refined"]

    composite, alpha_used = compose.compose(refl, weight, refined)

    # ---- export GeoTIFFs (scaled back to int16 DN, same convention as source) ----
    print("Exporting reconstructed monthly GeoTIFFs...")
    for i, m in enumerate(months):
        out = (composite[i] * sites.REFLECTANCE_SCALE).clip(0, 32000).astype(np.int16)
        io_utils.save_geotiff(site.recon_dir / f"nicfi_{m}_{site.tile_id}_reconstructed.tif",
                               out, transform, crs)

    # ---- figures ----
    print("Rendering figures...")
    visualize.plot_month_grid(refl, months, "Observed (raw monthly NICFI mosaics)",
                               site.fig_dir / "01_observed_grid.png")
    visualize.plot_month_grid(composite, months, "Reconstructed cloud-free composites",
                               site.fig_dir / "02_reconstructed_grid.png")
    visualize.plot_before_after(refl, composite, months,
                                 site.fig_dir / "03_before_after_full.png", quality=quality)
    visualize.plot_confidence_map(confidence[sites.NIR],
                                   site.fig_dir / "04_confidence_map.png")

    # sample pixels, chosen data-drivenly: lowest/highest-NDVI KMeans cluster
    # centroids (cropland-like vs. forest-like land cover) and the single
    # pixel with the highest cloud frequency across the year (6/12 months
    # flagged), to show the phenology+prior reconstruction under stress.
    sample_pixels = {
        "cropland_like": (903, 1174),
        "forest_like": (1100, 1171),
        "persistent_cloud_zone": (822, 175),
    }
    for name, rc in sample_pixels.items():
        visualize.plot_pixel_phenology(refl, weight, recon_curve, composite, months, rc,
                                        site.fig_dir / f"05_phenology_{name}.png")

    print("Done.")
    print(f"  GeoTIFFs -> {site.recon_dir}")
    print(f"  Figures  -> {site.fig_dir}")


if __name__ == "__main__":
    main()