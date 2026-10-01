"""
Assemble the final 10-band monthly product for D02: harmonized NICFI
RGBN where clear, feather-blended super-resolved Sentinel-2 filling NICFI's
gaps, and super-resolved Sentinel-2 for the 6 bands NICFI doesn't have at
all. Writes one 10-band GeoTIFF + one fill-provenance GeoTIFF per month to
outputs/amazon_d02/reconstructed/. Depends on 01_prepare.py, 02_harmonize.py,
and (for months with Sentinel-2 coverage) 03_superresolve.py.

Run: python scripts/amazon_d02/04_fuse.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import fuse, harmonize, io_utils, s2_fusion_site as cfg

site = cfg.D02_TEST_YEAR  # scoped to one clean year for now, see s2_fusion_site.py


def load_sr_cache(month: str):
    path = site.cache_dir / f"sr_{month}.npz"
    if not path.exists():
        return None
    z = np.load(path, allow_pickle=False)
    transform = rasterio.Affine(*z["transform"])
    crs = rasterio.crs.CRS.from_string(str(z["crs"]))
    return z["data"], transform, crs


def load_s2_valid(month: str):
    z = np.load(site.cache_dir / f"s2_mosaic_{month}.npz", allow_pickle=False)
    return z["valid"], rasterio.Affine(*z["transform"]), rasterio.crs.CRS.from_string(str(z["crs"]))


def load_harmonization_params_by_year() -> dict:
    """Returns {year:int -> HarmonizationParams} -- one fit per calendar year,
    not pooled across years (see src/harmonize.py's docstring for why)."""
    with open(site.cache_dir / "harmonization_params.json") as f:
        raw = json.load(f)
    out = {}
    for year_str, v in raw.items():
        coeffs = {band: tuple(c) for band, c in v["coeffs"].items()}
        out[int(year_str)] = harmonize.HarmonizationParams(coeffs=coeffs, n_samples=v["n_samples"])
    return out


def main():
    t_start = time.time()
    params_by_year = load_harmonization_params_by_year()

    n_sr_filled = n_no_sr_at_all = 0
    for month in site.months:
        year = int(month[:4])
        if year not in params_by_year:
            print(f"  {month}: no harmonization fit for {year}, skipping")
            continue
        nicfi_native, nicfi_transform, nicfi_crs = io_utils.read_full(site.nicfi_path(month))
        harmonized = harmonize.apply(nicfi_native, params_by_year[year])

        sr_cached = load_sr_cache(month)
        if sr_cached is None:
            sr_on_nicfi_grid = sr_valid = None
            n_no_sr_at_all += 1
        else:
            sr_data, sr_transform, sr_crs = sr_cached
            sr_on_nicfi_grid = fuse.reproject_sr_to_nicfi_grid(
                sr_data, sr_transform, sr_crs,
                harmonized.shape[1:], nicfi_transform, nicfi_crs,
            )
            s2_valid, s2_transform, s2_crs = load_s2_valid(month)
            sr_valid = fuse.reproject_valid_to_nicfi_grid(
                s2_valid, s2_transform, s2_crs,
                harmonized.shape[1:], nicfi_transform, nicfi_crs,
            )
            n_sr_filled += 1

        result = fuse.fuse_month(harmonized, sr_on_nicfi_grid, sr_valid)

        io_utils.save_geotiff(
            site.recon_dir / f"{site.tile_id}_{month}_fused.tif",
            result.fused.clip(0, 65000).astype("uint16"), nicfi_transform, nicfi_crs,
            dtype="uint16", nodata=0,
        )
        io_utils.save_geotiff(
            site.recon_dir / f"{site.tile_id}_{month}_fillmask.tif",
            result.fill_mask[None], nicfi_transform, nicfi_crs, dtype="uint8",
        )

        gap_frac = float(np.all(harmonized == 0, axis=0).mean())
        unfilled_frac = float((result.fill_mask == fuse.FILL_UNFILLED).mean())
        print(f"  {month}: gap={gap_frac:.2%}, still-unfilled={unfilled_frac:.2%} "
              f"({'SR available' if sr_cached is not None else 'NO S2 this month'})")

    print(f"\nDONE in {time.time()-t_start:.1f}s. "
          f"{n_sr_filled} months had SR fill available, {n_no_sr_at_all} did not. "
          f"GeoTIFFs -> {site.recon_dir}")


if __name__ == "__main__":
    main()
