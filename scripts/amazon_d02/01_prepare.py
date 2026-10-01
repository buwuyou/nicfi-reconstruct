"""
Build and cache the monthly Sentinel-2 mosaics for the D02 fusion pipeline
(one per-pixel-median mosaic per month, from however many D02_S2 frames
exist that month -- 0 to 8 in the data on disk), and sanity-check the NICFI
D02 tile against what gdalinfo showed directly (grid size, CRS, pixel
size). No GPU needed for this step.

Run: python scripts/amazon_d02/01_prepare.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import io_utils, s2_fusion_site as cfg, s2_mosaic

site = cfg.D02_TEST_YEAR  # scoped to one clean year for now, see s2_fusion_site.py


def main():
    t_start = time.time()

    print(f"Checking NICFI D02 tile grid ({site.months[0]}..{site.months[-1]}, "
          f"{len(site.months)} months)...")
    nicfi0, nicfi_transform, nicfi_crs = io_utils.read_full(site.nicfi_path(site.months[0]))
    print(f"  shape (C,H,W)={nicfi0.shape}, crs={nicfi_crs}, "
          f"px=({nicfi_transform.a:.6f}, {-nicfi_transform.e:.6f})")

    print("Building monthly Sentinel-2 mosaics (median across valid frames)...")
    coverage = {}
    s2_transform = s2_crs = s2_shape = None
    for month in site.months:
        t0 = time.time()
        mosaic = s2_mosaic.monthly_mosaic(site.s2_dir, month)
        if mosaic is None:
            coverage[month] = {"n_frames": 0, "valid_frac": 0.0}
            print(f"  {month}: 0 frames -- no Sentinel-2 coverage this month")
            continue

        if s2_transform is None:
            s2_transform, s2_crs, s2_shape = mosaic.transform, mosaic.crs, mosaic.data.shape[1:]

        np.savez_compressed(
            site.cache_dir / f"s2_mosaic_{month}.npz",
            data=mosaic.data, valid=mosaic.valid,
            transform=tuple(mosaic.transform)[:6], crs=mosaic.crs.to_string(),
        )
        valid_frac = float(mosaic.valid.mean())
        coverage[month] = {"n_frames": mosaic.n_frames, "valid_frac": round(valid_frac, 4)}
        print(f"  {month}: {mosaic.n_frames} frame(s), {valid_frac:.1%} valid "
              f"({time.time()-t0:.1f}s)")

    n_zero = sum(1 for v in coverage.values() if v["n_frames"] == 0)
    print(f"\n{n_zero}/{len(site.months)} months have zero Sentinel-2 coverage "
          f"(those NICFI gaps will stay unfilled in 04_fuse.py).")

    with open(site.cache_dir / "s2_coverage.json", "w") as f:
        json.dump(coverage, f, indent=2)

    print(f"\nDONE in {time.time()-t_start:.1f}s. Cache -> {site.cache_dir}")


if __name__ == "__main__":
    main()
