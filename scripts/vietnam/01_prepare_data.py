"""
Step 1: read the 12-month AOI stack (bounded-memory windowed reads) and run
cloud/shadow/haze detection on each month. Caches results to disk so later
steps (which we'll iterate on) don't re-pay the ~4 min OmniCloudMask cost.

This intentionally uses a *single* OmniCloudMask model version (4.0) --
it's the baseline mask, kept simple on purpose so 03_improve_masking.py
(model ensembling + continuous confidence scoring) has something to compare
against and quantify the improvement over.

Run: python scripts/vietnam/01_prepare_data.py
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import sites, io_utils, cloud_mask

site = sites.get_site("vietnam")


def main():
    t0 = time.time()
    print(f"Reading AOI stack: tile {site.tile_id}, window "
          f"row_off={site.aoi_row_off} col_off={site.aoi_col_off} size={site.aoi_size}")
    stack = io_utils.read_aoi_stack(site)
    print(f"  stack shape (T,C,H,W) = {stack.data.shape}, "
          f"{stack.data.nbytes/1e6:.0f} MB, {time.time()-t0:.1f}s")

    refl = io_utils.to_reflectance(stack.data)

    print("Running OmniCloudMask per month (this is the slow step, ~15-20s/month on CPU)...")
    t1 = time.time()
    quality, _disagreement = cloud_mask.compute_masks(refl, device="cpu", model_versions=(4.0,))
    print(f"  done in {time.time()-t1:.1f}s")

    for t, m in enumerate(stack.months):
        codes, counts = np.unique(quality[t], return_counts=True)
        frac = {int(c): round(float(n) / quality[t].size, 3) for c, n in zip(codes, counts)}
        print(f"  {m}: {frac}  (0=clear,1=thick,2=thin,3=shadow,4=haze,5=nodata)")

    np.savez_compressed(
        site.cache_dir / "aoi_stack.npz",
        data=stack.data.astype(np.float32),
        quality=quality.astype(np.uint8),
        months=np.array(stack.months),
        transform=np.array(stack.transform.to_gdal()),
        crs=str(stack.crs),
    )
    print(f"Saved cache -> {site.cache_dir / 'aoi_stack.npz'}  total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
