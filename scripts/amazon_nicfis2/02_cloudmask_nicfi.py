"""
Step 2: cloud/shadow/haze quality codes for every NICFI month (the same OCM
ensemble + haze heuristic as the temporal pipeline, src/cloud_mask.py).
Caches cache/nicfi_quality_<month>.npz. Resumable.

Run: python scripts/amazon_nicfis2/02_cloudmask_nicfi.py --tile D17
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import cloud_mask, io_utils
from src.nicfis2 import config as cfg, reconstruct


def main():
    ap = cfg.add_tile_args(argparse.ArgumentParser())
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    tile = cfg.tile_from_args(args)

    t0 = time.time()
    for month in tile.months:
        out = tile.cache_dir / f"nicfi_quality_{month}.npz"
        if out.exists():
            continue
        nicfi, _, _ = io_utils.read_full(tile.nicfi_path(month))
        q = reconstruct.nicfi_quality(nicfi, device=args.device)
        np.savez_compressed(out, quality=q)
        frac = {n: (q == c).mean() for n, c in (("thick", cloud_mask.CLOUD_THICK),
                ("thin", cloud_mask.CLOUD_THIN), ("shadow", cloud_mask.SHADOW),
                ("haze", cloud_mask.HAZE), ("nodata", cloud_mask.NODATA))}
        print(f"  {month}: " + " ".join(f"{k}={v:.1%}" for k, v in frac.items()))
    print(f"DONE in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
