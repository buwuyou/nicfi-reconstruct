"""
Step 1: OmniCloudMask (2-generation ensemble, see src/cloud_mask.py) on every
raw single-date Sentinel-2 frame of a tile. Caches the raw OCM classes per
frame (cache/s2_masks/<date>.npz) so buffer/threshold choices downstream can
change without re-running the model. Resumable: skips cached frames.

Run: python scripts/amazon_nicfis2/01_cloudmask_s2.py --tile D17 [--months 2023]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.nicfis2 import config as cfg, s2_composite


def main():
    ap = cfg.add_tile_args(argparse.ArgumentParser())
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    tile = cfg.tile_from_args(args)

    frames = [f for m in tile.months for f in tile.s2_frames(m)]
    print(f"{tile.tile_id}: {len(frames)} Sentinel-2 frames over {len(tile.months)} months")
    t0, n_done = time.time(), 0
    for f in frames:
        out = s2_composite.mask_cache_path(tile, f)
        if out.exists():
            continue
        data, _, _, names = s2_composite.read_frame(f)
        classes = s2_composite.ocm_classes(data, names, device=args.device)
        np.savez_compressed(out, classes=classes)
        n_done += 1
        print(f"  {f.stem}: clear(buffered)={s2_composite.clear_mask(classes).mean():.1%}")
    print(f"DONE: {n_done} frames masked in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
