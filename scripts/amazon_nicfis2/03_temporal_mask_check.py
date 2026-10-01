"""
Step 3: temporal post-check of both cloud-mask series (src/nicfis2/
temporal_mask.py) -- a spot flagged in most otherwise-clear observations,
and looking the same each time, is ground, not cloud. Adds `refined` and
`overridden` arrays next to the raw OCM classes in each cached mask file
(steps 4-6 read `refined`), plus cache/temporal_check_{nicfi,s2}.npz with
the per-pixel flag frequency and persistent-pixel map for QA.

Run: python scripts/amazon_nicfis2/03_temporal_mask_check.py --tile D17
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import io_utils
from src.nicfis2 import config as cfg, s2_composite, temporal_mask


def run(name, paths, key, load_blue_nir, out_path):
    classes = np.stack([np.load(p)[key] for p in paths])
    blue, nir = zip(*(load_blue_nir(i) for i in range(len(paths))))
    res = temporal_mask.recheck(classes, np.stack(blue), np.stack(nir))
    for i, p in enumerate(paths):
        z = dict(np.load(p))
        z.update(refined=res.refined[i], overridden=res.overridden[i])
        np.savez_compressed(p, **z)
    np.savez_compressed(out_path, flag_freq=res.flag_freq, persistent=res.persistent,
                        mostly_clear=res.mostly_clear, names=np.array([p.stem for p in paths]))
    dirty = np.isin(classes, temporal_mask.CONTAMINATED)
    print(f"{name}: {len(paths)} obs ({res.mostly_clear.sum()} mostly clear), persistent px "
          f"{res.persistent.mean():.2%}, flags overridden {res.overridden.sum() / max(dirty.sum(), 1):.2%} "
          f"of all flags ({res.overridden.sum():,} px-obs)")


def main():
    ap = cfg.add_tile_args(argparse.ArgumentParser())
    args = ap.parse_args()
    tile = cfg.tile_from_args(args)
    t0 = time.time()

    months = tile.months
    nicfi_paths = [tile.cache_dir / f"nicfi_quality_{m}.npz" for m in months]

    def nicfi_bn(i):
        a, _, _ = io_utils.read_full(tile.nicfi_path(months[i]))
        return a[0].astype(np.uint16), a[3].astype(np.uint16)
    run("NICFI", nicfi_paths, "quality", nicfi_bn, tile.cache_dir / "temporal_check_nicfi.npz")

    frames = [f for m in months for f in tile.s2_frames(m)]
    s2_paths = [s2_composite.mask_cache_path(tile, f) for f in frames]

    def s2_bn(i):
        with rasterio.open(frames[i]) as src:
            bi = s2_composite.band_index(list(src.descriptions))
            return src.read(bi["B2"] + 1), src.read(bi["B8"] + 1)
    run("Sentinel-2", s2_paths, "classes", s2_bn, tile.cache_dir / "temporal_check_s2.npz")
    print(f"DONE in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
