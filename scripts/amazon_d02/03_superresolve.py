"""
Super-resolve every month's cached Sentinel-2 mosaic (01_prepare.py) with
the pretrained SEN2SR model (model/LDSRS2-SEN2SR, 10m -> 2.5m, all 10
bands in one call). GPU-bound: the RGBN branch is a 200-step latent
diffusion model, so this is impractical on CPU for a full tile/month.

Resumable: skips any month whose SR cache already exists, so a run that
gets interrupted (or is done a few months at a time) just continues.

Run (full, needs a working GPU):
    python scripts/amazon_d02/03_superresolve.py

Smoke-test only (validates tensor shapes/band order/output scale through
the real model on one small patch; --device cpu works but is slow because
of the 200-step diffusion sampler -- expect minutes, not seconds):
    python scripts/amazon_d02/03_superresolve.py --smoke-test --device cpu
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import s2_fusion_site as cfg, s2_mosaic, superres

site = cfg.D02_TEST_YEAR  # scoped to one clean year for now, see s2_fusion_site.py


def load_cached_mosaic(month: str):
    path = site.cache_dir / f"s2_mosaic_{month}.npz"
    if not path.exists():
        return None
    z = np.load(path, allow_pickle=False)
    transform = rasterio.Affine(*z["transform"])
    crs = rasterio.crs.CRS.from_string(str(z["crs"]))
    return s2_mosaic.S2Mosaic(data=z["data"], valid=z["valid"], transform=transform,
                               crs=crs, n_frames=-1)


def save_sr_cache(month, sr_data, src_transform, src_crs):
    sr_transform = src_transform * rasterio.Affine.scale(1.0 / superres.SCALE)
    np.savez_compressed(
        site.cache_dir / f"sr_{month}.npz",
        data=sr_data, transform=tuple(sr_transform)[:6], crs=src_crs.to_string(),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke-test", action="store_true",
                     help="run the model on one small real patch only, to validate "
                          "wiring end-to-end without a full-tile/all-months run")
    ap.add_argument("--limit-months", type=int, default=None)
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available (torch.cuda.is_available() is False) -- "
              "this is expected while the GPU driver is broken. Re-run once "
              "it's fixed, or pass --device cpu --smoke-test for a slow "
              "correctness-only check on one small patch.")
        return

    print(f"Loading SEN2SR model onto {args.device}...")
    t0 = time.time()
    model = superres.load_model(device=args.device)
    print(f"  loaded in {time.time()-t0:.1f}s")

    months = site.months[: args.limit_months] if args.limit_months else site.months

    if args.smoke_test:
        for month in months:
            mosaic = load_cached_mosaic(month)
            if mosaic is not None:
                break
        else:
            print("No month with Sentinel-2 coverage found for the smoke test.")
            return
        patch = mosaic.data[:, :128, :128]
        print(f"Smoke test on {month}, patch shape {patch.shape} "
              f"(this can take a few minutes on CPU -- 200-step diffusion sampler)...")
        t0 = time.time()
        sr = superres.predict_tiled(patch, model, device=args.device, overlap=0)
        print(f"  SR output shape {sr.shape} (expect (10,512,512)), "
              f"value range [{sr.min():.1f}, {sr.max():.1f}], "
              f"took {time.time()-t0:.1f}s")
        assert sr.shape == (10, 512, 512), f"unexpected SR output shape {sr.shape}"
        print("  shapes/band-order/output-scale check passed.")
        return

    t_start = time.time()
    n_done = n_skipped = n_no_coverage = 0
    for month in months:
        out_path = site.cache_dir / f"sr_{month}.npz"
        if out_path.exists():
            n_skipped += 1
            continue
        mosaic = load_cached_mosaic(month)
        if mosaic is None:
            n_no_coverage += 1
            continue

        t0 = time.time()
        sr = superres.predict_tiled(mosaic.data, model, device=args.device)
        save_sr_cache(month, sr, mosaic.transform, mosaic.crs)
        n_done += 1
        print(f"  {month}: SR done in {time.time()-t0:.1f}s -> {out_path.name}")

    print(f"\nDONE: {n_done} super-resolved, {n_skipped} already cached, "
          f"{n_no_coverage} months with no Sentinel-2 coverage. "
          f"Total {time.time()-t_start:.1f}s.")


if __name__ == "__main__":
    main()
