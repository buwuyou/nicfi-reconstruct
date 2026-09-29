"""
Process the Amazon (D01) 5-year monthly series (2021-01 to 2025-12, 60
months) through the full validated pipeline: ensembled cloud/shadow/haze
detection -> land-cover cluster priors -> continuous temporal-residual
confidence scoring -> phenology reconstruction -> self-supervised DL
spatial refinement -> composition.

The tile (2096x2109px) is already ~AOI scale, so the whole tile is
processed, no sub-window cropping.

Run: python scripts/amazon/01_process.py
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import sites, io_utils, cloud_mask, phenology, robust_mask, inpaint, compose

site = sites.get_site("amazon")


def main():
    t_start = time.time()

    print(f"Reading full-tile stack: {site.tile_id}, {len(site.months)} months "
          f"({site.months[0]} to {site.months[-1]})")
    t0 = time.time()
    stack = io_utils.read_aoi_stack(site)
    print(f"  stack shape (T,C,H,W) = {stack.data.shape}, "
          f"{stack.data.nbytes/1e9:.2f} GB, {time.time()-t0:.1f}s")
    refl = io_utils.to_reflectance(stack.data)

    print("Running ensembled OmniCloudMask (model versions 3.0 + 4.0) across 60 months "
          "-- this is the slow step, expect ~60-70 min on CPU...")
    t0 = time.time()
    quality, disagreement = cloud_mask.compute_masks(refl, device="cpu", model_versions=(3.0, 4.0))
    print(f"  done in {time.time()-t0:.1f}s")
    weight = cloud_mask.quality_weight(quality, disagreement)

    for t, m in enumerate(stack.months):
        if t % 6 == 0:  # print every 6th month to keep the log readable
            codes, counts = np.unique(quality[t], return_counts=True)
            frac = {int(c): round(float(n) / quality[t].size, 3) for c, n in zip(codes, counts)}
            print(f"  {m}: {frac}  (0=clear,1=thick,2=thin,3=shadow,4=haze,5=nodata)")

    np.savez_compressed(site.cache_dir / "stack.npz",
                         quality=quality.astype(np.uint8), disagreement=disagreement.astype(np.float32))

    print("Fitting land-cover cluster priors...")
    t0 = time.time()
    cluster_labels, cluster_beta = phenology.fit_cluster_priors(refl, weight, n_clusters=10)
    print(f"  done in {time.time()-t0:.1f}s, cluster sizes: {np.bincount(cluster_labels.ravel())}")

    print("Continuous temporal-residual confidence scoring...")
    t0 = time.time()
    weight_final, severity, recon_curve, confidence = robust_mask.continuous_confidence(
        refl, weight, quality, cluster_labels, cluster_beta, severity50=6.0, power=2.5,
        n_iters=2, target_weight=10.0)
    print(f"  done in {time.time()-t0:.1f}s")

    np.savez_compressed(site.cache_dir / "phenology_recon.npz",
                         recon=recon_curve.astype(np.float32), conf=confidence.astype(np.float32),
                         labels=cluster_labels.astype(np.int32), weight_final=weight_final.astype(np.float32),
                         severity=severity.astype(np.float32))

    print("Training DL spatial refiner (self-supervised, this scene only)...")
    t0 = time.time()
    model = inpaint.train_refiner(refl, weight_final, recon_curve, n_iters=800, patch=192, batch=6,
                                   lr=2e-3, log_every=200, seed=0)
    print(f"  train time {time.time()-t0:.1f}s")
    import torch
    torch.save(model.state_dict(), site.cache_dir / "refiner.pt")

    print("Applying refiner across all 60 months...")
    t0 = time.time()
    refined = inpaint.apply_refiner(model, refl, weight_final, recon_curve, tile=512, overlap=64)
    print(f"  apply time {time.time()-t0:.1f}s")
    np.savez_compressed(site.cache_dir / "refined.npz", refined=refined.astype(np.float32))

    print("Composing final monthly composites and exporting GeoTIFFs...")
    composite, alpha_used = compose.compose(refl, weight_final, refined)
    for i, m in enumerate(stack.months):
        out = (composite[i] * sites.REFLECTANCE_SCALE).clip(0, 32000).astype(np.int16)
        io_utils.save_geotiff(site.recon_dir / f"{site.tile_id}_{m}_reconstructed.tif",
                               out, stack.transform, stack.crs)
    np.savez_compressed(site.cache_dir / "alpha_used.npz", alpha_used=alpha_used.astype(np.float32))

    print(f"ALL DONE in {time.time()-t_start:.1f}s total ({(time.time()-t_start)/60:.1f} min)")
    print(f"  GeoTIFFs -> {site.recon_dir}")


if __name__ == "__main__":
    main()