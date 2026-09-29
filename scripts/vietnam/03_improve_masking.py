"""
Step 3: improved cloud/shadow/haze masking, applied on top of the existing
AOI stack.

Two additions over 01+02 (kept separate rather than editing those in
place, so the original run stays available for before/after comparison):
  1. OmniCloudMask model-version ensembling (src/cloud_mask.py)
  2. Continuous temporal-residual confidence / opacity scoring
     (src/robust_mask.py) -- replaces both the old fixed-weight haze class
     and a discrete temporal-outlier flag with one continuous trust score.

Then re-runs reconstruction -> refiner -> composition -> figures with the
improved mask, and produces a comparison figure quantifying what changed.

Run: python scripts/vietnam/03_improve_masking.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import sites, cloud_mask, compose, io_utils, phenology, robust_mask, visualize, inpaint

site = sites.get_site("vietnam")


def main():
    t_start = time.time()

    # ---- reuse the already-read pixel stack (no need to re-read from disk) ----
    d = np.load(site.cache_dir / "aoi_stack.npz")
    data, months = d["data"], list(d["months"])
    transform = rasterio.Affine.from_gdal(*d["transform"])
    crs = str(d["crs"])
    refl = io_utils.to_reflectance(data)

    # ---- 1. ensembled cloud masking ----
    print("Running ensembled OmniCloudMask (model versions 3.0 + 4.0)...")
    t0 = time.time()
    quality_v1 = d["quality"]  # baseline, single-model mask from script 01
    quality_v2, disagreement = cloud_mask.compute_masks(refl, device="cpu", model_versions=(3.0, 4.0))
    weight_v2 = cloud_mask.quality_weight(quality_v2, disagreement)
    print(f"  done in {time.time()-t0:.1f}s")

    changed = (quality_v1 == cloud_mask.CLEAR) != (quality_v2 == cloud_mask.CLEAR)
    print(f"  ensembling changed clear/not-clear call on {changed.sum()} px-months "
          f"({100*changed.mean():.4f}%) vs. the single-model (v4 only) baseline")

    # ---- 2. cluster priors + temporal-residual robust refinement ----
    print("Fitting land-cover cluster priors...")
    cluster_labels, cluster_beta = phenology.fit_cluster_priors(refl, weight_v2, n_clusters=10)

    print("Continuous temporal-residual confidence scoring...")
    weight_v3, severity, recon_curve, confidence = robust_mask.continuous_confidence(
        refl, weight_v2, quality_v2, cluster_labels, cluster_beta, severity50=6.0, power=2.5, n_iters=2)

    np.savez_compressed(
        site.cache_dir / "aoi_stack_v2.npz",
        quality=quality_v2.astype(np.uint8), disagreement=disagreement.astype(np.float32),
        weight_v3=weight_v3.astype(np.float32), severity=severity.astype(np.float32),
    )
    np.savez_compressed(
        site.cache_dir / "phenology_recon_v2.npz",
        recon=recon_curve.astype(np.float32), conf=confidence.astype(np.float32),
        labels=cluster_labels.astype(np.int32),
    )

    # ---- 3. retrain spatial refiner on the improved mask/curve ----
    print("Retraining DL spatial refiner on improved mask...")
    t0 = time.time()
    model = inpaint.train_refiner(refl, weight_v3, recon_curve, n_iters=800, patch=192, batch=6,
                                   lr=2e-3, log_every=200, seed=0)
    print(f"  train time {time.time()-t0:.1f}s")
    refined = inpaint.apply_refiner(model, refl, weight_v3, recon_curve, tile=512, overlap=64)
    np.savez_compressed(site.cache_dir / "refined_v3.npz", refined=refined.astype(np.float32))

    # ---- 4. compose + export ----
    composite, alpha_used = compose.compose(refl, weight_v3, refined)
    recon_dir_v2 = site.out_root / "reconstructed_v2"
    recon_dir_v2.mkdir(exist_ok=True)
    for i, m in enumerate(months):
        out = (composite[i] * sites.REFLECTANCE_SCALE).clip(0, 32000).astype(np.int16)
        io_utils.save_geotiff(recon_dir_v2 / f"nicfi_{m}_{site.tile_id}_reconstructed_v2.tif",
                               out, transform, crs)

    # ---- 5. figures: what did the improved masking actually catch? ----
    print("Rendering comparison figures...")
    was_clear_v1 = quality_v1 == cloud_mask.CLEAR
    newly_downweighted_per_month = [
        int((was_clear_v1[i] & (weight_v3[i] < 0.5)).sum())
        for i in range(len(months))
    ]
    for m, n in zip(months, newly_downweighted_per_month):
        print(f"  {m}: {n} px newly down-weighted below 0.5 trust (was unconditionally 'clear' in v1)")

    visualize.plot_before_after(refl, composite, months,
                                 site.fig_dir / "06_before_after_v2.png", quality=quality_v2)
    visualize.plot_confidence_map(confidence[sites.NIR],
                                   site.fig_dir / "07_confidence_map_v2.png")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(range(len(months)), newly_downweighted_per_month, color="#e5383b")
    ax.set_xticks(range(len(months)))
    ax.set_xticklabels([m[:7] for m in months], rotation=45, ha="right")
    ax.set_ylabel("pixels newly down-weighted below 0.5 trust")
    ax.set_title("Contamination missed by the single-model baseline, caught by "
                  "ensembling + continuous temporal-residual confidence")
    plt.tight_layout()
    plt.savefig(site.fig_dir / "08_masking_improvement.png", dpi=130)

    # ---- 6. severity calibration figure: does severity actually separate
    # confirmed-haze pixels from the genuinely-clear population? ----
    clear_severity = severity[quality_v2 == cloud_mask.CLEAR]
    sample = np.random.default_rng(0).choice(clear_severity, size=min(500_000, clear_severity.size),
                                              replace=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].hist(sample, bins=100, range=(0, 15), color="#2274a5", alpha=0.8)
    for p in [99, 99.9]:
        axes[0].axvline(np.percentile(clear_severity, p), color="gray", ls="--", lw=1)
        axes[0].text(np.percentile(clear_severity, p), axes[0].get_ylim()[1]*0.9, f"p{p}",
                     rotation=90, va="top", fontsize=8)
    axes[0].axvline(6.0, color="#e5383b", lw=1.5, label="severity50=6.0")
    axes[0].set_xlabel("severity (clear-only population)")
    axes[0].set_title("Calibration: severity distribution of genuinely-clear px\n(vs. confirmed haze"
                       " residual, median 62-80 -- far off this axis)")
    axes[0].legend()

    sep_idx = months.index("2025-09-01")
    im = axes[1].imshow(severity[sep_idx], cmap="inferno", vmin=0, vmax=15)
    axes[1].set_title("Sept 2025 severity map")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046)
    plt.tight_layout()
    plt.savefig(site.fig_dir / "13_severity_calibration.png", dpi=130)

    print(f"Done in {time.time()-t_start:.1f}s total.")
    print(f"  GeoTIFFs -> {recon_dir_v2}")
    print(f"  Figures  -> {site.fig_dir}")


if __name__ == "__main__":
    main()
