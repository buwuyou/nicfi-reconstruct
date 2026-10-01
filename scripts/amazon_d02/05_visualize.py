"""
QA figures for the D02 NICFI+Sentinel-2 fusion pipeline:
  1. Per-month gap coverage vs. fill outcome (native/SR-filled/still-unfilled).
  2. Before/after RGB for the month with the largest NICFI gap that has
     Sentinel-2 coverage.
  3. Hard-constraint consistency check: downsample the SR output back to
     10m and compare against the real Sentinel-2 pixel it came from -- this
     checks that the model's built-in low-hallucination guardrail is
     actually holding, it doesn't add a new one.
  4. A false-color composite using the red-edge/SWIR bands NICFI never had,
     for the same month.

Run: python scripts/amazon_d02/05_visualize.py
"""
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import fuse, io_utils, s2_fusion_site as cfg

site = cfg.D02_TEST_YEAR  # scoped to one clean year for now, see s2_fusion_site.py


def stretch(rgb, lo=2, hi=98):
    out = np.empty_like(rgb, dtype=np.float32)
    for i in range(rgb.shape[-1]):
        band = rgb[..., i]
        valid = band[band > 0]
        if valid.size == 0:
            out[..., i] = 0
            continue
        p_lo, p_hi = np.percentile(valid, [lo, hi])
        out[..., i] = np.clip((band - p_lo) / max(p_hi - p_lo, 1e-6), 0, 1)
    return out


def load_fused(month):
    path = site.recon_dir / f"{site.tile_id}_{month}_fused.tif"
    if not path.exists():
        return None
    data, transform, crs = io_utils.read_full(path)
    return data


def load_fillmask(month):
    path = site.recon_dir / f"{site.tile_id}_{month}_fillmask.tif"
    if not path.exists():
        return None
    data, _, _ = io_utils.read_full(path)
    return data[0]


def main():
    t_start = time.time()

    months_with_output = [m for m in site.months if (site.recon_dir / f"{site.tile_id}_{m}_fused.tif").exists()]
    if not months_with_output:
        print("No fused GeoTIFFs found in outputs/amazon_d02/reconstructed/ yet -- "
              "run 01-04 first (04_fuse.py needs 03_superresolve.py's SR cache, "
              "which needs a working GPU).")
        return

    print(f"Found {len(months_with_output)} fused months. Building coverage figure...")
    native_frac, sr_frac, unfilled_frac = [], [], []
    for m in months_with_output:
        mask = load_fillmask(m)
        native_frac.append((mask == fuse.FILL_NATIVE).mean())
        sr_frac.append((mask == fuse.FILL_SR).mean())
        unfilled_frac.append((mask == fuse.FILL_UNFILLED).mean())

    fig, ax = plt.subplots(figsize=(14, 4))
    x = np.arange(len(months_with_output))
    ax.bar(x, native_frac, label="NICFI native", color="#2c7fb8")
    ax.bar(x, sr_frac, bottom=native_frac, label="Sentinel-2 SR filled", color="#41ab5d")
    ax.bar(x, unfilled_frac, bottom=np.array(native_frac) + np.array(sr_frac),
           label="still unfilled (no S2 that month)", color="#d7301f")
    step = max(1, len(months_with_output) // 24)
    ax.set_xticks(x[::step])
    ax.set_xticklabels([months_with_output[i] for i in x[::step]], rotation=90, fontsize=7)
    ax.set_ylabel("fraction of pixels")
    ax.set_title("D02: NICFI-native vs. Sentinel-2-SR-filled vs. unfilled, per month")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(site.fig_dir / "02_fill_coverage.png", dpi=130)
    print(f"  -> {site.fig_dir / '02_fill_coverage.png'}")

    gap_and_filled = [m for m, sr in zip(months_with_output, sr_frac) if sr > 0]
    if not gap_and_filled:
        print("No month had any Sentinel-2-filled pixels yet (SR cache likely empty "
              "-- 03_superresolve.py needs a working GPU). Skipping the remaining figures.")
        print(f"\nDONE in {time.time()-t_start:.1f}s.")
        return

    best_month = max(gap_and_filled, key=lambda m: sr_frac[months_with_output.index(m)])
    print(f"Before/after + false-color for {best_month} "
          f"(largest SR-filled fraction, {sr_frac[months_with_output.index(best_month)]:.2%})...")

    nicfi_native, _, _ = io_utils.read_full(site.nicfi_path(best_month))
    fused = load_fused(best_month)
    mask = load_fillmask(best_month)

    def rgb_from(bands_bgr_nir_order, idx):
        return stretch(np.stack([bands_bgr_nir_order[idx[0]], bands_bgr_nir_order[idx[1]],
                                  bands_bgr_nir_order[idx[2]]], axis=-1))

    before_rgb = rgb_from(nicfi_native, (2, 1, 0))  # R,G,B from NICFI's B,G,R,N order
    after_rgb = rgb_from(fused, (cfg.S2_INDEX["B4"], cfg.S2_INDEX["B3"], cfg.S2_INDEX["B2"]))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(before_rgb); axes[0].set_title(f"{best_month}: NICFI raw (gaps = black)")
    axes[1].imshow(after_rgb); axes[1].set_title(f"{best_month}: fused (gaps SR-filled)")
    im = axes[2].imshow(mask, cmap="viridis", vmin=0, vmax=2)
    axes[2].set_title("fill provenance (0=native, 1=SR, 2=unfilled)")
    fig.colorbar(im, ax=axes[2], ticks=[0, 1, 2], shrink=0.7)
    for a in axes:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(site.fig_dir / "03_before_after.png", dpi=130)
    print(f"  -> {site.fig_dir / '03_before_after.png'}")

    swir_rgb = rgb_from(fused, (cfg.S2_INDEX["B11"], cfg.S2_INDEX["B8"], cfg.S2_INDEX["B4"]))
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(swir_rgb)
    ax.set_title(f"{best_month}: SWIR1/NIR/Red false color (bands NICFI never had)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(site.fig_dir / "04_swir_false_color.png", dpi=130)
    print(f"  -> {site.fig_dir / '04_swir_false_color.png'}")

    print("Hard-constraint consistency check (downsample SR back to 10m, "
          "compare to the real Sentinel-2 pixel it came from)...")
    sr_path = site.cache_dir / f"sr_{best_month}.npz"
    if sr_path.exists():
        z = np.load(sr_path, allow_pickle=False)
        sr_data = z["data"]
        sr_transform = rasterio.Affine(*z["transform"])
        sr_crs = rasterio.crs.CRS.from_string(str(z["crs"]))

        s2_mosaic_path = site.cache_dir / f"s2_mosaic_{best_month}.npz"
        zm = np.load(s2_mosaic_path, allow_pickle=False)
        s2_native = zm["data"]
        s2_transform = rasterio.Affine(*zm["transform"])

        down = np.zeros_like(s2_native)
        reproject(source=sr_data, destination=down,
                   src_transform=sr_transform, src_crs=sr_crs,
                   dst_transform=s2_transform, dst_crs=sr_crs,
                   resampling=Resampling.average)

        valid = zm["valid"]
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        for i, band in enumerate(["B2", "B4", "B8", "B11"]):
            j = cfg.S2_INDEX[band]
            a, b = s2_native[j][valid], down[j][valid]
            axes[i].scatter(a, b, s=1, alpha=0.1)
            lims = [0, np.percentile(a, 99.5)]
            axes[i].plot(lims, lims, "r--", lw=1)
            axes[i].set_xlim(lims); axes[i].set_ylim(lims)
            axes[i].set_xlabel(f"real S2 {band}")
            axes[i].set_ylabel(f"SR {band} downsampled to 10m")
            corr = np.corrcoef(a, b)[0, 1]
            axes[i].set_title(f"{band} (r={corr:.3f})")
        fig.suptitle(f"{best_month}: hard-constraint consistency (should track y=x closely)")
        fig.tight_layout()
        fig.savefig(site.fig_dir / "05_hard_constraint_check.png", dpi=130)
        print(f"  -> {site.fig_dir / '05_hard_constraint_check.png'}")
    else:
        print(f"  no SR cache for {best_month}, skipping.")

    print(f"\nDONE in {time.time()-t_start:.1f}s.")


if __name__ == "__main__":
    main()
