"""
Before/after visualizations for the Amazon (D01) 5-year reconstruction.

Run: python scripts/amazon/02_visualize.py
"""
import sys
from pathlib import Path

import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import sites, io_utils, cloud_mask, compose, visualize

site = sites.get_site("amazon")


def rgb_of(a, gain=0.13, gamma=1.3):
    rgb = a[[sites.RED, sites.GREEN, sites.BLUE]] / gain
    rgb = np.clip(rgb, 0, 1)
    rgb = np.power(rgb, 1 / gamma)
    return rgb.transpose(1, 2, 0)


def main():
    months = site.months
    d = np.load(site.cache_dir / "stack.npz")
    quality = d["quality"]

    stack = io_utils.read_aoi_stack(site)
    refl = io_utils.to_reflectance(stack.data)

    # ---- 1. cloud-fraction time series across all 60 months ----
    contam_frac = (quality != cloud_mask.CLEAR).mean(axis=(1, 2))
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.bar(range(len(months)), contam_frac, color="#e5383b")
    ax.set_xticks(range(0, len(months), 3))
    ax.set_xticklabels([months[i] for i in range(0, len(months), 3)], rotation=45, ha="right")
    ax.set_ylabel("fraction of tile flagged non-clear")
    ax.set_title("D01 (Amazon): contamination fraction per month, 2021-2025")
    plt.tight_layout()
    plt.savefig(site.fig_dir / "01_contamination_timeseries.png", dpi=130)
    plt.close(fig)

    # ---- 2. curated before/after grid across a spread of severities/years ----
    highlight_months = ["2021-01", "2021-07", "2021-04", "2024-05", "2023-03", "2025-04", "2025-07", "2025-12"]
    fig, axes = plt.subplots(len(highlight_months), 2, figsize=(9, 4.2 * len(highlight_months)))
    for i, m in enumerate(highlight_months):
        t = months.index(m)
        obs = rgb_of(refl[t])
        recon_path = site.recon_dir / f"{site.tile_id}_{m}_reconstructed.tif"
        with rasterio.open(recon_path) as src:
            recon = src.read().astype(np.float32) / sites.REFLECTANCE_SCALE
        axes[i, 0].imshow(obs)
        axes[i, 0].set_title(f"{m} observed  ({100*contam_frac[t]:.0f}% flagged)")
        axes[i, 0].axis("off")
        axes[i, 1].imshow(rgb_of(recon))
        axes[i, 1].set_title(f"{m} reconstructed")
        axes[i, 1].axis("off")
    plt.tight_layout()
    plt.savefig(site.fig_dir / "02_before_after_highlights.png", dpi=120)
    plt.close(fig)

    # ---- 3. per-year full grids (observed and reconstructed) ----
    for year in range(2021, 2026):
        year_months = [m for m in months if m.startswith(str(year))]
        idxs = [months.index(m) for m in year_months]

        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        for ax, m, t in zip(axes.flat, year_months, idxs):
            ax.imshow(rgb_of(refl[t]))
            ax.set_title(f"{m} ({100*contam_frac[t]:.0f}% flagged)")
            ax.axis("off")
        fig.suptitle(f"{year} — observed", y=1.0, fontsize=14)
        plt.tight_layout()
        plt.savefig(site.fig_dir / f"03_observed_{year}.png", dpi=110)
        plt.close(fig)

        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        for ax, m in zip(axes.flat, year_months):
            recon_path = site.recon_dir / f"{site.tile_id}_{m}_reconstructed.tif"
            with rasterio.open(recon_path) as src:
                recon = src.read().astype(np.float32) / sites.REFLECTANCE_SCALE
            ax.imshow(rgb_of(recon))
            ax.set_title(m)
            ax.axis("off")
        fig.suptitle(f"{year} — reconstructed", y=1.0, fontsize=14)
        plt.tight_layout()
        plt.savefig(site.fig_dir / f"04_reconstructed_{year}.png", dpi=110)
        plt.close(fig)

    # ---- 4. phenology curve examples ----
    phen = np.load(site.cache_dir / "phenology_recon.npz")
    recon_curve, confidence, labels = phen["recon"], phen["conf"], phen["labels"]
    weight_final = phen["weight_final"]

    # pick two representative pixels: lowest & highest NDVI cluster centroids
    nir_mean = refl[:, sites.NIR].mean(axis=0)
    red_mean = refl[:, sites.RED].mean(axis=0)
    ndvi = (nir_mean - red_mean) / (nir_mean + red_mean + 1e-6)
    cluster_ndvi = [(k, ndvi[labels == k].mean()) for k in range(labels.max() + 1)]
    cluster_ndvi.sort(key=lambda x: x[1])
    low_k, high_k = cluster_ndvi[0][0], cluster_ndvi[-1][0]

    def sample_px(k):
        ys, xs = np.where(labels == k)
        j = len(ys) // 2
        return int(ys[j]), int(xs[j])

    refined = np.load(site.cache_dir / "refined.npz")["refined"]
    composite, _alpha = compose.compose(refl, weight_final, refined)

    for name, k in [("low_ndvi", low_k), ("high_ndvi", high_k)]:
        rc = sample_px(k)
        visualize.plot_pixel_phenology(refl, weight_final, recon_curve, composite, months, rc,
                                        site.fig_dir / f"05_phenology_{name}.png")

    print("Done. Figures ->", site.fig_dir)


if __name__ == "__main__":
    main()