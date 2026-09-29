"""
Annual composite for the Amazon site -- unlike Vietnam (a single year, so
"annual composite" and "the reconstructed stack" were the same thing), this
site has 5 calendar years of monthly data, so an annual composite here
means one composite *per year* (5 outputs per variant), each built from
that year's 12 reconstructed months. Same method as
scripts/vietnam/04_annual_composite.py (src/annual_composite.py): naive
(raw confidently-clear observations only, the standard approach and its
uneven-sample-count problem) vs. robust (weighted from the reconstructed
stack, comparable sample depth everywhere) vs. medoid (band-consistent,
one real month's vector per pixel).

Run: python scripts/amazon/03_annual_composite.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import annual_composite as ac
from src import compose, io_utils, sites, visualize

site = sites.get_site("amazon")
YEARS = range(2021, 2026)


def main():
    stack = io_utils.read_aoi_stack(site)
    refl = io_utils.to_reflectance(stack.data)
    months = stack.months

    phen = np.load(site.cache_dir / "phenology_recon.npz")
    weight_final = phen["weight_final"]
    refined = np.load(site.cache_dir / "refined.npz")["refined"]
    composite, alpha_used = compose.compose(refl, weight_final, refined)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_valid_by_year = {}
    for year in YEARS:
        idx = [i for i, m in enumerate(months) if m.startswith(str(year))]
        refl_y = refl[idx]
        weight_y = weight_final[idx]
        composite_y = composite[idx]
        alpha_y = alpha_used[idx]

        print(f"{year}: computing naive/robust/medoid annual composites...")
        naive_med, n_valid = ac.naive_composite(refl_y, weight_y, percentile=50)
        robust_med = ac.robust_composite(composite_y, alpha_y, percentile=50, recon_trust=0.35)
        medoid, best_t = ac.medoid_composite(composite_y, alpha_y, target=robust_med, recon_trust=0.35)
        print(f"  n_valid (confidently-clear months, out of 12): min={n_valid.min():.0f} "
              f"p5={np.percentile(n_valid,5):.1f} median={np.median(n_valid):.0f} max={n_valid.max():.0f}")
        n_valid_by_year[year] = n_valid

        for name, arr in [("naive_median", np.nan_to_num(naive_med, nan=0.0)),
                           ("robust_median", robust_med),
                           ("medoid", medoid)]:
            out = (arr * sites.REFLECTANCE_SCALE).clip(0, 32000).astype(np.int16)
            io_utils.save_geotiff(site.annual_dir / f"{site.tile_id}_{year}_annual_{name}.tif",
                                   out, stack.transform, stack.crs)

    # ---- figure 1: naive vs robust vs medoid vs n_valid, per year ----
    fig, axes = plt.subplots(len(list(YEARS)), 4, figsize=(18, 4.3 * len(list(YEARS))))
    for row, year in enumerate(YEARS):
        import rasterio
        def load(name):
            with rasterio.open(site.annual_dir / f"{site.tile_id}_{year}_annual_{name}.tif") as src:
                return src.read().astype(np.float32) / sites.REFLECTANCE_SCALE
        axes[row, 0].imshow(visualize.rgb_stretch(load("naive_median"), gain=0.13, gamma=1.3))
        axes[row, 0].set_title(f"{year} naive median")
        axes[row, 1].imshow(visualize.rgb_stretch(load("robust_median"), gain=0.13, gamma=1.3))
        axes[row, 1].set_title(f"{year} robust median")
        axes[row, 2].imshow(visualize.rgb_stretch(load("medoid"), gain=0.13, gamma=1.3))
        axes[row, 2].set_title(f"{year} medoid")
        im = axes[row, 3].imshow(n_valid_by_year[year], cmap="viridis", vmin=0, vmax=12)
        axes[row, 3].set_title(f"{year} n_valid (/12)")
        plt.colorbar(im, ax=axes[row, 3], fraction=0.046)
        for c in range(4):
            axes[row, c].axis("off")
    plt.tight_layout()
    plt.savefig(site.fig_dir / "06_annual_composite_by_year.png", dpi=110)
    plt.close(fig)

    # ---- figure 2: n_valid distribution per year, to show how much the
    # naive-vs-robust divergence should matter for each year ----
    fig, ax = plt.subplots(figsize=(9, 4.5))
    data = [n_valid_by_year[y].ravel() for y in YEARS]
    ax.boxplot(data, tick_labels=[str(y) for y in YEARS], showfliers=False)
    ax.set_ylabel("n confidently-clear months (out of 12)")
    ax.set_title("D01 (Amazon): per-year data sufficiency for the naive annual composite")
    plt.tight_layout()
    plt.savefig(site.fig_dir / "07_annual_n_valid_by_year.png", dpi=130)
    plt.close(fig)

    print(f"Done. GeoTIFFs -> {site.annual_dir}, figures -> {site.fig_dir}")


if __name__ == "__main__":
    main()