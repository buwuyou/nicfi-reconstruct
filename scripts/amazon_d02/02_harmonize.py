"""
Fit the NICFI -> Sentinel-2 spectral harmonization: one robust linear
regression per band, **per calendar year** (see src/harmonize.py's
docstring for why -- a first pooled-across-all-years version's QA plot
showed a sharp, calendar-aligned discontinuity in every band, isolated to
2022). Saves the fitted params (keyed by year) and plots a QA scatter +
per-year fit-line comparison + residual-vs-month check. No GPU needed for
this step. Depends on the Sentinel-2 mosaic cache built by 01_prepare.py.

Run: python scripts/amazon_d02/02_harmonize.py
"""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import harmonize, io_utils, s2_fusion_site as cfg, s2_mosaic

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


def main():
    t_start = time.time()
    rng = np.random.default_rng(0)

    # band -> year -> {"x": [...], "y": [...], "month": [...]}
    by_band_year = defaultdict(lambda: defaultdict(lambda: {"x": [], "y": [], "month": []}))
    # also keep an all-years pooled copy, only for the diagnostic residual-vs-month plot
    pooled = {band: {"x": [], "y": [], "month": []} for band in cfg.NICFI_BAND_NAMES}

    print(f"Collecting NICFI/Sentinel-2 overlap samples across {len(site.months)} months...")
    n_months_used = 0
    for month in site.months:
        mosaic = load_cached_mosaic(month)
        if mosaic is None:
            continue
        nicfi_native, nicfi_transform, nicfi_crs = io_utils.read_full(site.nicfi_path(month))
        samples = harmonize.collect_month_samples(nicfi_native, nicfi_transform, nicfi_crs,
                                                   mosaic, rng)
        if not samples:
            continue
        n_months_used += 1
        year = int(month[:4])
        for band, (x, y) in samples.items():
            by_band_year[band][year]["x"].append(x)
            by_band_year[band][year]["y"].append(y)
            by_band_year[band][year]["month"].append(np.full(x.shape, month))
            pooled[band]["x"].append(x)
            pooled[band]["y"].append(y)
            pooled[band]["month"].append(np.full(x.shape, month))

    print(f"  usable overlap in {n_months_used}/{len(site.months)} months")

    years = sorted({int(m[:4]) for m in site.months})
    print(f"Fitting robust (RANSAC) per-band, per-year regression ({years})...")
    params_by_year = {}
    for year in years:
        samples_by_band = {
            band: (np.concatenate(by_band_year[band][year]["x"]),
                   np.concatenate(by_band_year[band][year]["y"]))
            for band in cfg.NICFI_BAND_NAMES if by_band_year[band][year]["x"]
        }
        if len(samples_by_band) < len(cfg.NICFI_BAND_NAMES):
            print(f"  {year}: no usable overlap at all, skipping")
            continue
        params = harmonize.fit(samples_by_band)
        params_by_year[year] = params
        for band, (slope, intercept) in params.coeffs.items():
            print(f"  {year} {band:6s} ({cfg.NICFI_TO_S2_BAND[band]}): "
                  f"harmonized = raw*{slope:.4f} + {intercept:.1f}  "
                  f"(n={params.n_samples[band]})")

    with open(site.cache_dir / "harmonization_params.json", "w") as f:
        json.dump(
            {str(year): {"coeffs": p.coeffs, "n_samples": p.n_samples}
             for year, p in params_by_year.items()},
            f, indent=2,
        )

    print("Plotting QA figures...")
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    palette = plt.cm.viridis(np.linspace(0, 1, len(years)))
    for i, band in enumerate(cfg.NICFI_BAND_NAMES):
        x_all, y_all = np.concatenate(pooled[band]["x"]), np.concatenate(pooled[band]["y"])

        ax = axes[0, i]
        sub = rng.choice(len(x_all), size=min(20_000, len(x_all)), replace=False)
        ax.scatter(x_all[sub], y_all[sub], s=1, alpha=0.1, color="gray")
        lims = [0, np.percentile(np.concatenate([x_all, y_all]), 99.5)]
        ax.plot(lims, lims, "k--", lw=1, label="y=x")
        for year, color in zip(years, palette):
            if year not in params_by_year:
                continue
            slope, intercept = params_by_year[year].coeffs[band]
            ax.plot(lims, [v * slope + intercept for v in lims], "-", lw=1.3,
                     color=color, label=f"{year}: {slope:.2f}x+{intercept:.0f}")
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel(f"NICFI {band} (raw DN)")
        ax.set_ylabel(f"Sentinel-2 {cfg.NICFI_TO_S2_BAND[band]} (raw DN)")
        ax.set_title(band)
        ax.legend(fontsize=6, loc="upper left")

        ax2 = axes[1, i]
        months = np.concatenate(pooled[band]["month"])
        year_of = np.array([int(m[:4]) for m in months])
        pred = np.array([
            x_all[k] * params_by_year[year_of[k]].coeffs[band][0]
            + params_by_year[year_of[k]].coeffs[band][1]
            if year_of[k] in params_by_year else np.nan
            for k in range(len(x_all))
        ])
        resid = y_all - pred
        month_order = sorted(set(months.tolist()))
        month_idx = {m: k for k, m in enumerate(month_order)}
        mi = np.array([month_idx[m] for m in months])
        med_resid = [np.nanmedian(resid[mi == k]) for k in range(len(month_order))]
        ax2.plot(range(len(month_order)), med_resid, "-o", ms=2, lw=0.8)
        ax2.axhline(0, color="k", lw=0.8, ls="--")
        ax2.set_xlabel(f"month index ({site.months[0]} .. {site.months[-1]})")
        ax2.set_ylabel("median residual (per-year fit)")
        ax2.set_title(f"{band}: residual after per-year fit")

    fig.suptitle("D02 NICFI->Sentinel-2 harmonization: per-year fits + residual check")
    fig.tight_layout()
    fig.savefig(site.fig_dir / "01_harmonization_fit.png", dpi=130)
    print(f"  -> {site.fig_dir / '01_harmonization_fit.png'}")

    print(f"\nDONE in {time.time()-t_start:.1f}s.")


if __name__ == "__main__":
    main()
