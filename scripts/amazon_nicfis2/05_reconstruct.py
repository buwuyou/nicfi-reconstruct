"""
Step 5: reconstruct every NICFI month -- contaminated/nodata NICFI pixels
replaced by that month's clear Sentinel-2 composite, harmonized onto NICFI's
radiometry with a per-month fit (src/nicfis2/reconstruct.py).

Pass 1 fits S2->NICFI per month from clear-in-both pixels; a month with too
few (heavy cloud in either sensor) borrows the median coefficients of its
calendar year's fitted months (or of all months, failing that) -- recorded
as "fallback" in cache/harmonization.json. A month whose clear-in-both red
correlation is below MIN_FIT_R is marked "rejected": its S2 is not used at
all (contaminated NICFI is kept, quality source 3) rather than trusted. Pass 2 writes, per month:
  reconstructed/<tile>_<month>_recon.tif        4-band uint16, NICFI scale
  reconstructed/<tile>_<month>_quality.tif      5-band uint8 data-quality
                                                layer: source, nicfi_class,
                                                flags, s2_n_clear, score (codes
                                                in src/nicfis2/reconstruct.py
                                                and docs/amazon_nicfis2.md)

Run: python scripts/amazon_nicfis2/05_reconstruct.py --tile D17
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import io_utils
from src.nicfis2 import config as cfg, reconstruct as rc, s2_composite


def load_quality(tile, month):
    """NICFI classes after the temporal check (03_temporal_mask_check.py)."""
    z = np.load(tile.cache_dir / f"nicfi_quality_{month}.npz")
    if "refined" not in z:
        raise KeyError(f"{month}: no temporally re-checked NICFI mask -- run "
                       f"03_temporal_mask_check.py first")
    return z["refined"]


def load_overridden(tile, month):
    """Flags cleared by either post-check (temporal, or speckle blob removal;
    filled holes end up flagged, so they're not "overridden")."""
    z = np.load(tile.cache_dir / f"nicfi_quality_{month}.npz")
    return (z["overridden"] | z["despeckled"]) & (z["refined"] == 0)


def fit_all(tile) -> dict:
    rng = np.random.default_rng(0)
    fits = {}
    for month in tile.months:
        comp = s2_composite.load(tile, month)
        if comp is None:
            continue
        nicfi, n_tr, n_crs = io_utils.read_full(tile.nicfi_path(month))
        c = rc.fit_month(nicfi, load_quality(tile, month), n_tr, n_crs, comp, rng)
        if c is not None:
            fits[month] = {**c, "source": "month" if c["r_red"] >= rc.MIN_FIT_R else "rejected"}
    for month in tile.months:
        if month in fits or s2_composite.load(tile, month) is None:
            continue
        pool = [v for m, v in fits.items() if m[:4] == month[:4] and v["source"] == "month"] or \
               [v for v in fits.values() if v["source"] == "month"]
        if pool:
            fits[month] = {b: tuple(np.median([p[b] for p in pool], axis=0).tolist())
                           for b in cfg.NICFI_BAND_NAMES}
            fits[month].update(source="fallback", n_samples=0)
    return fits


def main():
    ap = cfg.add_tile_args(argparse.ArgumentParser())
    args = ap.parse_args()
    tile = cfg.tile_from_args(args)
    t0 = time.time()

    print("Pass 1: per-month S2 -> NICFI harmonization fits...")
    fits = fit_all(tile)
    (tile.cache_dir / "harmonization.json").write_text(json.dumps(fits, indent=1, sort_keys=True))
    for m, f in sorted(fits.items()):
        print(f"  {m} [{f['source']:8s} n={f['n_samples']:6d} r_red={f.get('r_red', float('nan')):.2f}] "
              + "  ".join(f"{b} {f[b][0]:.2f}x{f[b][1]:+.0f}" for b in cfg.NICFI_BAND_NAMES))

    print("Pass 2: reconstruct...")
    stats = {}
    for month in tile.months:
        nicfi, n_tr, n_crs = io_utils.read_full(tile.nicfi_path(month))
        q = load_quality(tile, month)
        comp = s2_composite.load(tile, month)
        f = fits.get(month)
        usable = f if f and f["source"] != "rejected" else None
        res = rc.reconstruct_month(nicfi, q, n_tr, n_crs, comp, usable,
                                   overridden=load_overridden(tile, month))
        io_utils.save_geotiff(tile.recon_dir / f"{tile.tile_id}_{month}_recon.tif",
                              res.recon.clip(0, 65535).astype("uint16"), n_tr, n_crs,
                              dtype="uint16", nodata=0)
        qpath = tile.recon_dir / f"{tile.tile_id}_{month}_quality.tif"
        io_utils.save_geotiff(qpath, res.quality, n_tr, n_crs, dtype="uint8")
        with rasterio.open(qpath, "r+") as dst:
            dst.descriptions = tuple(rc.QUALITY_BANDS)
        src = res.source
        stats[month] = {
            "native": float((src == rc.SRC_NICFI).mean()),
            "s2": float(np.isin(src, (rc.SRC_S2_SINGLE, rc.SRC_S2_MEDIAN)).mean()),
            "kept_dirty": float((src == rc.SRC_KEPT).mean()),
            "nodata": float((src == rc.SRC_NODATA).mean()),
            "overridden": float((res.quality[2] & rc.FLAG_OVERRIDDEN).astype(bool).mean()),
            "mean_score": float(res.quality[4].mean()),
            "s2_method": comp.method if comp else "none",
            "fit": f["source"] if f else "none",
        }
        s = stats[month]
        print(f"  {month}: replaced by S2 {s['s2']:.1%}, contaminated-kept {s['kept_dirty']:.1%}, "
              f"nodata {s['nodata']:.2%}  (S2 {s['s2_method']}, fit {s['fit']})")
    (tile.cache_dir / "reconstruction_stats.json").write_text(json.dumps(stats, indent=1))
    print(f"DONE in {time.time()-t0:.0f}s -> {tile.recon_dir}")


if __name__ == "__main__":
    main()
