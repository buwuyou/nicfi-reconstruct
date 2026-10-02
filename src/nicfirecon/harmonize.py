"""
Radiometric harmonization by robust, local quantile matching -- used for
Sentinel-2 -> NICFI (`reconstruct --method s2fill`) and NICFI -> NICFI
across years (`composite --type typical-year`).

**Quantile matching, not regression.** Slope = target p10-p90 spread /
source spread, intercept from the medians. A first version used RANSAC
regression: on a clean month (D17 2021-06, r=0.85-0.93 in the visible bands)
it returned slopes of only 0.3-0.65, because its default inlier band locks
onto the dense forest cluster where the relation looks flat (regression
dilution) -- every fill would have been washed out. For substituting one
sensor's values into another's image, matching the distributions of
clear-in-both pixels is the right target. The clear-in-both correlation is
kept as a trust signal: a month whose red band correlates below MIN_FIT_R
is treated as having no trustworthy S2.

**Per month, S2 -> NICFI.** NICFI's monthly basemap and the month's S2
composite are different acquisitions; a same-month fit absorbs their
date/illumination/phenology differences. Fit at S2's coarser 10m grid
(NICFI area-averaged down) so upsampling blur never ends up in the fit.

**Local.** One NICFI month is itself a mosaic of PlanetScope scenes that
don't share one radiometry (on D17 2021-01 the lower part of the tile has
~4x less green/red contrast than June over the same forest, the top
doesn't), so the whole-tile fit is only the prior for `local_fields`.

**Slope bounds.** Any fitted slope is clamped to SLOPE_RANGE: a block whose
source band is nearly flat gives target_spread / ~0, which shrinkage alone
doesn't bound -- the first typical-year December composite on D17 got red
= 13,804 DN (raw observations 127-431) from exactly this.

Upsampling S2 10m -> NICFI 4.77m is bilinear. A super-resolution model
(SEN2SR) was tested for this on an earlier tile: faithful to its own 10m
input, but no better than bilinear against NICFI on any band/month, at
~17 GPU-min per month (`docs/super-resolution.md`).
"""
from typing import Dict, Optional

import numpy as np
from rasterio.warp import Resampling, reproject
from scipy import ndimage

from .. import cloud_mask
from . import config as cfg
from .masking import band_index

COVER_FRAC = 0.999           # a cell counts as covered only if (almost) fully valid
MIN_FIT_SAMPLES = 2000
MAX_FIT_SAMPLES = 200_000
LOCAL_BLOCK = 64             # px of the grid the fit runs on (S2: ~640m)
LOCAL_MIN_N = 200            # clear-in-both px for a block to get its own fit
LOCAL_N0 = 1000              # shrinkage strength toward the whole-tile fit
SLOPE_RANGE = (0.25, 4.0)
MIN_FIT_R = 0.6


def reproject_to(src, s_tr, s_crs, shape, d_tr, d_crs, resampling, src_nodata=None):
    out = np.zeros((src.shape[0],) + tuple(shape), dtype=np.float32)
    reproject(source=src.astype(np.float32), destination=out, src_transform=s_tr, src_crs=s_crs,
              dst_transform=d_tr, dst_crs=d_crs, resampling=resampling,
              src_nodata=src_nodata, dst_nodata=0.0 if src_nodata is not None else None)
    return out


def s2_bands(comp, names) -> np.ndarray:
    bi = band_index(comp.band_names)
    return np.stack([comp.data[bi[b]] for b in names])


def s2_rgbn(comp) -> np.ndarray:
    return s2_bands(comp, [cfg.NICFI_TO_S2_BAND[b] for b in cfg.NICFI_BAND_NAMES])


def qstats(v):
    p10, p50, p90 = np.percentile(v, [10, 50, 90])
    return p50, max(p90 - p10, 1e-6)


def qmatch(x, y):
    """(slope, intercept) mapping the distribution of x onto that of y."""
    (x50, xs), (y50, ys) = qstats(x), qstats(y)
    slope = float(np.clip(ys / xs, *SLOPE_RANGE))
    return slope, float(y50 - slope * x50)


def clear_in_both(nicfi_raw, quality, n_tr, n_crs, comp):
    """Clear-NICFI area-averaged onto the S2 grid, the mask of S2 cells
    clear in both sensors, and the S2 RGBN bands."""
    clear = (quality == cloud_mask.CLEAR).astype(np.float32)
    shape = comp.data.shape[1:]
    nic = reproject_to(np.where(clear[None] > 0, nicfi_raw, 0), n_tr, n_crs, shape,
                       comp.transform, comp.crs, Resampling.average, src_nodata=0.0)
    frac = reproject_to(clear[None], n_tr, n_crs, shape, comp.transform, comp.crs,
                        Resampling.average)[0]
    return nic, (frac >= COVER_FRAC) & comp.valid, s2_rgbn(comp)


def fit_month(nicfi_raw, quality, n_tr, n_crs, comp, rng: np.random.Generator) -> Optional[Dict]:
    """Whole-tile per-band (slope, intercept), nicfi ~= s2*slope + intercept,
    plus r_<band> and n_samples; None if too few clear-in-both pixels."""
    nic, usable, s2 = clear_in_both(nicfi_raw, quality, n_tr, n_crs, comp)
    idx = np.flatnonzero(usable.ravel())
    if idx.size < MIN_FIT_SAMPLES:
        return None
    if idx.size > MAX_FIT_SAMPLES:
        idx = rng.choice(idx, MAX_FIT_SAMPLES, replace=False)
    coeffs = {}
    for i, band in enumerate(cfg.NICFI_BAND_NAMES):
        x, y = s2[i].ravel()[idx], nic[i].ravel()[idx]
        coeffs[band] = qmatch(x, y)
        coeffs[f"r_{band}"] = float(np.corrcoef(x, y)[0, 1])
    coeffs["n_samples"] = int(idx.size)
    return coeffs


def fit_all_months(tile, load_comp, load_classes, read_nicfi) -> Dict[str, Dict]:
    """Per-month fits; "source" is "month" (own fit, trusted), "rejected"
    (r_red < MIN_FIT_R: this month's S2 isn't used at all) or "fallback"
    (too few clear-in-both px: the median coefficients of the same year's
    trusted months, else of all trusted months)."""
    rng = np.random.default_rng(0)
    fits = {}
    for month in tile.months:
        comp = load_comp(month)
        if comp is None:
            continue
        nicfi, n_tr, n_crs = read_nicfi(month)
        c = fit_month(nicfi, load_classes(month), n_tr, n_crs, comp, rng)
        if c is not None:
            fits[month] = {**c, "source": "month" if c["r_red"] >= MIN_FIT_R else "rejected"}
    for month in tile.months:
        if month in fits or load_comp(month) is None:
            continue
        pool = [v for m, v in fits.items() if m[:4] == month[:4] and v["source"] == "month"] or \
               [v for v in fits.values() if v["source"] == "month"]
        if pool:
            fits[month] = {b: tuple(np.median([p[b] for p in pool], axis=0).tolist())
                           for b in cfg.NICFI_BAND_NAMES}
            fits[month].update(source="fallback", n_samples=0)
    return fits


def local_fields(target, usable, source, coeffs, block=LOCAL_BLOCK, n0=LOCAL_N0):
    """Spatially varying (slope, intercept) fields, each (4,h,w) on the grid
    of `target`/`source`: quantile matching per `block` x `block` block, the
    slope shrunk toward the whole-tile slope by n/(n+n0) so sparse blocks
    lean on the prior, then smoothed across blocks so parameters never jump
    at a block edge."""
    h, w = usable.shape
    nby, nbx = -(-h // block), -(-w // block)
    slope_b = np.zeros((4, nby, nbx), np.float32)
    inter_b = np.zeros((4, nby, nbx), np.float32)
    for i, band in enumerate(cfg.NICFI_BAND_NAMES):
        g_slope, g_inter = coeffs[band]
        slope_b[i], inter_b[i] = g_slope, g_inter
        for by in range(nby):
            for bx in range(nbx):
                sl = (slice(by * block, (by + 1) * block), slice(bx * block, (bx + 1) * block))
                m = usable[sl]
                n = int(m.sum())
                if n < LOCAL_MIN_N:
                    continue
                (x50, xs), (y50, ys) = qstats(source[i][sl][m]), qstats(target[i][sl][m])
                wgt = n / (n + n0)
                slope = np.clip(wgt * np.clip(ys / xs, *SLOPE_RANGE) + (1 - wgt) * g_slope,
                                *SLOPE_RANGE)
                slope_b[i, by, bx] = slope
                inter_b[i, by, bx] = y50 - slope * x50
    slope_b = ndimage.gaussian_filter(slope_b, (0, 1, 1), mode="nearest")
    inter_b = ndimage.gaussian_filter(inter_b, (0, 1, 1), mode="nearest")
    zoom = (1, h / nby, w / nbx)
    up = lambda f: ndimage.zoom(f, zoom, order=1, mode="nearest", grid_mode=True)[:, :h, :w]
    return up(slope_b), up(inter_b)


def harmonized_s2(nicfi_raw, quality, n_tr, n_crs, comp, coeffs) -> np.ndarray:
    """(4,h,w) S2 RGBN on the S2 grid mapped onto this month's NICFI
    radiometry (0 where S2 isn't valid). coeffs "source" == "month" ->
    refined locally around the whole-tile fit; otherwise (a borrowed
    fallback) applied as-is, since a month with too few clear-in-both pixels
    can't support a local fit either."""
    if coeffs.get("source") == "month":
        nic, usable, s2 = clear_in_both(nicfi_raw, quality, n_tr, n_crs, comp)
        slope, inter = local_fields(nic, usable, s2, coeffs)
    else:
        s2 = s2_rgbn(comp)
        slope = np.stack([np.full(s2.shape[1:], coeffs[b][0], np.float32) for b in cfg.NICFI_BAND_NAMES])
        inter = np.stack([np.full(s2.shape[1:], coeffs[b][1], np.float32) for b in cfg.NICFI_BAND_NAMES])
    return np.where(comp.valid[None], np.clip(s2 * slope + inter, 1.0, None), 0.0)
