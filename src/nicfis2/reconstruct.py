"""
Replace NICFI's cloud/shadow/haze-contaminated (and nodata) pixels with a
clear Sentinel-2 observation from the same month, mapped onto NICFI's
radiometry.

Harmonization is fit **per month**, Sentinel-2 -> NICFI (output stays a
NICFI-scale 4-band product), using only pixels clear in both sensors that
month. Per month rather than per year because NICFI's monthly basemap and
the month's Sentinel-2 composite are different acquisitions: their
relationship shifts with date/illumination/phenology, and a same-month fit
absorbs that instead of baking a yearly average into every fill. Fit at
Sentinel-2's coarser 10m grid (NICFI area-averaged down) so upsampling blur
never ends up in the slope/intercept.

The fit is robust quantile matching (slope from the 10th-90th percentile
spread, intercept from the medians), not a regression. A first version used
RANSAC regression: on a clean month (D17 2021-06, r=0.85-0.93 in the visible
bands) it returned slopes of only 0.3-0.65, because its default inlier band
locks onto the dense forest cluster where the relation looks flat
(regression dilution), which would have made every fill low-contrast. For
substituting one sensor's values into another's image, matching the
distributions of clear-in-both pixels is the right target. The fit's
clear-in-both correlation is kept as a quality signal: a month whose red
band correlates below MIN_FIT_R is treated as having no trustworthy S2.

On top of the whole-tile fit, the parameters are refined locally
(`local_fields`) because one NICFI month is itself a mosaic of scenes that
don't share one radiometry. Harmonization is applied at 10m, then the
harmonized S2 is upsampled to NICFI's 4.77m grid bilinearly. A super-resolution model
(`superres.py`, SEN2SR) was tested for this on an earlier tile: it preserved
its own 10m input well but did *not* agree with NICFI better than bilinear
on any band/month (its added sub-10m detail is plausible texture, not what
NICFI sees), at ~17 GPU-min per month, so it isn't used by default.
"""
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from rasterio.warp import Resampling, reproject
from scipy import ndimage

from .. import cloud_mask
from . import config as cfg
from .s2_composite import Composite, band_index

FEATHER_PX = 8               # ~40m at NICFI's 4.77m
NICFI_BUFFER_PX = 3          # ~15m dilation of NICFI contamination
COVER_FRAC = 0.999           # a cell counts as covered only if (almost) fully valid
MIN_FIT_SAMPLES = 2000
MAX_FIT_SAMPLES = 200_000
LOCAL_BLOCK = 64     # S2 px, ~640m
LOCAL_MIN_N = 200    # clear-in-both px for a block to get its own fit
LOCAL_N0 = 1000      # shrinkage strength toward the whole-tile fit
MIN_FIT_R = 0.6  # red-band clear-in-both correlation; below this the month's S2 isn't trusted

PROV_NATIVE = 0        # clear NICFI, kept
PROV_S2 = 1            # contaminated/nodata NICFI, replaced by harmonized S2
PROV_KEPT_DIRTY = 2    # contaminated NICFI, no clear S2 this month -> kept as-is
PROV_NODATA = 3        # NICFI nodata and no clear S2

CONTAMINATED = (cloud_mask.CLOUD_THICK, cloud_mask.CLOUD_THIN, cloud_mask.SHADOW,
                cloud_mask.HAZE)


def nicfi_quality(nicfi_raw: np.ndarray, device: str) -> np.ndarray:
    """(H,W) uint8 cloud_mask codes for one NICFI month (OCM ensemble + haze)."""
    q, _ = cloud_mask.compute_masks((nicfi_raw / cfg.REFLECTANCE_SCALE)[None], device=device)
    return q[0]


def replace_mask(quality: np.ndarray, buffer_px: int = NICFI_BUFFER_PX) -> np.ndarray:
    dirty = np.isin(quality, CONTAMINATED)
    if buffer_px > 0 and dirty.any():
        dirty = ndimage.binary_dilation(dirty, iterations=buffer_px)
    return dirty | (quality == cloud_mask.NODATA)


def _reproject(src, s_tr, s_crs, shape, d_tr, d_crs, resampling, src_nodata=None):
    out = np.zeros((src.shape[0],) + tuple(shape), dtype=np.float32)
    reproject(source=src.astype(np.float32), destination=out, src_transform=s_tr, src_crs=s_crs,
              dst_transform=d_tr, dst_crs=d_crs, resampling=resampling,
              src_nodata=src_nodata, dst_nodata=0.0 if src_nodata is not None else None)
    return out


def s2_rgbn(comp: Composite) -> np.ndarray:
    bi = band_index(comp.band_names)
    return np.stack([comp.data[bi[cfg.NICFI_TO_S2_BAND[b]]] for b in cfg.NICFI_BAND_NAMES])


def _pairs(nicfi_raw, quality, n_tr, n_crs, comp: Composite):
    """Clear-NICFI area-averaged onto the S2 grid, the mask of S2 cells that
    are clear in both sensors, and the S2 RGBN bands."""
    clear = (quality == cloud_mask.CLEAR).astype(np.float32)
    shape = comp.data.shape[1:]
    nic = _reproject(np.where(clear[None] > 0, nicfi_raw, 0), n_tr, n_crs, shape,
                     comp.transform, comp.crs, Resampling.average, src_nodata=0.0)
    frac = _reproject(clear[None], n_tr, n_crs, shape, comp.transform, comp.crs,
                      Resampling.average)[0]
    return nic, (frac >= COVER_FRAC) & comp.valid, s2_rgbn(comp)


def _qstats(v):
    p10, p50, p90 = np.percentile(v, [10, 50, 90])
    return p50, max(p90 - p10, 1e-6)


def fit_month(nicfi_raw, quality, n_tr, n_crs, comp: Composite,
              rng: np.random.Generator) -> Optional[Dict[str, Tuple[float, float]]]:
    """Whole-tile per-band (slope, intercept) with nicfi ~= s2*slope +
    intercept, from pixels clear in both this month, plus r_<band>
    correlations and n_samples; None if too few such pixels. Used as the
    prior for `local_fields`, as the trust check, and for QA."""
    nic, usable, s2 = _pairs(nicfi_raw, quality, n_tr, n_crs, comp)
    idx = np.flatnonzero(usable.ravel())
    if idx.size < MIN_FIT_SAMPLES:
        return None
    if idx.size > MAX_FIT_SAMPLES:
        idx = rng.choice(idx, MAX_FIT_SAMPLES, replace=False)
    coeffs = {}
    for i, band in enumerate(cfg.NICFI_BAND_NAMES):
        x, y = s2[i].ravel()[idx], nic[i].ravel()[idx]
        (x50, xs), (y50, ys) = _qstats(x), _qstats(y)
        slope = float(ys / xs)
        coeffs[band] = (slope, float(y50 - slope * x50))
        coeffs[f"r_{band}"] = float(np.corrcoef(x, y)[0, 1])
    coeffs["n_samples"] = int(idx.size)
    return coeffs


def local_fields(nic, usable, s2, coeffs, block=LOCAL_BLOCK, n0=LOCAL_N0):
    """Spatially varying (slope, intercept) fields, each (4,h,w) on the S2
    grid. Needed because a NICFI monthly basemap stitches several
    PlanetScope scenes with different contrast/brightness (on D17 2021-01 the
    lower part of the tile has ~4x less green/red contrast than June over the
    same forest, while the top doesn't), which no single whole-tile fit can
    match. Quantile matching per `block` x `block` S2-px block, the slope
    shrunk toward the whole-tile slope by n/(n+n0) so sparse blocks lean on
    the prior, then smoothed across blocks so parameters never jump at a
    block edge."""
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
                (x50, xs), (y50, ys) = _qstats(s2[i][sl][m]), _qstats(nic[i][sl][m])
                wgt = n / (n + n0)
                slope = wgt * (ys / xs) + (1 - wgt) * g_slope
                slope_b[i, by, bx] = slope
                inter_b[i, by, bx] = y50 - slope * x50
    slope_b = ndimage.gaussian_filter(slope_b, (0, 1, 1), mode="nearest")
    inter_b = ndimage.gaussian_filter(inter_b, (0, 1, 1), mode="nearest")
    zoom = (1, h / nby, w / nbx)
    up = lambda f: ndimage.zoom(f, zoom, order=1, mode="nearest", grid_mode=True)[:, :h, :w]
    return up(slope_b), up(inter_b)


def feather_alpha(mask: np.ndarray, feather_px: int = FEATHER_PX) -> np.ndarray:
    """1 everywhere inside `mask`, ramping to 0 over `feather_px` px *outward*
    into clear NICFI -- never a hard seam where the source changes, and the
    ramp only ever mixes harmonized S2 into clear pixels, never lets a
    contaminated NICFI pixel through at partial weight."""
    if not mask.any():
        return np.zeros(mask.shape, dtype=np.float32)
    if mask.all():
        return np.ones(mask.shape, dtype=np.float32)
    outside = ndimage.distance_transform_edt(~mask)
    return np.where(mask, 1.0, np.maximum(1.0 - outside / (feather_px + 1), 0.0)).astype(np.float32)


@dataclass
class Result:
    recon: np.ndarray        # (4,H,W) float32, NICFI scale
    provenance: np.ndarray   # (H,W) uint8 PROV_*
    s2_harmonized: np.ndarray  # (4,H,W) on NICFI grid (0 where no clear S2)


def reconstruct_month(nicfi_raw, quality, n_tr, n_crs, comp: Optional[Composite],
                      coeffs: Optional[dict]) -> Result:
    """coeffs: whole-tile fit (fit_month), plus "source": "month" -> refined
    locally (local_fields) around that prior; anything else (e.g. a borrowed
    year-median "fallback") -> applied as-is, since a month with too few
    clear-in-both pixels can't support a local fit either."""
    shape = nicfi_raw.shape[1:]
    replace = replace_mask(quality)
    nodata = quality == cloud_mask.NODATA
    prov = np.full(shape, PROV_NATIVE, dtype=np.uint8)
    if comp is None or coeffs is None:
        prov[replace] = PROV_KEPT_DIRTY
        prov[nodata] = PROV_NODATA
        return Result(nicfi_raw.astype(np.float32), prov, np.zeros_like(nicfi_raw, np.float32))

    if coeffs.get("source") == "month":
        nic, usable, s2 = _pairs(nicfi_raw, quality, n_tr, n_crs, comp)
        slope, inter = local_fields(nic, usable, s2, coeffs)
    else:
        s2 = s2_rgbn(comp)
        slope = np.stack([np.full(s2.shape[1:], coeffs[b][0], np.float32) for b in cfg.NICFI_BAND_NAMES])
        inter = np.stack([np.full(s2.shape[1:], coeffs[b][1], np.float32) for b in cfg.NICFI_BAND_NAMES])
    harm_s2 = np.where(comp.valid[None], np.clip(s2 * slope + inter, 1.0, None), 0.0)

    harm = _reproject(harm_s2, comp.transform, comp.crs, shape, n_tr, n_crs,
                      Resampling.bilinear, src_nodata=0.0)
    cover = _reproject(comp.valid[None].astype(np.float32), comp.transform, comp.crs, shape,
                       n_tr, n_crs, Resampling.average)[0] >= COVER_FRAC
    harm[:, ~cover] = 0.0

    fill = replace & cover
    alpha = feather_alpha(fill) * cover
    alpha[nodata & cover] = 1.0  # nothing real to blend toward
    recon = alpha * harm + (1.0 - alpha) * nicfi_raw
    prov[fill] = PROV_S2
    prov[replace & ~cover] = PROV_KEPT_DIRTY
    prov[nodata & ~cover] = PROV_NODATA
    recon[:, nodata & ~cover] = 0.0
    return Result(recon.astype(np.float32), prov, harm)
