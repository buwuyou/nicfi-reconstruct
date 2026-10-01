"""
Spectral harmonization: put NICFI's Blue/Green/Red/NIR bands onto
Sentinel-2's radiometric scale, so the whole 10-band fused output (real
NICFI + SR-filled gaps + SR-only extra bands) is self-consistent.

Fit at the coarser, common resolution (NICFI area-averaged down onto the
Sentinel-2 grid) rather than at NICFI's native resolution, so upsampling
blur never gets baked into the fitted transform -- only genuine radiometric/
spectral-response differences should end up in the slope/intercept.

A robust (RANSAC) fit is used per band rather than ordinary least squares,
because the input pairs come straight from each sensor's own delivered
"valid" pixels with no independent cloud check of our own -- some residual
cloud/shadow/haze contamination is expected to have survived Sentinel-2's
upstream mask (and NICFI's own basemap compositing), and OLS is far more
sensitive to that than RANSAC.

Fit **per calendar year**, not pooled across the whole 5-year series: the
first version of this module did pool everything into one fit, and the QA
plot it produced (`02_harmonize.py`'s residual-vs-month figure) showed a
sharp, calendar-aligned discontinuity in every band -- the NICFI/Sentinel-2
relationship shifts hard for exactly the 12 months of 2022, then reverts at
the 2023 boundary. That's too clean to be sampling noise; it's consistent
with e.g. a NICFI basemap processing-version change. A single pooled fit
would average across that discontinuity and mis-harmonize roughly a fifth
of the whole series, so each year gets its own fit instead (see
`docs/amazon_d02_s2fusion.md` for the figure and full writeup).
"""
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from rasterio.warp import Resampling, reproject
from sklearn.linear_model import RANSACRegressor

from . import s2_fusion_site as cfg

VALID_FRAC_THRESHOLD = 0.999  # require (near-)complete NICFI coverage of an S2 cell
MAX_SAMPLES_PER_MONTH = 50_000


@dataclass
class HarmonizationParams:
    # NICFI band name -> (slope, intercept), s.t. harmonized = raw*slope + intercept
    coeffs: Dict[str, Tuple[float, float]]
    n_samples: Dict[str, int]


def resample_nicfi_to_s2_grid(nicfi_data, nicfi_transform, nicfi_crs,
                               s2_shape, s2_transform, s2_crs):
    """Area-average NICFI reflectance (and the fraction of native-res pixels
    that were valid) onto the S2 grid. Returns (resampled (4,H,W) float32,
    valid_frac (H,W) float32 in [0,1])."""
    n_bands = nicfi_data.shape[0]
    valid_mask = np.any(nicfi_data != 0, axis=0).astype(np.float32)  # native res

    resampled = np.zeros((n_bands,) + s2_shape, dtype=np.float32)
    reproject(
        source=nicfi_data, destination=resampled,
        src_transform=nicfi_transform, src_crs=nicfi_crs,
        dst_transform=s2_transform, dst_crs=s2_crs,
        src_nodata=0.0, dst_nodata=0.0,
        resampling=Resampling.average,
    )

    valid_frac = np.zeros(s2_shape, dtype=np.float32)
    reproject(
        source=valid_mask, destination=valid_frac,
        src_transform=nicfi_transform, src_crs=nicfi_crs,
        dst_transform=s2_transform, dst_crs=s2_crs,
        resampling=Resampling.average,
    )
    return resampled, valid_frac


def collect_month_samples(nicfi_native, nicfi_transform, nicfi_crs, s2_mosaic,
                           rng: np.random.Generator):
    """Returns dict NICFI-band-name -> (x_nicfi_on_s2_grid, y_s2) sample arrays
    for this one month, or {} if there's no usable overlap."""
    if s2_mosaic is None:
        return {}

    resampled, valid_frac = resample_nicfi_to_s2_grid(
        nicfi_native, nicfi_transform, nicfi_crs,
        s2_mosaic.data.shape[1:], s2_mosaic.transform, s2_mosaic.crs,
    )
    usable = (valid_frac >= VALID_FRAC_THRESHOLD) & s2_mosaic.valid
    if not usable.any():
        return {}

    idx = np.flatnonzero(usable.ravel())
    if idx.size > MAX_SAMPLES_PER_MONTH:
        idx = rng.choice(idx, size=MAX_SAMPLES_PER_MONTH, replace=False)

    out = {}
    for band in cfg.NICFI_BAND_NAMES:
        i = cfg.NICFI_BAND_NAMES.index(band)
        j = cfg.S2_INDEX[cfg.NICFI_TO_S2_BAND[band]]
        x = resampled[i].ravel()[idx]
        y = s2_mosaic.data[j].ravel()[idx]
        out[band] = (x, y)
    return out


def fit(samples_by_band: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> HarmonizationParams:
    """samples_by_band: NICFI-band-name -> (x_pooled, y_pooled), already pooled
    across every month. Fits harmonized = x*slope + intercept ~= y per band."""
    coeffs, n_samples = {}, {}
    for band, (x, y) in samples_by_band.items():
        x = x.reshape(-1, 1)
        model = RANSACRegressor(random_state=0)
        model.fit(x, y)
        slope = float(model.estimator_.coef_[0])
        intercept = float(model.estimator_.intercept_)
        coeffs[band] = (slope, intercept)
        n_samples[band] = int(x.shape[0])
    return HarmonizationParams(coeffs=coeffs, n_samples=n_samples)


def apply(nicfi_native: np.ndarray, params: HarmonizationParams) -> np.ndarray:
    """nicfi_native: (4,H,W) raw DN, 0=gap. Returns harmonized (4,H,W) float32,
    with gap pixels kept at exactly 0 (never remapped by the intercept) so
    downstream gap detection stays a simple `==0` check."""
    out = np.zeros_like(nicfi_native, dtype=np.float32)
    for i, band in enumerate(cfg.NICFI_BAND_NAMES):
        slope, intercept = params.coeffs[band]
        valid = nicfi_native[i] != 0
        out[i] = np.where(valid, nicfi_native[i] * slope + intercept, 0.0)
    return out
