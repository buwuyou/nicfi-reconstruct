"""
Fuse harmonized NICFI + super-resolved Sentinel-2 into one 10-band monthly
product on NICFI's native grid.

RGBN bands (the 4 that NICFI has an equivalent for): keep the harmonized
NICFI value wherever NICFI has real data; substitute the SR-Sentinel-2
value, feather-blended across the gap boundary, only where NICFI is a true
gap (exact 0, the same convention used throughout this project). The other
6 bands (B5/B6/B7/B8A/B11/B12) have no NICFI equivalent, so they're
SR-Sentinel-2 everywhere.

The feather (not a hard cutoff at the mask edge) follows the same
philosophy already used in `src/compose.py` for the temporal-reconstruction
pipeline: never let a seam appear where trust in the two sources changes.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
from rasterio.warp import Resampling, reproject
from scipy.ndimage import distance_transform_edt

from . import s2_fusion_site as cfg

FEATHER_PX = 8  # ~40m at NICFI's ~4.77m/px resolution

FILL_NATIVE = 0    # real NICFI observation
FILL_SR = 1         # NICFI gap, filled from super-resolved Sentinel-2
FILL_UNFILLED = 2   # NICFI gap, no Sentinel-2 observation available


@dataclass
class FuseResult:
    fused: np.ndarray       # (10,H,W) float32, Sentinel-2 band order
    fill_mask: np.ndarray   # (H,W) uint8, FILL_* codes above


def reproject_sr_to_nicfi_grid(sr_data, sr_transform, sr_crs,
                                nicfi_shape, nicfi_transform, nicfi_crs) -> np.ndarray:
    """sr_data: (10,h,w) at ~2.5m/EPSG:4326. Returns (10,)+nicfi_shape on
    NICFI's exact grid. `average` because NICFI's pixel (~4.77m) is coarser
    than the SR pixel (~2.5m) -- this avoids aliasing that a nearest/bilinear
    resample would introduce going from fine to coarse."""
    out = np.zeros((sr_data.shape[0],) + tuple(nicfi_shape), dtype=np.float32)
    reproject(
        source=sr_data, destination=out,
        src_transform=sr_transform, src_crs=sr_crs,
        dst_transform=nicfi_transform, dst_crs=nicfi_crs,
        resampling=Resampling.average,
    )
    return out


# A NICFI pixel counts as S2-covered only if (almost) all of the 10m S2
# pixels it overlaps were valid -- this also trims the S2 mask edge, where the
# SR model saw zero-filled neighbours and its output is pulled toward them.
SR_VALID_MIN_FRACTION = 0.999


def reproject_valid_to_nicfi_grid(valid, s2_transform, s2_crs,
                                   nicfi_shape, nicfi_transform, nicfi_crs) -> np.ndarray:
    """valid: (H,W) bool Sentinel-2 mosaic validity at 10m. Returns (H,W)
    bool on NICFI's grid.

    Needed because the SR output is *not* zero where the input was invalid
    (forest-masked / no-data S2 pixels are fed to the model as 0 and come
    back as small non-zero values, ~30 DN median in B8 for 2022-01, vs.
    ~3300 over valid pixels), so `sr != 0` cannot tell real SR data from
    model output over no input at all."""
    frac = np.zeros(tuple(nicfi_shape), dtype=np.float32)
    reproject(
        source=valid.astype(np.float32), destination=frac,
        src_transform=s2_transform, src_crs=s2_crs,
        dst_transform=nicfi_transform, dst_crs=nicfi_crs,
        resampling=Resampling.average,
    )
    return frac >= SR_VALID_MIN_FRACTION


def feather_alpha(gap_mask: np.ndarray, feather_px: int = FEATHER_PX) -> np.ndarray:
    """gap_mask: (H,W) bool, True where NICFI is missing. Returns (H,W)
    float32 alpha in [0,1]: 1.0 deep inside a gap, ramping down to 0.0 over
    `feather_px` pixels as it crosses into real NICFI data."""
    if not gap_mask.any():
        return np.zeros(gap_mask.shape, dtype=np.float32)
    if gap_mask.all():
        return np.ones(gap_mask.shape, dtype=np.float32)
    dist_into_gap = distance_transform_edt(gap_mask)
    dist_into_real = distance_transform_edt(~gap_mask)
    alpha = np.where(
        gap_mask,
        np.minimum(dist_into_gap / feather_px, 1.0),
        np.maximum(1.0 - dist_into_real / feather_px, 0.0),
    )
    return alpha.astype(np.float32)


def fuse_month(nicfi_harmonized: np.ndarray, sr_on_nicfi_grid: Optional[np.ndarray],
               sr_valid: Optional[np.ndarray] = None) -> FuseResult:
    """nicfi_harmonized: (4,H,W) harmonized NICFI, 0=gap (see harmonize.apply).
    sr_on_nicfi_grid: (10,H,W) SR Sentinel-2 already reprojected onto NICFI's
    grid (see reproject_sr_to_nicfi_grid), or None if this month had zero
    Sentinel-2 frames at all. sr_valid: (H,W) bool, where the SR output is
    backed by a real S2 observation (see reproject_valid_to_nicfi_grid);
    required whenever sr_on_nicfi_grid is given."""
    h, w = nicfi_harmonized.shape[1:]
    n_bands = len(cfg.S2_BAND_NAMES)
    gap = np.all(nicfi_harmonized == 0, axis=0)
    fill_mask = np.zeros((h, w), dtype=np.uint8)

    if sr_on_nicfi_grid is None:
        fused = np.zeros((n_bands, h, w), dtype=np.float32)
        for band, s2_band in cfg.NICFI_TO_S2_BAND.items():
            i = cfg.NICFI_BAND_NAMES.index(band)
            j = cfg.S2_INDEX[s2_band]
            fused[j] = nicfi_harmonized[i]
        fill_mask[gap] = FILL_UNFILLED
        return FuseResult(fused=fused, fill_mask=fill_mask)

    if sr_valid is None:
        raise ValueError("sr_valid is required with sr_on_nicfi_grid -- SR output is "
                         "non-zero even where its input was invalid")
    fillable = gap & sr_valid
    # Feather only around gaps S2 can actually fill, and never pull SR values
    # into pixels where SR has no real observation behind it.
    alpha = feather_alpha(fillable) * sr_valid
    fused = sr_on_nicfi_grid.copy()  # the 6 SR-only bands are always this...
    fused[:, ~sr_valid] = 0.0        # ...except where there's no real S2 under it
    for band, s2_band in cfg.NICFI_TO_S2_BAND.items():
        i = cfg.NICFI_BAND_NAMES.index(band)
        j = cfg.S2_INDEX[s2_band]
        fused[j] = alpha * sr_on_nicfi_grid[j] + (1.0 - alpha) * nicfi_harmonized[i]

    fill_mask[fillable] = FILL_SR
    fill_mask[gap & ~sr_valid] = FILL_UNFILLED
    return FuseResult(fused=fused, fill_mask=fill_mask)
