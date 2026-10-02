"""
Temporal post-check of a cloud-mask time series (NICFI monthly or
Sentinel-2 single-date): clouds move, ground doesn't.

A single-date classifier repeatedly calls the same bright or dark *ground*
feature cloud/shadow (bare soil, flowering/pink canopy, roads, water).
Over a dense time series that shows up as a location flagged in a large
share of observations whose scene is otherwise mostly clear -- which a
real cloud essentially never does. On D17's NICFI, 4.4% of pixels are
flagged in >= 30% of the mostly-clear months (OCM at 4.77m is outside its
10-50m training range); Sentinel-2 at 10m has ~none.

Preprocessing, step "postcheck" (`run_postcheck`), on both sensors' mask
series. A second, spatial check (`despeckle_classes`) follows the temporal
one: clouds and their shadows are spatially coherent,
so a flagged blob smaller than MIN_CLOUD_AREA_M2 (8-connected, all
contaminated classes together) is classifier noise and is cleared, and a
clear hole that small inside a cloud is filled. This matters beyond the mask
itself: the cloud buffer dilates every flag by 3 px, turning each isolated
flagged pixel into a 7x7 hole -- the speckle seen in the S2 composites.

Temporal check, two-stage, so a real cloud passing over such a spot is
still caught:
1. **Persistent pixels**: flagged in >= `min_freq` of mostly-clear
   observations (scene contamination <= `mostly_clear_max`), and at least
   `min_count` times.
2. **Per-observation appearance test**: for those pixels only, the
   "typical appearance" is the median blue/NIR over mostly-clear
   observations (i.e. what the ground looks like, flagged or not). A flag
   is overridden to clear only where that observation looks like the
   ground -- blue and NIR both within tolerance. A genuine cloud (much
   brighter blue) or shadow (much darker NIR) on the same spot keeps its
   flag.
"""
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

import rasterio

from .. import cloud_mask as cm, io_utils
from . import config as cfg, masking

CONTAMINATED = (cm.CLOUD_THICK, cm.CLOUD_THIN, cm.SHADOW, cm.HAZE)

MOSTLY_CLEAR_MAX = 0.3
MIN_FREQ = 0.25
MIN_COUNT = 3
TOL_BLUE = (80.0, 0.35)   # DN floor, relative -- haze/cloud raise blue most
TOL_NIR = (400.0, 0.25)   # shadows lower NIR most
MIN_CLOUD_AREA_M2 = 2500.0  # 50 m x 50 m; smaller flagged blobs are speckle


def pixel_area_m2(transform, crs) -> float:
    """Approximate ground area of one pixel (geographic CRS: at the grid's
    first-row latitude, adequate for a ~10 km tile)."""
    dx, dy = abs(transform.a), abs(transform.e)
    if crs.is_geographic:
        lat = np.deg2rad(transform.f)
        return dx * 111_320 * np.cos(lat) * dy * 110_574
    return dx * dy


def min_blob_px(transform, crs, area_m2: float = MIN_CLOUD_AREA_M2) -> int:
    return max(1, int(round(area_m2 / pixel_area_m2(transform, crs))))


def despeckle(mask: np.ndarray, min_px: int) -> np.ndarray:
    """mask with 8-connected components smaller than `min_px` removed."""
    if not mask.any():
        return mask
    lab, n = ndimage.label(mask, structure=np.ones((3, 3), bool))
    sizes = np.bincount(lab.ravel())
    keep = sizes >= min_px
    keep[0] = False
    return keep[lab]


def despeckle_classes(classes: np.ndarray, min_px: int):
    """Spatial clean-up of one (H,W) class map, symmetric in both directions:
    - contaminated blobs smaller than `min_px` -> CLEAR (speckle, not cloud);
    - clear holes smaller than `min_px` enclosed by contamination -> the
      class of the nearest contaminated pixel (a 50 m gap inside a cloud is
      not a trustworthy clear view; left in, it leaves isolated unreplaced
      NICFI pixels inside a fill, or isolated "valid" S2 pixels at cloud
      edges -- the speckle seen in the fills).
    Returns (cleaned classes, bool mask of pixels changed)."""
    flagged = np.isin(classes, CONTAMINATED)
    removed = flagged & ~despeckle(flagged, min_px)
    out = classes.copy()
    out[removed] = cm.CLEAR

    flagged = flagged & ~removed
    clear = ~flagged & (classes != cm.NODATA)
    holes = clear & ~despeckle(clear, min_px)
    if holes.any() and flagged.any():
        _, (ri, ci) = ndimage.distance_transform_edt(~flagged, return_indices=True)
        out[holes] = classes[ri[holes], ci[holes]]
    return out, removed | holes


@dataclass
class RecheckResult:
    refined: np.ndarray      # (T,H,W) uint8 classes after the check
    overridden: np.ndarray   # (T,H,W) bool, flag -> clear by the check
    flag_freq: np.ndarray    # (H,W) float32, flag frequency in mostly-clear obs
    persistent: np.ndarray   # (H,W) bool
    mostly_clear: np.ndarray  # (T,) bool


def _within(v, typ, tol):
    return np.abs(v - typ) <= np.maximum(tol[0], tol[1] * typ)


def recheck(classes: np.ndarray, blue: np.ndarray, nir: np.ndarray,
            mostly_clear_max=MOSTLY_CLEAR_MAX, min_freq=MIN_FREQ,
            min_count=MIN_COUNT) -> RecheckResult:
    """classes: (T,H,W) cloud_mask codes; blue/nir: (T,H,W) DN, same grid."""
    valid = classes != cm.NODATA
    dirty = np.isin(classes, CONTAMINATED)
    frame_frac = (dirty & valid).sum((1, 2)) / np.maximum(valid.sum((1, 2)), 1)
    mc = frame_frac <= mostly_clear_max

    n_obs = valid[mc].sum(0)
    n_flag = dirty[mc].sum(0)
    freq = (n_flag / np.maximum(n_obs, 1)).astype(np.float32)
    persistent = (freq >= min_freq) & (n_flag >= min_count)

    refined = classes.copy()
    overridden = np.zeros(classes.shape, dtype=bool)
    idx = np.flatnonzero(persistent.ravel())
    if idx.size:
        T = classes.shape[0]
        b = blue.reshape(T, -1)[:, idx].astype(np.float32)
        n = nir.reshape(T, -1)[:, idx].astype(np.float32)
        v = valid.reshape(T, -1)[:, idx]
        use = v & mc[:, None]
        typ_b = np.nanmedian(np.where(use, b, np.nan), axis=0)
        typ_n = np.nanmedian(np.where(use, n, np.nan), axis=0)
        ground_like = _within(b, typ_b, TOL_BLUE) & _within(n, typ_n, TOL_NIR)
        ov = dirty.reshape(T, -1)[:, idx] & ground_like
        flat_ov = overridden.reshape(T, -1)
        flat_ov[:, idx] = ov
        flat_ref = refined.reshape(T, -1)
        sub = flat_ref[:, idx]
        sub[ov] = cm.CLEAR
        flat_ref[:, idx] = sub
    return RecheckResult(refined, overridden, freq, persistent, mc)


def _run_series(name, paths, key, load_blue_nir, out_path, min_px, temporal=True, log=print):
    classes = np.stack([np.load(p)[key] for p in paths])
    if temporal:
        blue, nir = zip(*(load_blue_nir(i) for i in range(len(paths))))
        res = recheck(classes, np.stack(blue), np.stack(nir))
    else:
        z = np.zeros(classes.shape, bool)
        res = RecheckResult(classes.copy(), z, np.zeros(classes.shape[1:], np.float32),
                            z[0], np.ones(len(paths), bool))
    n_speckle = 0
    for i, p in enumerate(paths):
        refined, despeckled = despeckle_classes(res.refined[i], min_px) if min_px > 1 else \
            (res.refined[i], np.zeros(classes.shape[1:], bool))
        n_speckle += int(despeckled.sum())
        z = dict(np.load(p))
        z.update(refined=refined, overridden=res.overridden[i], despeckled=despeckled)
        np.savez_compressed(p, **z)
    np.savez_compressed(out_path, flag_freq=res.flag_freq, persistent=res.persistent,
                        mostly_clear=res.mostly_clear, names=np.array([p.stem for p in paths]))
    dirty = np.isin(classes, CONTAMINATED)
    log(f"  {name}: {len(paths)} obs ({res.mostly_clear.sum()} mostly clear), persistent px "
        f"{res.persistent.mean():.2%}, flags overridden {res.overridden.sum() / max(dirty.sum(), 1):.2%} "
        f"of all flags; px changed by speckle clean-up (blobs/holes < {min_px} px): {n_speckle:,}")


def run_postcheck(tile: cfg.Tile, sensors=("nicfi", "s2"), temporal=True,
                  min_area_m2: float = MIN_CLOUD_AREA_M2, log=print):
    """Always over the *whole* series on disk (tile.all_months): the temporal
    check is only meaningful with every observation in it."""
    months = tile.all_months
    if "nicfi" in sensors:
        paths = [masking.nicfi_mask_path(tile, m) for m in months]

        def nicfi_bn(i):
            a, _, _ = io_utils.read_full(tile.nicfi_path(months[i]))
            return a[0].astype(np.uint16), a[3].astype(np.uint16)
        with rasterio.open(tile.nicfi_path(months[0])) as src:
            min_px = min_blob_px(src.transform, src.crs, min_area_m2)
        _run_series("NICFI", paths, "quality", nicfi_bn, tile.cache_dir / "postcheck_nicfi.npz",
                    min_px, temporal, log)

    if "s2" in sensors and tile.has_s2:
        frames = [f for m in months for f in tile.s2_frames(m)]
        paths = [masking.s2_mask_path(tile, f) for f in frames]

        def s2_bn(i):
            with rasterio.open(frames[i]) as src:
                bi = masking.band_index(list(src.descriptions))
                return src.read(bi["B2"] + 1), src.read(bi["B8"] + 1)
        tr, crs = masking.s2_grid(tile)
        _run_series("Sentinel-2", paths, "classes", s2_bn, tile.cache_dir / "postcheck_s2.npz",
                    min_blob_px(tr, crs, min_area_m2), temporal, log)
