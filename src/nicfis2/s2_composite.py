"""
Raw single-date Sentinel-2 frames -> one clear monthly composite.

1. **Cloud masking per frame** with the same OmniCloudMask ensemble the
   NICFI side uses (`src/cloud_mask.ocm_ensemble`, Red/Green/NIR input --
   Sentinel-2 at 10m is squarely inside OmniCloudMask's training domain).
   Thick cloud, thin cloud and shadow are all treated as not-clear, then
   buffered by `buffer_px` (cloud edges and faint shadow fringes are what a
   per-pixel classifier misses most). The raw OCM classes are cached per
   frame so the buffer/threshold can change without re-running the model.
2. **Compositing per month**:
   - if the clearest frame that month is (near-)cloud-free -- clear over
     >= `clear_thresh` of the tile -- that *single* frame is used as-is: one
     real acquisition, spatially coherent, no cross-date mixing. Its few
     remaining masked pixels (if any) are filled from the median of the
     month's other clear observations;
   - otherwise the per-pixel median of every clear observation that month.
   A per-pixel source map records which applied (see SOURCE_* below).

**Haze rejection on top of OCM.** OCM alone was not enough: on D17 2021-01
its "clear" pixels (mostly a single observation each) had blue median 393 DN
vs ~190 for clear forest in a clean month -- residual haze, which then broke
the S2->NICFI fit (r=0.19). Haze is additive and strongest in blue, so each
OCM-clear observation is also compared against a per-pixel, per-year blue
reference (25th percentile of that pixel's OCM-clear blue that calendar
year) and rejected if brighter by more than max(HAZE_ABS_DN, HAZE_REL x
reference). Per *year* rather than whole-archive so that a genuine
brightening (e.g. a new clearing) is only ever compared against the same
year's observations, limiting how long it could be mistaken for haze; if it
is, the effect is conservative -- that pixel keeps its NICFI value instead
of being replaced, never a wrong fill.
"""
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import rasterio
from scipy import ndimage

from .. import cloud_mask
from . import config as cfg

SOURCE_NONE = 0     # no clear observation this month
SOURCE_SINGLE = 1   # the month's single cloud-free frame
SOURCE_MEDIAN = 2   # per-pixel median of clear observations

DEFAULT_CLEAR_THRESH = 0.99
DEFAULT_BUFFER_PX = 3  # 30m at 10m
HAZE_REF_PCT = 25
HAZE_ABS_DN = 100
HAZE_REL = 0.5
NODATA_CLASS = cloud_mask.NODATA


def band_index(descriptions) -> Dict[str, int]:
    idx = {d: i for i, d in enumerate(descriptions) if d}
    missing = [b for b in cfg.NICFI_TO_S2_BAND.values() if b not in idx]
    if missing:
        raise ValueError(f"Sentinel-2 file lacks band descriptions for {missing} "
                         f"(has {descriptions}); can't map bands safely.")
    return idx


def read_frame(path: Path):
    """Returns (data (C,H,W) float32 DN with nodata as 0, transform, crs,
    band names). Nodata = all bands 0 or any band non-finite."""
    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)
        transform, crs, names = src.transform, src.crs, list(src.descriptions)
    bad = ~np.all(np.isfinite(data), axis=0) | np.all(data == 0, axis=0)
    data[:, bad] = 0.0
    return data, transform, crs, names


def ocm_classes(data: np.ndarray, names: List[str], device: str,
                model_versions=(3.0, 4.0)) -> np.ndarray:
    """(H,W) uint8: cloud_mask codes CLEAR/CLOUD_THICK/CLOUD_THIN/SHADOW/NODATA."""
    bi = band_index(names)
    rgn = np.stack([data[bi["B4"]], data[bi["B3"]], data[bi["B8"]]]) / cfg.REFLECTANCE_SCALE
    nodata = np.all(data == 0, axis=0)
    pred, _ = cloud_mask.ocm_ensemble(rgn, device=device, model_versions=model_versions)
    q = np.zeros(pred.shape, dtype=np.uint8)
    q[pred == 1] = cloud_mask.CLOUD_THICK
    q[pred == 2] = cloud_mask.CLOUD_THIN
    q[pred == 3] = cloud_mask.SHADOW
    q[nodata] = NODATA_CLASS
    return q


def clear_mask(classes: np.ndarray, buffer_px: int = DEFAULT_BUFFER_PX) -> np.ndarray:
    contaminated = np.isin(classes, (cloud_mask.CLOUD_THICK, cloud_mask.CLOUD_THIN,
                                     cloud_mask.SHADOW))
    if buffer_px > 0 and contaminated.any():
        contaminated = ndimage.binary_dilation(contaminated, iterations=buffer_px)
    return ~contaminated & (classes != NODATA_CLASS)


def blue_reference(tile: cfg.Tile, year: str, buffer_px: int = DEFAULT_BUFFER_PX) -> np.ndarray:
    """(H,W) float32 per-pixel HAZE_REF_PCT-th percentile of OCM-clear blue
    over every frame of `year` (NaN where never clear). Cached."""
    path = tile.cache_dir / f"s2_blue_ref_{year}_b{buffer_px}.npz"
    if path.exists():
        return np.load(path)["ref"]
    blues = []
    for f in tile.s2_frames(year):
        with rasterio.open(f) as src:
            b2 = src.read(band_index(list(src.descriptions))["B2"] + 1).astype(np.float32)
        clear = clear_mask(np.load(mask_cache_path(tile, f))["classes"], buffer_px)
        blues.append(np.where(clear, b2, np.nan))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        ref = np.nanpercentile(np.stack(blues), HAZE_REF_PCT, axis=0).astype(np.float32)
    np.savez_compressed(path, ref=ref)
    return ref


def haze_free(blue: np.ndarray, ref: np.ndarray) -> np.ndarray:
    tol = np.maximum(HAZE_ABS_DN, HAZE_REL * ref)
    with np.errstate(invalid="ignore"):
        return ~(blue > ref + tol)  # NaN ref (never clear that year) -> not rejected


def mask_cache_path(tile: cfg.Tile, frame: Path) -> Path:
    d = tile.cache_dir / "s2_masks"
    d.mkdir(exist_ok=True)
    return d / f"{frame.stem}.npz"


@dataclass
class Composite:
    data: np.ndarray        # (C,H,W) float32 DN, 0 where no clear obs
    valid: np.ndarray       # (H,W) bool
    source: np.ndarray      # (H,W) uint8, SOURCE_*
    n_clear: np.ndarray     # (H,W) uint8, clear observations that month
    method: str             # "single:<date>" or "median"
    frame_clear: Dict[str, float]
    transform: object
    crs: object
    band_names: List[str]


def monthly_composite(tile: cfg.Tile, month: str, clear_thresh: float = DEFAULT_CLEAR_THRESH,
                      buffer_px: int = DEFAULT_BUFFER_PX) -> Optional[Composite]:
    frames = tile.s2_frames(month)
    if not frames:
        return None
    ref = blue_reference(tile, month[:4], buffer_px)
    stack, clears, frame_clear = [], [], {}
    transform = crs = names = None
    for f in frames:
        mpath = mask_cache_path(tile, f)
        if not mpath.exists():
            raise FileNotFoundError(f"no cloud mask cached for {f.name} -- run "
                                    f"01_cloudmask_s2.py first")
        data, f_tr, f_crs, f_names = read_frame(f)
        if transform is None:
            transform, crs, names = f_tr, f_crs, f_names
        elif data.shape[1:] != stack[0].shape[1:] or f_tr != transform or f_names != names:
            raise ValueError(f"{f.name}: grid/bands differ from {frames[0].name} -- "
                             f"frames must share one grid; refusing to misalign")
        clear = clear_mask(np.load(mpath)["classes"], buffer_px)
        clear &= haze_free(data[band_index(f_names)["B2"]], ref)
        clear = ndimage.binary_opening(clear, iterations=1)  # drop haze-test speckle
        stack.append(data)
        clears.append(clear)
        frame_clear[f.stem] = float(clear.mean())

    stack = np.stack(stack)           # (N,C,H,W)
    clears = np.stack(clears)         # (N,H,W)
    n_clear = clears.sum(axis=0).astype(np.uint8)

    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        median = np.nanmedian(np.where(clears[:, None], stack, np.nan), axis=0)
    valid = n_clear > 0
    median = np.nan_to_num(median, nan=0.0).astype(np.float32)

    best = int(np.argmax([frame_clear[f.stem] for f in frames]))
    source = np.where(valid, SOURCE_MEDIAN, SOURCE_NONE).astype(np.uint8)
    if frame_clear[frames[best].stem] >= clear_thresh:
        method = f"single:{frames[best].stem}"
        use = clears[best]
        data = np.where(use[None], stack[best], median)
        source[use] = SOURCE_SINGLE
    else:
        method = "median"
        data = median

    return Composite(data=data, valid=valid, source=source, n_clear=n_clear, method=method,
                     frame_clear=frame_clear, transform=transform, crs=crs, band_names=names)


def composite_cache_path(tile: cfg.Tile, month: str) -> Path:
    return tile.cache_dir / f"s2_composite_{month}.npz"


def save(tile: cfg.Tile, month: str, c: Composite):
    np.savez_compressed(
        composite_cache_path(tile, month),
        data=np.clip(c.data, 0, 65535).astype(np.uint16), valid=c.valid, source=c.source,
        n_clear=c.n_clear, method=c.method, frame_clear=json.dumps(c.frame_clear),
        transform=tuple(c.transform)[:6], crs=c.crs.to_string(), band_names=np.array(c.band_names),
    )


def load(tile: cfg.Tile, month: str) -> Optional[Composite]:
    path = composite_cache_path(tile, month)
    if not path.exists():
        return None
    z = np.load(path, allow_pickle=False)
    return Composite(
        data=z["data"].astype(np.float32), valid=z["valid"], source=z["source"],
        n_clear=z["n_clear"], method=str(z["method"]), frame_clear=json.loads(str(z["frame_clear"])),
        transform=rasterio.Affine(*z["transform"]), crs=rasterio.crs.CRS.from_string(str(z["crs"])),
        band_names=[str(b) for b in z["band_names"]],
    )
