"""
Preprocessing, step "s2composite": raw single-date Sentinel-2 frames -> one
clear monthly composite (needed by `reconstruct --method s2fill`).

1. **Clear mask per frame** from the post-checked OCM classes
   (`masking.load_s2_classes`): thick cloud, thin cloud and shadow are
   not-clear, buffered by `buffer_px` (cloud edges and faint shadow fringes
   are what a per-pixel classifier misses most).
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
from . import config as cfg, postcheck
from .masking import band_index, load_s2_classes, read_frame, s2_mask_path

SOURCE_NONE = 0     # no clear observation this month
SOURCE_SINGLE = 1   # the month's single cloud-free frame
SOURCE_MEDIAN = 2   # per-pixel median of clear observations

DEFAULT_CLEAR_THRESH = 0.99
DEFAULT_BUFFER_PX = 3  # 30m at 10m
HAZE_REF_PCT = 25
HAZE_ABS_DN = 100
HAZE_REL = 0.5


def clear_mask(classes: np.ndarray, buffer_px: int = DEFAULT_BUFFER_PX) -> np.ndarray:
    contaminated = np.isin(classes, (cloud_mask.CLOUD_THICK, cloud_mask.CLOUD_THIN,
                                     cloud_mask.SHADOW))
    if buffer_px > 0 and contaminated.any():
        contaminated = ndimage.binary_dilation(contaminated, iterations=buffer_px)
    return ~contaminated & (classes != cloud_mask.NODATA)


def blue_reference(tile: cfg.Tile, year: str, buffer_px: int = DEFAULT_BUFFER_PX) -> np.ndarray:
    """(H,W) float32 per-pixel HAZE_REF_PCT-th percentile of OCM-clear blue
    over every frame of `year` (NaN where never clear). Cached within one
    `run_s2_composites` call (which clears it first)."""
    path = tile.cache_dir / f"s2_blue_ref_{year}_b{buffer_px}.npz"
    if path.exists():
        return np.load(path)["ref"]
    blues = []
    for f in tile.s2_frames(year):
        with rasterio.open(f) as src:
            b2 = src.read(band_index(list(src.descriptions))["B2"] + 1).astype(np.float32)
        clear = clear_mask(load_s2_classes(tile, f), buffer_px)
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
    with rasterio.open(frames[0]) as src:
        min_px = postcheck.min_blob_px(src.transform, src.crs)
    stack, clears, frame_clear = [], [], {}
    transform = crs = names = None
    for f in frames:
        if not s2_mask_path(tile, f).exists():
            raise FileNotFoundError(f"no cloud mask cached for {f.name} -- run "
                                    f"`preprocess --steps mask` first")
        data, f_tr, f_crs, f_names = read_frame(f)
        if transform is None:
            transform, crs, names = f_tr, f_crs, f_names
        elif data.shape[1:] != stack[0].shape[1:] or f_tr != transform or f_names != names:
            raise ValueError(f"{f.name}: grid/bands differ from {frames[0].name} -- "
                             f"frames must share one grid; refusing to misalign")
        clear = clear_mask(load_s2_classes(tile, f), buffer_px)
        # haze-test rejections: like cloud flags, isolated ones are speckle
        hazy = clear & ~haze_free(data[band_index(f_names)["B2"]], ref)
        clear &= ~postcheck.despeckle(hazy, min_px)
        clear = postcheck.despeckle(clear, min_px)  # and isolated clear specks
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


def run_s2_composites(tile: cfg.Tile, clear_thresh: float = DEFAULT_CLEAR_THRESH,
                      buffer_px: int = DEFAULT_BUFFER_PX, log=print):
    # The blue references are derived from the (post-checked) masks, so they
    # are rebuilt on every run -- an earlier version cached them forever,
    # which silently kept references built from the raw, pre-post-check
    # masks after the masks had changed.
    for f in tile.cache_dir.glob("s2_blue_ref_*.npz"):
        f.unlink()
    summary_path = tile.cache_dir / "s2_composite_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    for month in tile.months:
        c = monthly_composite(tile, month, clear_thresh, buffer_px)
        if c is None:
            summary[month] = {"method": "none", "n_frames": 0, "valid_frac": 0.0}
            log(f"  {month}: no Sentinel-2 frames")
            continue
        save(tile, month, c)
        summary[month] = {"method": c.method, "n_frames": len(c.frame_clear),
                          "valid_frac": float(c.valid.mean()), "frame_clear": c.frame_clear}
        log(f"  {month}: {c.method:18s} frames={len(c.frame_clear)} "
            f"best-frame clear={max(c.frame_clear.values()):.1%} composite valid={c.valid.mean():.1%}")
    summary_path.write_text(json.dumps(summary, indent=1, sort_keys=True))
