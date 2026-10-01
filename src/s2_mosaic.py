"""
Build one monthly Sentinel-2 mosaic per D02 month from however many
single-date frames exist that month (0-8 in the data on disk, verified
directly against D02_S2's filenames).

D02_S2 turns out not to be one uniform format, found only by checking the
actual pixel values rather than trusting the file format to be consistent:

1. **Invalid-pixel sentinel**: most of the archive (2021, 2023-2025) uses
   exact 0 as its invalid/nodata sentinel, but the 49 files dated 2022 use
   NaN instead (confirmed directly: those 49 are exactly the ones with any
   NaN at all; every other file has none). Checking only `arr != 0` mishandles
   this: in IEEE754, `NaN != 0` is `True`, so every NaN pixel in the 2022
   files was being counted *valid*.
2. **Reflectance scale**: those same 49 2022 files store true 0-1 reflectance
   floats (p99 ~0.3-0.5 across all of them), while every other file in the
   archive stores raw DN scaled by 10000 like NICFI (p99 in the thousands,
   confirmed for all 141 non-2022 files) -- a ~10000x unit mismatch, not a
   subtle one.

Both were only caught because `02_harmonize.py`'s regression fit on the raw,
unfixed data looked like NICFI and Sentinel-2 had a real one-year
radiometric discontinuity in 2022 -- they didn't; it was these two bugs
compounding. See `docs/amazon_d02_s2fusion.md` for the full writeup. The
fixes here are format-auto-detecting (NaN-or-zero validity, rescale if the
99th percentile of a file's real values is suspiciously low) rather than a
hardcoded "if year == 2022", so this is robust if the same issue turns up
anywhere else in the archive.
"""
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import rasterio


@dataclass
class S2Mosaic:
    data: np.ndarray       # (10,H,W) float32 raw DN, 0 = no valid observation
    valid: np.ndarray      # (H,W) bool, True where >=1 frame had data
    transform: "rasterio.Affine"
    crs: object
    n_frames: int


def s2_files_for_month(s2_dir: Path, month: str) -> List[Path]:
    return sorted(p for p in s2_dir.glob(f"{month}-*.tif") if p.suffix == ".tif")


# A real raw-DN Sentinel-2 pixel (scaled by 10000, like the rest of this
# archive and like NICFI) essentially never has its 99th percentile below
# this; a file storing 0-1 reflectance floats instead does (~0.3-0.5, see
# module docstring) -- there's a ~10000x gap between the two, so this
# threshold has a lot of margin either way.
_LOW_SCALE_P99_THRESHOLD = 50.0
_REFLECTANCE_RESCALE = 10000.0


def _read_frame(path: Path):
    """Read one frame, auto-correcting the 0-1-reflectance-scale format some
    D02_S2 files use instead of the archive's usual raw-DN scale (see module
    docstring). NaN is left as NaN here -- validity is handled by the caller.
    Returns (arr (10,H,W) float32, transform, crs)."""
    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)  # (10,H,W)
        transform, crs = src.transform, src.crs
    finite = np.isfinite(arr)
    # The percentile must be computed over *real* observations only (finite
    # AND nonzero) -- a first version used `finite` alone, which silently
    # breaks for a heavily-clouded frame (e.g. 6 real pixels out of >1M):
    # the 99th percentile of "finite" values there is dominated by
    # legitimate zeros, reads as ~0, and wrongly triggered a x10000 rescale
    # on an already-correctly-scaled file. Caught the same way as the other
    # two bugs in this module: by seeing an impossible value (a >1e7 "raw
    # DN") downstream in a harmonization QA plot, not by inspection here.
    real = finite & (arr != 0)
    if real.any() and np.percentile(arr[real], 99) < _LOW_SCALE_P99_THRESHOLD:
        arr = np.where(finite, arr * _REFLECTANCE_RESCALE, arr)
    return arr, transform, crs


def monthly_mosaic(s2_dir: Path, month: str) -> Optional[S2Mosaic]:
    """Per-pixel median across all valid frames for `month`. Returns None if
    there are zero Sentinel-2 frames that month (some months in D02_S2 have
    none) -- callers must handle that rather than assuming coverage."""
    files = s2_files_for_month(s2_dir, month)
    if not files:
        return None

    frames, valids = [], []
    transform = crs = ref_shape = None
    for f in files:
        arr, f_transform, f_crs = _read_frame(f)
        if transform is None:
            transform, crs, ref_shape = f_transform, f_crs, arr.shape
        elif arr.shape != ref_shape:
            raise ValueError(
                f"{month}: frame {f.name} shape {arr.shape} != {ref_shape} "
                f"(from {files[0].name}) -- D02_S2 frames for one month are "
                f"expected to share one grid; refusing to silently misalign."
            )
        valid_px = np.any(np.isfinite(arr) & (arr != 0), axis=0)  # (H,W) bool
        arr = np.nan_to_num(arr, nan=0.0)  # so a NaN-sentinel frame behaves like a 0-sentinel one below
        frames.append(arr)
        valids.append(valid_px)

    stack = np.stack(frames, axis=0)             # (N,10,H,W)
    valid_stack = np.stack(valids, axis=0)       # (N,H,W)
    valid_any = valid_stack.any(axis=0)          # (H,W)

    masked = np.where(valid_stack[:, None, :, :], stack, np.nan)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        # expected wherever valid_any is False (no frame had data at all there)
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        mosaic = np.nanmedian(masked, axis=0)    # (10,H,W); nan where valid_any is False
    mosaic = np.nan_to_num(mosaic, nan=0.0).astype(np.float32)

    return S2Mosaic(data=mosaic, valid=valid_any, transform=transform, crs=crs,
                     n_frames=len(files))
