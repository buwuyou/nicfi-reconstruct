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

Two-stage, so a real cloud passing over such a spot is still caught:
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

from .. import cloud_mask as cm

CONTAMINATED = (cm.CLOUD_THICK, cm.CLOUD_THIN, cm.SHADOW, cm.HAZE)

MOSTLY_CLEAR_MAX = 0.3
MIN_FREQ = 0.25
MIN_COUNT = 3
TOL_BLUE = (80.0, 0.35)   # DN floor, relative -- haze/cloud raise blue most
TOL_NIR = (400.0, 0.25)   # shadows lower NIR most


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
