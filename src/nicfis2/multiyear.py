"""
Single-sensor alternative: one "typical year" of monthly NICFI images built
from all years of NICFI alone (no Sentinel-2). For calendar month k, every
year's month-k observation is a sample of what that place looks like in
month k; a pixel cloudy in one year is usually clear in another.

Per calendar month:
1. **Clear observations** -- NICFI classes after the temporal post-check
   (`temporal_mask.py`), CLEAR and outside the cloud buffer
   (`reconstruct.replace_mask`).
2. **Least-hazy selection, not the median.** Per pixel, clear observations
   are ranked by blue; the darkest one is set aside as a possible unflagged
   cloud shadow when there are >= 3, and the next one is taken -- the whole
   4-band observation, so the spectrum stays physically consistent. A
   first version used the per-band median: on D17 the February composite
   came out as a hazy blue veil, because 4 of the 5 Februaries carry a thin
   haze neither mask catches -- a median then *is* haze; haze raises blue,
   so ranking by blue selects against it. A second version took the lower
   quartile with floor(), which for the 3-4 clear years typical of wet
   months is the *minimum* -- dark blotches from unflagged shadows in the
   Jan/Feb/Nov/Dec composites.
3. **Radiometric normalization across years.** NICFI years differ (e.g.
   D17 2021-01 has ~4x flatter green/red contrast than other Januaries
   over the same forest), and a per-pixel median whose contributing years
   change from pixel to pixel turns that into patchiness. So a first-pass
   composite is computed, then every observation is mapped onto it with the
   same local quantile matching used for S2 -> NICFI
   (`reconstruct.local_fields`, here NICFI -> NICFI), and the composite is
   recomputed from the normalized observations.
4. **Fallback chain** (recorded per pixel as the `tier`):
   1 = clear same-month observations (all years)
   2 = none clear that month in any year -> clear observations
       from the adjacent months (k-1, k+1, all years)
   3 = still none -> the least-contaminated same-month observation
       (buffer-only < haze < thin < shadow < thick cloud), flagged as such
   0 = no data at all
"""
from typing import Dict, List

import numpy as np

from .. import cloud_mask
from . import config as cfg
from .reconstruct import SLOPE_RANGE, local_fields, replace_mask

TIER_NONE, TIER_SAME, TIER_ADJACENT, TIER_LEAST_BAD = 0, 1, 2, 3
TIER_NAMES = {TIER_SAME: "clear, same month", TIER_ADJACENT: "clear, adjacent month",
              TIER_LEAST_BAD: "least-contaminated obs", TIER_NONE: "no data"}
NORM_BLOCK = 128          # NICFI px, ~610 m
MIN_NORM_SAMPLES = 2000

# lower = less contaminated, used only for tier 3 (lookup table by class code)
_NO_OBS = 99
_BADNESS = np.full(256, _NO_OBS, np.uint8)
for _code, _rank in ((cloud_mask.CLEAR, 0), (cloud_mask.HAZE, 1), (cloud_mask.CLOUD_THIN, 2),
                     (cloud_mask.SHADOW, 3), (cloud_mask.CLOUD_THICK, 4)):
    _BADNESS[_code] = _rank


def clear_obs(classes: np.ndarray) -> np.ndarray:
    return (classes == cloud_mask.CLEAR) & ~replace_mask(classes)


def _low_blue_pick(stack, use):
    """stack (N,4,H,W), use (N,H,W). Per pixel, its usable observations
    sorted by blue: the second-lowest if there are >= 3 (lowest set aside as
    a possible shadow), else the lowest; NaN where there are none."""
    blue = np.where(use, stack[:, 0], np.inf)
    order = np.argsort(blue, axis=0)
    n = use.sum(0)
    rank = (n >= 3).astype(np.int64)
    pick = np.take_along_axis(order, rank[None], axis=0)  # (1,H,W)
    out = np.take_along_axis(stack, pick[:, None].repeat(stack.shape[1], axis=1), axis=0)[0]
    return np.where(n[None] > 0, out, np.nan).astype(np.float32)


def _qmatch(x, y):
    x10, x50, x90 = np.percentile(x, [10, 50, 90])
    y10, y50, y90 = np.percentile(y, [10, 50, 90])
    slope = float(np.clip((y90 - y10) / max(x90 - x10, 1e-6), *SLOPE_RANGE))
    return slope, float(y50 - slope * x50)


def normalize(obs: np.ndarray, clear: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Map one (4,H,W) observation onto `target` radiometry, fitted on pixels
    clear in `obs` where `target` is defined. Returned unchanged if there
    are too few such pixels to fit."""
    usable = clear & np.all(np.isfinite(target), axis=0)
    if usable.sum() < MIN_NORM_SAMPLES:
        return obs
    tgt = np.nan_to_num(target)
    coeffs = {b: _qmatch(obs[i][usable], tgt[i][usable]) for i, b in enumerate(cfg.NICFI_BAND_NAMES)}
    slope, inter = local_fields(tgt, usable, obs, coeffs, block=NORM_BLOCK)
    return np.where(obs > 0, np.clip(obs * slope + inter, 1.0, None), 0.0).astype(np.float32)


def composite_calendar_month(same: List[np.ndarray], same_cls: List[np.ndarray],
                             adj: List[np.ndarray], adj_cls: List[np.ndarray]) -> Dict:
    """same/adj: lists of (4,H,W) NICFI DN for this calendar month / the two
    adjacent months over all years, with their (H,W) class maps."""
    S, A = np.stack(same).astype(np.float32), np.stack(adj).astype(np.float32)
    Sc, Ac = np.stack(same_cls), np.stack(adj_cls)
    s_clear = np.stack([clear_obs(c) for c in Sc])
    a_clear = np.stack([clear_obs(c) for c in Ac])

    # pass 1: un-normalized target, only to normalize against
    target = _low_blue_pick(S, s_clear)
    target = np.where(np.isnan(target), _low_blue_pick(A, a_clear), target)

    # pass 2: normalize every observation onto it, recomposite
    Sn = np.stack([normalize(o, c, target) for o, c in zip(S, s_clear)])
    An = np.stack([normalize(o, c, target) for o, c in zip(A, a_clear)])
    med_s, med_a = _low_blue_pick(Sn, s_clear), _low_blue_pick(An, a_clear)

    h, w = S.shape[2:]
    tier = np.full((h, w), TIER_NONE, np.uint8)
    out = np.zeros((4, h, w), np.float32)
    has_s = s_clear.any(0)
    has_a = a_clear.any(0) & ~has_s
    out[:, has_s], tier[has_s] = med_s[:, has_s], TIER_SAME
    out[:, has_a], tier[has_a] = med_a[:, has_a], TIER_ADJACENT

    rest = ~has_s & ~has_a
    badness = _BADNESS[Sc]
    best = np.argmin(badness, axis=0)
    usable_rest = rest & (np.min(badness, axis=0) < _NO_OBS)
    pick = np.take_along_axis(Sn, best[None, None].repeat(4, axis=1), axis=0)[0]
    out[:, usable_rest], tier[usable_rest] = pick[:, usable_rest], TIER_LEAST_BAD

    return {"data": out, "tier": tier,
            "n_clear_same": s_clear.sum(0).astype(np.uint8),
            "n_clear_adjacent": a_clear.sum(0).astype(np.uint8)}
