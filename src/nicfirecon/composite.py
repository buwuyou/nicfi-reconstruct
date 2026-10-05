"""
Composites -- two types:

**annual** (`run_annual`): one image per calendar year from a monthly
source -- raw NICFI + post-checked masks (`--source nicfi`) or any
reconstruction method's output (`mask`, `s2fill`, `phenology`). Every
source carries a per-pixel 0-100 quality score (`reconstruct.py`; raw NICFI:
100 if clear, else 0), so one rule serves all of them: among the year's
months with score >= `min_score`, the least-hazy observation (`pick`, below;
or the per-band median); a pixel with no such month takes its best-scoring
month (tier 2). No cross-month normalization -- months within a year differ
for real (phenology), not just radiometrically.

**typical-year** (`run_typical_year`): one "typical year" of 12 monthly
images built from all years -- of raw NICFI (`--source nicfi`, no
Sentinel-2, described below) or of any method's monthly output, with the
same score rule as annual. For calendar month k, every
year's month-k observation is a sample of what that place looks like in
month k; a pixel cloudy in one year is usually clear in another.

Per calendar month:
1. **Clear observations** -- NICFI classes after the post-check
   (`postcheck.py`), CLEAR and outside the cloud buffer
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
   (`harmonize.local_fields`, here NICFI -> NICFI), and the composite is
   recomputed from the normalized observations.
4. **Fallback chain** (recorded per pixel as the `tier`):
   1 = clear same-month observations (all years)
   2 = none clear that month in any year -> clear observations
       from the adjacent months (k-1, k+1, all years)
   3 = still none -> the best-scoring same-month observation (raw NICFI:
       buffer-only > haze > thin > shadow > thick cloud), flagged as such
   0 = no data at all
"""
import json
import warnings
from collections import defaultdict
from typing import Dict, List

import numpy as np
import rasterio

from .. import cloud_mask, io_utils
from . import config as cfg, masking
from .harmonize import local_fields, qmatch
from .reconstruct import replace_mask

STATS = ("lowblue", "median")

TIER_NONE, TIER_SAME, TIER_ADJACENT, TIER_LEAST_BAD = 0, 1, 2, 3
TIER_NAMES = {TIER_SAME: "clear, same month", TIER_ADJACENT: "clear, adjacent month",
              TIER_LEAST_BAD: "least-contaminated obs", TIER_NONE: "no data"}
NORM_BLOCK = 128          # NICFI px, ~610 m
MIN_NORM_SAMPLES = 2000

def clear_obs(classes: np.ndarray) -> np.ndarray:
    return (classes == cloud_mask.CLEAR) & ~replace_mask(classes)


def _median(stack, use):
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN where nothing usable
        return np.nanmedian(np.where(use[:, None], stack, np.nan), axis=0).astype(np.float32)


def pick(stack, use, stat="lowblue"):
    """Per-pixel composite of (N,C,H,W) `stack` over `use` (N,H,W); NaN where
    nothing is usable. `lowblue`: one whole observation (see
    `_low_blue_pick`); `median`: per-band median."""
    if stat not in STATS:
        raise ValueError(f"stat must be one of {STATS}")
    return _low_blue_pick(stack, _drop_outliers(stack, use)) if stat == "lowblue" else _median(stack, use)


OUTLIER_BLUE_LOW = 0.5    # x the pixel's median blue: defective dark scene / unflagged shadow
OUTLIER_RED_HIGH = 3.0    # x the pixel's median red: red artifact


def _drop_outliers(stack, use):
    """`use` minus per-pixel spectral outliers, where a pixel has >= 3
    usable observations. Needed because the least-hazy rule *prefers* low
    blue: on D17 a defective 2021-11 NICFI scene (blue 79 / red 85 DN vs
    ~215 / ~230 usual, NIR normal, called 100% clear by OCM) was picked for
    the 2021 annual composite as a black blob, and a red artifact likewise.
    Pixels whose every observation would be dropped keep the original set."""
    if use.sum(0).max() < 3:
        return use
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(np.where(use[:, None], stack[:, [0, 2]], np.nan), axis=0)  # blue, red
    out = (stack[:, 0] < OUTLIER_BLUE_LOW * med[0]) | (stack[:, 2] > OUTLIER_RED_HIGH * med[1])
    kept = use & ~out
    enough = (use.sum(0) >= 3) & kept.any(0)
    return np.where(enough[None], kept, use)


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


def normalize(obs: np.ndarray, clear: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Map one (4,H,W) observation onto `target` radiometry, fitted on pixels
    clear in `obs` where `target` is defined. Returned unchanged if there
    are too few such pixels to fit."""
    usable = clear & np.all(np.isfinite(target), axis=0)
    if usable.sum() < MIN_NORM_SAMPLES:
        return obs
    tgt = np.nan_to_num(target)
    coeffs = {b: qmatch(obs[i][usable], tgt[i][usable]) for i, b in enumerate(cfg.NICFI_BAND_NAMES)}
    slope, inter = local_fields(tgt, usable, obs, coeffs, block=NORM_BLOCK)
    return np.where(obs > 0, np.clip(obs * slope + inter, 1.0, None), 0.0).astype(np.float32)


def composite_calendar_month(S: np.ndarray, s_use: np.ndarray, s_score: np.ndarray,
                             s_valid: np.ndarray, A: np.ndarray, a_use: np.ndarray,
                             stat: str = "lowblue") -> Dict:
    """S (N,4,H,W): this calendar month over all years, with its usable mask,
    0-100 score and valid (has-a-value) mask; A/a_use: the two adjacent
    calendar months over all years. Tier 3 takes the best-scoring valid
    same-month observation."""
    S, A = S.astype(np.float32), A.astype(np.float32)

    # pass 1: un-normalized target, only to normalize against
    target = pick(S, s_use, stat)
    target = np.where(np.isnan(target), pick(A, a_use, stat), target)

    # pass 2: normalize every observation onto it, recomposite
    Sn = np.stack([normalize(o, c, target) for o, c in zip(S, s_use)])
    An = np.stack([normalize(o, c, target) for o, c in zip(A, a_use)])
    med_s, med_a = pick(Sn, s_use, stat), pick(An, a_use, stat)

    h, w = S.shape[2:]
    tier = np.full((h, w), TIER_NONE, np.uint8)
    out = np.zeros((4, h, w), np.float32)
    has_s = s_use.any(0)
    has_a = a_use.any(0) & ~has_s
    out[:, has_s], tier[has_s] = med_s[:, has_s], TIER_SAME
    out[:, has_a], tier[has_a] = med_a[:, has_a], TIER_ADJACENT

    rest = ~has_s & ~has_a & s_valid.any(0)
    best = np.argmax(np.where(s_valid, s_score.astype(np.int16), -1), axis=0)
    least_bad = np.take_along_axis(Sn, best[None, None].repeat(4, axis=1), axis=0)[0]
    out[:, rest], tier[rest] = least_bad[:, rest], TIER_LEAST_BAD

    return {"data": out, "tier": tier,
            "n_clear_same": s_use.sum(0).astype(np.uint8),
            "n_clear_adjacent": a_use.sum(0).astype(np.uint8)}


def _write(path, data, layer, layer_names, tr, crs, band_names):
    io_utils.save_geotiff(path, data.clip(0, 65535).astype("uint16"), tr, crs,
                          dtype="uint16", nodata=0)
    qpath = path.with_name(path.stem + "_quality.tif")
    io_utils.save_geotiff(qpath, layer, tr, crs, dtype="uint8")
    for p, names in ((path, band_names), (qpath, layer_names)):
        with rasterio.open(p, "r+") as dst:
            dst.descriptions = tuple(names)


def run_typical_year(tile: cfg.Tile, source: str = "nicfi", stat: str = "lowblue",
                     min_score: int = 50, log=print) -> Dict:
    """12 calendar-month composites over the whole series on disk
    (tile.all_months), from raw NICFI (+ masks) or any method's output;
    an observation is usable if its score >= min_score."""
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}")
    out_dir = tile.composite_dir("typical_year", source)
    by_cal = defaultdict(list)
    for m in tile.all_months:
        by_cal[int(m[5:])].append(m)

    def load(months):
        ins = [source_inputs(tile, source, m) for m in months]
        score = np.stack([x[1] for x in ins])
        valid = np.stack([x[2] for x in ins])
        return (np.stack([x[0] for x in ins]), valid & (score >= min_score), score, valid,
                ins[0][3], ins[0][4])

    stats = {}
    for k in range(1, 13):
        if not by_cal[k]:
            continue
        S, s_use, s_score, s_valid, tr, crs = load(by_cal[k])
        A, a_use, _, _, _, _ = load(by_cal[(k - 2) % 12 + 1] + by_cal[k % 12 + 1])
        res = composite_calendar_month(S, s_use, s_score, s_valid, A, a_use, stat)
        _write(out_dir / f"{tile.tile_id}_m{k:02d}.tif", res["data"],
               np.stack([res["tier"], res["n_clear_same"], res["n_clear_adjacent"]]),
               ["tier", "n_usable_same", "n_usable_adjacent"], tr, crs, cfg.NICFI_BAND_NAMES)
        t = res["tier"]
        stats[k] = {name: float((t == code).mean()) for code, name in
                    ((TIER_SAME, "same"), (TIER_ADJACENT, "adjacent"),
                     (TIER_LEAST_BAD, "least_bad"), (TIER_NONE, "none"))}
        stats[k]["mean_n_clear_same"] = float(res["n_clear_same"].mean())
        log(f"  m{k:02d} ({len(by_cal[k])} yrs, source {source}): usable same month "
            f"{stats[k]['same']:.1%}, adjacent {stats[k]['adjacent']:.2%}, least-contaminated "
            f"{stats[k]['least_bad']:.2%}, mean usable yrs/px {stats[k]['mean_n_clear_same']:.1f}")
    (out_dir / "stats.json").write_text(json.dumps(
        {"source": source, "stat": stat, "min_score": min_score, "months": stats}, indent=1))
    return stats


SOURCES = ("nicfi", "mask", "s2fill", "phenology")
ANNUAL_TIER_OK, ANNUAL_TIER_BEST = 1, 2
# Raw NICFI has no quality layer; its score is set so that score >= 50 is
# exactly "clear and outside the cloud buffer" and the score order is the
# least-contaminated order (buffer-only < haze < thin < shadow < thick).
_NICFI_SCORE = np.zeros(256, np.uint8)
for _code, _sc in ((cloud_mask.CLEAR, 40), (cloud_mask.HAZE, 30), (cloud_mask.CLOUD_THIN, 20),
                   (cloud_mask.SHADOW, 10), (cloud_mask.CLOUD_THICK, 5)):
    _NICFI_SCORE[_code] = _sc


def source_inputs(tile: cfg.Tile, source: str, month: str):
    """(data (4,H,W), score (H,W) 0-100, valid (H,W), transform, crs) for one
    month of `source`: raw NICFI + post-checked masks, or a reconstruction
    method's output with its quality layer."""
    if source == "nicfi":
        data, tr, crs = io_utils.read_full(tile.nicfi_path(month))
        cls = masking.load_nicfi_classes(tile, month)
        score = np.where(clear_obs(cls), 100, _NICFI_SCORE[cls]).astype(np.uint8)
        return data, score, score > 0, tr, crs
    d = tile.recon_dir(source)
    path = d / f"{tile.tile_id}_{month}.tif"
    if not path.exists():
        raise FileNotFoundError(f"{path} -- run `reconstruct --method {source}` first")
    data, tr, crs = io_utils.read_full(path)
    q, _, _ = io_utils.read_full(d / f"{tile.tile_id}_{month}_quality.tif")
    return data[:4], q[4].astype(np.uint8), np.all(data[:4] > 0, axis=0), tr, crs


def run_annual(tile: cfg.Tile, source: str = "s2fill", stat: str = "lowblue",
               min_score: int = 50, log=print) -> Dict:
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}")
    out_dir = tile.composite_dir("annual", source)
    stats = {}
    for year in tile.years:
        months = [m for m in tile.months if m.startswith(year)]
        ins = [source_inputs(tile, source, m) for m in months]
        data = np.stack([x[0] for x in ins]).astype(np.float32)
        score = np.stack([x[1] for x in ins])
        valid = np.stack([x[2] for x in ins])
        use = valid & (score >= min_score)
        out = pick(data, use, stat)
        tier = np.where(use.any(0), ANNUAL_TIER_OK, 0).astype(np.uint8)
        rest = ~use.any(0) & valid.any(0)
        if rest.any():
            best = np.argmax(np.where(valid, score.astype(np.int16), -1), axis=0)
            fb = np.take_along_axis(data, best[None, None].repeat(4, axis=1), axis=0)[0]
            out[:, rest] = fb[:, rest]
            tier[rest] = ANNUAL_TIER_BEST
        out = np.nan_to_num(out, nan=0.0)
        _write(out_dir / f"{tile.tile_id}_{year}.tif", out,
               np.stack([tier, use.sum(0).astype(np.uint8)]), ["tier", "n_months_used"],
               ins[0][3], ins[0][4], cfg.NICFI_BAND_NAMES)
        stats[year] = {"n_months": len(months), "tier_ok": float((tier == ANNUAL_TIER_OK).mean()),
                       "tier_best": float((tier == ANNUAL_TIER_BEST).mean()),
                       "mean_months_used": float(use.sum(0).mean())}
        log(f"  {year} ({len(months)} months, source {source}): >= {min_score} score "
            f"{stats[year]['tier_ok']:.1%}, best-month fallback {stats[year]['tier_best']:.2%}, "
            f"mean months used {stats[year]['mean_months_used']:.1f}")
    (out_dir / "stats.json").write_text(json.dumps(
        {"source": source, "stat": stat, "min_score": min_score, "years": stats}, indent=1))
    return stats
