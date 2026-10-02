# NICFI + Sentinel-2: replacing cloudy NICFI pixels with same-month Sentinel-2

A third method alongside the temporal pipeline (`docs/vietnam.md`,
`docs/amazon.md`). Instead of interpolating a cloudy NICFI pixel from other
months, it substitutes a real, clear Sentinel-2 observation from the *same
month*, mapped onto NICFI's radiometry. Site-agnostic: code in
`src/nicfis2/`, scripts in `scripts/amazon_nicfis2/`, every script takes
`--tile <ID>`. First (and so far only) test tile: **D17**, 60 months
(2021-01..2025-12) of NICFI + 379 raw, unmasked single-date Sentinel-2
frames.

```bash
python scripts/amazon_nicfis2/01_cloudmask_s2.py        --tile D17   # GPU, ~13 min
python scripts/amazon_nicfis2/02_cloudmask_nicfi.py     --tile D17   # GPU, ~3 min
python scripts/amazon_nicfis2/03_temporal_mask_check.py --tile D17   # ~1 min
python scripts/amazon_nicfis2/04_s2_composite.py        --tile D17
python scripts/amazon_nicfis2/05_reconstruct.py         --tile D17
python scripts/amazon_nicfis2/06_visualize.py           --tile D17
python scripts/amazon_nicfis2/07_nicfi_multiyear_monthly.py --tile D17   # NICFI-only alternative
```
Data layout (defaults, overridable): `<data-root>/<TILE>/*YYYY-MM.tif`
(NICFI) and `<data-root>/<TILE>_S2/YYYY-MM-DD.tif` (Sentinel-2, band
descriptions B2..B12). `--months 2023 2024-05` restricts to month prefixes.

## Method

1. **Sentinel-2 cloud mask per frame** -- OmniCloudMask, the same
   2-generation ensemble as NICFI (`cloud_mask.ocm_ensemble`), thick/thin
   cloud + shadow, buffered 3 px (30 m).
2. **Haze rejection on top of OCM** -- OCM-clear observations whose blue
   is > max(100 DN, 0.5x) above that pixel's per-year reference (25th
   percentile of its OCM-clear blue) are dropped, then the mask is opened
   to remove speckle. Needed: on 2021-01, OCM-"clear" S2 had blue median
   393 vs ~190 for clear forest.
3. **Temporal post-check of both cloud masks** (`temporal_mask.py`) --
   clouds move, ground doesn't. A pixel flagged in >= 25% (and >= 3) of the
   *mostly-clear* observations (scene <= 30% contaminated) is "persistent";
   for those pixels only, a flag is overridden to clear in any observation
   where it looks like the pixel's typical appearance (median blue/NIR over
   mostly-clear observations; blue within max(80 DN, 35%), NIR within
   max(400 DN, 25%)). A real cloud (much brighter blue) or shadow (much
   darker NIR) over the same spot keeps its flag. D17: NICFI 6.1% of pixels
   persistent, 15% of all NICFI flags overridden -- mostly pink/bright
   canopy and bare patches OCM calls cloud at 4.77m; Sentinel-2 at 10m:
   none. Then a **spatial** check: clouds are spatially coherent, so
   flagged blobs smaller than 2500 m² (110 NICFI px / 26 S2 px) are
   cleared as speckle, and clear holes that small inside a cloud are filled
   with the nearest flagged class (D17: 633k NICFI and 63k S2 pixel-obs
   changed). Without it, every isolated flag became a 7x7 hole after the
   3 px buffer, and every isolated clear pixel inside a cloud stayed
   unreplaced -- the speckle in the fills. The same clean-up is applied to
   the S2 haze-test rejections and the final S2 clear mask. Raw OCM classes
   are kept in the cache next to the re-checked ones.
4. **Monthly S2 composite** -- if the clearest frame is clear over >= 99%
   of the tile, that *single* frame is used (one coherent acquisition); else
   the per-pixel median of clear observations. D17: 17 months single-frame,
   43 median.
5. **NICFI quality mask** -- the existing OCM ensemble + haze heuristic
   (`cloud_mask.compute_masks`), buffered 3 px (~15 m).
6. **Harmonization S2 -> NICFI, per month and local** -- robust quantile
   matching (slope = NICFI p10-p90 spread / S2 spread, intercept from the
   medians) on pixels clear in both, at 10 m. First a whole-tile fit (the
   prior + a trust check), then refined per ~640 m block, shrunk toward the
   prior where a block has few clear-in-both pixels, and smoothed across
   blocks. Applied at 10 m, then upsampled bilinearly to NICFI's 4.77 m.
7. **Replacement** -- contaminated/nodata NICFI pixels with clear S2 under
   them get the harmonized S2; the blend ramps *outward* over 8 px into
   clear NICFI (never lets a cloudy pixel through at partial weight).

Outputs per month in `outputs/amazon_nicfis2/<tile>/reconstructed/`:
`<tile>_<month>_recon.tif` (4-band uint16, NICFI scale) and
`<tile>_<month>_quality.tif`, a 5-band uint8 **data-quality layer** (band
descriptions set in the file):

| band | name | values |
|---|---|---|
| 1 | `source` | 0 NICFI clear · 1 S2 single cloud-free frame · 2 S2 median of clear obs · 3 contaminated NICFI kept (no clear S2) · 4 nodata |
| 2 | `nicfi_class` | NICFI cloud class after the temporal check: 0 clear · 1 thick · 2 thin · 3 shadow · 4 haze · 5 nodata |
| 3 | `flags` | bit 1: NICFI flag cleared by the post-check (temporal or speckle) · 2: edge blend (clear NICFI mixed with S2) · 4: S2 fit borrowed (year median) · 8: replaced only as cloud buffer |
| 4 | `s2_n_clear` | clear S2 observations that month |
| 5 | `score` | 0-100 heuristic confidence: NICFI clear 100 (overridden flag 90, edge blend 95); S2 single frame 80; S2 median 70 (>= 3 obs) / 60 (2) / 50 (1); -15 if the fit was borrowed; kept contaminated: buffer-only 60, haze 30, thin 20, shadow 10, thick 0; nodata 0 |

The score is a ranking aid for users, not a calibrated accuracy -- there's
no ground truth behind it. Figures in `.../figures/`.

## Why these choices (bugs/failures found on the way)

- **RANSAC regression -> quantile matching.** RANSAC returned slopes of
  0.3-0.65 even on a clean month (2021-06, r=0.85-0.93 in the visible
  bands): its default inlier band locks onto the dense forest cluster where
  the relation looks flat (regression dilution) -- every fill would have
  been washed out. On a noisy month it went negative. Quantile matching
  gives 0.74-1.02 on 2021-06 and can't flip sign.
- **Whole-tile fit -> local fit.** A NICFI monthly basemap stitches several
  PlanetScope scenes: on 2021-01 the lower tile has ~4x less green/red
  contrast than June over the same forest (p10-p90 green 31 vs 123 DN, same
  medians), the upper tile doesn't. One fit can't match both.
- **Super-resolution dropped.** SEN2SR (10 m -> 2.5 m) was tested on an
  earlier tile: faithful to its own input but no better than bilinear
  against NICFI, at ~17 GPU-min/month -- see `docs/super-resolution.md`.

## Results on D17 (`figures/01_overview.png`, `03_clouds_<year>.png`, `04_full_tile.png`, `05_temporal_check.png`)

- **Works where S2 saw the ground.** Thick cloud removed cleanly with no
  visible seam in e.g. 2024-04 (S2 median), 2025-02 (single frame
  2025-02-20, 40.6% of the tile replaced), 2025-05 (single frame). Fits are
  well-determined: clear-in-both red r 0.70-0.98, median 0.94, no month
  below the 0.6 trust threshold; 4 months had too few clear-in-both pixels
  and borrowed their year's median fit.
- **Main limitation: the wet season.** When NICFI is cloudy, S2 usually is
  too: 2021-12 (70.8% contaminated, S2 composite 0% clear -> 0% replaced),
  2021-02, 2022-03, 2024-12, 2025-12 all keep most contamination. Of all
  NICFI contamination over 60 months, about 24.5% gets replaced (after the temporal + spatial checks; mean per-pixel quality score over all months 91/100).
- **2024-09 is smoke, not cloud** (fire season): 98% of the tile flagged
  thin cloud in NICFI, S2 equally affected; nothing to replace it with.
- **Residual S2 contamination in medians.** Some median-composite fills
  still carry faint cloud (e.g. 2024-03, 2025-09 windows). Isolated speckle
  is gone after the spatial check; what remains in hazy wet-season months
  are larger, worm-shaped gaps -- S2's own fragmented clear area, a real
  data limit rather than noise.
- **Thin cloud is still over-called on NICFI in some months** (e.g. the
  2023-02 window: a fairly clear-looking scene mostly flagged thin cloud).
  Not persistent in time, so the temporal check can't catch it.
- The temporal check shares the limitation noted in `docs/amazon.md`: haze
  that recurs at the same spot often enough becomes part of that spot's
  "typical appearance" and could be overridden too.
- No independent ground truth: judged visually and via the fit statistics.

## Single-sensor alternative: one typical year from all NICFI years (step 7)

`07_nicfi_multiyear_monthly.py` (`src/nicfis2/multiyear.py`) builds 12
monthly images from NICFI only, no Sentinel-2: for calendar month k, every
year's month-k observation is a sample of that place in month k, and a
pixel cloudy in one year is usually clear in another.

- Clear = NICFI class after both post-checks, outside the cloud buffer.
- Per pixel, clear observations are ranked by blue (haze raises blue); the
  darkest is set aside as a possible unflagged shadow when there are >= 3,
  and the next one is taken as a whole 4-band observation. Two earlier
  versions failed visibly: the per-band median made February a hazy veil
  (4 of 5 Februaries carry undetected thin haze, so the median *is* haze);
  the lower quartile with floor() is the minimum for 3-4 clear years and
  left dark shadow blotches in the wet-season months.
- Every year is first normalized onto a first-pass composite with the same
  local quantile matching (NICFI -> NICFI), so pixels drawn from different
  years don't turn into patchiness. Fitted slopes are hard-clamped to
  [0.25, 4] (also for the S2 fill): a block whose band is nearly flat in one
  observation otherwise blew up -- the first December composite had a
  saturated red patch (13,804 DN from raw values of 127-431).
- Fallbacks, per pixel (`_quality.tif` band `tier`): clear same month ->
  clear adjacent months -> least-contaminated same-month observation.

Outputs: `nicfi_multiyear/<tile>_m<MM>.tif` + `_quality.tif` (tier,
n_clear_same, n_clear_adjacent); figures `multiyear_01_tiers.png`,
`multiyear_02_full_tile.png`, and `multiyear_03_<year>_r<row>_c<col>.png`
for the same example areas as `03_clouds_<year>.png` (every year's original
NICFI for each month, the composite, and the clear-year count behind it).

**D17:** every calendar month is 98.8-98.9% filled from clear same-month
observations (the remaining ~1.05% is the tile-edge nodata border; adjacent
months are needed for <= 0.16%, least-contaminated never), with on average
3.3 (Dec) to 4.9 (May-Aug) clear years per pixel. Compared to the
S2-fusion route (~24.5% of contamination replaced), this is the much more
complete product -- at the cost of being a *typical* year, not a specific
one: it shows seasonality (e.g. the pink canopy flushes Jun-Oct), not
year-specific change such as a new clearing or the 2024 fire-season
smoke.

## Possible next steps

1. Widen the S2 window for cloudy months (e.g. +-15-30 days into adjacent
   months) -- the most direct way to cut the "kept" fraction in the wet
   season, at the cost of a temporal mismatch.
2. Use NICFI's own texture: take the nearest clear NICFI observation of a
   pixel and apply the S2-measured change between that date and this
   month (a STARFM-like approach) -- 4.77 m detail from NICFI, temporal
   signal from S2.
3. Stricter residual-cloud check on median composites (e.g. require >= 2
   agreeing observations, or a temporal consistency test per pixel).
