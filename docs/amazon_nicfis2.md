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
python scripts/amazon_nicfis2/01_cloudmask_s2.py   --tile D17   # GPU, ~13 min
python scripts/amazon_nicfis2/02_s2_composite.py   --tile D17
python scripts/amazon_nicfis2/03_cloudmask_nicfi.py --tile D17   # GPU, ~3 min
python scripts/amazon_nicfis2/04_reconstruct.py    --tile D17
python scripts/amazon_nicfis2/05_visualize.py      --tile D17
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
3. **Monthly S2 composite** -- if the clearest frame is clear over >= 99%
   of the tile, that *single* frame is used (one coherent acquisition); else
   the per-pixel median of clear observations. D17: 17 months single-frame,
   43 median.
4. **NICFI quality mask** -- the existing OCM ensemble + haze heuristic
   (`cloud_mask.compute_masks`), buffered 3 px (~15 m).
5. **Harmonization S2 -> NICFI, per month and local** -- robust quantile
   matching (slope = NICFI p10-p90 spread / S2 spread, intercept from the
   medians) on pixels clear in both, at 10 m. First a whole-tile fit (the
   prior + a trust check), then refined per ~640 m block, shrunk toward the
   prior where a block has few clear-in-both pixels, and smoothed across
   blocks. Applied at 10 m, then upsampled bilinearly to NICFI's 4.77 m.
6. **Replacement** -- contaminated/nodata NICFI pixels with clear S2 under
   them get the harmonized S2; the blend ramps *outward* over 8 px into
   clear NICFI (never lets a cloudy pixel through at partial weight).
   Per-pixel provenance: 0 NICFI clear, 1 replaced by S2, 2 contaminated
   but kept (no clear S2 that month), 3 nodata.

Outputs: `outputs/amazon_nicfis2/<tile>/reconstructed/<tile>_<month>_recon.tif`
(4-band uint16, NICFI scale) + `_provenance.tif`. Figures in `.../figures/`.

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

## Results on D17 (`figures/01_overview.png`, `03_clouds_<year>.png`, `04_full_tile.png`)

- **Works where S2 saw the ground.** Thick cloud removed cleanly with no
  visible seam in e.g. 2024-04 (S2 median), 2025-02 (single frame
  2025-02-20, 40.6% of the tile replaced), 2025-05 (single frame). Fits are
  well-determined: clear-in-both red r 0.70-0.98, median 0.94, no month
  below the 0.6 trust threshold; 4 months had too few clear-in-both pixels
  and borrowed their year's median fit.
- **Main limitation: the wet season.** When NICFI is cloudy, S2 usually is
  too: 2021-12 (70.8% contaminated, S2 composite 0% clear -> 0% replaced),
  2021-02, 2022-03, 2024-12, 2025-12 all keep most contamination. Of all
  NICFI contamination over 60 months, about a quarter (25%) gets replaced.
- **2024-09 is smoke, not cloud** (fire season): 98% of the tile flagged
  thin cloud in NICFI, S2 equally affected; nothing to replace it with.
- **Residual S2 contamination in medians.** Some median-composite fills
  still carry faint cloud (e.g. 2024-03, 2025-09 windows) and speckle where
  only one or two observations survived masking.
- No independent ground truth: judged visually and via the fit statistics.

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
