# D02: filling real NICFI gaps with super-resolved Sentinel-2

`docs/super-resolution.md` investigated using Sentinel-2 fusion to
reconstruct NICFI and left it as "a possible future direction, not
attempted." This is that direction, built for a new test tile, **D02**
(`/mnt/warehouse/amazon/D02`, 60 months) paired with a matching Sentinel-2
archive extract (`/mnt/warehouse/amazon/D02_S2`, 190 single-date frames).

This is a genuinely different problem from the rest of this repo's pipeline
(`amazon`/`vietnam`, tile D01): that pipeline reconstructs pixels NICFI
*captured but got wrong* (residual cloud/shadow/haze) via temporal
interpolation + a learned spatial residual. This one fills pixels NICFI's
own monthly mosaic simply has *no data for at all* (exact 0 across all 4
bands) — and does it with a second sensor's real observation, not a
model's temporal guess, because that's a strictly more defensible source
of truth for a genuine gap.

> **Update (2026-10-01): D02 turned out to have no genuine NICFI gaps.** Its
> only exact-0 pixels are a fixed 6-7px border along the four tile edges
> (0.589% of the tile, 4 blobs), pixel-for-pixel identical in all 60 months
> 2021-2025 -- a tile-edge footprint, not missing observations -- and
> Sentinel-2 doesn't cover it either (268 px fillable in 2022-01). So the
> gap-fill as designed has essentially nothing to do on this tile. See
> "Performance check" below for what the SR output *is* good for, and the
> open decision on where to take this next.

## D02_S2 is not one uniform, clean dataset -- three real bugs found by checking actual pixel values

The person who prepared D02_S2 masked it to forest land only, not just
cloud/shadow (confirmed directly by the person, after this was flagged as
suspicious: the invalid fraction sits at a near-constant ~18.5% of the tile
in almost every month regardless of frame count, which a cloud mask alone
wouldn't produce). A reprocessed, unmasked extract is expected eventually.

Separately, and only found by tracing an impossible-looking harmonization
fit back through the pipeline rather than trusting the fitted numbers,
every file dated **2022** (49 of the archive's 190 files) turned out to use
a different on-disk format than the rest of the archive:

1. **Invalid-pixel sentinel**: 2021/2023-2025 files use exact `0`; the 2022
   files use `NaN`. `src/s2_mosaic.py`'s first version only checked
   `arr != 0`, which silently mis-handles this -- in IEEE754, `NaN != 0` is
   `True`, so every NaN pixel in the 2022 files was being counted *valid*.
2. **Reflectance scale**: those same 49 files store true 0-1 reflectance
   floats (99th percentile of real values ~0.3-0.5), while the other 141
   files store raw DN scaled by 10000 like NICFI (99th percentile in the
   thousands) -- a ~10000x unit mismatch.
3. **A bug in the fix for #2**: the first version of the auto-rescale
   heuristic computed its "is this file low-scale?" percentile over all
   *finite* values, which for a heavily-clouded frame (e.g. 6 real pixels
   out of >1,000,000) is dominated by legitimate zeros and reads as
   near-zero, wrongly triggering a x10000 rescale on an already-correctly-
   scaled file and producing a >30,000,000 "raw DN" value.

All three were caught the same way: `02_harmonize.py`'s QA plot looked
wrong (a whole year collapsed to slope~intercept~0, or one month spiked the
residual axis to 1e7), and each was traced back to the actual pixel data
rather than accepted as a fitted number. `src/s2_mosaic.py`'s docstring has
the full detail. The fixes auto-detect the format per file rather than
hardcoding "if year == 2022", so they're robust if the same issue turns up
elsewhere in the archive.

## Scope: one year, for now

Given the forest-only masking (a real, external data-prep choice that a
reprocessing will change) and to avoid spending more time on quirks that
belong to specific bad months rather than the method itself, this is
currently tested against **2022 only** (`src/s2_fusion_site.py`'s
`D02_TEST_YEAR`) -- the one year in the archive where every one of its 12
months has consistent ~80-82% valid coverage (no missing or near-empty
months, unlike every other year: 2021/2023/2024/2025 each have 2-4 months
that are <10% valid or missing entirely). Once the reprocessed, unmasked
Sentinel-2 extract is available, `D02_TEST_YEAR` can be widened back to
`D02` (all 60 months) with no other code changes.

## Method

1. **Monthly Sentinel-2 mosaic** (`src/s2_mosaic.py`) — median across
   whatever single-date D02_S2 frames exist that month.
2. **Spectral harmonization, NICFI -> Sentinel-2** (`src/harmonize.py`) —
   NICFI's Blue/Green/Red/NIR bands correspond to Sentinel-2's B2/B3/B4/B8.
   A robust (RANSAC) per-band linear fit, pooled per calendar year (see
   below), maps NICFI's DN scale onto Sentinel-2's, so the final 10-band
   product (native NICFI + SR-filled gaps + the 6 SR-only bands) is
   radiometrically self-consistent. Fit at the coarser Sentinel-2
   resolution (NICFI area-averaged down onto the 10m grid) so upsampling
   blur never gets baked into the fitted slope/intercept.

   Fit *per calendar year* rather than pooled across the whole archive:
   this was originally meant as a robustness margin, but turned out to be
   what actually surfaced the 2022 format bugs above (a single pooled fit
   would have quietly averaged over them). With the bugs fixed and the
   scope narrowed to one year, this degenerates to exactly one fit — kept
   as per-year rather than ripped out, so widening `months` back to the
   full archive later doesn't require touching this logic again. 2022's
   fitted coefficients (`harmonized = raw*slope + intercept`): blue
   0.53x+111, green 0.89x+92, red 1.49x-190, nir 0.68x+694 — see
   `outputs/amazon_d02/figures/01_harmonization_fit.png` for the fit
   against the actual scatter and the per-month residual check (residuals
   are now bounded to a few hundred DN on a thousands-DN-scale range, a
   plausible level of real seasonal/BRDF noise, not a bug signature).
3. **Super-resolution** (`src/superres.py`) — the pretrained SEN2SR model
   (`model/LDSRS2-SEN2SR`, weights already on disk) takes the monthly
   10-band/10m Sentinel-2 mosaic and returns all 10 bands at 2.5m in one
   call (RGBN via a latent-diffusion x4 branch, red-edge/SWIR via a staged
   Swin2SR x2-then-x4 branch). Its `HardConstraint` module forces each
   output patch's low-frequency content back to what the input actually
   measured — this *is* the low-hallucination guardrail (the model can
   only add texture around a real measurement, not invent a new mean
   reflectance); nothing extra was built on top of it, just verified via
   `05_visualize.py`'s consistency-check figure.
4. **Fusion** (`src/fuse.py`) — the SR output is reprojected from its
   native 2.5m/EPSG:4326 grid onto NICFI's exact 4.77m/EPSG:3857 grid
   (area-average, since NICFI's pixel is coarser). Per band:
   - Blue/Green/Red/NIR: harmonized NICFI wherever NICFI has real data;
     SR-Sentinel-2 only where NICFI is a true gap, feather-blended across
     an ~8px ramp at the gap boundary (never a hard cutoff, same
     philosophy as `src/compose.py`'s confidence blending elsewhere in
     this repo).
   - Red-edge/SWIR (B5/B6/B7/B8A/B11/B12): SR-Sentinel-2 everywhere — pure
     additive value, NICFI has no equivalent band.
   A per-pixel provenance mask (0=NICFI-native, 1=SR-filled,
   2=still-unfilled) is written alongside every monthly product.

## Honest limitations / current status

- **Only 2022 is in scope right now** (see above) — the pipeline hasn't
  been run against the rest of the archive since the forest-only masking
  makes most other months' coverage too sparse to be a fair test of the
  method itself.
- **Super-resolution has run for 2022-01..06 on GPU** (RTX 3090, ~17.5 min
  per month, ~1040s; 07-12 still to do, resumable). 04/05 have not been
  run yet -- see "Performance check" below. Original note from before the
  GPU was available:
- **Not run end-to-end yet (superseded).** Steps 1-2 (mosaic building, harmonization
  fit) need no GPU and have been run for real against 2022. Step 3
  (super-resolution) needs a working GPU driver, which wasn't available
  while this was built. `src/superres.py` was smoke-tested for real on one
  128x128 patch of actual 2022-01 data with `--device cpu`: model +
  pretrained weights load correctly, output shape is exactly
  `(10, 512, 512)` as expected (4x), values land in a plausible raw-DN
  range (min/max -562/7695 -- the small negative excursion is a normal
  overshoot from the model's `HardConstraint` correction step, not an
  error; `04_fuse.py` already clips to `[0, 65000]` before writing output
  so it never reaches the final product). Took 196s for this one patch on
  CPU (200-step diffusion sampler), confirming a full tile/month run needs
  the GPU, not just "works but slow." Once the driver is fixed, run the
  real `03_superresolve.py`, then `04_fuse.py` and `05_visualize.py`, and
  look at the actual figures before trusting any month's output.
- **No independent cloud check on either sensor's "valid" pixels** beyond
  the format bugs above. The harmonization fit uses RANSAC specifically
  because some residual contamination surviving either mask is still
  expected, but it isn't detected and flagged the way `src/cloud_mask.py`
  does for the D01 pipeline.
- **D02_S2's forest-only masking** means, even once unfilled-gap counts
  look good on paper, the *type* of gap-fill being tested here is narrower
  than the eventual real use case (which needs non-forest fill too). Revisit
  once the reprocessed extract lands.

## Bug fixed in fusion: SR output is not zero where its input was invalid

`03_superresolve.py` feeds invalid Sentinel-2 pixels (forest-masked / no
data) to the model as 0, but the model does not return 0 there: on
2022-01, 99.96% of SR pixels over invalid input are non-zero, with B8
median ~31 DN (vs ~3347 over valid input). `src/fuse.py` originally
decided "SR has data" with `sr != 0`, which would have labelled every
NICFI gap as SR-filled and painted near-black values into it, and its
feather would have blended that junk into real NICFI pixels next to gaps.
Fix: `fuse.reproject_valid_to_nicfi_grid` carries the S2 mosaic's own
`valid` mask onto NICFI's grid (a NICFI pixel counts as covered only if
>=99.9% of the S2 area under it was valid, which also trims the mask edge
where the model saw zero-filled neighbours); fill, feather and the 6
SR-only bands are all restricted to it, and `fuse_month` now refuses SR
data without a mask.

## Performance check (2022-01..06, `scripts/amazon_d02/06_sr_performance.py`)

Run on the 6 super-resolved months to inform the next step. Figures:
`outputs/amazon_d02/figures/perf_01..04_*.png`.

1. **Visual (`perf_01_visual_crops.png`)** -- NICFI 4.77m vs S2 10m vs
   SR 2.5m on the same ground, shared stretch per row. SR is clearly
   sharper than 10m, structures (roads, clearing edges) line up with
   NICFI, no obvious artefacts.
2. **Hard constraint holds (`perf_02_hard_constraint.png`)** -- SR
   block-averaged back to 10m vs the real S2 mosaic: r = 0.974 / 0.988 /
   0.993 for B2 / B4 / B8, bias < 0.4%; SWIR B11 looser (r = 0.882, MAE
   6.5%).
3. **SR does not beat plain interpolation against NICFI
   (`perf_03_vs_nicfi.png`)** -- on ~2.7M clear-in-both px/month,
   compared against a bilinear upsample of the 10m S2 mosaic, SR is never
   better on any band/month. Mean r (SR vs bilinear): blue 0.64 vs 0.68,
   green 0.72 vs 0.75, red 0.80 vs 0.82, NIR 0.68 vs 0.70. High-pass r
   (fine-detail agreement, x - gaussian(x, sigma=3px)): blue 0.27 vs
   0.40, green 0.35 vs 0.43, red 0.39 vs 0.48, NIR 0.42 vs 0.45. I.e. the
   sub-10m detail SR adds is plausible-looking but does not match what
   NICFI actually sees. Caveat: NICFI and the S2 median mosaic are from
   different dates (hence the month-to-month swings, March poor for
   both), but the SR-vs-baseline comparison is like-for-like within each
   month.
4. **Contamination candidates (`perf_04_contamination.png`)** -- robust
   z of blue (NICFI - SR), |z| > 4. NICFI-brighter (likely NICFI
   haze/cloud): 0.03-0.24% of px, e.g. 29 blobs in 2022-01 including a
   ~6000 px hazy patch. SR-brighter (likely S2 contamination): 0.7-1.6%
   every month, several times more frequent -- the S2 mosaics carry more
   residual contamination than NICFI.

**Implications / open decision:**
- (a) Gap-fill as designed: nothing real to fill on D02.
- (b) Replace *contaminated-but-present* NICFI pixels with S2: feasible
  for patches like 2022-01's, but the replacement carries ~10m-effective
  information, not 2.5m, and needs its own S2-side cloud check, since S2
  is the dirtier source here. On these numbers, bilinear-upsampled S2
  would do about as well as SR at a fraction of the GPU cost.
- Clearest value of SR: the 6 extra bands (red-edge/SWIR) on NICFI's
  grid; red-edge/NIR well-constrained, SWIR less so.
