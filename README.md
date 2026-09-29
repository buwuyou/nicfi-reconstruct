# NICFI monthly cloud-free reconstruction

A general pipeline for reconstructing monthly, (near-)cloud-free composites
from [NICFI Planet Basemap](https://www.planet.com/nicfi/) monthly mosaics
— which are already heavily cloud-filtered by Planet but still carry real
residual cloud, cloud-shadow, and haze contamination. The method is
site-agnostic (it operates on plain reflectance arrays + a cloud mask, and
knows nothing about where the data came from); it's been run end-to-end on
two independent test sites so far, chosen to be genuinely different
(mountainous vs. lowland rainforest, one year vs. five, a cropped AOI vs. a
whole tile) rather than two easy variations on the same case.

## Project layout


```
src/                          # shared, site-agnostic pipeline code
  sites.py                    # per-site config: paths, tile naming, AOI, months
  cloud_mask.py               # OmniCloudMask ensembling + haze heuristic
  robust_mask.py              # continuous temporal-residual confidence scoring
  phenology.py                # land-cover cluster priors + harmonic reconstruction
  inpaint.py                  # self-supervised DL spatial refiner
  compose.py                  # final observed/reconstructed blend
  annual_composite.py         # reconstruction-informed annual compositing
  io_utils.py, visualize.py
scripts/
  vietnam/                    # run scripts for the Vietnam test site
  amazon/                     # run scripts for the Amazon test site
outputs/
  vietnam/{figures,reconstructed,reconstructed_v2,annual,cache}/
  amazon/{figures,reconstructed,annual,cache}/
    # figures/ is tracked in git; reconstructed*/, annual/, and cache/ are
    # regenerable (rerun the scripts against the same source data) and
    # gitignored -- see "Reproducing" below.
```

Adding a third test site means adding one `Site` entry to `src/sites.py`
and a `scripts/<new_site>/` folder with run scripts that call into the same
`src/` modules -- nothing in `src/` needs to change.

## Test sites

| | Vietnam | Amazon |
|---|---|---|
| Tile | `00115_00023`, cropped to a 2048x2048px AOI | `D01`, whole 2096x2109px tile |
| Location | Sơn La / Yên Bái mountains, N. Vietnam | ~63°W 0.5°N, Amazon basin |
| Period | 12 months, 2025 | 60 months, 2021-2025 |
| Land cover | Mixed forest + cropland | Rainforest + wetland/floodplain |
| CRS | EPSG:4326 (geographic) | EPSG:3857 (Web Mercator) |
| Cloud pattern | Wet-season spikes (Jun/Jul/Sep/Oct) | Persistent, more year-to-year variable |

The Vietnam AOI was chosen (see `outputs/vietnam/figures/candidate_aoi_months.png`)
for a mix of forest + cropland (a real seasonal signal) *and* genuine
cloud/haze contamination in several months, rather than a scene that's
already clean. The Amazon tile was added later specifically to check the
pipeline generalizes to a different CRS, a much longer time series, and a
site that's persistently cloudier -- see "Amazon results" below for what
that stress test actually found, including a real limitation it surfaced.

**Memory discipline**: every read goes through `rasterio` windowed reads
against the on-disk COGs (`src/io_utils.py`); source tiles are never loaded
whole. `io_utils.iter_tiling_windows` documents how the same approach
extends to a much larger production tile with bounded memory (chunked,
overlap-feathered processing) -- not exercised by either current test site
(the Vietnam AOI is already a bounded crop; the Amazon tile is already
AOI-scale), but the pipeline doesn't need to change to scale up.

## Method

Five stages, each addressing a different part of the problem. All code
referenced here is site-agnostic; `scripts/<site>/*.py` just supply the
data and file paths.

### 1. Cloud / shadow / haze detection — `src/cloud_mask.py`, `src/robust_mask.py`

[OmniCloudMask](https://github.com/DPIRD-DMA/OmniCloudMask) — a small
(~1.2M-param), pretrained, sensor-agnostic CNN built specifically for 10-50m
imagery (Sentinel-2/Landsat/PlanetScope) — classifies each month into
clear / thick cloud / thin cloud / cloud shadow. This is the one place the
pipeline uses a model trained on external data. Two independent
improvements over a single-model, single-date call:

- **Model ensembling**: average softmax probabilities from two independent
  OmniCloudMask generations (3.0 and 4.0) instead of trusting one network's
  verdict. On the Vietnam AOI this changed the clear/not-clear call on
  1.25% of all pixel-months vs. a single model.
- **Continuous temporal-residual confidence scoring** (`robust_mask.py`,
  TMask-style detection generalized to a severity *score* rather than a
  hard flag): for every pixel-month not already hard-blocked (thick
  cloud/shadow/nodata), measure how anomalous it is against a robust
  reference -- the pooled land-cover-cluster seasonal shape (see stage 2)
  plus that pixel's own median offset from it, *not* an individually-overfit
  per-pixel curve (an earlier version used that and over-triggered on 5-9%
  of all pixel-months, because a 5-parameter curve fit to ~11-60 points is
  close to interpolating them, so its in-sample residuals are tiny almost
  by construction) -- then convert that anomaly into a continuous [0,1]
  trust weight instead of a fixed per-class constant. Calibrated against
  each site's own clear-pixel population (Vietnam: 99.9th-percentile
  severity ~6 in these units vs. independently-confirmed haze residual at
  35-80 -- a big enough gap that severity alone cleanly separates the two).
  OmniCloudMask's own CLOUD_THIN/HAZE calls still set an upper bound on
  trust (0.3 / 0.4) that severity can only push *down* from, never up past
  -- an earlier version let severity fully override the categorical call
  and it silently let a real, OmniCloudMask-confirmed contamination event
  back in because severity's brightening-based signature didn't happen to
  flag that particular case (see "Vietnam results" for how this was found
  and fixed, and "Amazon results" for a related failure mode this doesn't
  cover).

A simple physically-motivated haze index (blue-band brightening + depressed
NIR-red contrast) provides an initial categorical HAZE flag on top of
OmniCloudMask's own classes, for the severity scoring above to refine.

### 2. Land-cover cluster priors + phenology reconstruction — `src/phenology.py`

Per-pixel, per-band harmonic (Fourier) regression across all available
months, weighted-least-squares fit using the confidence weights from stage 1:

```
y(t) = a0 + a1*cos(2*pi*t/12) + b1*sin(2*pi*t/12) + a2*cos(4*pi*t/12) + b2*sin(4*pi*t/12)
```

(`t` is the absolute month index, so the annual period holds regardless of
whether the series is 12 months or 60.) Evaluating the fitted curve at any
month gives the expected clear-sky value -- this *is* "follow the seasonal
pattern of the geography" made concrete.

**Handling pixels with too little of their own clear data** (the original
brief's "borrow phenology from a nearby cloud-free place" idea): instead of
importing a curve from a separate geographic location (which drags in its
own registration/BRDF/atmosphere-matching problems), pixels are clustered
by weighted-mean spectral signature (KMeans, 10 clusters -- a cheap
land-cover proxy), and a robust cluster-level harmonic curve is fit by
pooling *all* member pixels' observations. A pixel's own fit is then shrunk
toward its cluster's curve in proportion to how little confident data it
personally has. This is the spatial-analog realization of "borrow phenology
from somewhere cloud-free nearby": it draws on every spectrally-similar
pixel in the same scene (same sensor, date, atmosphere) rather than one
hand-picked external site.

### 3. Self-supervised DL spatial refinement — `src/inpaint.py`

The harmonic model is fit independently per pixel through time, so it has
no notion of spatial texture -- reconstructed cloud gaps can look a little
flat next to genuinely observed neighbors. A small (~0.5M-param) **Partial
Convolution U-Net** (Liu et al. 2018, *Image Inpainting for Irregular Holes
Using Partial Convolutions*) adds spatial detail: partial convs renormalize
by how much valid signal actually fell under each kernel, so holes are
filled from real spatial context rather than corrupted by masked-out zeros.

Trained **self-supervised, per-scene, from scratch** (test-time training):
there's no ground truth for the real clouds, so the network repeatedly sees
patches that are already clear, has random cloud-shaped holes punched into
them, and is trained to recover the hidden pixels from context. No external
labels or pretrained weights are used here -- everything it learns comes
from that scene's own clear-ish months. Two robustness choices, both added
after an initial attempt produced degenerate output on the largest cloud gaps:

- **GroupNorm, not BatchNorm** -- batch statistics swing wildly when holes
  range from 0% to 90% of a tile; GroupNorm normalizes each sample alone.
- **Bounded residual around the phenology curve**: the network predicts
  `guide + tanh(delta) * 0.15` rather than raw reflectance, and its last
  layer is zero-initialized. Worst case (undertrained, or a tile with
  almost no valid context) it contributes ~nothing and the output reduces
  to the already-sensible harmonic estimate -- it can never hallucinate an
  unbounded or wildly wrong color the way the first (BatchNorm,
  unconstrained-output) version did.

### 4. Composition — `src/compose.py`

Final value = the **real observation** wherever confidently clear (never
replace a good pixel with a model's guess), smoothly blended down to the
refined reconstruction as confidence drops (`alpha = weight`, directly --
see "Vietnam results" for a real bug this design replaced) -- no hard mask
seams.

### 5. Annual compositing — `src/annual_composite.py`

The standard way to build an annual median/percentile composite -- compute
the statistic per pixel from whatever confidently-clear observations that
pixel happens to have -- has a well-known problem in persistently cloudy
areas: sample depth varies wildly pixel to pixel, so the "median" means
something different at every pixel, and a pixel that only ever saw
dry-season clear sky gets a value biased toward dry-season color rather
than a genuine annual summary.

Fix: compute the percentile from the already-reconstructed monthly stack
instead of the raw sparse observations, weighted so a confidently-observed
month keeps full trust but a reconstructed month gets a fixed, modest,
non-zero weight (`recon_trust=0.35`) rather than zero. Every pixel ends up
with a full, comparable sample depth -- enough to stabilize poorly-observed
pixels without letting model-filled months outvote real ones wherever real
data exists. A band-consistent **medoid** variant is also provided: rather
than picking each band's percentile independently (which can synthesize a
spectrally-impossible pixel -- red from one date, NIR from another), it
selects the one actual month whose full 4-band vector is closest to the
weighted target, so the output is always a genuine observed-or-reconstructed
spectrum from one real date. (Run for Vietnam only so far --
`scripts/vietnam/04_annual_composite.py`.)

## Vietnam results

Run: `scripts/vietnam/01_prepare_data.py` (baseline single-model mask) →
`02_run_pipeline.py` (baseline reconstruction) → `03_improve_masking.py`
(ensembling + continuous confidence, the improvements described in stage 1
above) → `04_annual_composite.py`.

- `outputs/vietnam/reconstructed_v2/*.tif` — 12 final reconstructed monthly
  GeoTIFFs (recommended version; `reconstructed/*.tif` is the earlier
  single-model baseline, kept for comparison).
- `outputs/vietnam/annual/*.tif` — naive/robust/medoid annual composites.
- `outputs/vietnam/figures/` — see the figure-by-figure index below.

**Masking improvement, checked quantitatively and re-checked after each fix,
not just eyeballed once and trusted** (`19_v4_final_compare.png`): two real
bugs were found while verifying this against actual before/after images
rather than aggregate counts alone --

- **Compose-time alpha rescaling.** `compose.py` originally computed
  `alpha = clip(weight / 0.6, 0, 1)`, calibrated for an older scheme where
  weight only took a few near-discrete values. Once weight became
  genuinely continuous, that rescaling amplified ordinary pixel-to-pixel
  weight noise into large blend-ratio swings right at the 0.6 threshold.
  Fixed by using `alpha = weight` directly.
- **Severity silently overriding a correct categorical call.** The first
  version of the continuous scorer let severity fully determine trust for
  any non-hard-blocked pixel, including ones OmniCloudMask had already
  (correctly) called CLOUD_THIN. Severity specifically measures coherent
  multi-band *brightening*; it's blind to contamination that shifts color
  without broadly brightening every band. This AOI has exactly that case: a
  chromatic/thin-cirrus-like speckle artifact that OmniCloudMask's ensemble
  correctly flagged CLOUD_THIN, but which scored near-zero severity, so its
  weight climbed back to ~0.78 mean -- visibly reintroducing a magenta
  speckle into the composite that OmniCloudMask had already caught. Found
  by tracing one such artifact pixel back through every stage (observed →
  phenology curve → DL-refined → composite) rather than assuming the
  aggregate numbers told the whole story. Fixed with the categorical caps
  described in stage 1.

Counting residual bright (>0.3 mean RGB reflectance) pixels in the same
crop used throughout this investigation, across three pipeline versions:

| month | v1 (baseline) | v2 (ensembling + hard-flag outlier, first attempt) | v4 (continuous severity + caps, final) |
|---|---|---|---|
| Sep | 1049 | 957 (-9%) | **0** |
| Oct | 2179 | 1428 (-34%) | **0** |

Both months' previously-residual pixels are fully gone, verified at every
threshold tested (0.35/0.3/0.25). Dry-season months (Jan, Mar) are
essentially untouched by the change (mean absolute difference ~0.0001, at
the noise floor), confirming the continuous scoring isn't discounting
genuinely clear data.

**Annual composite**: at n_valid=12 (fully clear all year) naive and robust
are identical by construction -- no real data to override. As n_valid drops
the two composites diverge, up to ~0.03 reflectance (mean absolute, across
bands) at the low end here -- but the divergence doesn't scale cleanly with
n_valid alone (`12_annual_naive_vs_robust_diff.png`); it also depends on
how much a pixel's phenology amplitude actually differs between its
observed and reconstructed months. This AOI's cloud cover, while real,
never drops any single pixel below 5/12 clear months, so the demonstrated
effect here is real but modest -- the mechanism would matter far more in a
more persistently cloudy scene, which is part of why the Amazon site was
added next.

### Vietnam figure index (`outputs/vietnam/figures/`)

- `candidate_aoi_months.png` — AOI selection rationale (all 12 months).
- `01_observed_grid.png` / `02_reconstructed_grid.png` — full-year
  before/after (baseline pipeline).
- `03_before_after_full.png` — per-month observed | cloud mask |
  reconstructed (baseline).
- `04_confidence_map.png` — where the baseline reconstruction leans on the
  cluster prior vs. the pixel's own data.
- `05_phenology_*.png` — example per-pixel time series (raw observations
  colored by confidence, fitted curve, final composite) for a cropland
  pixel, a forest pixel, and a pixel from the persistently-cloudy zone.
- `06_before_after_v2.png` / `07_confidence_map_v2.png` — same as
  `03`/`04` but for the final masking pipeline.
- `08_masking_improvement.png` — px newly down-weighted per month by
  ensembling + continuous confidence vs. the single-model baseline.
- `09_v1_vs_v2_crop_compare.png` — first-attempt (discrete hard-flag)
  masking result, kept for the record; superseded by `19`.
- `10_annual_composite_compare.png` / `11_annual_composite_zoom.png` /
  `12_annual_naive_vs_robust_diff.png` — annual composite comparison.
- `13_severity_calibration.png` — clear-pixel severity distribution vs.
  the calibration point used for continuous confidence weighting.
- `18_artifact_stage_trace.png` — evidence that a chromatic artifact
  traced back to the raw observed data, not something the pipeline
  introduced.
- `19_v4_final_compare.png` — final verified before/after on both
  previously-residual crops, after both bugs above were fixed.

## Amazon results

To check the pipeline generalizes beyond a single hand-picked AOI/year, it
was run unmodified (aside from the `sites.py` refactor that made it
possible to point the same code at a different dataset at all) on a second,
independent dataset: `/mnt/warehouse/amazon/D01`, 60 monthly tiles
(2021-01 to 2025-12) -- same band layout (B,G,R,N) and reflectance scale as
Vietnam, but a different CRS (EPSG:3857) and a tile already at ~AOI scale,
processed whole.

Run: `scripts/amazon/01_process.py` (full pipeline in one pass -- ensembled
masking, continuous confidence, phenology, DL refiner, composition, ~69 min
on CPU) → `02_visualize.py` (before/after figures).

**A real environment bug was caught here before it could corrupt output**:
a global `PROJ_DATA` environment variable in this shell (set for an
unrelated conda env) silently degraded CRS handling. For Vietnam's simple
geographic CRS (EPSG:4326) this only broke `to_epsg()` resolution while
leaving the actual WKT intact (verified after the fact: the already-pushed
Vietnam outputs are correctly georeferenced regardless). For Amazon's
projected CRS (EPSG:3857) it was worse -- rasterio silently downgraded it
to a bogus `LOCAL_CS` with no relationship to real-world coordinates.
Caught by checking `to_epsg()` on a fresh read *before* running the full
60-month pipeline, rather than discovering it after writing 60
mis-georeferenced files. Fixed by unsetting `PROJ_DATA` for every command
touching this dataset.

**Results**: `02_before_after_highlights.png` shows 8 examples spanning
different years and severities, including the most extreme case in the
whole 5 years -- July 2021, 97% of the tile flagged contaminated by a
single large cloud mass -- reconstructing into plausible, detailed terrain
consistent with the surrounding pattern. Already-clean months (e.g. Jan
2021) are correctly left near-untouched. `01_contamination_timeseries.png`
shows a real, plausible climatology: recurring contamination spikes
clustered in Mar-Jul across multiple years, near-zero in other months,
rather than noise.

**A genuine limitation surfaced by this larger, more varied test**: one
highlighted month (April 2021) still shows a large, visibly hazy patch
after reconstruction. Tracing it back, that region is classified mostly
"clear" by both OmniCloudMask and the continuous severity scorer (mean
weight ~0.87, mean severity ~1.0 -- well under the anomaly threshold), so
the hazy observation passes through nearly unmodified. Checking the same
location across multiple Januaries shows this is a **recurring, localized
haze/fog pattern** (likely a wetland or lake microclimate), not a one-off
event. This is a structural blind spot of temporal-residual anomaly
detection: if a contamination pattern recurs often enough at one location,
it gets partly absorbed into that pixel's own "expected normal" reference
and stops looking anomalous by the time severity is scored. This is a
different failure mode than the Vietnam haze case above (which was
non-recurring at a given location) and isn't fixed by that work -- flagged,
not solved, here.

### Amazon figure index (`outputs/amazon/figures/`)

- `00_sample_months.png` — initial visual survey used to sanity-check the
  dataset before committing to the full run.
- `01_contamination_timeseries.png` — contamination fraction per month,
  all 5 years.
- `02_before_after_highlights.png` — 8 curated before/after pairs across
  years/severities (the main results figure for this site).
- `03_observed_YYYY.png` / `04_reconstructed_YYYY.png` — full per-year
  grids, all 60 months.
- `05_phenology_*.png` — two 5-year pixel time series (low- and
  high-NDVI cluster examples) showing the seasonal curve fit.

## Honest limitations

- No held-out ground truth on either site -- "quality" is judged visually
  (does it look plausible and cloud-free) and by the self-supervised
  reconstruction loss, not a quantitative accuracy metric.
- OmniCloudMask occasionally under-calls very thin cirrus; the haze index
  is a heuristic backstop, not a learned detector, and the continuous
  severity scorer is a brightening-based signature that's blind to
  contamination recurring often enough at one location to bias its own
  reference (see Amazon results).
- The harmonic model assumes a smooth annual seasonal cycle; a genuinely
  abrupt one-off event (e.g. sudden clear-cut) in an otherwise cloudy month
  would be smoothed over until a subsequent month's clear observation
  updates the picture -- this is a compositing method, not a
  change-detection method.
- The DL refiner's contribution is intentionally small by design (a
  bounded residual around the phenology curve) -- most of the
  reconstruction quality comes from the phenology model, with the network
  adding modest texture on top.
- A handful of Vietnam pixels (106/4.19M in Sept, 193/4.19M in Oct --
  ~0.003-0.005%, from an earlier pipeline version) showed a faint
  brightness trace from the heuristic "haze" class's deliberate partial
  trust; resolved by the continuous severity scoring described above.
- Annual compositing has only been run for Vietnam so far, not Amazon.
- These two test sites, while deliberately different from each other, are
  still both NICFI analytic 4-band products at the same native resolution
  and processing level -- generalization to a different sensor or
  processing level hasn't been tested.

## Multi-frame super-resolution: investigated, not implemented

The hypothesis -- that sub-pixel geolocation differences between months
could be exploited to reconstruct detail finer than the native ~4.5m pixel,
the way classical multi-image super-resolution (MISR) does -- is a real
technique, but it was measured against the Vietnam data before building
anything, and it doesn't hold up well enough there to justify building it:

1. **All 12 monthly GeoTIFFs share a bit-identical pixel grid** (same
   origin, same transform) -- expected for a tiled basemap product, but it
   means Planet's per-month compositing already resampled everything onto
   one fixed grid before delivery, which destroys most of the raw sub-pixel
   sampling diversity MISR needs before it ever reaches this dataset.
2. **Measured actual sub-pixel registration** via phase correlation between
   clear months: shifts are tiny (~0.02-0.2px) and, critically, *not
   spatially consistent within a single month-pair* -- different patches of
   the same image-pair gave shifts varying by up to 0.3px, meaning it isn't
   a clean rigid geometric offset, it's noise/local content change. Between
   temporally-adjacent months with minimal phenology change (Nov vs Dec),
   shifts shrank to ~0.05px with comparable-sized noise -- at the edge of
   measurement precision, not a reliable signal.

Classical MISR needs consistent, well-characterized sub-pixel diversity
across many frames to recover real high-frequency detail. Attempting it on
what's actually here would mostly manufacture the appearance of sharpness
without real new ground information -- the same hallucination risk flagged
above for the DL spatial refiner, just in a different guise.

**Decision: not pursued.** The `sen2sr` conda env this project uses already
has `opensr-model`/`sen2sr` installed -- a legitimate pretrained
single-image super-resolution model (diffusion-based, from the openSR
project) that takes a different, defensible approach (a learned image
prior rather than multi-frame geometry). It's trained on Sentinel-2 (10m
native), so applying it to NICFI's already-finer 4.77m Planet imagery would
be a real domain shift in both resolution and sensor radiometry that would
need honest evaluation before trusting it -- flagged as a possible future
direction, not attempted.

## Reproducing

Environment: `sen2sr` conda env (torch, rasterio, omnicloudmask,
scikit-learn; see that env for the exact package set). Every command that
touches the Amazon site (or any projected/non-EPSG:4326 CRS) should be run
with `PROJ_DATA` unset -- a stray global env var pointing at a mismatched
PROJ database silently corrupts CRS handling for projected CRSes
(see "Amazon results").

```bash
unset PROJ_DATA
python scripts/vietnam/01_prepare_data.py
python scripts/vietnam/02_run_pipeline.py
python scripts/vietnam/03_improve_masking.py
python scripts/vietnam/04_annual_composite.py

python scripts/amazon/01_process.py
python scripts/amazon/02_visualize.py
```

`outputs/<site>/{reconstructed,reconstructed_v2,annual,cache}/` are
gitignored (regenerable, and large -- the Amazon cache alone is ~8.5GB);
`outputs/<site>/figures/` is tracked.