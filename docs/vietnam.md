# Vietnam results (detail)

Run: `scripts/vietnam/01_prepare_data.py` (baseline single-model mask) →
`02_run_pipeline.py` (baseline reconstruction) → `03_improve_masking.py`
(ensembling + continuous confidence) → `04_annual_composite.py`.

- `outputs/vietnam/reconstructed_v2/*.tif` — 12 final reconstructed monthly
  GeoTIFFs (recommended version; `reconstructed/*.tif` is the earlier
  single-model baseline, kept for comparison).
- `outputs/vietnam/annual/*.tif` — naive/robust/medoid annual composites.

## Masking improvement: two real bugs found and fixed

Found while verifying against actual before/after images, not just
aggregate counts:

- **Compose-time alpha rescaling.** `compose.py` originally computed
  `alpha = clip(weight / 0.6, 0, 1)`, calibrated for an older scheme where
  weight only took a few near-discrete values. Once weight became
  genuinely continuous, that rescaling amplified ordinary pixel-to-pixel
  weight noise into large blend-ratio swings right at the 0.6 threshold.
  Fixed by using `alpha = weight` directly.
- **Severity silently overriding a correct categorical call.** The first
  version of the continuous confidence scorer let severity fully determine
  trust for any non-hard-blocked pixel, including ones OmniCloudMask had
  already (correctly) called CLOUD_THIN. Severity specifically measures
  coherent multi-band *brightening*; it's blind to contamination that
  shifts color without broadly brightening every band. This AOI has
  exactly that case: a chromatic/thin-cirrus-like speckle artifact that
  OmniCloudMask's ensemble correctly flagged CLOUD_THIN, but which scored
  near-zero severity, so its weight climbed back to ~0.78 mean — visibly
  reintroducing a magenta speckle into the composite that OmniCloudMask had
  already caught. Found by tracing one such artifact pixel back through
  every stage (observed → phenology curve → DL-refined → composite) rather
  than assuming the aggregate numbers told the whole story. Fixed with
  categorical caps: CLOUD_THIN can never exceed 0.3 trust, HAZE never
  exceeds 0.4, regardless of how low severity scores it.

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

## Calibrating the severity threshold

The continuous confidence scorer needed a "how anomalous is too
anomalous" threshold. Calibrated against this AOI's own clear-pixel
population: 99.9th-percentile severity ~6 (in cluster/band-normalized
z-score units) vs. independently-confirmed haze residual at severity 35-80
— a big enough gap (`13_severity_calibration.png`) that severity alone
cleanly separates the two, rather than needing a hand-tuned spectral
threshold.

## Annual composite

At n_valid=12 (fully clear all year) naive and robust annual composites are
identical by construction — no real data to override. As n_valid drops the
two composites diverge, up to ~0.03 reflectance (mean absolute, across
bands) at the low end here — but the divergence doesn't scale cleanly with
n_valid alone (`12_annual_naive_vs_robust_diff.png`); it also depends on
how much a pixel's phenology amplitude actually differs between its
observed and reconstructed months. This AOI's cloud cover, while real,
never drops any single pixel below 5/12 clear months, so the demonstrated
effect here is real but modest — the mechanism would matter far more in a
more persistently cloudy scene, which is part of why the Amazon site was
added next (see `docs/amazon.md`).

## Figure index (`outputs/vietnam/figures/`)

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