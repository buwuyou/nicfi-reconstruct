# NICFI monthly cloud-free reconstruction

Reconstructs monthly, (near-)cloud-free composites from
[NICFI Planet Basemap](https://www.planet.com/nicfi/) monthly mosaics,
which are already heavily cloud-filtered by Planet but still carry real
residual cloud, cloud-shadow, and haze contamination. The pipeline is
site-agnostic — it operates on plain reflectance arrays + a cloud mask —
and has been run end-to-end on two independent, deliberately different
test sites: **Vietnam** (mountains, 1 year, cropped AOI) and **Amazon**
(rainforest, 5 years, whole tile). Full write-ups for each are in `docs/`.

## Method, in one pass per pixel-month

1. **Cloud/shadow/haze detection** (`src/cloud_mask.py`) — ensembled
   [OmniCloudMask](https://github.com/DPIRD-DMA/OmniCloudMask) (2 model
   generations averaged) + a spectral haze heuristic.
2. **Continuous confidence scoring** (`src/robust_mask.py`) — how anomalous
   is this observation against a robust land-cover-cluster + per-pixel
   reference, converted to a [0,1] trust weight instead of a hard flag.
3. **Phenology reconstruction** (`src/phenology.py`) — per-pixel harmonic
   regression across all months, shrunk toward a land-cover cluster's
   pooled curve where a pixel's own data is too sparse to trust alone.
4. **Self-supervised DL spatial refinement** (`src/inpaint.py`) — a small
   partial-conv U-Net adds local texture, trained per-scene with no
   external data, as a bounded residual around the phenology curve.
5. **Composition** (`src/compose.py`) — real observation where confidently
   clear, smoothly blended toward the reconstruction as confidence drops.

A 6th module, `src/annual_composite.py`, builds annual composites from the
reconstructed monthly stack rather than raw sparse observations, so every
pixel gets a comparable sample depth instead of the usual
more-cloud-means-fewer-samples problem. Run on both sites (Vietnam: one
year; Amazon: one composite per calendar year, 5 total).

Design rationale, and two real bugs found/fixed while verifying results
against actual images rather than trusting aggregate numbers, are in
`docs/vietnam.md`.

## Layout

```
src/            shared, site-agnostic pipeline code (sites.py = per-site config)
scripts/vietnam/, scripts/amazon/   run scripts per test site
outputs/vietnam/, outputs/amazon/   figures (tracked) + reconstructed GeoTIFFs/cache (gitignored, regenerable)
docs/           detailed write-ups: vietnam.md, amazon.md, super-resolution.md
```

A third, independent method lives alongside the above: `scripts/amazon_nicfis2/`
replaces cloudy NICFI pixels with a clear Sentinel-2 observation from the
*same month* (OmniCloudMask on raw Sentinel-2, single cloud-free frame when
available else a median of clear observations, local quantile-matching onto
NICFI's radiometry). Site-agnostic (`--tile <ID>`), first run on tile D17,
all 60 months -- see `docs/amazon_nicfis2.md`. Code in `src/nicfis2/`, kept
separate from `sites.py`'s NICFI-only `Site` since it's genuinely
dual-sensor.

## Quick start

```bash
unset PROJ_DATA   # see "Gotchas" below
python scripts/vietnam/01_prepare_data.py
python scripts/vietnam/02_run_pipeline.py
python scripts/vietnam/03_improve_masking.py   # recommended final version
python scripts/vietnam/04_annual_composite.py

python scripts/amazon/01_process.py
python scripts/amazon/02_visualize.py
python scripts/amazon/03_annual_composite.py
python scripts/amazon/04_deciduous_ndvi.py

python scripts/amazon_nicfis2/01_cloudmask_s2.py    --tile D17   # GPU
python scripts/amazon_nicfis2/02_s2_composite.py    --tile D17
python scripts/amazon_nicfis2/03_cloudmask_nicfi.py --tile D17   # GPU
python scripts/amazon_nicfis2/04_reconstruct.py     --tile D17
python scripts/amazon_nicfis2/05_visualize.py       --tile D17
```

Environment: `sen2sr` conda env (torch, rasterio, omnicloudmask,
scikit-learn).

## Results at a glance

- **Vietnam**: `outputs/vietnam/figures/19_v4_final_compare.png` — cloud
  and haze removed with two previously-residual patches verified fully
  gone (0 bright px at every threshold tested, down from 1049/2179).
  Full write-up + the two bugs that got it there: `docs/vietnam.md`.
- **Amazon**: `outputs/amazon/figures/02_before_after_highlights.png` —
  even a 97%-contaminated month (Jul 2021) reconstructs into plausible
  terrain. One genuine unfixed limitation found here (recurring localized
  haze that the detector's own reference absorbs): `docs/amazon.md`. Its
  5-year annual composite also gives the sharpest evidence yet for why the
  robust method matters: the naive median is literally undefined (black
  holes) at ~0.04-0.08% of pixels each year, versus zero gaps for the
  robust/medoid versions (`08_annual_naive_undefined_zoom.png`). Per-pixel
  NDVI phenology checks (`10-13_ndvi_*.png`), relevant to this site's
  deciduous-tree-mapping use case, found the reconstruction preserves real
  recurring seasonal signal rather than smoothing it away — but also found
  that the highest-amplitude candidates sit right on the forest/non-forest
  boundary, an unresolved confound between genuine deciduous phenology and
  a boundary artifact. A follow-up unbiased check (10 random dense-forest-
  interior pixels, no amplitude selection, `14-23_ndvi_random_*.png`) found
  no comparable recurring signal in any of the 10 — consistent with
  deciduous trees being genuinely rare here, not a method failure. See
  `docs/amazon.md` for both.

## Honest limitations

- No held-out ground truth on either site — judged visually and by
  self-supervised reconstruction loss, not a quantitative accuracy metric.
- Severity scoring measures coherent multi-band *brightening*; it's blind
  to contamination that recurs often enough at one spot to bias its own
  reference (see `docs/amazon.md`).
- The harmonic model assumes a smooth seasonal cycle — a genuine abrupt
  event (e.g. clear-cut) in a cloudy month gets smoothed over until a later
  clear observation updates the picture. This is a compositing method, not
  a change-detection method.
- Both test sites are NICFI analytic 4-band products at the same
  resolution/processing level — a different sensor hasn't been tested.
- Multi-frame super-resolution of NICFI *itself* was investigated and
  deliberately not built — see `docs/super-resolution.md` for why. A
  related idea (super-resolving *Sentinel-2* to fill NICFI) was tested
  and also not adopted -- it was no better than bilinear upsampling
  against NICFI; same doc.
- The NICFI+Sentinel-2 method can only replace a cloudy NICFI pixel when
  Sentinel-2 saw that pixel clear the same month -- in the wet season it
  usually didn't (D17: ~25% of all NICFI contamination replaced; see
  `docs/amazon_nicfis2.md`).

## Gotchas

Unset `PROJ_DATA` before any command touching the Amazon site (or any
non-EPSG:4326 CRS) — a stray global env var pointing at a mismatched PROJ
database silently corrupts CRS handling for projected CRSes. Caught before
it could corrupt output; see `docs/amazon.md`.