# Amazon results (detail)

To check the pipeline generalizes beyond a single hand-picked AOI/year, it
was run unmodified (aside from the `sites.py` refactor that made it
possible to point the same code at a different dataset at all) on a second,
independent dataset: `/mnt/warehouse/amazon/D01`, 60 monthly tiles
(2021-01 to 2025-12) — same band layout (B,G,R,N) and reflectance scale as
Vietnam, but a different CRS (EPSG:3857) and a tile already at ~AOI scale,
processed whole.

Run: `scripts/amazon/01_process.py` (full pipeline in one pass — ensembled
masking, continuous confidence, phenology, DL refiner, composition, ~69 min
on CPU) → `02_visualize.py` (before/after figures).

## A real environment bug, caught before it could corrupt output

A global `PROJ_DATA` environment variable in this shell (set for an
unrelated conda env) silently degraded CRS handling. For Vietnam's simple
geographic CRS (EPSG:4326) this only broke `to_epsg()` resolution while
leaving the actual WKT intact (verified after the fact: the already-pushed
Vietnam outputs are correctly georeferenced regardless). For Amazon's
projected CRS (EPSG:3857) it was worse — rasterio silently downgraded it
to a bogus `LOCAL_CS` with no relationship to real-world coordinates.
Caught by checking `to_epsg()` on a fresh read *before* running the full
60-month pipeline, rather than discovering it after writing 60
mis-georeferenced files. Fixed by unsetting `PROJ_DATA` for every command
touching this dataset.

## Results

`02_before_after_highlights.png` shows 8 examples spanning different years
and severities, including the most extreme case in the whole 5 years —
July 2021, 97% of the tile flagged contaminated by a single large cloud
mass — reconstructing into plausible, detailed terrain consistent with the
surrounding pattern. Already-clean months (e.g. Jan 2021) are correctly
left near-untouched. `01_contamination_timeseries.png` shows a real,
plausible climatology: recurring contamination spikes clustered in Mar-Jul
across multiple years, near-zero in other months, rather than noise.

## A genuine limitation surfaced by this larger, more varied test

One highlighted month (April 2021) still shows a large, visibly hazy patch
after reconstruction. Tracing it back, that region is classified mostly
"clear" by both OmniCloudMask and the continuous severity scorer (mean
weight ~0.87, mean severity ~1.0 — well under the anomaly threshold), so
the hazy observation passes through nearly unmodified. Checking the same
location across multiple Januaries shows this is a **recurring, localized
haze/fog pattern** (likely a wetland or lake microclimate), not a one-off
event.

This is a structural blind spot of temporal-residual anomaly detection: if
a contamination pattern recurs often enough at one location, it gets
partly absorbed into that pixel's own "expected normal" reference and
stops looking anomalous by the time severity is scored. This is a
different failure mode than the Vietnam haze case (see `docs/vietnam.md`,
which was non-recurring at a given location) and isn't fixed by that
work — flagged, not solved, here.

## Figure index (`outputs/amazon/figures/`)

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