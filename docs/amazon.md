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

## Annual composite: per calendar year (5 years, not 1)

Unlike Vietnam (a single year, where "annual composite" and "the
reconstructed stack" were the same thing), this site has 5 calendar years
of monthly data, so an annual composite here means one composite *per
year* — run via `scripts/amazon/03_annual_composite.py`, same method as
`src/annual_composite.py` (naive / robust / medoid).

This is a much sharper test of the naive-vs-robust problem than Vietnam
ever produced. Vietnam's AOI never dropped below 5/12 confidently-clear
months for any pixel; here, **every single year has pixels with literally
zero confidently-clear months** (2021: 1797 px, 0.041%; 2025: 3657 px,
0.083% -- worth noting 2025 has *more* zero-coverage pixels than 2021
despite 2021 looking cloudier in aggregate, because this is about whether
a given pixel ever gets one lucky clear glimpse, not overall cloudiness).

At those pixels the naive median is **undefined** -- not just noisy or
biased, literally NaN, rendered as black holes in
`08_annual_naive_undefined_zoom.png`. The robust composite has no such
gaps anywhere, by construction (every month contributes at least
`recon_trust=0.35` weight, so there's always something to compute a
percentile from). `06_annual_composite_by_year.png` shows all 5 years
side by side with their `n_valid` maps -- 2021 and 2025 visibly have the
patchiest coverage, matching the two years with real contamination spikes
in `01_contamination_timeseries.png`.

## NDVI phenology at forest pixels (deciduous-mapping relevance)

This site's stated downstream use is mapping deciduous trees within the
forest, which lives or dies on whether the reconstruction preserves real
seasonal NDVI signal rather than smoothing it into the generic cluster
curve — the exact risk flagged early in this project for confusable/rare
classes. `scripts/amazon/04_deciduous_ndvi.py` checks this directly:

1. **"Forest"** = mean NDVI (confidently-clear months only) > 0.6, which
   cleanly separates dense canopy from this tile's lower-NDVI non-forest
   (open water, wetland/campina, bare ground) -- picked from the actual
   distribution (median 0.66, most of the mass above 0.6).
2. **Seasonal amplitude** = mean annual (max NDVI − min NDVI) of the
   *reconstructed* series (gap-free, so amplitude isn't itself an artifact
   of missing data) per forest pixel, averaged over 5 years.
3. Four pixels picked at amplitude percentiles 10/50/90/98 among forest
   pixels with strong data support (≥45/60 confidently-clear months), so
   the amplitude estimate is trustworthy rather than a sparse-data fluke.
   `09_ndvi_pixel_locations.png` shows the amplitude map has real spatial
   coherence (patches, not salt-and-pepper noise) -- evidence it's tracking
   a genuine property of the forest, not sensor noise.

This is a **diagnostic proxy for deciduousness, not a species-level
classification** -- elevated NDVI amplitude with a recurring within-year
dip is *consistent with* deciduous/semi-deciduous behavior, but confirming
it would need ground reference data this project doesn't have.

Each figure now also shows an 80x80px RGB zoom (January of each year, exact
pixel marked with a red crosshair) above its NDVI panel, specifically to
ground-truth what's being measured -- and it surfaced a real caveat, visible
consistently across all four pixels: **the two low-amplitude pixels
(`evergreen_like`, `typical`) both sit solidly inside a large, uniform
forest interior in every year's zoom, while both higher-amplitude pixels
(`elevated_amplitude`, `deciduous_candidate`) sit right at the boundary
between dense forest and this tile's lower-NDVI tan/beige land-cover type**
(a green patch against a lighter background in both figures' zoom rows).
That's not proof of anything on its own (n=4, hand-inspected), but it's a
real, unresolved ambiguity worth stating plainly: elevated NDVI amplitude
at an edge pixel could reflect genuine deciduous phenology, *or* it could
be a boundary/mixed-pixel effect (the 4.77m footprint spanning two
different covers, with sub-pixel misregistration between months shifting
how much of each cover falls inside the pixel) that has nothing to do with
leaf phenology at all. The amplitude-based selection used here can't
distinguish the two; doing so would need either finer-resolution imagery,
an explicit edge/interior mask before ranking candidates, or checking
whether the recurring dip's timing matches known deciduous phenology for
this region -- none of which this project has done.

**Result, per pixel** (`10_ndvi_evergreen_like.png` through
`13_ndvi_deciduous_candidate.png`, RGB zoom + raw/masked/reconstructed
NDVI, one panel per year):

- **evergreen_like** and **typical**: flat NDVI (~0.78-0.85) across all 5
  years, with a handful of sharp downward spikes in the raw series (clear
  cloud/shadow contamination) that are correctly absent from the masked
  points and correctly ignored by the reconstruction, which stays flat.
- **elevated_amplitude**: a real, consistent within-year cycle repeating
  across all 5 independent years (low Jan-Mar, peak Jun-Aug) — not noise,
  since most months are confidently observed (dense orange markers) and
  the pattern recurs at the same calendar position every year.
- **deciduous_candidate**: the strongest and most textbook pattern — a
  recurring Feb-Mar dip in 4 of 5 years, again mostly built from real
  confidently-clear observations rather than reconstructed gaps, with the
  reconstruction correctly preserving the dip shape while still catching
  genuine one-off outliers (e.g. a clear cloud-contaminated spike in
  Oct 2025, pulled back toward the seasonal pattern rather than left in).

The reassuring finding, independent of the edge-effect caveat above: at
both high-amplitude pixels, the recurring shape is present *in the real
observations themselves* (confidently-clear months directly show the
dip/peak), and the reconstruction tracks it rather than erasing it —
whatever is physically driving the signal (deciduous phenology, a mixed
forest/non-forest pixel, or something else), the pipeline is preserving
it rather than flattening it into the generic cluster curve. That's the
narrower, defensible claim; it doesn't retire the earlier concern about
reconstruction erasing rare-class signal (a pixel with much sparser real
data than these two, selected specifically for good data support, could
still be shrunk harder toward the cluster mean), and it doesn't resolve
whether either pixel is actually showing deciduous behavior -- but it is a
positive data point on the narrower question of whether real signal
survives reconstruction, rather than a purely theoretical worry.

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
- `06_annual_composite_by_year.png` — naive/robust/medoid/n_valid, all 5
  years.
- `07_annual_n_valid_by_year.png` — per-year data-sufficiency boxplot.
- `08_annual_naive_undefined_zoom.png` — naive median's literal undefined
  (black) pixels vs. the robust median at the same location.
- `09_ndvi_pixel_locations.png` — mean-NDVI and seasonal-amplitude maps
  with the 4 selected forest pixels marked.
- `10_ndvi_evergreen_like.png` / `11_ndvi_typical.png` /
  `12_ndvi_elevated_amplitude.png` / `13_ndvi_deciduous_candidate.png` —
  raw/masked/reconstructed monthly NDVI, one panel per year, for each
  selected pixel.