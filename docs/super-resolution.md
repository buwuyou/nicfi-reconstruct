# Multi-frame super-resolution: investigated, not implemented

The hypothesis — that sub-pixel geolocation differences between months
could be exploited to reconstruct detail finer than the native ~4.5m pixel,
the way classical multi-image super-resolution (MISR) does — is a real
technique, but it was measured against the Vietnam data before building
anything, and it doesn't hold up well enough there to justify building it:

1. **All 12 monthly GeoTIFFs share a bit-identical pixel grid** (same
   origin, same transform) — expected for a tiled basemap product, but it
   means Planet's per-month compositing already resampled everything onto
   one fixed grid before delivery, which destroys most of the raw sub-pixel
   sampling diversity MISR needs before it ever reaches this dataset.
2. **Measured actual sub-pixel registration** via phase correlation between
   clear months: shifts are tiny (~0.02-0.2px) and, critically, *not
   spatially consistent within a single month-pair* — different patches of
   the same image-pair gave shifts varying by up to 0.3px, meaning it isn't
   a clean rigid geometric offset, it's noise/local content change. Between
   temporally-adjacent months with minimal phenology change (Nov vs Dec),
   shifts shrank to ~0.05px with comparable-sized noise — at the edge of
   measurement precision, not a reliable signal.

Classical MISR needs consistent, well-characterized sub-pixel diversity
across many frames to recover real high-frequency detail. Attempting it on
what's actually here would mostly manufacture the appearance of sharpness
without real new ground information — the same hallucination risk flagged
for the DL spatial refiner (see main README), just in a different guise.

**Decision: not pursued as multi-frame NICFI self-super-resolution.** The
`sen2sr` conda env this project uses already has `opensr-model`/`sen2sr`
installed — a legitimate pretrained single-image super-resolution model
(diffusion-based, from the openSR project) that takes a different,
defensible approach (a learned image prior rather than multi-frame
geometry). It's trained on Sentinel-2 (10m native), so applying it to
NICFI's already-finer 4.77m Planet imagery would be a real domain shift in
both resolution and sensor radiometry that would need honest evaluation
before trusting it — flagged at the time as a possible future direction,
not attempted.

That future direction has since been built, in a different shape than
"upsample NICFI directly": rather than super-resolving NICFI itself, this
model is applied to Sentinel-2 (its native domain, no resolution/radiometry
domain shift) to fill NICFI's genuine data *gaps* — a different problem
than the multi-frame reconstruction investigated above, for a new test
tile (D02). See `docs/amazon_d02_s2fusion.md`.