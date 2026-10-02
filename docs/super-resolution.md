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

A related idea was later tested: applying this model to *Sentinel-2* (its
native domain) and using the 2.5m output to fill NICFI. On a test tile it
reproduced its own 10m input faithfully (r 0.97-0.99 for B2/B4/B8 after
block-averaging back), but agreed with NICFI *no better than a plain
bilinear upsample* of the 10m Sentinel-2 on any band/month -- and worse on
fine detail (high-pass r, e.g. blue 0.27 vs 0.40): the sub-10m texture it
adds is plausible, not what NICFI sees -- at ~17 GPU-min per tile-month. So
the NICFI+Sentinel-2 pipeline (`docs/nicfirecon.md`) upsamples
bilinearly and doesn't use SR.
