"""
Final monthly cloud-free composite = keep the real observation wherever it's
confidently clear (preserves genuine radiometric detail -- we should never
replace a good pixel with a model's guess of it), and fall back to the
DL-refined phenology reconstruction wherever it isn't.

A soft blend, proportional to the observation-confidence weight itself, is
used across the confidence transition rather than a hard cutoff, to avoid
seams at mask boundaries.

`alpha = weight` directly, with no rescaling. An earlier version applied
`alpha = clip(weight / 0.6, 0, 1)`, calibrated back when `weight` only took
a handful of near-discrete values (1.0 / 0.25 / 0.15 / 0.1 / 0 from
`cloud_mask.quality_weight`) and dividing by 0.6 gave a reasonable ramp
across those few levels. Once `weight` became a genuinely continuous,
per-pixel-calibrated trust score (`robust_mask.continuous_confidence`), that
same rescaling was actively harmful: it amplifies ordinary pixel-to-pixel
weight noise into large blend-ratio swings right around the 0.6 threshold
(anything above it got snapped to fully-observed regardless of whether it
was 0.61 or 0.99), which produced a visible speckled/discolored artifact in
moderate-trust transition zones once weight varied continuously and finely
across space -- confirmed by checking that the artifact's location matched
exactly where the weight map showed spatially noisy mid-range values, and
that using the weight directly (this version) removes it. If a future
weight source is *not* pre-calibrated as a direct trust probability, rescale
it into that form before calling this function, rather than adding a
threshold back in here.
"""
import numpy as np


def compose(refl, weight, refined):
    """refl,weight,refined: (T,4,H,W)/(T,H,W)/(T,4,H,W). `weight` must
    already be a calibrated [0,1] trust score (this is what
    `cloud_mask.quality_weight` and `robust_mask.continuous_confidence`
    both produce). Returns (T,4,H,W) composite reflectance, clipped to a
    physically plausible range, plus the blend weight actually used
    (T,H,W) for visualization/QA.
    """
    alpha = weight[:, None, :, :]  # (T,1,H,W)
    composite = alpha * refl + (1 - alpha) * refined
    composite = np.clip(composite, 0.0, 1.3)
    return composite.astype(np.float32), alpha[:, 0]