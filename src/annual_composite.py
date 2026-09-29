"""
Annual composite (median/percentile) that doesn't inherit the classic
uneven-sample-count problem.

**The problem**: a standard annual median/percentile composite is computed
independently per pixel from whatever clear observations that pixel
happens to have. In a persistently cloudy tropical area, that count varies
wildly pixel to pixel -- one pixel might have 10 clear months, its neighbor
under a recurring cloud shadow only 2. A "median" of 2 points is really just
picking one of them; a pixel with only dry-season clear observations gets a
median biased toward dry-season color, not a true annual summary. The
result is spatially inconsistent, patchy composites that quietly encode
"how cloudy was this pixel's year," not just "what does it look like."

**The fix**: compute the percentile from the already-gap-filled monthly
stack (`compose.compose`'s output) instead of the raw sparse observations,
weighted by how much each month's value should be trusted:
  - a confidently-observed month keeps its full weight (this is real data,
    it should dominate wherever it exists)
  - a reconstructed month gets a fixed, modest, non-zero weight
    (`recon_trust`, default 0.35) -- enough to stabilize the statistic for
    poorly-observed pixels (which is the entire point), not enough to let
    12 "phenology-model-informed" months outvote a handful of real ones.

Every pixel effectively gets n=12 weighted samples instead of n=(however
many happened to be clear), which is what actually fixes the inconsistency
-- not "gap-filling" for its own sake, but guaranteeing a comparable,
unbiased sample depth everywhere.

A `medoid_composite` is also provided: rather than computing each band's
percentile independently (which can synthesize a spectrally-impossible
pixel -- red from one date, NIR from another), it picks, per pixel, the
single actual month whose full 4-band vector is closest to the weighted
target, preserving a physically consistent spectrum.
"""
import numpy as np


def weighted_percentile_along_time(values, weights, q):
    """values, weights: (T, ...) matching shape. Returns (...) the
    weighted q-th percentile (0-100) along axis 0, using the standard
    weighted-percentile midpoint interpolation. Fully vectorized.
    """
    T = values.shape[0]
    order = np.argsort(values, axis=0)
    v_sorted = np.take_along_axis(values, order, axis=0)
    w_sorted = np.take_along_axis(weights, order, axis=0)

    cw = np.cumsum(w_sorted, axis=0)
    cw_total = cw[-1:]
    cw_mid = (cw - 0.5 * w_sorted) / np.clip(cw_total, 1e-6, None)

    target = q / 100.0
    idx_hi = np.clip((cw_mid < target).sum(axis=0), 0, T - 1)
    idx_lo = np.clip(idx_hi - 1, 0, T - 1)

    v_hi = np.take_along_axis(v_sorted, idx_hi[None], axis=0)[0]
    v_lo = np.take_along_axis(v_sorted, idx_lo[None], axis=0)[0]
    cw_hi = np.take_along_axis(cw_mid, idx_hi[None], axis=0)[0]
    cw_lo = np.take_along_axis(cw_mid, idx_lo[None], axis=0)[0]

    denom = np.clip(cw_hi - cw_lo, 1e-6, None)
    frac = np.clip((target - cw_lo) / denom, 0, 1)
    result = v_lo + frac * (v_hi - v_lo)
    return np.where(idx_hi == idx_lo, v_hi, result)


def naive_composite(refl, weight, percentile=50, clear_thresh=0.6):
    """The standard approach, for comparison: percentile computed only from
    confidently-clear real observations, ignoring reconstruction entirely.
    Returns (composite (4,H,W), n_valid (H,W)) -- n_valid is the thing that
    varies unevenly and drives the inconsistency.
    """
    conf_mask = (weight >= clear_thresh).astype(np.float32)  # (T,H,W)
    n_valid = conf_mask.sum(axis=0)
    C = refl.shape[1]
    comp = np.full((C,) + weight.shape[1:], np.nan, dtype=np.float32)
    for c in range(C):
        w = np.where(conf_mask > 0, conf_mask, 0.0)
        # pixels with zero valid observations: leave as NaN (undefined -- this
        # is itself part of the problem the reconstruction-informed version fixes)
        has_any = n_valid > 0
        vals = refl[:, c]
        res = weighted_percentile_along_time(vals, w, percentile)
        comp[c] = np.where(has_any, res, np.nan)
    return comp, n_valid


def robust_composite(composite, alpha_used, percentile=50, recon_trust=0.35):
    """composite: (T,4,H,W) final monthly composites (observed where
    confident, reconstructed elsewhere -- from compose.compose).
    alpha_used: (T,H,W) in [0,1], the observed-vs-reconstructed trust used
    for that same composite.

    Weight = alpha_used, floored at `recon_trust` rather than 0 -- every
    month contributes *something*, so every pixel gets a full, comparable
    n=12 sample depth regardless of how cloudy its year was.
    """
    C = composite.shape[1]
    w = np.maximum(alpha_used, recon_trust)  # (T,H,W)
    comp = np.zeros((C,) + alpha_used.shape[1:], dtype=np.float32)
    for c in range(C):
        comp[c] = weighted_percentile_along_time(composite[:, c], w, percentile)
    return comp


def medoid_composite(composite, alpha_used, target=None, recon_trust=0.35):
    """Band-consistent alternative: pick, per pixel, the actual month whose
    full spectral vector is closest (weighted L2) to `target` (defaults to
    the robust_composite median), rather than mixing independently-chosen
    per-band statistics. Avoids ever synthesizing a spectrally-impossible
    pixel; the output is always a genuine observed-or-reconstructed vector
    from one specific month.
    """
    if target is None:
        target = robust_composite(composite, alpha_used, percentile=50, recon_trust=recon_trust)
    T, C, H, W = composite.shape
    dist2 = ((composite - target[None]) ** 2).sum(axis=1)  # (T,H,W)
    # among months with decent trust if any exist for this pixel, else fall
    # back to all months (better a low-confidence real pick than nothing)
    has_confident = (alpha_used >= 0.5).any(axis=0)
    penalty = np.where(alpha_used >= 0.5, 0.0, np.inf)
    dist2_restricted = dist2 + np.where(has_confident[None], penalty, 0.0)
    best_t = np.argmin(dist2_restricted, axis=0)  # (H,W)

    comp = np.take_along_axis(composite, best_t[None, None], axis=0)[0]
    return comp, best_t