"""
Temporal-residual confidence estimation (TMask-style detection, after
Zhu & Woodcock 2014, generalized to a continuous severity/opacity score
instead of a hard flag).

A single-date classifier -- however good -- can only ever look at one image
in isolation, so it structurally cannot catch a cloud/haze/shadow that
happens to look locally plausible on its own but is inconsistent with that
same pixel's own history. This module closes the loop by using the pixel's
own multi-month record as a second, temporal detector:

    build a reference curve -> measure how anomalous each observation is
    against it -> convert that into a continuous trust weight -> repeat

Only pixels not already hard-blocked by OmniCloudMask (thick cloud / shadow
/ nodata) are touched -- this pass can only add caution to an otherwise
"clear/ambiguous" call, never override an obvious categorical one.

**Why the reference curve is *not* the per-pixel harmonic fit used for
reconstruction**: an order-2 harmonic (5 free parameters) fit to a single
pixel's ~10-12 clear-ish monthly points is close to interpolating them --
its in-sample residuals are tiny almost by construction (empirically,
std ~0.006 reflectance here), which makes *every* pixel look like a
razor-sharp outlier detector and floods the flagged set with false
positives (an earlier version of this module flagged >5% of all
pixel-months this way -- clearly wrong). Instead, the reference here is a
**cluster shape + a single robust per-pixel offset**: the pooled,
high-degrees-of-freedom cluster curve from `phenology.fit_cluster_priors`
(fit from thousands of pixels, not overfit to one), shifted by that pixel's
own median offset from it. This has much more honest residual statistics
because the model doing the predicting was never allowed to chase this one
pixel's noise.

**Why this replaced a hard z-score threshold**: an earlier version of this
module hard-flagged pixels above a z-score cutoff (same-sign-across-4-bands,
z>4) as a discrete "temporal outlier" class, on top of a separately fixed
0.25 trust weight for anything the spectral haze heuristic in `cloud_mask.py`
called "haze". That combination left a real, measured gap: pixels the
detectors correctly called "haze" but never revisited kept a flat 25% trust
regardless of how hazy they actually were -- a thin veil and a nearly-opaque
one got treated identically. Measured against this AOI's own data, the
z-score signal cleanly separates the two: pixels independently confirmed as
residual haze score severity ~35-80 in these units, while the *entire*
population of confidently-clear pixels sits under 12 even at the 99.99th
percentile (`outputs/figures/13_severity_calibration.png`). That's a big
enough gap to use severity directly as a continuous, per-pixel-month
opacity estimate instead of routing everything through fixed per-class
weights -- a thin veil and a thick one now get different trust
automatically, and a discrete "outlier" class isn't needed at all.
"""
import numpy as np

from . import cloud_mask, phenology


def _cluster_shape_plus_offset(refl, weight, cluster_labels, cluster_beta, conf_thresh=0.9):
    """Reference curve = pooled cluster shape + a robust (median) per-pixel
    level shift, fit only from this pixel's confidently-clear points. Much
    higher effective degrees of freedom than the per-pixel harmonic fit, so
    its residuals are a meaningful "how surprising is this observation"
    signal rather than near-zero by construction.
    """
    T = refl.shape[0]
    X = phenology.design_matrix(T)
    beta_px = cluster_beta[cluster_labels]           # (H,W,4,k)
    cluster_curve = np.einsum("tk,hwck->tchw", X, beta_px)  # (T,4,H,W)

    conf_mask = weight >= conf_thresh                # (T,H,W)
    C = refl.shape[1]
    offset = np.zeros((C,) + cluster_labels.shape, dtype=np.float32)
    for c in range(C):
        diff = refl[:, c] - cluster_curve[:, c]       # (T,H,W)
        diff_masked = np.where(conf_mask, diff, np.nan)
        with np.errstate(invalid="ignore"):
            off = np.nanmedian(diff_masked, axis=0)
        offset[c] = np.nan_to_num(off, nan=0.0)

    return cluster_curve + offset[None]


def _cluster_band_scale(residual, weight, cluster_labels, conf_thresh=0.9, min_samples=20):
    """residual:(T,4,H,W) weight:(T,H,W) cluster_labels:(H,W) -> (K,4) robust
    (MAD-based) scale of "normal" residual spread, per cluster per band,
    estimated only from confidently-clear observations.
    """
    T, C, H, W = residual.shape
    N = H * W
    resid_flat = residual.reshape(T, C, N)
    w_flat = weight.reshape(T, N)
    labels_flat = cluster_labels.reshape(N)
    K = int(labels_flat.max()) + 1

    conf = w_flat >= conf_thresh
    scale = np.full((K, C), np.nan, dtype=np.float32)
    for k in range(K):
        member = labels_flat == k
        if member.sum() == 0:
            continue
        conf_k = conf[:, member]
        for c in range(C):
            vals = resid_flat[:, c, :][:, member][conf_k]
            if vals.size >= min_samples:
                mad = np.median(np.abs(vals - np.median(vals)))
                scale[k, c] = max(float(mad) * 1.4826, 1e-4)
    global_scale = np.nanmedian(scale, axis=0)
    global_scale = np.where(np.isnan(global_scale), 0.05, global_scale)
    nan_mask = np.isnan(scale)
    scale = np.where(nan_mask, global_scale[None, :], scale)
    return scale


def severity_score(refl, weight, cluster_labels, cluster_beta):
    """Continuous "how anomalously bright/contaminated is this observation"
    score, in cluster-and-band-normalized z-score units, per pixel-month.

    Positive-only and averaged across all 4 bands (not per-band max): real
    cloud/haze brightens blue/green/red/NIR *together*, so averaging both
    rewards coherent multi-band brightening (a real event) and automatically
    suppresses a single noisy band (which only contributes 1/4 of the mean)
    -- a softer, continuous version of the old hard "all 4 bands must agree
    in sign" rule, without needing a hand-picked band-count threshold.
    """
    ref_curve = _cluster_shape_plus_offset(refl, weight, cluster_labels, cluster_beta)
    residual = refl - ref_curve
    scale = _cluster_band_scale(residual, weight, cluster_labels)  # (K,4)
    scale_px = scale[cluster_labels]  # (H,W,4)
    z = residual / np.transpose(scale_px, (2, 0, 1))[None, ...]  # (T,4,H,W)
    return np.clip(z, 0, None).mean(axis=1)  # (T,H,W)


def continuous_confidence(refl, weight_categorical, quality, cluster_labels, cluster_beta,
                           severity50=6.0, power=2.5, n_iters=2, target_weight=10.0,
                           thin_cap=0.3, haze_cap=0.4):
    """Iteratively: build reference -> score severity -> convert to a smooth
    [0,1] trust weight -> refit cluster priors against the new weight ->
    repeat.

    Severity is a *brightening* signature specifically (see `severity_score`)
    -- it's blind to contamination that shifts color without broadly
    brightening every band (a real case found in this AOI: a chromatic
    speckle artifact, thin-cirrus-like, that OmniCloudMask's ensemble
    correctly called CLOUD_THIN but which scored near-zero severity because
    it isn't a uniform brightening event). An earlier version of this
    function let severity fully override the categorical call for anything
    short of CLOUD_THICK/SHADOW/NODATA, which let that case's weight climb
    back to ~0.78 mean -- silently discarding OmniCloudMask's own correct
    verdict. Fixed by treating the categorical class as a trust *cap*, not
    something severity can override upward: CLOUD_THIN can never exceed
    `thin_cap`, HAZE never exceeds `haze_cap`, regardless of how low
    severity scores it -- severity can only push trust *down* from there,
    catching contamination the categorical detectors missed, never up past
    what they already flagged.

    `severity50` is calibrated against this AOI's own clear-pixel
    population (its 99.9th percentile is ~6 in these units -- see module
    docstring), so a pixel at that severity keeps ~50% trust, while the
    independently-confirmed haze residual (severity 35-80) drops to <1%.

    Returns: weight (updated, continuous), severity (last iteration, for
    diagnostics/visualization), recon_curve, confidence.
    """
    hard_block = np.isin(quality, [cloud_mask.CLOUD_THICK, cloud_mask.SHADOW, cloud_mask.NODATA])
    cap = np.ones(quality.shape, dtype=np.float32)
    cap[quality == cloud_mask.CLOUD_THIN] = thin_cap
    cap[quality == cloud_mask.HAZE] = haze_cap
    weight = weight_categorical.copy()
    severity = None

    for it in range(n_iters):
        severity = severity_score(refl, weight, cluster_labels, cluster_beta)
        continuous_w = 1.0 / (1.0 + (severity / severity50) ** power)
        weight = np.where(hard_block, 0.0, np.minimum(continuous_w, cap)).astype(np.float32)
        print(f"  [robust_mask] iter {it+1}/{n_iters}: mean weight over non-hard-blocked px "
              f"= {weight[~hard_block].mean():.4f} (severity mean={severity[~hard_block].mean():.3f})")

    recon_curve, confidence = phenology.reconstruct(refl, weight, cluster_labels, cluster_beta,
                                                      target_weight=target_weight)
    return weight, severity, recon_curve, confidence