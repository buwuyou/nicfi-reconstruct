"""
Cloud / cloud-shadow / haze masking for each monthly AOI.

Primary detector: OmniCloudMask (Wright et al. 2024/2025) — a lightweight
(~1.2M-param) CNN trained across many sensors (Sentinel-2, Landsat, PlanetScope)
specifically for 10-50m imagery, distributed as pretrained weights via
`pip install omnicloudmask`. This is a genuine, small, fast, sota-ish choice
for the detection half of the problem, and it's the only DL model we treat as
"trained on external data" in this pipeline — no other component needs
labelled cloud masks or a pretrained satellite foundation model.

Output classes (model_version >= 2): 0=clear, 1=thick cloud, 2=thin cloud,
3=cloud shadow.

Two independent improvements over a single-model, single-date call:

1. **Model ensembling**: OmniCloudMask ships several model generations
   (`model_version` 1.0-4.0, trained separately). We run two of them
   (3.0 = last "legacy" fastai-trained generation, 4.0 = current) and average
   their softmax class probabilities rather than trusting one network's
   verdict. The two versions disagree on a non-trivial fraction of
   ambiguous/edge pixels (thin cirrus, cloud boundaries) — averaging is a
   cheap variance reduction on exactly the cases a single model is least
   reliable on.
2. **Temporal-residual outlier rejection** (`robust_mask.py`) — a single-date
   classifier structurally cannot catch a cloud/haze/shadow that happens to
   look locally plausible; but a pixel's own 12-month history can, by
   flagging months that don't fit its (robustly-estimated) seasonal curve.
   That pass runs *after* this module, once an initial phenology fit exists.

We additionally flag residual haze that OmniCloudMask tends to under-call
(thin, brightening veils) using a simple physically motivated haze index:
haze inflates blue/green reflectance and compresses the NIR-red contrast.
This is cheap, has no learned parameters, and only ever *adds* pixels to the
"uncertain" set, so it can't hide obvious cloud errors from the DL model.
"""
import numpy as np

from . import sites as config

# Validity/quality codes used throughout the pipeline
CLEAR = 0
CLOUD_THICK = 1
CLOUD_THIN = 2
SHADOW = 3
HAZE = 4
NODATA = 5
TEMPORAL_OUTLIER = 6  # assigned later, by robust_mask.py


def _haze_index(refl: np.ndarray) -> np.ndarray:
    """refl: (4,H,W) reflectance (blue,green,red,nir). Higher = hazier.
    Combines blue brightness (haze/aerosol scattering is strongest at blue)
    with a depressed NDVI-like NIR-red contrast (haze desaturates vegetation
    contrast). Purely heuristic, used only to catch thin veils the CNN misses.
    """
    blue, green, red, nir = refl[config.BLUE], refl[config.GREEN], refl[config.RED], refl[config.NIR]
    ndvi_contrast = (nir - red) / (nir + red + 1e-6)
    haze = np.clip(blue - 0.5 * ndvi_contrast.clip(min=0), 0, None)
    return haze


def ocm_ensemble(rgn: np.ndarray, device: str = "cpu", model_versions=(3.0, 4.0)):
    """rgn: (3,H,W) Red/Green/NIR reflectance, 0 = nodata. Runs each
    OmniCloudMask generation in `model_versions` and averages their softmax
    class probabilities (see module docstring). Returns (pred (H,W) int in
    {0=clear,1=thick,2=thin,3=shadow}, disagreement (H,W) float32 = fraction
    of models whose own argmax differs from the consensus). Sensor-agnostic:
    used for NICFI here and for Sentinel-2 by `src/nicfirecon/masking.py`."""
    from omnicloudmask import predict_from_array

    probs = []
    for v in model_versions:
        probs.append(predict_from_array(
            rgn.astype(np.float32), patch_size=1000, patch_overlap=300, batch_size=1,
            inference_device=device, no_data_value=0, apply_no_data_mask=True,
            export_confidence=True, softmax_output=True, model_version=v,
        ))  # (4,H,W) class probabilities
    probs = np.stack(probs, axis=0)  # (n_models,4,H,W)
    pred = probs.mean(axis=0).argmax(axis=0)  # (H,W) in {0,1,2,3}
    disagreement = (probs.argmax(axis=1) != pred[None]).mean(axis=0).astype(np.float32)
    return pred, disagreement


def compute_masks(stack_refl: np.ndarray, device: str = "cpu", haze_thresh: float = 0.14,
                   model_versions=(3.0, 4.0)) -> np.ndarray:
    """stack_refl: (T,4,H,W) reflectance in [0,~1.2]. Returns (T,H,W) uint8
    quality codes (see constants above), one map per month.

    Ensembles `model_versions` OmniCloudMask generations by averaging their
    softmax class probabilities before taking the class with highest
    consensus confidence (see module docstring).
    """
    T = stack_refl.shape[0]
    quality = np.zeros((T,) + stack_refl.shape[2:], dtype=np.uint8)
    disagreement = np.zeros((T,) + stack_refl.shape[2:], dtype=np.float32)

    for t in range(T):
        refl = stack_refl[t]
        nodata_mask = np.all(refl <= 0, axis=0)
        rgn = np.stack([refl[config.RED], refl[config.GREEN], refl[config.NIR]], axis=0)
        pred, disagreement[t] = ocm_ensemble(rgn, device=device, model_versions=model_versions)

        q = np.zeros_like(pred, dtype=np.uint8)
        q[pred == 1] = CLOUD_THICK
        q[pred == 3] = SHADOW
        q[(pred == 2) & (q == 0)] = CLOUD_THIN

        haze = _haze_index(refl)
        q[(q == 0) & (haze > haze_thresh)] = HAZE
        q[nodata_mask] = NODATA

        quality[t] = q

    return quality, disagreement


def quality_weight(quality: np.ndarray, disagreement: np.ndarray = None) -> np.ndarray:
    """Map quality codes to a [0,1] observation-confidence weight used by the
    temporal reconstruction. Thin cloud/haze aren't binary-excluded — they're
    strongly downweighted, since they still carry some real signal.

    If `disagreement` (fraction of ensembled models that disagreed with the
    consensus class, from `compute_masks`) is given, a pixel called "clear"
    only because of a close vote gets partial credit rather than full trust
    -- ensemble disagreement is itself a confidence signal, not just a
    tie-breaker.
    """
    w = np.ones(quality.shape, dtype=np.float32)
    w[quality == CLOUD_THICK] = 0.0
    w[quality == SHADOW] = 0.0
    w[quality == NODATA] = 0.0
    w[quality == CLOUD_THIN] = 0.15
    w[quality == HAZE] = 0.25
    w[quality == TEMPORAL_OUTLIER] = 0.1
    if disagreement is not None:
        w = w * (1.0 - 0.5 * disagreement)
    return w