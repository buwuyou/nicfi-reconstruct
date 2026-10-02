"""
Monthly NICFI reconstruction -- three methods, one output format:

- `mask`      contaminated (cloud/shadow/haze, buffered) and nodata NICFI
              pixels set to nodata (0); everything else untouched. The
              cheapest, most conservative product.
- `s2fill`    contaminated pixels replaced by the same month's clear
              Sentinel-2 composite, harmonized onto NICFI radiometry
              (`harmonize.py`), blended outward over FEATHER_PX into clear
              NICFI. Contaminated pixels with no clear S2 are kept as-is.
              `--add-s2-bands` appends S2's B5/B6/B7/B8A/B11/B12 (no NICFI
              counterpart, so not harmonized; S2 DN scale, 0 where no clear
              S2) for every pixel -> a 10-band product.
- `phenology` the temporal method (see docs/amazon.md): per-pixel harmonic
              phenology shrunk toward land-cover cluster curves, continuous
              confidence weights, an optional self-supervised partial-conv
              refiner for spatial texture, then observation blended with the
              reconstruction by confidence (`src/phenology.py`,
              `robust_mask.py`, `inpaint.py`, `compose.py`). Needs the whole
              (contiguous) series at once.

All methods read the post-checked NICFI classes (`masking.load_nicfi_classes`)
and write, to outputs/nicfirecon/<tile>/reconstructed/<method>/:
    <tile>_<month>.tif           uint16 DN (4 bands, or 10 with --add-s2-bands)
    <tile>_<month>_quality.tif   5-band uint8 data-quality layer:
      source       SRC_* below
      nicfi_class  NICFI class after the post-check (cloud_mask codes)
      flags        FLAG_* bit field below
      s2_n_clear   clear S2 observations that month (s2fill only)
      score        0-100 heuristic confidence -- a ranking aid, not a
                   calibrated accuracy
"""
import json
from typing import Dict, Optional

import numpy as np
import rasterio
from rasterio.warp import Resampling
from scipy import ndimage

from .. import cloud_mask, io_utils
from . import config as cfg, harmonize as hz, masking, s2

FEATHER_PX = 8               # ~40m at NICFI's 4.77m
NICFI_BUFFER_PX = 3          # ~15m dilation of NICFI contamination
METHODS = ("mask", "s2fill", "phenology")

QUALITY_BANDS = ["source", "nicfi_class", "flags", "s2_n_clear", "score"]
SRC_NICFI = 0          # clear NICFI, kept
SRC_S2_SINGLE = 1      # replaced by harmonized S2, the month's single cloud-free frame
SRC_S2_MEDIAN = 2      # replaced by harmonized S2, median of clear observations
SRC_CONTAMINATED = 3   # contaminated NICFI: kept as-is (s2fill, no clear S2) / masked (mask)
SRC_NODATA = 4         # no data
SRC_PHENO = 5          # phenology reconstruction (observation weight < 5%)
SRC_PHENO_BLEND = 6    # observation blended with the phenology reconstruction
SOURCE_NAMES = ["NICFI clear", "S2 single frame", "S2 median", "contaminated NICFI",
                "nodata", "phenology", "phenology blend"]
FLAG_CLEARED = 1       # NICFI flag cleared by the post-check (temporal: ground; spatial: speckle)
FLAG_BLEND = 2         # clear NICFI mixed with S2 in the edge ramp around a fill
FLAG_FIT_FALLBACK = 4  # S2 harmonized with a borrowed (year-median) fit
FLAG_BUFFER = 8        # NICFI class clear, treated as contaminated only as cloud buffer

SCORE_NICFI, SCORE_NICFI_CLEARED, SCORE_BLEND = 100, 90, 95
SCORE_S2_SINGLE = 80
SCORE_S2_MEDIAN = {1: 50, 2: 60}   # n_clear -> score; >= 3 -> 70
SCORE_S2_MEDIAN_MANY = 70
SCORE_FALLBACK_PENALTY = 15
SCORE_KEPT = {cloud_mask.CLEAR: 60, cloud_mask.HAZE: 30, cloud_mask.CLOUD_THIN: 20,
              cloud_mask.SHADOW: 10, cloud_mask.CLOUD_THICK: 0}
SCORE_PHENO = 45       # score of a fully model-reconstructed pixel


def replace_mask(quality: np.ndarray, buffer_px: int = NICFI_BUFFER_PX) -> np.ndarray:
    """Contaminated (buffered) or nodata -- what every method treats as not
    trustworthy NICFI."""
    dirty = np.isin(quality, masking.CONTAMINATED)
    if buffer_px > 0 and dirty.any():
        dirty = ndimage.binary_dilation(dirty, iterations=buffer_px)
    return dirty | (quality == cloud_mask.NODATA)


def feather_alpha(mask: np.ndarray, feather_px: int = FEATHER_PX) -> np.ndarray:
    """1 everywhere inside `mask`, ramping to 0 over `feather_px` px *outward*
    into clear NICFI -- never a hard seam where the source changes, and the
    ramp only ever mixes the fill into clear pixels, never lets a
    contaminated NICFI pixel through at partial weight."""
    if not mask.any():
        return np.zeros(mask.shape, dtype=np.float32)
    if mask.all():
        return np.ones(mask.shape, dtype=np.float32)
    outside = ndimage.distance_transform_edt(~mask)
    return np.where(mask, 1.0, np.maximum(1.0 - outside / (feather_px + 1), 0.0)).astype(np.float32)


def _base_layer(quality, cleared, replace):
    """source/flags/score for NICFI-only pixels; methods overwrite where they fill."""
    h, w = quality.shape
    src = np.full((h, w), SRC_NICFI, np.uint8)
    flags = np.zeros((h, w), np.uint8)
    flags[cleared] |= FLAG_CLEARED
    flags[replace & (quality == cloud_mask.CLEAR)] |= FLAG_BUFFER
    score = np.full((h, w), SCORE_NICFI, np.int16)
    score[cleared] = SCORE_NICFI_CLEARED
    kept = replace & (quality != cloud_mask.NODATA)
    src[kept] = SRC_CONTAMINATED
    for cls, sc in SCORE_KEPT.items():
        score[kept & (quality == cls)] = sc
    src[quality == cloud_mask.NODATA] = SRC_NODATA
    score[quality == cloud_mask.NODATA] = 0
    return src, flags, score


def _stack_layer(src, quality, flags, n_clear, score):
    return np.stack([src, quality.astype(np.uint8), flags, n_clear.astype(np.uint8),
                     np.clip(score, 0, 100).astype(np.uint8)])


# ---------------------------------------------------------------- method: mask
def mask_month(nicfi_raw, quality, cleared):
    replace = replace_mask(quality)
    src, flags, score = _base_layer(quality, cleared, replace)
    score[replace] = 0  # masked: no value at all
    out = np.where(replace[None], 0.0, nicfi_raw).astype(np.float32)
    return out, _stack_layer(src, quality, flags, np.zeros_like(src), score)


# -------------------------------------------------------------- method: s2fill
def s2fill_month(nicfi_raw, quality, cleared, n_tr, n_crs, comp, coeffs, add_s2_bands=False):
    """coeffs: from harmonize.fit_all_months, or None (no usable S2 this
    month -> contaminated NICFI is kept)."""
    shape = nicfi_raw.shape[1:]
    replace = replace_mask(quality)
    nodata = quality == cloud_mask.NODATA
    src, flags, score = _base_layer(quality, cleared, replace)
    n_clear = np.zeros(shape, np.uint8)
    extra = np.zeros((len(cfg.S2_EXTRA_BANDS),) + shape, np.float32) if add_s2_bands else None
    if comp is None or coeffs is None:
        out = nicfi_raw.astype(np.float32)
    else:
        harm = hz.reproject_to(hz.harmonized_s2(nicfi_raw, quality, n_tr, n_crs, comp, coeffs),
                               comp.transform, comp.crs, shape, n_tr, n_crs,
                               Resampling.bilinear, src_nodata=0.0)
        cover = hz.reproject_to(comp.valid[None].astype(np.float32), comp.transform, comp.crs,
                                shape, n_tr, n_crs, Resampling.average)[0] >= hz.COVER_FRAC
        harm[:, ~cover] = 0.0
        src_s2, n_clear = hz.reproject_to(np.stack([comp.source, comp.n_clear]).astype(np.float32),
                                          comp.transform, comp.crs, shape, n_tr, n_crs,
                                          Resampling.nearest).astype(np.uint8)
        fill = replace & cover
        alpha = feather_alpha(fill) * cover
        alpha[nodata & cover] = 1.0  # nothing real to blend toward
        out = alpha * harm + (1.0 - alpha) * nicfi_raw
        out[:, nodata & ~cover] = 0.0

        flags[(alpha > 0) & ~fill] |= FLAG_BLEND
        score[(alpha > 0) & ~fill] = SCORE_BLEND
        single = fill & (src_s2 == s2.SOURCE_SINGLE)
        median = fill & ~single
        src[single], src[median] = SRC_S2_SINGLE, SRC_S2_MEDIAN
        score[single] = SCORE_S2_SINGLE
        med_score = np.where(n_clear >= 3, SCORE_S2_MEDIAN_MANY,
                             np.where(n_clear == 2, SCORE_S2_MEDIAN[2], SCORE_S2_MEDIAN[1]))
        score[median] = med_score[median]
        if coeffs.get("source") != "month":
            flags[fill] |= FLAG_FIT_FALLBACK
            score[fill] -= SCORE_FALLBACK_PENALTY
        if add_s2_bands:
            extra = hz.reproject_to(hz.s2_bands(comp, cfg.S2_EXTRA_BANDS), comp.transform, comp.crs,
                                    shape, n_tr, n_crs, Resampling.bilinear, src_nodata=0.0)
            extra[:, ~cover] = 0.0
    if add_s2_bands:
        out = np.concatenate([out, extra])
    return out.astype(np.float32), _stack_layer(src, quality, flags, n_clear, score)


# ----------------------------------------------------------- method: phenology
def phenology_series(refl, quality, disagreement, device="cuda", n_clusters=10,
                     refiner=True, refiner_iters=800, log=print):
    """refl (T,4,H,W) reflectance of a contiguous monthly series. Returns
    (composite reflectance (T,4,H,W), observation weight used (T,H,W))."""
    from .. import compose, inpaint, phenology, robust_mask

    weight = cloud_mask.quality_weight(quality, disagreement)
    log("  phenology: land-cover cluster priors...")
    labels, beta = phenology.fit_cluster_priors(refl, weight, n_clusters=n_clusters)
    log("  phenology: continuous confidence + harmonic reconstruction...")
    weight_final, _, recon_curve, _ = robust_mask.continuous_confidence(
        refl, weight, quality, labels, beta, severity50=6.0, power=2.5, n_iters=2,
        target_weight=10.0)
    if refiner:
        log(f"  phenology: training the spatial refiner ({refiner_iters} iters, {device})...")
        model = inpaint.train_refiner(refl, weight_final, recon_curve, n_iters=refiner_iters,
                                      patch=192, batch=6, lr=2e-3, device=device,
                                      log_every=max(refiner_iters // 4, 1), seed=0)
        refined = inpaint.apply_refiner(model, refl, weight_final, recon_curve, device=device,
                                        tile=512, overlap=64)
    else:
        refined = recon_curve
    return compose.compose(refl, weight_final, refined)


def phenology_layer(quality, cleared, alpha):
    # no cloud buffer here: phenology decides per pixel via its own weights,
    # so a clear-class pixel it kept is NICFI clear, not "contaminated"
    replace = replace_mask(quality, buffer_px=0)
    src, flags, score = _base_layer(quality, cleared, replace)
    nodata = quality == cloud_mask.NODATA
    pheno = (alpha < 0.05) & ~nodata
    blend = (alpha >= 0.05) & (alpha < 0.95) & ~nodata
    src[pheno], src[blend] = SRC_PHENO, SRC_PHENO_BLEND
    score_alpha = np.round(100 * alpha + SCORE_PHENO * (1 - alpha)).astype(np.int16)
    score[pheno | blend] = score_alpha[pheno | blend]
    return _stack_layer(src, quality, flags, np.zeros_like(src), score)


# ------------------------------------------------------------------- runner
def _write(tile, method, month, data, quality_layer, n_tr, n_crs, band_names):
    d = tile.recon_dir(method)
    path = d / f"{tile.tile_id}_{month}.tif"
    io_utils.save_geotiff(path, data.clip(0, 65535).astype("uint16"), n_tr, n_crs,
                          dtype="uint16", nodata=0)
    qpath = d / f"{tile.tile_id}_{month}_quality.tif"
    io_utils.save_geotiff(qpath, quality_layer, n_tr, n_crs, dtype="uint8")
    for p, names in ((path, band_names), (qpath, QUALITY_BANDS)):
        with rasterio.open(p, "r+") as dst:
            dst.descriptions = tuple(names)


def _month_stats(layer) -> Dict:
    src = layer[0]
    return {"source": {n: float((src == i).mean()) for i, n in enumerate(SOURCE_NAMES)},
            "cleared": float((layer[2] & FLAG_CLEARED).astype(bool).mean()),
            "mean_score": float(layer[4].mean())}


def run_reconstruct(tile: cfg.Tile, method: str, add_s2_bands: bool = False, device: str = "cuda",
                    n_clusters: int = 10, refiner: bool = True, refiner_iters: int = 800, log=print):
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    bands = list(cfg.NICFI_BAND_NAMES)
    stats, meta = {}, {"method": method}

    if method == "mask":
        for month in tile.months:
            nicfi, n_tr, n_crs = io_utils.read_full(tile.nicfi_path(month))
            out, layer = mask_month(nicfi, masking.load_nicfi_classes(tile, month),
                                    masking.load_nicfi_cleared(tile, month))
            _write(tile, method, month, out, layer, n_tr, n_crs, bands)
            stats[month] = _month_stats(layer)
            log(f"  {month}: masked {stats[month]['source']['contaminated NICFI']:.1%}")

    elif method == "s2fill":
        if not tile.has_s2:
            raise FileNotFoundError(f"s2fill needs Sentinel-2 frames in {tile.s2_dir}")
        if add_s2_bands:
            bands += cfg.S2_EXTRA_BANDS
        read = lambda m: io_utils.read_full(tile.nicfi_path(m))
        fits = hz.fit_all_months(tile, lambda m: s2.load(tile, m),
                                 lambda m: masking.load_nicfi_classes(tile, m), read)
        (tile.cache_dir / "harmonization.json").write_text(json.dumps(fits, indent=1, sort_keys=True))
        for month in tile.months:
            nicfi, n_tr, n_crs = read(month)
            f = fits.get(month)
            usable = f if f and f["source"] != "rejected" else None
            comp = s2.load(tile, month)
            out, layer = s2fill_month(nicfi, masking.load_nicfi_classes(tile, month),
                                      masking.load_nicfi_cleared(tile, month), n_tr, n_crs,
                                      comp, usable, add_s2_bands)
            _write(tile, method, month, out, layer, n_tr, n_crs, bands)
            st = stats[month] = _month_stats(layer)
            st.update(s2_method=comp.method if comp else "none", fit=f["source"] if f else "none")
            replaced = st["source"]["S2 single frame"] + st["source"]["S2 median"]
            log(f"  {month}: replaced by S2 {replaced:.1%}, contaminated kept "
                f"{st['source']['contaminated NICFI']:.1%} (S2 {st['s2_method']}, fit {st['fit']})")
        meta["add_s2_bands"] = add_s2_bands

    else:  # phenology
        months = tile.months
        idx = [int(m[:4]) * 12 + int(m[5:]) for m in months]
        if idx != list(range(idx[0], idx[0] + len(idx))):
            raise ValueError("phenology needs a contiguous monthly series (check --months)")
        reads = [io_utils.read_full(tile.nicfi_path(m)) for m in months]
        n_tr, n_crs = reads[0][1], reads[0][2]
        refl = np.stack([r[0] for r in reads]) / cfg.REFLECTANCE_SCALE
        del reads
        quality = np.stack([masking.load_nicfi_classes(tile, m) for m in months])
        dis = [masking.load_nicfi_disagreement(tile, m) for m in months]
        dis = np.stack(dis).astype(np.float32) if all(d is not None for d in dis) else None
        comp, alpha = phenology_series(refl, quality, dis, device, n_clusters, refiner,
                                       refiner_iters, log)
        for t, month in enumerate(months):
            nodata = quality[t] == cloud_mask.NODATA
            out = np.where(nodata[None], 0.0, comp[t] * cfg.REFLECTANCE_SCALE)
            layer = phenology_layer(quality[t], masking.load_nicfi_cleared(tile, month), alpha[t])
            _write(tile, method, month, out, layer, n_tr, n_crs, bands)
            stats[month] = _month_stats(layer)
        meta.update(n_clusters=n_clusters, refiner=refiner, refiner_iters=refiner_iters)
        log(f"  phenology: wrote {len(months)} months")

    meta["bands"] = bands
    (tile.recon_dir(method) / "stats.json").write_text(json.dumps({"meta": meta, "months": stats},
                                                                   indent=1))
