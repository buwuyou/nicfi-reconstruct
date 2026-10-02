"""
Preprocessing, step "mask": cloud/shadow/haze classes per observation, for
both sensors, with the same OmniCloudMask 2-generation ensemble
(`src/cloud_mask.ocm_ensemble`, Red/Green/NIR input).

- NICFI (monthly): OCM + the haze heuristic (`cloud_mask.compute_masks`),
  cached as cache/nicfi_quality_<month>.npz {quality, disagreement}.
- Sentinel-2 (single-date frames): OCM, cached as
  cache/s2_masks/<date>.npz {classes}. Sentinel-2 at 10m is squarely inside
  OCM's training domain; NICFI at 4.77m is not, which is why its masks need
  the post-check (`postcheck.py`) much more.

The post-check adds `refined` (+ `overridden`, `despeckled`) next to the raw
classes in the same files; everything downstream reads `refined` through
the loaders here. Resumable: cached observations are skipped.
"""
from pathlib import Path
from typing import Dict, List

import numpy as np
import rasterio

from .. import cloud_mask, io_utils
from . import config as cfg

CONTAMINATED = (cloud_mask.CLOUD_THICK, cloud_mask.CLOUD_THIN, cloud_mask.SHADOW,
                cloud_mask.HAZE)


# ---------------------------------------------------------------- Sentinel-2
def band_index(descriptions) -> Dict[str, int]:
    idx = {d: i for i, d in enumerate(descriptions) if d}
    missing = [b for b in cfg.NICFI_TO_S2_BAND.values() if b not in idx]
    if missing:
        raise ValueError(f"Sentinel-2 file lacks band descriptions for {missing} "
                         f"(has {descriptions}); can't map bands safely.")
    return idx


def read_frame(path: Path):
    """Returns (data (C,H,W) float32 DN with nodata as 0, transform, crs,
    band names). Nodata = all bands 0 or any band non-finite."""
    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)
        transform, crs, names = src.transform, src.crs, list(src.descriptions)
    bad = ~np.all(np.isfinite(data), axis=0) | np.all(data == 0, axis=0)
    data[:, bad] = 0.0
    return data, transform, crs, names


def s2_ocm_classes(data: np.ndarray, names: List[str], device: str,
                   model_versions=(3.0, 4.0)) -> np.ndarray:
    """(H,W) uint8: cloud_mask codes CLEAR/CLOUD_THICK/CLOUD_THIN/SHADOW/NODATA."""
    bi = band_index(names)
    rgn = np.stack([data[bi["B4"]], data[bi["B3"]], data[bi["B8"]]]) / cfg.REFLECTANCE_SCALE
    nodata = np.all(data == 0, axis=0)
    pred, _ = cloud_mask.ocm_ensemble(rgn, device=device, model_versions=model_versions)
    q = np.zeros(pred.shape, dtype=np.uint8)
    q[pred == 1] = cloud_mask.CLOUD_THICK
    q[pred == 2] = cloud_mask.CLOUD_THIN
    q[pred == 3] = cloud_mask.SHADOW
    q[nodata] = cloud_mask.NODATA
    return q


def s2_mask_path(tile: cfg.Tile, frame: Path) -> Path:
    d = tile.cache_dir / "s2_masks"
    d.mkdir(exist_ok=True)
    return d / f"{frame.stem}.npz"


def load_s2_classes(tile: cfg.Tile, frame: Path) -> np.ndarray:
    """Frame classes after the post-check."""
    z = np.load(s2_mask_path(tile, frame))
    if "refined" not in z:
        raise KeyError(f"{frame.name}: no post-checked mask -- run "
                       f"`preprocess --steps postcheck` first")
    return z["refined"]


def run_mask_s2(tile: cfg.Tile, device: str = "cuda", log=print):
    frames = [f for m in tile.months for f in tile.s2_frames(m)]
    n_done = 0
    for f in frames:
        out = s2_mask_path(tile, f)
        if out.exists():
            continue
        data, _, _, names = read_frame(f)
        np.savez_compressed(out, classes=s2_ocm_classes(data, names, device=device))
        n_done += 1
    log(f"  Sentinel-2: {n_done} frames masked ({len(frames) - n_done} already cached)")


# --------------------------------------------------------------------- NICFI
def nicfi_mask_path(tile: cfg.Tile, month: str) -> Path:
    return tile.cache_dir / f"nicfi_quality_{month}.npz"


def load_nicfi_classes(tile: cfg.Tile, month: str) -> np.ndarray:
    """NICFI classes after the post-check."""
    z = np.load(nicfi_mask_path(tile, month))
    if "refined" not in z:
        raise KeyError(f"{month}: no post-checked NICFI mask -- run "
                       f"`preprocess --steps postcheck` first")
    return z["refined"]


def load_nicfi_cleared(tile: cfg.Tile, month: str) -> np.ndarray:
    """Pixels whose flag the post-check cleared (temporal: ground, or a
    removed speckle blob); filled holes end up flagged, so they're not
    counted."""
    z = np.load(nicfi_mask_path(tile, month))
    return (z["overridden"] | z["despeckled"]) & (z["refined"] == cloud_mask.CLEAR)


def load_nicfi_disagreement(tile: cfg.Tile, month: str):
    z = np.load(nicfi_mask_path(tile, month))
    return z["disagreement"] if "disagreement" in z else None


def run_mask_nicfi(tile: cfg.Tile, device: str = "cuda", log=print):
    n_done = 0
    for month in tile.months:
        out = nicfi_mask_path(tile, month)
        if out.exists():
            continue
        nicfi, _, _ = io_utils.read_full(tile.nicfi_path(month))
        q, dis = cloud_mask.compute_masks((nicfi / cfg.REFLECTANCE_SCALE)[None], device=device)
        np.savez_compressed(out, quality=q[0], disagreement=dis[0].astype(np.float16))
        n_done += 1
        log(f"  NICFI {month}: contaminated {np.isin(q[0], CONTAMINATED).mean():.1%}")
    log(f"  NICFI: {n_done} months masked ({len(tile.months) - n_done} already cached)")


def s2_grid(tile: cfg.Tile):
    """(transform, crs) of the S2 frames (all share one grid)."""
    with rasterio.open(tile.s2_frames()[0]) as src:
        return src.transform, src.crs
