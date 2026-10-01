"""
Thin wrapper around the pretrained SEN2SR model (LDSRS2-SEN2SR, already
downloaded at model/LDSRS2-SEN2SR -- confirmed present, not re-downloaded
here). Input: (10,H,W) raw-DN Sentinel-2, band order
[B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12] (matches D02_S2 exactly). Output:
(10,4H,4W) raw-DN-scaled super-resolved Sentinel-2, same band order.

Deliberately doesn't use `sen2sr.predict_large`: reading
`sen2sr/utils.py` directly shows it allocates its output buffer as
`(X.shape[1], X.shape[1])` -- it repeats the *first* spatial dim, which is
wrong for the D02_S2 mosaics' non-square shape (~1002x1045). `predict_tiled`
below is a corrected version of the same 128x128-patch/overlap idea, with a
smooth (raised-cosine) blend across overlaps instead of a hard crop.

The model's own `HardConstraint` module (applied inside `compiled_model`,
see model/LDSRS2-SEN2SR/load.py) already forces each output patch's
low-frequency content back to what the input patch actually measured --
this *is* the low-hallucination guardrail; nothing extra is added here.
"""
import sys
from pathlib import Path

import numpy as np
import torch

MODEL_DIR = Path("/mnt/super/code/model/LDSRS2-SEN2SR")
PATCH = 128
SCALE = 4


def load_model(device: str = "cuda"):
    if str(MODEL_DIR) not in sys.path:
        sys.path.insert(0, str(MODEL_DIR))
    import load as _load  # the model's own load.py, colocated with its weights
    return _load.compiled_model(MODEL_DIR, device=device)


def _tile_starts(padded_len: int, step: int) -> list:
    starts = list(range(0, padded_len - PATCH + 1, step))
    if not starts:
        starts = [0]
    if starts[-1] != padded_len - PATCH:
        starts.append(padded_len - PATCH)
    return starts


def _blend_ramp(overlap_px: int) -> np.ndarray:
    ramp = np.ones(PATCH * SCALE, dtype=np.float32)
    if overlap_px > 0:
        ov = overlap_px * SCALE
        t = np.linspace(0.0, 1.0, ov, endpoint=False, dtype=np.float32)
        edge = 0.5 - 0.5 * np.cos(np.pi * t)
        ramp[:ov] = edge
        ramp[-ov:] = edge[::-1]
    return ramp


def predict_tiled(x: np.ndarray, model, device: str = "cuda", overlap: int = 32) -> np.ndarray:
    """x: (10,H,W) raw DN (not yet /10000, matches the D02_S2 on-disk scale).
    Returns (10,4H,4W) raw-DN-scaled SR output, cropped to exactly 4x the
    input size (no padding artifacts left in)."""
    c, h, w = x.shape
    x_refl = np.nan_to_num(x / 10000.0, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    step = PATCH - overlap
    pad_h = (-(h - PATCH) % step) if h > PATCH else PATCH - h
    pad_w = (-(w - PATCH) % step) if w > PATCH else PATCH - w
    x_pad = np.pad(x_refl, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
    _, hp, wp = x_pad.shape

    out = np.zeros((c, hp * SCALE, wp * SCALE), dtype=np.float32)
    weight_sum = np.zeros((hp * SCALE, wp * SCALE), dtype=np.float32)
    ramp = _blend_ramp(overlap)
    tile_weight = ramp[:, None] * ramp[None, :]

    rows = _tile_starts(hp, step)
    cols = _tile_starts(wp, step)

    with torch.no_grad():
        for r in rows:
            for cx in cols:
                patch = x_pad[:, r:r + PATCH, cx:cx + PATCH]
                t_in = torch.from_numpy(patch).float().to(device)[None]
                sr = model(t_in).squeeze(0).detach().cpu().numpy()  # (10,512,512)
                rr, cc = r * SCALE, cx * SCALE
                out[:, rr:rr + PATCH * SCALE, cc:cc + PATCH * SCALE] += sr * tile_weight
                weight_sum[rr:rr + PATCH * SCALE, cc:cc + PATCH * SCALE] += tile_weight

    out /= np.clip(weight_sum, 1e-6, None)[None]
    out = out[:, :h * SCALE, :w * SCALE]
    return (out * 10000.0).astype(np.float32)
