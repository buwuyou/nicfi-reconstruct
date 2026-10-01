"""
Step 4: one clear Sentinel-2 composite per month (src/nicfis2/s2_composite.py):
the month's single cloud-free frame if one exists (clear over >=
--clear-thresh of the tile), else the per-pixel median of clear
observations. Writes cache/s2_composite_<month>.npz and a per-month summary
(cache/s2_composite_summary.json).

Run: python scripts/amazon_nicfis2/04_s2_composite.py --tile D17
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.nicfis2 import config as cfg, s2_composite


def main():
    ap = cfg.add_tile_args(argparse.ArgumentParser())
    ap.add_argument("--clear-thresh", type=float, default=s2_composite.DEFAULT_CLEAR_THRESH)
    ap.add_argument("--buffer-px", type=int, default=s2_composite.DEFAULT_BUFFER_PX)
    args = ap.parse_args()
    tile = cfg.tile_from_args(args)

    summary_path = tile.cache_dir / "s2_composite_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    t0 = time.time()
    for month in tile.months:
        c = s2_composite.monthly_composite(tile, month, args.clear_thresh, args.buffer_px)
        if c is None:
            summary[month] = {"method": "none", "n_frames": 0, "valid_frac": 0.0}
            print(f"  {month}: no Sentinel-2 frames")
            continue
        s2_composite.save(tile, month, c)
        summary[month] = {"method": c.method, "n_frames": len(c.frame_clear),
                          "valid_frac": float(c.valid.mean()), "frame_clear": c.frame_clear}
        best = max(c.frame_clear.values())
        print(f"  {month}: {c.method:18s} frames={len(c.frame_clear)} "
              f"best-frame clear={best:.1%} composite valid={c.valid.mean():.1%}")
    summary_path.write_text(json.dumps(summary, indent=1, sort_keys=True))
    print(f"DONE in {time.time()-t0:.0f}s -> {summary_path}")


if __name__ == "__main__":
    main()
