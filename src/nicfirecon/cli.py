"""
Command line for the NICFI reconstruction pipeline. Run from the repo root:

    python -m src.nicfirecon preprocess  --tile D17
    python -m src.nicfirecon reconstruct --tile D17 --method s2fill --add-s2-bands
    python -m src.nicfirecon composite   --tile D17 --type typical-year
    python -m src.nicfirecon run --config configs/D17.yaml      # all of it

Stages (each writes its QA figures unless --no-figures):

preprocess   --steps mask,postcheck,s2composite (default: all)
             mask         OmniCloudMask ensemble per observation (GPU)
             postcheck    temporal (persistent "cloud" that looks the same
                          every time = ground) + spatial (speckle) check
             s2composite  monthly clear Sentinel-2 composite (for s2fill)
             Sentinel-2 steps are skipped if the tile has no S2 frames.
reconstruct  --method mask | s2fill | phenology   (one product per method,
             reconstructed/<method>/, each with a 5-band quality layer)
composite    --type annual       --source nicfi|mask|s2fill|phenology
             --type typical-year (from all years of NICFI observations)
visualize    re-draw figures only: --what preprocess | reconstruct (--method)
             | typical-year | annual (--source)
"""
import argparse
import sys
import time
from pathlib import Path

from . import composite, config as cfg, masking, postcheck, reconstruct, s2, viz


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def do_preprocess(tile, steps=("mask", "postcheck", "s2composite"), sensors=None, device="cuda",
                  temporal=True, min_cloud_area_m2=postcheck.MIN_CLOUD_AREA_M2,
                  s2_clear_thresh=s2.DEFAULT_CLEAR_THRESH, s2_buffer_px=s2.DEFAULT_BUFFER_PX,
                  figures=True):
    sensors = list(sensors or (["nicfi", "s2"] if tile.has_s2 else ["nicfi"]))
    if "s2" in sensors and not tile.has_s2:
        _log(f"no Sentinel-2 frames in {tile.s2_dir} -- skipping S2 steps")
        sensors.remove("s2")
    if "mask" in steps:
        _log(f"preprocess/mask: {', '.join(sensors)}")
        if "nicfi" in sensors:
            masking.run_mask_nicfi(tile, device, _log)
        if "s2" in sensors:
            masking.run_mask_s2(tile, device, _log)
    if "postcheck" in steps:
        _log(f"preprocess/postcheck (temporal={temporal}, min cloud area {min_cloud_area_m2:g} m2)")
        postcheck.run_postcheck(tile, sensors, temporal, min_cloud_area_m2, _log)
    if "s2composite" in steps and "s2" in sensors:
        _log("preprocess/s2composite")
        s2.run_s2_composites(tile, s2_clear_thresh, s2_buffer_px, _log)
    if figures:
        viz.run_visualize(tile, "preprocess", log=_log)


def do_reconstruct(tile, method, add_s2_bands=False, device="cuda", n_clusters=10, refiner=True,
                   refiner_iters=800, figures=True):
    _log(f"reconstruct --method {method}" + (" --add-s2-bands" if add_s2_bands else ""))
    reconstruct.run_reconstruct(tile, method, add_s2_bands, device, n_clusters, refiner,
                                refiner_iters, _log)
    if figures:
        viz.run_visualize(tile, "reconstruct", method=method, log=_log)


def do_composite(tile, type, source="nicfi", stat="lowblue", min_score=50, figures=True):
    if type == "typical-year":
        _log(f"composite --type typical-year (stat {stat})")
        composite.run_typical_year(tile, stat, _log)
        if figures:
            viz.run_visualize(tile, "typical-year", log=_log)
    elif type == "annual":
        _log(f"composite --type annual --source {source} (stat {stat}, min score {min_score})")
        composite.run_annual(tile, source, stat, min_score, _log)
        if figures:
            viz.run_visualize(tile, "annual", source=source, log=_log)
    else:
        raise ValueError(f"unknown composite type {type!r}")


def do_run(config_path: Path):
    """Run a whole chain from a YAML config, e.g.

        tile: D17
        data_root: /mnt/warehouse/amazon        # optional, like the CLI flags
        preprocess: {device: cuda}              # or false to skip
        reconstruct:
          - {method: s2fill, add_s2_bands: true}
          - {method: phenology}
        composite:
          - {type: typical-year}
          - {type: annual, source: s2fill}
    """
    import yaml
    c = yaml.safe_load(Path(config_path).read_text())
    tile_args = argparse.Namespace(
        tile=c["tile"], data_root=Path(c.get("data_root", cfg.DEFAULT_DATA_ROOT)),
        nicfi_dir=Path(c["nicfi_dir"]) if c.get("nicfi_dir") else None,
        s2_dir=Path(c["s2_dir"]) if c.get("s2_dir") else None, months=c.get("months"),
        out_base=Path(c.get("out_base", cfg.PROJECT_DIR / "outputs" / cfg.PIPELINE_NAME)))
    tile = cfg.tile_from_args(tile_args)
    t0 = time.time()
    if c.get("preprocess", {}) is not False:
        do_preprocess(tile, **(c.get("preprocess") or {}))
    for r in c.get("reconstruct", []) or []:
        do_reconstruct(tile, **r)
    for k in c.get("composite", []) or []:
        do_composite(tile, **k)
    _log(f"run finished in {(time.time() - t0) / 60:.1f} min")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m src.nicfirecon", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = cfg.add_tile_args(sub.add_parser("preprocess", help="cloud masks, post-check, S2 composites"))
    p.add_argument("--steps", default="mask,postcheck,s2composite")
    p.add_argument("--sensors", default=None, help="nicfi,s2 (default: both if S2 frames exist)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-temporal", action="store_true", help="skip the temporal post-check")
    p.add_argument("--min-cloud-area", type=float, default=postcheck.MIN_CLOUD_AREA_M2,
                   help="m2; smaller cloud blobs / clear holes are speckle (0 disables)")
    p.add_argument("--s2-clear-thresh", type=float, default=s2.DEFAULT_CLEAR_THRESH)
    p.add_argument("--s2-buffer-px", type=int, default=s2.DEFAULT_BUFFER_PX)
    p.add_argument("--no-figures", action="store_true")

    p = cfg.add_tile_args(sub.add_parser("reconstruct", help="monthly NICFI reconstruction"))
    p.add_argument("--method", required=True, choices=reconstruct.METHODS)
    p.add_argument("--add-s2-bands", action="store_true",
                   help="s2fill: append S2 B5/B6/B7/B8A/B11/B12 (10-band output)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--n-clusters", type=int, default=10, help="phenology: land-cover clusters")
    p.add_argument("--no-refiner", action="store_true", help="phenology: skip the DL refiner")
    p.add_argument("--refiner-iters", type=int, default=800)
    p.add_argument("--no-figures", action="store_true")

    p = cfg.add_tile_args(sub.add_parser("composite", help="annual / typical-year composites"))
    p.add_argument("--type", required=True, choices=("annual", "typical-year"))
    p.add_argument("--source", default="nicfi", choices=composite.ANNUAL_SOURCES,
                   help="annual: monthly input (raw NICFI+masks or a reconstruction method)")
    p.add_argument("--stat", default="lowblue", choices=composite.STATS)
    p.add_argument("--min-score", type=int, default=50, help="annual: min quality score to use a month")
    p.add_argument("--no-figures", action="store_true")

    p = cfg.add_tile_args(sub.add_parser("visualize", help="re-draw QA figures only"))
    p.add_argument("--what", required=True, choices=("preprocess", "reconstruct", "typical-year", "annual"))
    p.add_argument("--method", choices=reconstruct.METHODS)
    p.add_argument("--source", choices=composite.ANNUAL_SOURCES)

    p = sub.add_parser("run", help="run a chain of stages from a YAML config")
    p.add_argument("--config", required=True, type=Path)

    a = ap.parse_args(argv)
    if a.cmd == "run":
        return do_run(a.config)
    tile = cfg.tile_from_args(a)
    _log(f"{tile.tile_id}: {len(tile.months)} NICFI months, "
         f"{len(tile.s2_frames())} S2 frames -> {tile.out_root}")
    t0 = time.time()
    if a.cmd == "preprocess":
        do_preprocess(tile, a.steps.split(","), a.sensors.split(",") if a.sensors else None,
                      a.device, not a.no_temporal, a.min_cloud_area, a.s2_clear_thresh,
                      a.s2_buffer_px, not a.no_figures)
    elif a.cmd == "reconstruct":
        do_reconstruct(tile, a.method, a.add_s2_bands, a.device, a.n_clusters, not a.no_refiner,
                       a.refiner_iters, not a.no_figures)
    elif a.cmd == "composite":
        do_composite(tile, a.type, a.source, a.stat, a.min_score, not a.no_figures)
    elif a.cmd == "visualize":
        viz.run_visualize(tile, a.what, a.method, a.source, _log)
    _log(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    sys.exit(main())
