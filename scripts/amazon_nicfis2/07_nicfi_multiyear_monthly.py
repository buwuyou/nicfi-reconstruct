"""
Step 7 (single-sensor alternative): one typical year of 12 monthly images
from all years of NICFI only -- no Sentinel-2 (src/nicfis2/multiyear.py).
Needs steps 2-3 (NICFI cloud masks + temporal check) and, for the example
areas, step 6 (cache/example_windows.json).

Writes nicfi_multiyear/<tile>_m<MM>.tif (4-band uint16, NICFI scale) and
<tile>_m<MM>_quality.tif (tier, n_clear_same, n_clear_adjacent), plus
figures/multiyear_*.png:
  multiyear_01_tiers.png        per calendar month, % of tile per fallback tier
  multiyear_02_full_tile.png    the 12 composites, whole tile
  multiyear_03_<area>.png       per example area (one per year from step 6):
                                every year's original NICFI for each month,
                                then the multi-year composite and the number
                                of clear years behind each pixel

Run: python scripts/amazon_nicfis2/07_nicfi_multiyear_monthly.py --tile D17
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import io_utils
from src.nicfis2 import config as cfg, multiyear as my

C1, C2, C8, INK, GRID = "#2a78d6", "#eb6834", "#e34948", "#52514e", "#e6e5e0"
MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def load(tile, month):
    data, tr, crs = io_utils.read_full(tile.nicfi_path(month))
    cls = np.load(tile.cache_dir / f"nicfi_quality_{month}.npz")["refined"]
    return data, cls, tr, crs


def rgb(b4, lims):
    x = np.moveaxis(b4[[2, 1, 0]], 0, -1)
    return np.clip((x - lims[:, 0]) / np.maximum(lims[:, 1] - lims[:, 0], 1e-6), 0, 1)


def lims_of(stack4):
    x = np.moveaxis(stack4[:, [2, 1, 0]], 1, 0).reshape(3, -1)
    x = x[:, np.all(x > 0, axis=0)]
    return np.percentile(x, [2, 98], axis=1).T


def tidy(ax):
    ax.grid(axis="y", color=GRID, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main():
    ap = cfg.add_tile_args(argparse.ArgumentParser())
    args = ap.parse_args()
    tile = cfg.tile_from_args(args)
    out_dir = tile._mkdir("nicfi_multiyear")
    t0 = time.time()

    by_cal = defaultdict(list)
    for m in tile.months:
        by_cal[int(m[5:])].append(m)
    years = sorted({m[:4] for m in tile.months})

    stats, comps = {}, {}
    for k in range(1, 13):
        same = by_cal[k]
        adj = by_cal[(k - 2) % 12 + 1] + by_cal[k % 12 + 1]
        S = [load(tile, m) for m in same]
        A = [load(tile, m) for m in adj]
        tr, crs = S[0][2], S[0][3]
        res = my.composite_calendar_month([s[0] for s in S], [s[1] for s in S],
                                          [a[0] for a in A], [a[1] for a in A])
        io_utils.save_geotiff(out_dir / f"{tile.tile_id}_m{k:02d}.tif",
                              res["data"].clip(0, 65535).astype("uint16"), tr, crs,
                              dtype="uint16", nodata=0)
        qpath = out_dir / f"{tile.tile_id}_m{k:02d}_quality.tif"
        io_utils.save_geotiff(qpath, np.stack([res["tier"], res["n_clear_same"], res["n_clear_adjacent"]]),
                              tr, crs, dtype="uint8")
        with rasterio.open(qpath, "r+") as dst:
            dst.descriptions = ("tier", "n_clear_same", "n_clear_adjacent")
        t = res["tier"]
        stats[k] = {name: float((t == code).mean()) for code, name in
                    ((my.TIER_SAME, "same"), (my.TIER_ADJACENT, "adjacent"),
                     (my.TIER_LEAST_BAD, "least_bad"), (my.TIER_NONE, "none"))}
        stats[k]["mean_n_clear_same"] = float(res["n_clear_same"].mean())
        comps[k] = res
        print(f"  {MONTH_ABBR[k-1]} ({len(same)} yrs, {len(adj)} adjacent obs): "
              f"same-month clear {stats[k]['same']:.1%}, adjacent {stats[k]['adjacent']:.2%}, "
              f"least-bad {stats[k]['least_bad']:.2%}, none {stats[k]['none']:.2%}, "
              f"mean clear yrs/px {stats[k]['mean_n_clear_same']:.1f}")
    (tile.cache_dir / "multiyear_stats.json").write_text(json.dumps(stats, indent=1))

    # ---- figure 1: tiers per calendar month
    x = np.arange(12)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    bottom = np.zeros(12)
    for key, col, label in (("adjacent", C1, "clear only in adjacent months (tier 2)"),
                            ("least_bad", C8, "never clear: least-contaminated obs (tier 3)")):
        v = np.array([stats[k + 1][key] for k in x]) * 100
        ax.bar(x, v, bottom=bottom, color=col, width=0.8, label=label, edgecolor="white", linewidth=0.5)
        bottom += v
    ax.set_xticks(x); ax.set_xticklabels(MONTH_ABBR)
    ax.set_ylabel("% of tile", color=INK)
    ax.set_title(f"{tile.tile_id}: multi-year NICFI monthly composite -- pixels not clear in any "
                 f"{years[0]}-{years[-1]} observation of that month (rest = tier 1)", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    tidy(ax)
    fig.tight_layout()
    fig.savefig(tile.fig_dir / "multiyear_01_tiers.png", dpi=120)
    plt.close(fig)

    # ---- figure 2: 12 composites, whole tile
    lims = lims_of(np.stack([comps[k]["data"][:, ::8, ::8] for k in range(1, 13)]))
    fig, axes = plt.subplots(3, 4, figsize=(20, 15.5))
    for k, ax in zip(range(1, 13), axes.ravel()):
        ax.imshow(rgb(comps[k]["data"][:, ::3, ::3], lims))
        ax.set_title(f"{MONTH_ABBR[k-1]}  (tier 1: {stats[k]['same']:.1%})", fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{tile.tile_id}: one typical year from NICFI {years[0]}-{years[-1]} only "
                 "(shared stretch)", fontsize=15)
    fig.tight_layout()
    fig.savefig(tile.fig_dir / "multiyear_02_full_tile.png", dpi=80)
    plt.close(fig)

    # ---- figure 3: example areas from step 6, one per year
    windows = json.loads((tile.cache_dir / "example_windows.json").read_text())
    chosen = {}
    for w in windows:
        chosen.setdefault(w["month"][:4], w)
    for year, w in sorted(chosen.items()):
        sl = (slice(None), slice(w["r0"], w["r0"] + w["win"]), slice(w["c0"], w["c0"] + w["win"]))
        orig = {}
        for m in tile.months:
            a, cls, _, _ = load(tile, m)
            orig[m] = a[sl]
        comp = np.stack([comps[k]["data"][sl] for k in range(1, 13)])
        lims = lims_of(comp)
        n_rows = len(years) + 2
        fig, axes = plt.subplots(n_rows, 12, figsize=(24, 2.1 * n_rows + 1.2),
                                 gridspec_kw=dict(hspace=0.08, wspace=0.04, left=0.04, right=0.99,
                                                  top=0.94, bottom=0.1))
        for r, y in enumerate(years):
            for k in range(1, 13):
                m = f"{y}-{k:02d}"
                ax = axes[r, k - 1]
                if m in orig:
                    ax.imshow(rgb(orig[m], lims), interpolation="nearest")
                if m == w["month"]:
                    for s in ax.spines.values():
                        s.set_edgecolor(C2); s.set_linewidth(3)
        for k in range(1, 13):
            axes[-2, k - 1].imshow(rgb(comp[k - 1], lims), interpolation="nearest")
            nc = axes[-1, k - 1].imshow(comps[k]["n_clear_same"][sl[1:]], cmap="Blues", vmin=0,
                                        vmax=len(years), interpolation="nearest")
            if (comps[k]["tier"][sl[1:]] != my.TIER_SAME).any():
                axes[-1, k - 1].contour(comps[k]["tier"][sl[1:]] != my.TIER_SAME, levels=[0.5],
                                        colors=C8, linewidths=0.8)
            axes[0, k - 1].set_title(MONTH_ABBR[k - 1], fontsize=11)
        for r, label in enumerate(years + ["multi-year\ncomposite", "clear years\n(of 5)"]):
            axes[r, 0].set_ylabel(label, fontsize=10, color=INK)
        for ax in axes.ravel():
            ax.set_xticks([]); ax.set_yticks([])
        cbar = fig.colorbar(nc, ax=axes[-1, :].tolist(), orientation="horizontal", fraction=0.04,
                            pad=0.25, aspect=60)
        cbar.set_label(f"clear same-month observations behind each composite pixel (of "
                       f"{len(years)}); red outline = none, fell back to adjacent months / "
                       "least-contaminated", fontsize=9)
        fig.suptitle(f"{tile.tile_id} area r{w['r0']} c{w['c0']} (~{w['win'] * 4.77 / 1000:.1f} km; "
                     f"orange frame = month shown in 03_clouds_{year}.png): every NICFI month vs. "
                     "the NICFI-only multi-year composite", fontsize=13)
        name = f"multiyear_03_{year}_r{w['r0']}_c{w['c0']}.png"
        fig.savefig(tile.fig_dir / name, dpi=80)
        plt.close(fig)
        print(f"-> {name}")
    print(f"DONE in {time.time()-t0:.0f}s -> {out_dir}")


if __name__ == "__main__":
    main()
