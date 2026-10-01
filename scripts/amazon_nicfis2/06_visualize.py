"""
Step 6: QA figures (outputs/amazon_nicfis2/<tile>/figures/):
  01_overview.png          per month: % of tile replaced by S2 vs contaminated
                           but kept (no clear S2), and the S2 composite's clear
                           coverage + method (single cloud-free frame / median)
  02_harmonization.png     per-month S2->NICFI slope/intercept per band
  03_clouds_<year>.png     for each year, the cloudiest windows actually
                           reconstructed (one per month): original NICFI |
                           NICFI with its (temporally re-checked) cloud mask |
                           harmonized S2 used | reconstruction | data-quality
                           source layer, shared stretch per row
  04_full_tile.png         the most contaminated months, whole tile: original
                           NICFI | reconstruction | quality source | score
  05_temporal_check.png    the cloud-mask temporal post-check: per-pixel flag
                           frequency in mostly-clear observations (NICFI, S2)
                           and the month with the most flags overridden,
                           raw vs re-checked

Run: python scripts/amazon_nicfis2/06_visualize.py --tile D17
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import cloud_mask, io_utils
from src.nicfis2 import config as cfg, reconstruct as rc, s2_composite

# categorical slots from the dataviz reference palette, fixed order
C1, C2, C3, C4, C5, C7, C8 = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
                              "#4a3aa7", "#e34948")
INK, GRID = "#52514e", "#e6e5e0"
MASK_COLORS = {cloud_mask.CLOUD_THICK: (C8, "thick cloud"), cloud_mask.CLOUD_THIN: (C4, "thin cloud"),
               cloud_mask.SHADOW: (C7, "shadow"), cloud_mask.HAZE: (C3, "haze")}
OVERRIDE_COLOR = (C1, "flag overridden (ground)")
SOURCE_CMAP = ListedColormap(["#d9d8d3", C1, C3, C8, "#0b0b0b"])  # rc.SRC_* order
WIN = 256          # NICFI px, ~1.2 km
PER_YEAR = 4


def tidy(ax):
    ax.grid(axis="y", color=GRID, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def rgb(bands4, lims):
    x = np.moveaxis(bands4[[2, 1, 0]], 0, -1)
    return np.clip((x - lims[:, 0]) / np.maximum(lims[:, 1] - lims[:, 0], 1e-6), 0, 1)


def lims_from(bands4, mask=None):
    x = bands4[[2, 1, 0]].reshape(3, -1)
    if mask is not None and mask.sum() > 100:
        x = x[:, mask.ravel()]
    x = x[:, np.all(x > 0, axis=0)]
    return np.percentile(x, [2, 98], axis=1).T


def _hex(h):
    return np.array([int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)])


def overlay(img, quality, overridden=None, alpha=0.55):
    out = img.copy()
    layers = [(quality == code, hexc) for code, (hexc, _) in MASK_COLORS.items()]
    if overridden is not None:
        layers.append((overridden, OVERRIDE_COLOR[0]))
    for m, hexc in layers:
        out[m] = (1 - alpha) * out[m] + alpha * _hex(hexc)
    return out


def show_source(ax, src):
    ax.imshow(src, cmap=SOURCE_CMAP, vmin=-0.5, vmax=4.5, interpolation="nearest")


def source_legend():
    return [Patch(color=c, label=l) for c, l in zip(SOURCE_CMAP.colors, rc.SOURCE_NAMES)]


def load_month(tile, month):
    nicfi, _, _ = io_utils.read_full(tile.nicfi_path(month))
    recon, _, _ = io_utils.read_full(tile.recon_dir / f"{tile.tile_id}_{month}_recon.tif")
    qual, _, _ = io_utils.read_full(tile.recon_dir / f"{tile.tile_id}_{month}_quality.tif")
    qual = qual.astype(np.uint8)
    overridden = (qual[2] & rc.FLAG_OVERRIDDEN).astype(bool)
    return nicfi, recon, qual, overridden


def fig_overview(tile, stats, out):
    months = sorted(stats)
    x = np.arange(len(months))
    s2 = np.array([stats[m]["s2"] for m in months]) * 100
    kept = np.array([stats[m]["kept_dirty"] for m in months]) * 100
    summ = json.loads((tile.cache_dir / "s2_composite_summary.json").read_text())
    valid = np.array([summ.get(m, {}).get("valid_frac", 0) for m in months]) * 100
    single = np.array([summ.get(m, {}).get("method", "").startswith("single") for m in months])

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    a1.bar(x, s2, color=C1, width=0.8, label="contaminated NICFI, replaced by S2")
    a1.bar(x, kept, bottom=s2, color=C2, width=0.8, edgecolor="white", linewidth=0.5,
           label="contaminated NICFI, kept (no clear S2 that month)")
    a1.set_ylabel("% of tile", color=INK)
    a1.set_title(f"{tile.tile_id}: NICFI contamination (OCM cloud/shadow + haze, buffered) "
                 "and how much was reconstructed", fontsize=12)
    a1.legend(frameon=False, fontsize=9, loc="upper left")
    tidy(a1)
    a2.plot(x, valid, color=INK, lw=1, zorder=1)
    a2.scatter(x[single], valid[single], s=64, color=C1, zorder=2, label="single cloud-free frame")
    a2.scatter(x[~single], valid[~single], s=64, facecolor="white", edgecolor=C1, lw=2, zorder=2,
               label="median of clear observations")
    a2.set_ylabel("S2 composite clear coverage, %", color=INK)
    a2.set_ylim(-3, 103)
    a2.legend(frameon=False, fontsize=9, loc="lower left")
    tidy(a2)
    a2.set_xticks(x[::3]); a2.set_xticklabels(months[::3], rotation=90, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def fig_harmonization(tile, out):
    fits = json.loads((tile.cache_dir / "harmonization.json").read_text())
    months = sorted(fits)
    x = np.arange(len(months))
    src = np.array([fits[m]["source"] for m in months])
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    for k, (ax, label) in enumerate(zip(axes[:2], ("slope", "intercept, DN"))):
        for band, col in zip(cfg.NICFI_BAND_NAMES, (C1, C2, C3, C4)):
            v = np.array([fits[m][band][k] for m in months])
            ax.plot(x, v, color=col, lw=2, marker="o", ms=4, label=band)
        ax.set_ylabel(label, color=INK)
        tidy(ax)
    r = np.array([fits[m].get("r_red", np.nan) for m in months])
    ax = axes[2]
    ax.axhline(rc.MIN_FIT_R, color=INK, lw=1, ls="--")
    ax.annotate(f"trust threshold r={rc.MIN_FIT_R}", (x[-1], rc.MIN_FIT_R), xytext=(0, 4),
                textcoords="offset points", ha="right", fontsize=9, color=INK)
    ax.plot(x, r, color=INK, lw=1, zorder=1)
    for name, style in (("month", dict(color=C1)), ("rejected", dict(facecolor="white", edgecolor=C8, lw=2))):
        sel = src == name
        ax.scatter(x[sel], r[sel], s=56, zorder=2, label={"month": "fitted, used",
                   "rejected": "S2 not trusted, not used"}[name], **style)
    for xi in x[src == "fallback"]:
        ax.axvline(xi, color=C4, lw=3, alpha=0.4)
    ax.set_ylabel("red r, clear-in-both px", color=INK)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    tidy(ax)
    axes[0].legend(frameon=False, ncol=4, fontsize=9)
    axes[0].set_title(f"{tile.tile_id}: per-month S2→NICFI quantile-matching fit "
                      "(NICFI ≈ S2·slope + intercept); yellow bands = year-median fallback "
                      "(too few clear-in-both px)", fontsize=12)
    axes[2].set_xticks(x[::3]); axes[2].set_xticklabels(months[::3], rotation=90, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def best_window(qual):
    """Window with the most real cloud that was replaced, but not entirely
    cloud (some clear context, so seams/consistency are visible)."""
    replaced = ndimage.uniform_filter(np.isin(qual[0], (rc.SRC_S2_SINGLE, rc.SRC_S2_MEDIAN))
                                      .astype(np.float32), WIN)
    thick = ndimage.uniform_filter(np.isin(qual[1], (cloud_mask.CLOUD_THICK, cloud_mask.CLOUD_THIN,
                                                     cloud_mask.SHADOW)).astype(np.float32), WIN)
    h, w = qual.shape[1:]
    ok = np.zeros_like(replaced, dtype=bool)
    ok[WIN // 2:h - WIN // 2, WIN // 2:w - WIN // 2] = True
    score = np.where(ok & (replaced <= 0.8), np.minimum(thick, replaced), -1)
    r, c = np.unravel_index(np.argmax(score), score.shape)
    return float(score[r, c]), r - WIN // 2, c - WIN // 2


def fig_year(tile, year, months, stats, out):
    cands = []
    for m in months:
        if stats[m]["s2"] < 0.002:
            continue
        _, _, qual, _ = load_month(tile, m)
        score, r0, c0 = best_window(qual)
        if score > 0:
            cands.append((score, m, r0, c0))
    cands = sorted(cands, reverse=True)[:PER_YEAR]
    if not cands:
        return False
    cands.sort(key=lambda t: t[1])
    fig, axes = plt.subplots(len(cands), 5, figsize=(22, 4.7 * len(cands)), squeeze=False)
    summ = json.loads((tile.cache_dir / "s2_composite_summary.json").read_text())
    for i, (score, m, r0, c0) in enumerate(cands):
        nicfi, recon, qual, ovr = load_month(tile, m)
        sl = (slice(r0, r0 + WIN), slice(c0, c0 + WIN))
        n, rcn, qq = nicfi[:, sl[0], sl[1]], recon[:, sl[0], sl[1]], qual[:, sl[0], sl[1]]
        replaced = np.isin(qq[0], (rc.SRC_S2_SINGLE, rc.SRC_S2_MEDIAN))
        s2h = np.where(replaced[None], rcn, 0)  # harmonized S2 exactly as used
        lims = lims_from(rcn)
        method = summ[m]["method"].replace("single:", "single frame ")
        panels = [(rgb(n, lims), f"{m}  original NICFI"),
                  (overlay(rgb(n, lims), qq[1], ovr[sl]), "NICFI + cloud mask (re-checked)"),
                  (rgb(s2h, lims), f"S2 used ({method})"),
                  (rgb(rcn, lims), "reconstructed")]
        for j, (img, title) in enumerate(panels):
            axes[i, j].imshow(img, interpolation="nearest")
            axes[i, j].set_title(title, fontsize=10)
        show_source(axes[i, 4], qq[0])
        axes[i, 4].set_title(f"data quality: source (mean score {qq[4].mean():.0f}/100)", fontsize=10)
        for a in axes[i]:
            a.set_xticks([]); a.set_yticks([])
        axes[i, 0].set_ylabel(f"r{r0} c{c0}, ~{WIN * 4.77 / 1000:.1f} km", fontsize=9, color=INK)
    mask_handles = [Patch(color=c, label=l) for c, l in list(MASK_COLORS.values()) + [OVERRIDE_COLOR]]
    fig.legend(handles=mask_handles, loc="lower left", ncol=5, frameon=False, fontsize=10,
               title="cloud mask", bbox_to_anchor=(0.02, 0))
    fig.legend(handles=source_legend(), loc="lower right", ncol=5, frameon=False, fontsize=10,
               title="data quality: source", bbox_to_anchor=(0.98, 0))
    fig.suptitle(f"{tile.tile_id} {year}: cloudiest reconstructed windows (shared stretch per row; "
                 "black in 'S2 used' = not replaced)", fontsize=13)
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    fig.savefig(out, dpi=95)
    plt.close(fig)
    return True


def fig_full_tile(tile, stats, out, n=3):
    worst = sorted(stats, key=lambda m: -(stats[m]["s2"] + stats[m]["kept_dirty"]))[:n]
    fig, axes = plt.subplots(len(worst), 4, figsize=(24, 6 * len(worst)), squeeze=False)
    for i, m in enumerate(sorted(worst)):
        nicfi, recon, qual, _ = load_month(tile, m)
        lims = lims_from(recon, qual[0] == rc.SRC_NICFI)
        axes[i, 0].imshow(rgb(nicfi[:, ::2, ::2], lims)); axes[i, 0].set_title(f"{m} original NICFI")
        axes[i, 1].imshow(rgb(recon[:, ::2, ::2], lims)); axes[i, 1].set_title(f"{m} reconstructed")
        show_source(axes[i, 2], qual[0, ::2, ::2])
        axes[i, 2].set_title(f"source: S2 {stats[m]['s2']:.1%}, kept {stats[m]['kept_dirty']:.1%}")
        im = axes[i, 3].imshow(qual[4, ::2, ::2], cmap="Blues", vmin=0, vmax=100,
                               interpolation="nearest")
        axes[i, 3].set_title(f"score (mean {qual[4].mean():.0f}/100)")
        fig.colorbar(im, ax=axes[i, 3], shrink=0.7)
        for a in axes[i]:
            a.set_xticks([]); a.set_yticks([])
    fig.legend(handles=source_legend(), loc="lower center", ncol=5, frameon=False, fontsize=11)
    fig.suptitle(f"{tile.tile_id}: most contaminated months, whole tile", fontsize=14)
    fig.tight_layout(rect=(0, 0.02, 1, 0.98))
    fig.savefig(out, dpi=80)
    plt.close(fig)


def fig_temporal_check(tile, stats, out):
    zn = np.load(tile.cache_dir / "temporal_check_nicfi.npz")
    zs = np.load(tile.cache_dir / "temporal_check_s2.npz")
    m = max(stats, key=lambda k: stats[k]["overridden"])
    nicfi, _, _, ovr = load_month(tile, m)
    raw = np.load(tile.cache_dir / f"nicfi_quality_{m}.npz")["quality"]
    score = ndimage.uniform_filter(ovr.astype(np.float32), WIN)
    r, c = np.unravel_index(np.argmax(score[WIN // 2:-WIN // 2, WIN // 2:-WIN // 2]),
                            (score.shape[0] - WIN, score.shape[1] - WIN))
    sl = (slice(r, r + WIN), slice(c, c + WIN))
    crop = nicfi[:, sl[0], sl[1]]
    lims = lims_from(crop)
    refined = np.where(ovr[sl], cloud_mask.CLEAR, raw[sl])

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    for ax, z, name in ((axes[0, 0], zn, "NICFI"), (axes[1, 0], zs, "Sentinel-2")):
        im = ax.imshow(z["flag_freq"][::2, ::2], cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
        ax.contour(z["persistent"][::2, ::2], levels=[0.5], colors=C8, linewidths=0.6)
        ax.set_title(f"{name}: flag frequency in {int(z['mostly_clear'].sum())} mostly-clear obs\n"
                     f"red = persistent ({z['persistent'].mean():.2%} of px)", fontsize=11)
        fig.colorbar(im, ax=ax, shrink=0.7)
    axes[0, 1].imshow(overlay(rgb(crop, lims), raw[sl]))
    axes[0, 1].set_title(f"{m}: raw OCM mask ({stats[m]['overridden']:.1%} of tile overridden)")
    axes[0, 2].imshow(overlay(rgb(crop, lims), refined, ovr[sl]))
    axes[0, 2].set_title("after temporal check")
    axes[1, 1].imshow(rgb(crop, lims)); axes[1, 1].set_title(f"{m}: NICFI")
    months = sorted(stats)
    x = np.arange(len(months))
    axes[1, 2].bar(x, [stats[k]["overridden"] * 100 for k in months], color=C1, width=0.8)
    axes[1, 2].set_ylabel("% of tile", color=INK)
    axes[1, 2].set_title("NICFI flags overridden per month")
    axes[1, 2].set_xticks(x[::6]); axes[1, 2].set_xticklabels(months[::6], rotation=90, fontsize=8)
    tidy(axes[1, 2])
    for a in list(axes.ravel()[:5]):
        a.set_xticks([]); a.set_yticks([])
    fig.legend(handles=[Patch(color=c, label=l) for c, l in list(MASK_COLORS.values()) + [OVERRIDE_COLOR]],
               loc="lower center", ncol=5, frameon=False, fontsize=10)
    fig.suptitle(f"{tile.tile_id}: cloud-mask temporal post-check -- a spot flagged in most clear "
                 "observations, looking the same each time, is ground", fontsize=13)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(out, dpi=90)
    plt.close(fig)


def main():
    ap = cfg.add_tile_args(argparse.ArgumentParser())
    args = ap.parse_args()
    tile = cfg.tile_from_args(args)
    stats = json.loads((tile.cache_dir / "reconstruction_stats.json").read_text())
    stats = {m: v for m, v in stats.items() if m in tile.months}

    fig_overview(tile, stats, tile.fig_dir / "01_overview.png")
    print("-> 01_overview.png")
    fig_harmonization(tile, tile.fig_dir / "02_harmonization.png")
    print("-> 02_harmonization.png")
    by_year = defaultdict(list)
    for m in sorted(stats):
        by_year[m[:4]].append(m)
    for year, months in by_year.items():
        if fig_year(tile, year, months, stats, tile.fig_dir / f"03_clouds_{year}.png"):
            print(f"-> 03_clouds_{year}.png")
    fig_full_tile(tile, stats, tile.fig_dir / "04_full_tile.png")
    print("-> 04_full_tile.png")
    fig_temporal_check(tile, stats, tile.fig_dir / "05_temporal_check.png")
    print("-> 05_temporal_check.png")


if __name__ == "__main__":
    main()
