"""
Step 5: QA figures (outputs/amazon_nicfis2/<tile>/figures/):
  01_overview.png          per month: % of tile replaced by S2 vs contaminated
                           but kept (no clear S2), and the S2 composite's clear
                           coverage + method (single cloud-free frame / median)
  02_harmonization.png     per-month S2->NICFI slope/intercept per band
  03_clouds_<year>.png     for each year, the cloudiest windows actually
                           reconstructed (one per month): original NICFI |
                           NICFI with its cloud mask | harmonized S2 used |
                           reconstruction, shared stretch per row
  04_full_tile.png         the most contaminated months, whole tile, original
                           NICFI vs reconstruction vs provenance

Run: python scripts/amazon_nicfis2/05_visualize.py --tile D17
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
C1, C2, C3, C4, C7, C8 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7", "#e34948"
INK, GRID = "#52514e", "#e6e5e0"
MASK_COLORS = {cloud_mask.CLOUD_THICK: (C8, "thick cloud"), cloud_mask.CLOUD_THIN: (C4, "thin cloud"),
               cloud_mask.SHADOW: (C7, "shadow"), cloud_mask.HAZE: (C3, "haze")}
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


def overlay(img, quality, alpha=0.55):
    out = img.copy()
    for code, (hexc, _) in MASK_COLORS.items():
        m = quality == code
        col = np.array([int(hexc[i:i + 2], 16) / 255 for i in (1, 3, 5)])
        out[m] = (1 - alpha) * out[m] + alpha * col
    return out


def load_month(tile, month):
    nicfi, _, _ = io_utils.read_full(tile.nicfi_path(month))
    recon, _, _ = io_utils.read_full(tile.recon_dir / f"{tile.tile_id}_{month}_recon.tif")
    prov, _, _ = io_utils.read_full(tile.recon_dir / f"{tile.tile_id}_{month}_provenance.tif")
    q = np.load(tile.cache_dir / f"nicfi_quality_{month}.npz")["quality"]
    return nicfi, recon, prov[0].astype(np.uint8), q


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


def best_window(prov, q):
    """Window with the most real cloud that was replaced, but not entirely
    cloud (some clear context, so seams/consistency are visible)."""
    replaced = ndimage.uniform_filter((prov == rc.PROV_S2).astype(np.float32), WIN)
    thick = ndimage.uniform_filter(np.isin(q, (cloud_mask.CLOUD_THICK, cloud_mask.CLOUD_THIN,
                                               cloud_mask.SHADOW)).astype(np.float32), WIN)
    h, w = prov.shape
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
        nicfi, recon, prov, q = load_month(tile, m)
        score, r0, c0 = best_window(prov, q)
        if score > 0:
            cands.append((score, m, r0, c0))
    cands = sorted(cands, reverse=True)[:PER_YEAR]
    if not cands:
        return False
    cands.sort(key=lambda t: t[1])
    fig, axes = plt.subplots(len(cands), 4, figsize=(18, 4.7 * len(cands)), squeeze=False)
    summ = json.loads((tile.cache_dir / "s2_composite_summary.json").read_text())
    for i, (score, m, r0, c0) in enumerate(cands):
        nicfi, recon, prov, q = load_month(tile, m)
        sl = (slice(None), slice(r0, r0 + WIN), slice(c0, c0 + WIN))
        n, rcn, p, qq = nicfi[sl], recon[sl], prov[sl[1:]], q[sl[1:]]
        s2h = np.where(p[None] == rc.PROV_S2, rcn, 0)  # harmonized S2 exactly as used
        lims = lims_from(rcn)
        method = summ[m]["method"].replace("single:", "single frame ")
        panels = [(rgb(n, lims), f"{m}  original NICFI"),
                  (overlay(rgb(n, lims), qq), "NICFI + cloud mask"),
                  (rgb(s2h, lims), f"S2 used ({method})"),
                  (rgb(rcn, lims), "reconstructed")]
        for j, (img, title) in enumerate(panels):
            ax = axes[i, j]
            ax.imshow(img, interpolation="nearest")
            if j == 3:
                ax.contour(p == rc.PROV_S2, levels=[0.5], colors="white", linewidths=0.6)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(title, fontsize=10)
        axes[i, 0].set_ylabel(f"r{r0} c{c0}, ~{WIN * 4.77 / 1000:.1f} km", fontsize=9, color=INK)
    handles = [Patch(color=c, label=l) for c, l in MASK_COLORS.values()]
    handles.append(Patch(facecolor="none", edgecolor=INK, label="white line: replaced area"))
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=10)
    fig.suptitle(f"{tile.tile_id} {year}: cloudiest reconstructed windows (shared stretch per row; "
                 "black in 'S2 used' = kept NICFI)", fontsize=13)
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    fig.savefig(out, dpi=100)
    plt.close(fig)
    return True


def fig_full_tile(tile, stats, out, n=3):
    worst = sorted(stats, key=lambda m: -(stats[m]["s2"] + stats[m]["kept_dirty"]))[:n]
    cmap = ListedColormap(["#d9d8d3", C1, C2, "#0b0b0b"])
    fig, axes = plt.subplots(len(worst), 3, figsize=(18, 6 * len(worst)), squeeze=False)
    for i, m in enumerate(sorted(worst)):
        nicfi, recon, prov, q = load_month(tile, m)
        lims = lims_from(recon, prov == rc.PROV_NATIVE)
        axes[i, 0].imshow(rgb(nicfi[:, ::2, ::2], lims)); axes[i, 0].set_title(f"{m} original NICFI")
        axes[i, 1].imshow(rgb(recon[:, ::2, ::2], lims)); axes[i, 1].set_title(f"{m} reconstructed")
        axes[i, 2].imshow(prov[::2, ::2], cmap=cmap, vmin=-0.5, vmax=3.5, interpolation="nearest")
        axes[i, 2].set_title(f"provenance: S2 {stats[m]['s2']:.1%}, kept {stats[m]['kept_dirty']:.1%}")
        for a in axes[i]:
            a.set_xticks([]); a.set_yticks([])
    fig.legend(handles=[Patch(color=c, label=l) for c, l in zip(
        cmap.colors, ["NICFI clear", "replaced by S2", "contaminated, kept", "nodata"])],
        loc="lower center", ncol=4, frameon=False, fontsize=11)
    fig.suptitle(f"{tile.tile_id}: most contaminated months, whole tile", fontsize=14)
    fig.tight_layout(rect=(0, 0.02, 1, 0.98))
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


if __name__ == "__main__":
    main()
