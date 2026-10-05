"""
QA figures, in three sections under outputs/nicfirecon/<tile>/figures/. Each
section compares every method/source that exists on disk, on the same
example windows -- per year, the months whose ~1.2 km window holds the most
cloud/shadow with some clear context left, picked once from the
post-checked NICFI masks (cache/example_windows.json).

1_cloudmask/
    mask_stats.png           per month: % of tile per cloud class after the
                             post-check, vs. the raw OCM total
    postcheck_frequency.png  how often each pixel is flagged in mostly-clear
                             obs (NICFI, S2) + NICFI flags cleared per month
    mask_effect_<year>.png   example windows: NICFI | raw OCM mask |
                             post-checked mask (cleared flags in blue)
    s2_coverage.png          S2 composite clear coverage + method per month
2_reconstruct/
    sources_by_month.png     per method: % of tile by data source per month
    harmonization.png        (s2fill) per-month S2->NICFI fit + trust
    compare_<year>.png       example windows: original NICFI (+ mask) and
                             each method's result, its quality source below
    compare_full_tile.png    the most contaminated months, whole tile
3_composite/
    typical_year.png         12 monthly composites, one row per source
    annual.png               annual composites, one row per source
    coverage.png             per source, % of tile from usable observations
    area_<year>_r<row>_c<col>.png  one example area: typical year + annual,
                             one row per source
"""
import json
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy import ndimage  # noqa: E402

from .. import cloud_mask, io_utils  # noqa: E402
from . import config as cfg, harmonize as hz, masking, reconstruct as rc  # noqa: E402
from . import composite as cp  # noqa: E402

# categorical slots from the dataviz reference palette, fixed order
C1, C2, C3, C4, C5, C7, C8 = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
                              "#4a3aa7", "#e34948")
INK, GRID, LIGHT = "#52514e", "#e6e5e0", "#d9d8d3"
MASK_COLORS = {cloud_mask.CLOUD_THICK: (C8, "thick cloud"), cloud_mask.CLOUD_THIN: (C4, "thin cloud"),
               cloud_mask.SHADOW: (C7, "shadow"), cloud_mask.HAZE: (C3, "haze")}
CLEARED_COLOR = (C1, "flag cleared by post-check")
SOURCE_COLORS = [LIGHT, C1, C3, C8, "#0b0b0b", C7, C5]  # rc.SRC_* order
SOURCE_CMAP = ListedColormap(SOURCE_COLORS)
MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
WIN = 256          # NICFI px, ~1.2 km
PER_YEAR = 4
CLOUDY = (cloud_mask.CLOUD_THICK, cloud_mask.CLOUD_THIN, cloud_mask.SHADOW)


# ------------------------------------------------------------------ helpers
def tidy(ax):
    ax.grid(axis="y", color=GRID, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def month_axis(ax, labels, step=3):
    x = np.arange(len(labels))
    ax.set_xticks(x[::step])
    ax.set_xticklabels(labels[::step], rotation=90, fontsize=8)


def rgb(bands4, lims):
    x = np.moveaxis(bands4[[2, 1, 0]], 0, -1)
    return np.clip((x - lims[:, 0]) / np.maximum(lims[:, 1] - lims[:, 0], 1e-6), 0, 1)


def lims_from(bands4, mask=None):
    x = bands4[[2, 1, 0]].reshape(3, -1)
    if mask is not None and mask.sum() > 100:
        x = x[:, mask.ravel()]
    x = x[:, np.all(x > 0, axis=0)]
    if x.shape[1] == 0:
        return np.array([[0, 1]] * 3, float)
    return np.percentile(x, [2, 98], axis=1).T


def stretch3(arr3):
    """(3,H,W) shown as RGB in the given band order, 2-98% stretch per band."""
    x = arr3.reshape(3, -1)
    x = x[:, np.all(x > 0, axis=0)]
    lims = np.percentile(x, [2, 98], axis=1).T if x.shape[1] else np.array([[0, 1]] * 3, float)
    img = np.moveaxis(arr3, 0, -1)
    return np.clip((img - lims[:, 0]) / np.maximum(lims[:, 1] - lims[:, 0], 1e-6), 0, 1)


def lims_stack(stack4):
    x = np.moveaxis(stack4[:, [2, 1, 0]], 1, 0).reshape(3, -1)
    x = x[:, np.all(x > 0, axis=0)]
    return np.percentile(x, [2, 98], axis=1).T


def _hex(h):
    return np.array([int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)])


def overlay(img, quality, cleared=None, alpha=0.55):
    out = img.copy()
    layers = [(quality == code, hexc) for code, (hexc, _) in MASK_COLORS.items()]
    if cleared is not None:
        layers.append((cleared, CLEARED_COLOR[0]))
    for m, hexc in layers:
        out[m] = (1 - alpha) * out[m] + alpha * _hex(hexc)
    return out


def show_source(ax, src):
    ax.imshow(src, cmap=SOURCE_CMAP, vmin=-0.5, vmax=len(SOURCE_COLORS) - 0.5,
              interpolation="nearest")


def source_legend(codes):
    return [Patch(color=SOURCE_COLORS[c], label=rc.SOURCE_NAMES[c]) for c in codes]


def mask_legend():
    return [Patch(color=c, label=l) for c, l in list(MASK_COLORS.values()) + [CLEARED_COLOR]]


def _noticks(axes):
    for a in np.atleast_1d(axes).ravel():
        a.set_xticks([]); a.set_yticks([])


def _save(fig, path, dpi=100):
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


# ---------------------------------------------------------- example windows
def example_windows(tile: cfg.Tile, refresh=False):
    path = tile.cache_dir / "example_windows.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    cands = defaultdict(list)
    for m in tile.all_months:
        q = masking.load_nicfi_classes(tile, m)
        frac = ndimage.uniform_filter(np.isin(q, CLOUDY).astype(np.float32), WIN)
        h, w = q.shape
        ok = np.zeros_like(frac, bool)
        ok[WIN // 2:h - WIN // 2, WIN // 2:w - WIN // 2] = True
        score = np.where(ok & (frac <= 0.8), frac, -1)  # some clear context left
        r, c = np.unravel_index(np.argmax(score), score.shape)
        if score[r, c] > 0.05:
            cands[m[:4]].append((float(score[r, c]), m, int(r - WIN // 2), int(c - WIN // 2)))
    windows = []
    for year in sorted(cands):
        for s, m, r0, c0 in sorted(sorted(cands[year], reverse=True)[:PER_YEAR], key=lambda t: t[1]):
            windows.append({"month": m, "r0": r0, "c0": c0, "win": WIN, "cloud_frac": round(s, 3)})
    path.write_text(json.dumps(windows, indent=1))
    return windows


def _crop(a, w):
    return a[..., w["r0"]:w["r0"] + w["win"], w["c0"]:w["c0"] + w["win"]]



# --------------------------------------------------------------- shared figs
def fig_s2_coverage(tile, out_dir):
    path = tile.cache_dir / "s2_composite_summary.json"
    if not path.exists():
        return None
    summ = json.loads(path.read_text())
    months = [m for m in tile.all_months if m in summ]
    x = np.arange(len(months))
    valid = np.array([summ[m].get("valid_frac", 0) for m in months]) * 100
    single = np.array([summ[m].get("method", "").startswith("single") for m in months])
    fig, ax = plt.subplots(figsize=(16, 4.5))
    ax.plot(x, valid, color=INK, lw=1, zorder=1)
    ax.scatter(x[single], valid[single], s=64, color=C1, zorder=2, label="single cloud-free frame")
    ax.scatter(x[~single], valid[~single], s=64, facecolor="white", edgecolor=C1, lw=2, zorder=2,
               label="median of clear observations")
    ax.set_ylabel("S2 composite clear coverage, %", color=INK)
    ax.set_ylim(-3, 103)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax.set_title(f"{tile.tile_id}: monthly Sentinel-2 composite -- clear coverage and method", fontsize=12)
    month_axis(ax, months)
    tidy(ax)
    fig.tight_layout()
    return _save(fig, out_dir / "s2_coverage.png", 120)


def _recon_stats(tile, method):
    p = tile.recon_dir(method) / "stats.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} -- run `reconstruct --method {method}` first")
    return json.loads(p.read_text())


def _load_recon(tile, method, month):
    d = tile.recon_dir(method)
    data, _, _ = io_utils.read_full(d / f"{tile.tile_id}_{month}.tif")
    q, _, _ = io_utils.read_full(d / f"{tile.tile_id}_{month}_quality.tif")
    return data, q.astype(np.uint8)


def fig_harmonization(tile, out_dir):
    fits = json.loads((tile.cache_dir / "harmonization.json").read_text())
    months = sorted(fits)
    x = np.arange(len(months))
    src = np.array([fits[m]["source"] for m in months])
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    for k, (ax, label) in enumerate(zip(axes[:2], ("slope", "intercept, DN"))):
        for band, col in zip(cfg.NICFI_BAND_NAMES, (C1, C2, C3, C4)):
            ax.plot(x, [fits[m][band][k] for m in months], color=col, lw=2, marker="o", ms=4, label=band)
        ax.set_ylabel(label, color=INK)
        tidy(ax)
    r = np.array([fits[m].get("r_red", np.nan) for m in months])
    ax = axes[2]
    ax.axhline(hz.MIN_FIT_R, color=INK, lw=1, ls="--")
    ax.annotate(f"trust threshold r={hz.MIN_FIT_R}", (x[-1], hz.MIN_FIT_R), xytext=(0, 4),
                textcoords="offset points", ha="right", fontsize=9, color=INK)
    ax.plot(x, r, color=INK, lw=1, zorder=1)
    for name, style, label in (("month", dict(color=C1), "fitted, used"),
                               ("rejected", dict(facecolor="white", edgecolor=C8, lw=2),
                                "S2 not trusted, not used")):
        sel = src == name
        ax.scatter(x[sel], r[sel], s=56, zorder=2, label=label, **style)
    for xi in x[src == "fallback"]:
        ax.axvline(xi, color=C4, lw=3, alpha=0.4)
    ax.set_ylabel("red r, clear-in-both px", color=INK)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    tidy(ax)
    axes[0].legend(frameon=False, ncol=4, fontsize=9)
    axes[0].set_title(f"{tile.tile_id}: per-month S2→NICFI quantile-matching fit "
                      "(NICFI ≈ S2·slope + intercept); yellow bands = year-median fallback", fontsize=12)
    month_axis(axes[2], months)
    fig.tight_layout()
    return _save(fig, out_dir / "harmonization.png", 120)



SOURCE_LABEL = {"nicfi": "NICFI (raw, masked)", "mask": "mask", "s2fill": "s2fill",
                "phenology": "phenology"}


def _methods(tile):
    return [m for m in rc.METHODS if (tile.recon_dir(m) / "stats.json").exists()]


def _sources(tile, kind):
    return [s for s in cp.SOURCES if (tile.composite_dir(kind, s) / "stats.json").exists()]


# ------------------------------------------------------------- 1 cloud mask
def _mask_month_stats(tile):
    rows = {}
    for m in tile.all_months:
        z = np.load(masking.nicfi_mask_path(tile, m))
        raw, ref = z["quality"], z["refined"]
        rows[m] = {"raw": float(np.isin(raw, masking.CONTAMINATED).mean()),
                   **{code: float((ref == code).mean()) for code in MASK_COLORS},
                   "cleared": float(masking.load_nicfi_cleared(tile, m).mean())}
    return rows


def fig_mask_stats(tile, st, out_dir):
    months = list(st)
    x = np.arange(len(months))
    fig, ax = plt.subplots(figsize=(16, 4.8))
    bottom = np.zeros(len(months))
    for code, (col, label) in MASK_COLORS.items():
        v = np.array([st[m][code] for m in months]) * 100
        ax.bar(x, v, bottom=bottom, color=col, width=0.8, label=label + " (post-checked)",
               edgecolor="white", linewidth=0.5)
        bottom += v
    ax.plot(x, [st[m]["raw"] * 100 for m in months], color=INK, lw=1.5, marker="o", ms=4,
            label="raw OCM total (before post-check)")
    ax.set_ylabel("% of tile", color=INK)
    ax.set_title(f"{tile.tile_id}: NICFI cloud mask per month -- raw OmniCloudMask vs. after the "
                 "temporal + speckle post-check", fontsize=12)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    month_axis(ax, months)
    tidy(ax)
    fig.tight_layout()
    return _save(fig, out_dir / "mask_stats.png", 120)


def fig_postcheck_frequency(tile, st, out_dir):
    months = list(st)
    fig, axes = plt.subplots(1, 3, figsize=(21, 6.2), gridspec_kw=dict(width_ratios=[1, 1, 1.3]))
    for ax, name in ((axes[0], "nicfi"), (axes[1], "s2")):
        zp = tile.cache_dir / f"postcheck_{name}.npz"
        if not zp.exists():
            ax.set_visible(False)
            continue
        z = np.load(zp)
        im = ax.imshow(z["flag_freq"][::2, ::2], cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
        ax.contour(z["persistent"][::2, ::2], levels=[0.5], colors=C8, linewidths=0.6)
        ax.set_title(f"{'NICFI' if name == 'nicfi' else 'Sentinel-2'}: flag frequency in "
                     f"{int(z['mostly_clear'].sum())} mostly-clear obs\nred = persistent "
                     f"({z['persistent'].mean():.2%} of px)", fontsize=11)
        fig.colorbar(im, ax=ax, shrink=0.75)
        _noticks(ax)
    x = np.arange(len(months))
    axes[2].bar(x, [st[m]["cleared"] * 100 for m in months], color=C1, width=0.8)
    axes[2].set_ylabel("% of tile", color=INK)
    axes[2].set_title("NICFI flags cleared by the post-check, per month", fontsize=11)
    month_axis(axes[2], months, 6)
    tidy(axes[2])
    fig.suptitle(f"{tile.tile_id}: temporal post-check -- a spot flagged in most clear observations, "
                 "looking the same each time, is ground", fontsize=13)
    fig.tight_layout()
    return _save(fig, out_dir / "postcheck_frequency.png", 100)


def fig_mask_effect(tile, out_dir):
    by_year = defaultdict(list)
    for w in example_windows(tile):
        by_year[w["month"][:4]].append(w)
    paths = []
    for year, ws in sorted(by_year.items()):
        fig, axes = plt.subplots(len(ws), 3, figsize=(13.5, 4.5 * len(ws)), squeeze=False)
        for i, w in enumerate(ws):
            m = w["month"]
            n = _crop(io_utils.read_full(tile.nicfi_path(m))[0], w)
            z = np.load(masking.nicfi_mask_path(tile, m))
            raw, ref = _crop(z["quality"], w), _crop(z["refined"], w)
            cleared = _crop(masking.load_nicfi_cleared(tile, m), w)
            lims = lims_from(n)
            for j, (img, title) in enumerate((
                    (rgb(n, lims), f"{m}  NICFI"),
                    (overlay(rgb(n, lims), raw), f"raw OCM ({np.isin(raw, masking.CONTAMINATED).mean():.0%} flagged)"),
                    (overlay(rgb(n, lims), ref, cleared),
                     f"post-checked ({np.isin(ref, masking.CONTAMINATED).mean():.0%} flagged)"))):
                axes[i, j].imshow(img, interpolation="nearest")
                axes[i, j].set_title(title, fontsize=10)
            axes[i, 0].set_ylabel(f"r{w['r0']} c{w['c0']}, ~{WIN * 4.77 / 1000:.1f} km", fontsize=9, color=INK)
        _noticks(axes)
        fig.legend(handles=mask_legend(), loc="lower center", ncol=5, frameon=False, fontsize=10)
        fig.suptitle(f"{tile.tile_id} {year}: NICFI cloud mask before and after the post-check",
                     fontsize=13)
        fig.tight_layout(rect=(0, 0.04, 1, 0.98))
        paths.append(_save(fig, out_dir / f"mask_effect_{year}.png", 85))
    return paths


# ------------------------------------------------------------ 2 reconstruct
def fig_sources_by_month(tile, methods, out_dir):
    fig, axes = plt.subplots(len(methods), 1, figsize=(16, 3.4 * len(methods) + 0.6), sharex=True,
                             squeeze=False)
    months = None
    for ax, method in zip(axes[:, 0], methods):
        st = _recon_stats(tile, method)
        months = [m for m in tile.all_months if m in st["months"]]
        x = np.arange(len(months))
        bottom = np.zeros(len(months))
        for code in (rc.SRC_S2_SINGLE, rc.SRC_S2_MEDIAN, rc.SRC_PHENO, rc.SRC_PHENO_BLEND,
                     rc.SRC_CONTAMINATED):
            name = rc.SOURCE_NAMES[code]
            v = np.array([st["months"][m]["source"][name] for m in months]) * 100
            if not v.any():
                continue
            label = name + {rc.SRC_CONTAMINATED: " (masked)" if method == "mask" else " (kept)"}.get(code, "")
            ax.bar(x, v, bottom=bottom, color=SOURCE_COLORS[code], width=0.8, edgecolor="white",
                   linewidth=0.5, label=label)
            bottom += v
        ax.set_ylabel("% of tile", color=INK)
        ax.set_title(f"{method}", fontsize=12, loc="left", color=INK)
        ax.legend(frameon=False, fontsize=9, loc="upper right")
        tidy(ax)
    month_axis(axes[-1, 0], months)
    fig.suptitle(f"{tile.tile_id}: pixels not taken from clear NICFI, by data source "
                 "(rest = NICFI clear / nodata border)", fontsize=13)
    fig.tight_layout()
    return _save(fig, out_dir / "sources_by_month.png", 110)


def fig_compare(tile, methods, out_dir):
    by_year = defaultdict(list)
    for w in example_windows(tile):
        by_year[w["month"][:4]].append(w)
    paths = []
    ncol = 1 + len(methods)
    for year, ws in sorted(by_year.items()):
        fig, axes = plt.subplots(2 * len(ws), ncol, figsize=(3.9 * ncol, 7.4 * len(ws)), squeeze=False,
                                 gridspec_kw=dict(height_ratios=[1, 0.55] * len(ws)))
        codes = set()
        for i, w in enumerate(ws):
            m = w["month"]
            n = _crop(io_utils.read_full(tile.nicfi_path(m))[0], w)
            ref = _crop(masking.load_nicfi_classes(tile, m), w)
            lims = lims_from(n, ~np.isin(ref, masking.CONTAMINATED + (cloud_mask.NODATA,)))
            top, bot = axes[2 * i], axes[2 * i + 1]
            top[0].imshow(rgb(n, lims), interpolation="nearest")
            top[0].set_title(f"{m}  original NICFI", fontsize=10)
            bot[0].imshow(overlay(rgb(n, lims), ref), interpolation="nearest")
            bot[0].set_title("cloud mask (post-checked)", fontsize=9)
            for j, method in enumerate(methods, start=1):
                data, q = _load_recon(tile, method, m)
                d, qq = _crop(data, w), _crop(q, w)
                top[j].imshow(rgb(d[:4], lims), interpolation="nearest")
                top[j].set_title(method, fontsize=11, weight="bold")
                show_source(bot[j], qq[0])
                bot[j].set_title(f"source (mean score {qq[4].mean():.0f})", fontsize=9)
                codes |= set(np.unique(qq[0]).tolist())
            top[0].set_ylabel(f"r{w['r0']} c{w['c0']}", fontsize=9, color=INK)
        _noticks(axes)
        fig.legend(handles=mask_legend(), loc="lower left", ncol=3, frameon=False, fontsize=9,
                   title="cloud mask", bbox_to_anchor=(0.01, 0))
        fig.legend(handles=source_legend(sorted(codes)), loc="lower right", ncol=4, frameon=False,
                   fontsize=9, title="data quality: source", bbox_to_anchor=(0.99, 0))
        fig.suptitle(f"{tile.tile_id} {year}: monthly reconstruction, methods side by side "
                     "(shared stretch per row; black = nodata)", fontsize=13)
        fig.tight_layout(rect=(0, 0.035, 1, 0.985))
        paths.append(_save(fig, out_dir / f"compare_{year}.png", 80))
    return paths


def fig_compare_full_tile(tile, methods, mask_st, out_dir, n=2):
    worst = sorted(sorted(mask_st, key=lambda m: -mask_st[m]["raw"])[:n])
    ncol = 1 + len(methods)
    fig, axes = plt.subplots(len(worst), ncol, figsize=(5.2 * ncol, 5.6 * len(worst)), squeeze=False)
    for i, m in enumerate(worst):
        nicfi = io_utils.read_full(tile.nicfi_path(m))[0]
        ref = masking.load_nicfi_classes(tile, m)
        lims = lims_from(nicfi, ~np.isin(ref, masking.CONTAMINATED + (cloud_mask.NODATA,)))
        axes[i, 0].imshow(rgb(nicfi[:, ::3, ::3], lims))
        axes[i, 0].set_title(f"{m}  original NICFI ({mask_st[m]['raw']:.0%} flagged raw)", fontsize=11)
        for j, method in enumerate(methods, start=1):
            data, _ = _load_recon(tile, method, m)
            axes[i, j].imshow(rgb(data[:4, ::3, ::3], lims))
            axes[i, j].set_title(method, fontsize=12, weight="bold")
    _noticks(axes)
    fig.suptitle(f"{tile.tile_id}: most contaminated months, whole tile, methods side by side",
                 fontsize=14)
    fig.tight_layout()
    return _save(fig, out_dir / "compare_full_tile.png", 70)


# -------------------------------------------------------------- 3 composite
def _composite_paths(tile, kind, source):
    d = tile.composite_dir(kind, source)
    keys = [f"m{k:02d}" for k in range(1, 13)] if kind == "typical_year" else tile.years
    return {k: d / f"{tile.tile_id}_{k}.tif" for k in keys if (d / f"{tile.tile_id}_{k}.tif").exists()}


def _grid_figure(tile, kind, sources, title, path, step=6):
    imgs = {s: {k: io_utils.read_full(p)[0][:, ::step, ::step]
                for k, p in _composite_paths(tile, kind, s).items()} for s in sources}
    keys = sorted({k for v in imgs.values() for k in v})
    lims = lims_stack(np.stack([a for v in imgs.values() for a in v.values()]))
    fig, axes = plt.subplots(len(sources), len(keys), figsize=(2.3 * len(keys), 2.45 * len(sources) + 0.6),
                             squeeze=False)
    for i, s in enumerate(sources):
        for j, k in enumerate(keys):
            if k in imgs[s]:
                axes[i, j].imshow(rgb(imgs[s][k], lims))
            if i == 0:
                axes[i, j].set_title(MONTH_ABBR[int(k[1:]) - 1] if kind == "typical_year" else k, fontsize=11)
        axes[i, 0].set_ylabel(SOURCE_LABEL[s], fontsize=11, color=INK)
    _noticks(axes)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return _save(fig, path, 80)


def fig_coverage(tile, ty_sources, an_sources, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(16, 4.4), gridspec_kw=dict(width_ratios=[2.2, 1]))
    cols = dict(zip(cp.SOURCES, (INK, C8, C1, C7)))
    for s in ty_sources:
        st = json.loads((tile.composite_dir("typical_year", s) / "stats.json").read_text())["months"]
        ks = sorted(int(k) for k in st)
        axes[0].plot(ks, [100 * st[str(k)]["same"] for k in ks], color=cols[s], lw=2, marker="o", ms=6,
                     ls="--" if s == "nicfi" else "-", label=SOURCE_LABEL[s])
    axes[0].set_xticks(range(1, 13)); axes[0].set_xticklabels(MONTH_ABBR)
    axes[0].set_ylabel("% of tile from usable same-month obs", color=INK)
    axes[0].set_ylim(90, 100.5)
    axes[0].set_title("typical year", fontsize=12, loc="left")
    axes[0].legend(frameon=False, fontsize=9)
    tidy(axes[0])
    for s in an_sources:
        st = json.loads((tile.composite_dir("annual", s) / "stats.json").read_text())["years"]
        ys = sorted(st)
        axes[1].plot(ys, [st[y]["mean_months_used"] for y in ys], color=cols[s], lw=2, marker="o", ms=6,
                     ls="--" if s == "nicfi" else "-", label=SOURCE_LABEL[s])
    axes[1].set_ylabel("usable months per pixel (of 12)", color=INK)
    axes[1].set_ylim(0, 12.5)
    axes[1].set_title("annual", fontsize=12, loc="left")
    axes[1].legend(frameon=False, fontsize=9)
    tidy(axes[1])
    fig.suptitle(f"{tile.tile_id}: how much usable data (score >= threshold) each composite rests on "
                 "(raw NICFI and mask coincide by construction; ~1% of the tile is nodata border)",
                 fontsize=12)
    fig.tight_layout()
    return _save(fig, out_dir / "coverage.png", 110)


def fig_composite_areas(tile, ty_sources, an_sources, out_dir):
    sources = [s for s in cp.SOURCES if s in ty_sources or s in an_sources]
    chosen = {}
    for w in example_windows(tile):
        chosen.setdefault(w["month"][:4], w)
    paths = []
    for year, w in sorted(chosen.items()):
        ty = {s: {k: _crop(io_utils.read_full(p)[0], w) for k, p in _composite_paths(tile, "typical_year", s).items()}
              for s in ty_sources}
        an = {s: {k: _crop(io_utils.read_full(p)[0], w) for k, p in _composite_paths(tile, "annual", s).items()}
              for s in an_sources}
        mk = [f"m{k:02d}" for k in range(1, 13)]
        yk = tile.years
        all_crops = [a for v in list(ty.values()) + list(an.values()) for a in v.values()]
        lims = lims_stack(np.stack(all_crops))
        ncol = len(mk) + 1 + len(yk)
        fig, axes = plt.subplots(len(sources), ncol, figsize=(1.9 * ncol, 2.05 * len(sources) + 0.9),
                                 squeeze=False, gridspec_kw=dict(wspace=0.04, hspace=0.06, left=0.06,
                                                                 right=0.995, top=0.86, bottom=0.02))
        for i, s in enumerate(sources):
            for j, k in enumerate(mk):
                if k in ty.get(s, {}):
                    axes[i, j].imshow(rgb(ty[s][k], lims), interpolation="nearest")
            axes[i, len(mk)].axis("off")
            for j, k in enumerate(yk, start=len(mk) + 1):
                if k in an.get(s, {}):
                    axes[i, j].imshow(rgb(an[s][k], lims), interpolation="nearest")
            axes[i, 0].set_ylabel(SOURCE_LABEL[s], fontsize=10, color=INK)
        for j, k in enumerate(mk):
            axes[0, j].set_title(MONTH_ABBR[j], fontsize=10)
        for j, k in enumerate(yk, start=len(mk) + 1):
            axes[0, j].set_title(k, fontsize=10)
        _noticks(axes)
        fig.text(0.06 + 0.94 * len(mk) / ncol / 2, 0.95, "typical year (12 months, all years)",
                 ha="center", fontsize=12, weight="bold", color=C2)
        fig.text(0.06 + 0.94 * (len(mk) + 1 + len(yk) / 2) / ncol, 0.95, "annual", ha="center",
                 fontsize=12, weight="bold", color=C2)
        fig.suptitle(f"{tile.tile_id} area r{w['r0']} c{w['c0']} (~{w['win'] * 4.77 / 1000:.1f} km): "
                     "composites by source (shared stretch)", fontsize=13, y=0.995)
        paths.append(_save(fig, out_dir / f"area_{year}_r{w['r0']}_c{w['c0']}.png", 80))
    return paths


# ------------------------------------------------------------------- runner
def run_visualize(tile: cfg.Tile, what: str = "all", log=print):
    paths = []
    if what in ("cloudmask", "all"):
        out = tile.fig_dir("1_cloudmask")
        st = _mask_month_stats(tile)
        paths += [fig_mask_stats(tile, st, out), fig_postcheck_frequency(tile, st, out)]
        paths += fig_mask_effect(tile, out)
        paths.append(fig_s2_coverage(tile, out))
    if what in ("reconstruct", "all"):
        methods = _methods(tile)
        if methods:
            out = tile.fig_dir("2_reconstruct")
            paths.append(fig_sources_by_month(tile, methods, out))
            if "s2fill" in methods:
                paths.append(fig_harmonization(tile, out))
            paths += fig_compare(tile, methods, out)
            paths.append(fig_compare_full_tile(tile, methods, _mask_month_stats(tile), out))
    if what in ("composite", "all"):
        ty, an = _sources(tile, "typical_year"), _sources(tile, "annual")
        if ty or an:
            out = tile.fig_dir("3_composite")
            if ty:
                paths.append(_grid_figure(tile, "typical_year", ty, f"{tile.tile_id}: typical-year "
                                          "monthly composites by source (shared stretch)",
                                          out / "typical_year.png"))
            if an:
                paths.append(_grid_figure(tile, "annual", an, f"{tile.tile_id}: annual composites by "
                                          "source (shared stretch)", out / "annual.png", step=4))
            paths.append(fig_coverage(tile, ty, an, out))
            paths += fig_composite_areas(tile, ty, an, out)
    for p in paths:
        if p:
            log(f"  -> {p.relative_to(tile.out_root)}")
