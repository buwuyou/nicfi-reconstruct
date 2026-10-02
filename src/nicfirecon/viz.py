"""
QA figures for every stage, written to outputs/nicfirecon/<tile>/figures/<stage>/.

Example areas ("windows") are chosen once per tile from the post-checked
NICFI masks alone -- per year, the months whose ~1.2 km window holds the
most cloud/shadow while still showing some clear context -- and cached in
cache/example_windows.json, so every method and composite is shown on the
same ground and can be compared side by side.

preprocess/   postcheck.png        flag frequency in mostly-clear obs (NICFI,
                                   S2), raw vs post-checked example, flags
                                   cleared per month
              s2_coverage.png      S2 composite clear coverage + method/month
reconstruct/<method>/
              overview.png         per month, % of tile by data source
              harmonization.png    (s2fill) per-month S2->NICFI fit + trust
              clouds_<year>.png    example windows: original | cloud mask |
                                   [S2 used] | reconstructed | [S2 SWIR false
                                   colour] | quality source
              full_tile.png        the most contaminated months, whole tile
composite/    typical_year_*.png   tiers, whole tile, example areas (every
                                   year's NICFI per month vs the composite)
              annual_<source>_*.png  whole tile per year, example areas
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


# --------------------------------------------------------------- preprocess
def fig_postcheck(tile, out_dir):
    months = tile.all_months
    zn = np.load(tile.cache_dir / "postcheck_nicfi.npz")
    zs_path = tile.cache_dir / "postcheck_s2.npz"
    cleared_frac = {m: float(masking.load_nicfi_cleared(tile, m).mean()) for m in months}
    m = max(cleared_frac, key=cleared_frac.get)
    nicfi, _, _ = io_utils.read_full(tile.nicfi_path(m))
    raw = np.load(masking.nicfi_mask_path(tile, m))["quality"]
    refined = masking.load_nicfi_classes(tile, m)
    cleared = masking.load_nicfi_cleared(tile, m)
    score = ndimage.uniform_filter(cleared.astype(np.float32), WIN)
    r, c = np.unravel_index(np.argmax(score[WIN // 2:-WIN // 2, WIN // 2:-WIN // 2]),
                            (score.shape[0] - WIN, score.shape[1] - WIN))
    w = {"r0": r, "c0": c, "win": WIN}
    crop = _crop(nicfi, w)
    lims = lims_from(crop)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    for ax, zp, name in ((axes[0, 0], tile.cache_dir / "postcheck_nicfi.npz", "NICFI"),
                         (axes[1, 0], zs_path, "Sentinel-2")):
        if not zp.exists():
            ax.set_visible(False)
            continue
        z = np.load(zp)
        im = ax.imshow(z["flag_freq"][::2, ::2], cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
        ax.contour(z["persistent"][::2, ::2], levels=[0.5], colors=C8, linewidths=0.6)
        ax.set_title(f"{name}: flag frequency in {int(z['mostly_clear'].sum())} mostly-clear obs\n"
                     f"red = persistent ({z['persistent'].mean():.2%} of px)", fontsize=11)
        fig.colorbar(im, ax=ax, shrink=0.7)
    axes[0, 1].imshow(overlay(rgb(crop, lims), _crop(raw, w)))
    axes[0, 1].set_title(f"{m}: raw OCM mask")
    axes[0, 2].imshow(overlay(rgb(crop, lims), _crop(refined, w), _crop(cleared, w)))
    axes[0, 2].set_title(f"after post-check ({cleared_frac[m]:.1%} of tile cleared)")
    axes[1, 1].imshow(rgb(crop, lims)); axes[1, 1].set_title(f"{m}: NICFI")
    x = np.arange(len(months))
    axes[1, 2].bar(x, [cleared_frac[k] * 100 for k in months], color=C1, width=0.8)
    axes[1, 2].set_ylabel("% of tile", color=INK)
    axes[1, 2].set_title("NICFI flags cleared per month")
    month_axis(axes[1, 2], months, 6)
    tidy(axes[1, 2])
    _noticks(axes.ravel()[[0, 1, 2, 3, 4]])
    fig.legend(handles=mask_legend(), loc="lower center", ncol=5, frameon=False, fontsize=10)
    fig.suptitle(f"{tile.tile_id}: cloud-mask post-check -- temporal (a spot flagged in most clear "
                 f"obs, looking the same each time, is ground) + spatial (speckle)", fontsize=13)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    return _save(fig, out_dir / "postcheck.png", 90)


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


# -------------------------------------------------------------- reconstruct
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


def fig_recon_overview(tile, method, st, out_dir):
    months = [m for m in tile.months if m in st["months"]]
    x = np.arange(len(months))
    fig, ax = plt.subplots(figsize=(16, 4.8))
    bottom = np.zeros(len(months))
    shown = []
    for code in (rc.SRC_S2_SINGLE, rc.SRC_S2_MEDIAN, rc.SRC_PHENO, rc.SRC_PHENO_BLEND,
                 rc.SRC_CONTAMINATED):
        name = rc.SOURCE_NAMES[code]
        v = np.array([st["months"][m]["source"][name] for m in months]) * 100
        if not v.any():
            continue
        label = name + (" (masked)" if method == "mask" and code == rc.SRC_CONTAMINATED else
                        " (kept)" if code == rc.SRC_CONTAMINATED else "")
        ax.bar(x, v, bottom=bottom, color=SOURCE_COLORS[code], width=0.8, edgecolor="white",
               linewidth=0.5, label=label)
        bottom += v
        shown.append(code)
    ax.set_ylabel("% of tile", color=INK)
    ax.set_title(f"{tile.tile_id} [{method}]: pixels not taken from clear NICFI, by data source "
                 "(rest = NICFI clear / nodata border)", fontsize=12)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    month_axis(ax, months)
    tidy(ax)
    fig.tight_layout()
    return _save(fig, out_dir / "overview.png", 120)


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


def fig_recon_windows(tile, method, st, out_dir):
    windows = example_windows(tile)
    bands = st["meta"]["bands"]
    has_extra = len(bands) > 4
    by_year = defaultdict(list)
    for w in windows:
        if w["month"] in st["months"]:
            by_year[w["month"][:4]].append(w)
    paths = []
    for year, ws in sorted(by_year.items()):
        cols = ["original NICFI", "NICFI + cloud mask (post-checked)"]
        if method == "s2fill":
            cols.append("S2 used (harmonized)")
        cols.append("reconstructed")
        if has_extra:
            cols.append("S2 SWIR1/NIR/Red (extra bands)")
        cols.append("data quality: source")
        fig, axes = plt.subplots(len(ws), len(cols), figsize=(4.4 * len(cols), 4.5 * len(ws)),
                                 squeeze=False)
        codes = set()
        for i, w in enumerate(ws):
            m = w["month"]
            nicfi, _, _ = io_utils.read_full(tile.nicfi_path(m))
            data, q = _load_recon(tile, method, m)
            n, rcn, qq = _crop(nicfi, w), _crop(data, w), _crop(q, w)
            cleared = (qq[2] & rc.FLAG_CLEARED).astype(bool)
            lims = lims_from(rcn[:4] if method != "mask" else n)
            panels = [rgb(n, lims), overlay(rgb(n, lims), qq[1], cleared)]
            if method == "s2fill":
                filled = np.isin(qq[0], (rc.SRC_S2_SINGLE, rc.SRC_S2_MEDIAN))
                panels.append(rgb(np.where(filled[None], rcn[:4], 0), lims))
            panels.append(rgb(rcn[:4], lims))
            if has_extra:
                panels.append(stretch3(np.stack([rcn[bands.index(b)] for b in ("B11", "nir", "red")])))
            for j, img in enumerate(panels):
                axes[i, j].imshow(img, interpolation="nearest")
            show_source(axes[i, -1], qq[0])
            codes |= set(np.unique(qq[0]).tolist())
            for j, title in enumerate(cols):
                axes[i, j].set_title((f"{m}  " if j == 0 else "") + title +
                                     (f" (mean score {qq[4].mean():.0f})" if j == len(cols) - 1 else ""),
                                     fontsize=10)
            axes[i, 0].set_ylabel(f"r{w['r0']} c{w['c0']}, ~{WIN * 4.77 / 1000:.1f} km", fontsize=9,
                                  color=INK)
        _noticks(axes)
        fig.legend(handles=mask_legend(), loc="lower left", ncol=5, frameon=False, fontsize=10,
                   title="cloud mask", bbox_to_anchor=(0.02, 0))
        fig.legend(handles=source_legend(sorted(codes)), loc="lower right", ncol=4, frameon=False,
                   fontsize=10, title="data quality: source", bbox_to_anchor=(0.98, 0))
        fig.suptitle(f"{tile.tile_id} {year} [{method}]: cloudiest example windows "
                     "(shared stretch per row)", fontsize=13)
        fig.tight_layout(rect=(0, 0.05, 1, 0.98))
        paths.append(_save(fig, out_dir / f"clouds_{year}.png", 85))
    return paths


def fig_recon_full_tile(tile, method, st, out_dir, n=3):
    months = [m for m in tile.months if m in st["months"]]

    def dirty(m):
        s = st["months"][m]["source"]
        return sum(v for k, v in s.items() if k not in ("NICFI clear", "nodata"))
    worst = sorted(sorted(months, key=lambda m: -dirty(m))[:n])
    fig, axes = plt.subplots(len(worst), 4, figsize=(24, 6 * len(worst)), squeeze=False)
    codes = set()
    for i, m in enumerate(worst):
        nicfi, _, _ = io_utils.read_full(tile.nicfi_path(m))
        data, q = _load_recon(tile, method, m)
        lims = lims_from(nicfi, q[0] == rc.SRC_NICFI)
        axes[i, 0].imshow(rgb(nicfi[:, ::2, ::2], lims)); axes[i, 0].set_title(f"{m} original NICFI")
        axes[i, 1].imshow(rgb(data[:4, ::2, ::2], lims)); axes[i, 1].set_title(f"{m} [{method}]")
        show_source(axes[i, 2], q[0, ::2, ::2])
        codes |= set(np.unique(q[0]).tolist())
        axes[i, 2].set_title("data quality: source")
        im = axes[i, 3].imshow(q[4, ::2, ::2], cmap="Blues", vmin=0, vmax=100, interpolation="nearest")
        axes[i, 3].set_title(f"score (mean {q[4].mean():.0f}/100)")
        fig.colorbar(im, ax=axes[i, 3], shrink=0.7)
    _noticks(axes)
    fig.legend(handles=source_legend(sorted(codes)), loc="lower center", ncol=6, frameon=False,
               fontsize=11)
    fig.suptitle(f"{tile.tile_id} [{method}]: most contaminated months, whole tile", fontsize=14)
    fig.tight_layout(rect=(0, 0.02, 1, 0.98))
    return _save(fig, out_dir / "full_tile.png", 75)


# ---------------------------------------------------------------- composite
def _area_grid(tile, w, rows, row_labels, title, path, extra_row=None):
    """rows: list of lists of (4,h,w) crops or None, 12 per row (or fewer)."""
    n_cols = max(len(r) for r in rows)
    n_rows = len(rows) + (1 if extra_row is not None else 0)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2.1 * n_rows + 1.2),
                             gridspec_kw=dict(hspace=0.08, wspace=0.04, left=0.05, right=0.99,
                                              top=0.93, bottom=0.1 if extra_row else 0.03),
                             squeeze=False)
    stack = [c for r in rows[-1:] for c in r if c is not None]
    lims = lims_stack(np.stack(stack)) if stack else np.array([[0, 1]] * 3, float)
    for i, row in enumerate(rows):
        for j, crop in enumerate(row):
            if crop is not None:
                axes[i, j].imshow(rgb(crop, lims), interpolation="nearest")
    if extra_row is not None:
        imgs, vmax, label = extra_row
        for j, img in enumerate(imgs):
            im = axes[-1, j].imshow(img, cmap="Blues", vmin=0, vmax=vmax, interpolation="nearest")
        cb = fig.colorbar(im, ax=axes[-1, :].tolist(), orientation="horizontal", fraction=0.04,
                          pad=0.25, aspect=60)
        cb.set_label(label, fontsize=9)
    for i, label in enumerate(row_labels):
        axes[i, 0].set_ylabel(label, fontsize=10, color=INK)
    _noticks(axes)
    fig.suptitle(title, fontsize=13)
    return _save(fig, path, 80)


def figs_typical_year(tile, out_dir):
    d = tile.composite_dir("typical_year")
    st = json.loads((d / "stats.json").read_text())["months"]
    ks = sorted(int(k) for k in st)
    x = np.arange(len(ks))
    fig, ax = plt.subplots(figsize=(12, 4.5))
    bottom = np.zeros(len(ks))
    for key, col, label in (("adjacent", C1, "clear only in adjacent months (tier 2)"),
                            ("least_bad", C8, "never clear: least-contaminated obs (tier 3)")):
        v = np.array([st[str(k)][key] for k in ks]) * 100
        ax.bar(x, v, bottom=bottom, color=col, width=0.8, label=label, edgecolor="white", linewidth=0.5)
        bottom += v
    ax.set_xticks(x); ax.set_xticklabels([MONTH_ABBR[k - 1] for k in ks])
    ax.set_ylabel("% of tile", color=INK)
    ax.set_title(f"{tile.tile_id}: typical-year composite -- pixels not clear in any observation "
                 "of that calendar month (rest = tier 1)", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    tidy(ax)
    fig.tight_layout()
    paths = [_save(fig, out_dir / "typical_year_tiers.png", 120)]

    comps = {k: io_utils.read_full(d / f"{tile.tile_id}_m{k:02d}.tif")[0] for k in ks}
    quals = {k: io_utils.read_full(d / f"{tile.tile_id}_m{k:02d}_quality.tif")[0] for k in ks}
    lims = lims_stack(np.stack([comps[k][:, ::8, ::8] for k in ks]))
    fig, axes = plt.subplots(3, 4, figsize=(20, 15.5))
    for k, ax in zip(ks, axes.ravel()):
        ax.imshow(rgb(comps[k][:, ::3, ::3], lims))
        ax.set_title(f"{MONTH_ABBR[k - 1]}  (tier 1: {st[str(k)]['same']:.1%})", fontsize=12)
    _noticks(axes)
    years = sorted({m[:4] for m in tile.all_months})
    fig.suptitle(f"{tile.tile_id}: typical year from NICFI {years[0]}-{years[-1]} only "
                 "(shared stretch)", fontsize=15)
    fig.tight_layout()
    paths.append(_save(fig, out_dir / "typical_year_full_tile.png", 80))

    chosen = {}
    for w in example_windows(tile):
        chosen.setdefault(w["month"][:4], w)
    for year, w in sorted(chosen.items()):
        rows, labels = [], []
        for y in years:
            row = []
            for k in range(1, 13):
                m = f"{y}-{k:02d}"
                row.append(_crop(io_utils.read_full(tile.nicfi_path(m))[0], w)
                           if m in tile.all_months else None)
            rows.append(row)
            labels.append(y)
        rows.append([_crop(comps[k], w) if k in comps else None for k in range(1, 13)])
        labels.append("typical-year\ncomposite")
        ncl = [_crop(quals[k][1], w) for k in ks]
        paths.append(_area_grid(
            tile, w, rows, labels + [f"clear years\n(of {len(years)})"],
            f"{tile.tile_id} area r{w['r0']} c{w['c0']} (~{w['win'] * 4.77 / 1000:.1f} km): every NICFI "
            f"month vs. the typical-year composite (columns Jan..Dec)",
            out_dir / f"typical_year_area_{year}_r{w['r0']}_c{w['c0']}.png",
            extra_row=(ncl, len(years), "clear same-month observations behind each composite pixel")))
    return paths


def figs_annual(tile, source, out_dir):
    d = tile.composite_dir("annual", source)
    st = json.loads((d / "stats.json").read_text())
    years = sorted(st["years"])
    comps = {y: io_utils.read_full(d / f"{tile.tile_id}_{y}.tif")[0] for y in years}
    lims = lims_stack(np.stack([comps[y][:, ::8, ::8] for y in years]))
    fig, axes = plt.subplots(1, len(years), figsize=(5 * len(years), 5.6), squeeze=False)
    for y, ax in zip(years, axes[0]):
        ax.imshow(rgb(comps[y][:, ::3, ::3], lims))
        ax.set_title(f"{y}  (score>={st['min_score']}: {st['years'][y]['tier_ok']:.1%})", fontsize=12)
    _noticks(axes)
    fig.suptitle(f"{tile.tile_id}: annual composites from [{source}], stat={st['stat']} "
                 "(shared stretch)", fontsize=14)
    fig.tight_layout()
    paths = [_save(fig, out_dir / f"annual_{source}_full_tile.png", 80)]

    chosen = {}
    for w in example_windows(tile):
        chosen.setdefault(w["month"][:4], w)
    rows = [[_crop(comps[y], w) for y in years] for w in chosen.values()]
    fig, axes = plt.subplots(len(rows), len(years), figsize=(3 * len(years), 3.1 * len(rows)),
                             squeeze=False)
    for i, (row, w) in enumerate(zip(rows, chosen.values())):
        lims = lims_stack(np.stack(row))
        for j, crop in enumerate(row):
            axes[i, j].imshow(rgb(crop, lims), interpolation="nearest")
            if i == 0:
                axes[i, j].set_title(years[j], fontsize=11)
        axes[i, 0].set_ylabel(f"r{w['r0']} c{w['c0']}", fontsize=9, color=INK)
    _noticks(axes)
    fig.suptitle(f"{tile.tile_id}: annual composites [{source}] on the example areas", fontsize=13)
    fig.tight_layout()
    paths.append(_save(fig, out_dir / f"annual_{source}_areas.png", 85))
    return paths


# ------------------------------------------------------------------- runner
def run_visualize(tile: cfg.Tile, what: str, method: str = None, source: str = None, log=print):
    if what == "preprocess":
        out = tile.fig_dir("preprocess")
        paths = [fig_postcheck(tile, out), fig_s2_coverage(tile, out)]
    elif what == "reconstruct":
        out = tile.fig_dir(f"reconstruct/{method}")
        st = _recon_stats(tile, method)
        paths = [fig_recon_overview(tile, method, st, out)]
        if method == "s2fill":
            paths.append(fig_harmonization(tile, out))
        paths += fig_recon_windows(tile, method, st, out)
        paths.append(fig_recon_full_tile(tile, method, st, out))
    elif what == "typical-year":
        paths = figs_typical_year(tile, tile.fig_dir("composite"))
    elif what == "annual":
        paths = figs_annual(tile, source, tile.fig_dir("composite"))
    else:
        raise ValueError(what)
    for p in paths:
        if p:
            log(f"  -> {p.relative_to(tile.out_root)}")
