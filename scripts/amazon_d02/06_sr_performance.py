"""
Performance check of the super-resolved Sentinel-2 (SR) on D02, run on
whatever months 03_superresolve.py has cached so far. Written to inform the
decision in docs/amazon_d02_s2fusion.md: D02's NICFI has no real data gaps
(only a fixed 6-7px tile-edge border, identical in all 60 months), so the
open question is whether SR is good enough to *replace* NICFI pixels that
are present but contaminated, not just fill holes.

Figures (outputs/amazon_d02/figures/perf_*.png):
  1. Visual crops: NICFI (4.77m) vs Sentinel-2 (10m) vs SR (2.5m) on the same
     ground, plus an SR SWIR false colour.
  2. Hard-constraint check: SR block-averaged back to 10m vs the real S2
     mosaic it came from (pooled over all cached months).
  3. SR vs NICFI agreement on NICFI's grid, per band/month, against a plain
     bilinear upsample of the 10m S2 mosaic as the baseline. The high-pass
     correlation is the key number: it measures whether SR's added fine
     detail matches what NICFI actually sees, i.e. real detail rather than
     plausible-looking texture.
  4. Contamination candidates: blobs where NICFI blue is anomalously
     brighter than SR blue (haze/cloud in NICFI that SR could replace), and
     per-month counts of both signs (the opposite sign = S2 contaminated).

Caveat for 3 and 4: NICFI is a monthly mosaic and the S2 mosaic is a median
of different dates, so some disagreement is real temporal/BRDF difference,
not SR error. Comparing SR against the bilinear baseline under the same
caveat is what makes the numbers interpretable.

Run: python scripts/amazon_d02/06_sr_performance.py
"""
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import fuse, harmonize, io_utils, s2_fusion_site as cfg

site = cfg.D02_TEST_YEAR

C_SR = "#2a78d6"        # categorical slot 1
C_BASE = "#eb6834"      # categorical slot 2
INK = "#52514e"
EVAL_ERODE_PX = 10      # keep away from S2-mask/tile edges (filter + SR edge effects)
HP_SIGMA = 3.0          # NICFI px; high-pass = x - gaussian(x)
BANDS = [("blue", "B2"), ("green", "B3"), ("red", "B4"), ("nir", "B8")]


def load_params():
    import json
    with open(site.cache_dir / "harmonization_params.json") as f:
        raw = json.load(f)
    return {int(y): harmonize.HarmonizationParams(
        coeffs={b: tuple(c) for b, c in v["coeffs"].items()}, n_samples=v["n_samples"])
        for y, v in raw.items()}


def load_npz(path):
    z = np.load(path, allow_pickle=False)
    return z, rasterio.Affine(*z["transform"]), rasterio.crs.CRS.from_string(str(z["crs"]))


def to_grid(src, s_tr, s_crs, shape, d_tr, d_crs, resampling):
    out = np.zeros((src.shape[0],) + tuple(shape), dtype=np.float32)
    reproject(source=src.astype(np.float32), destination=out, src_transform=s_tr, src_crs=s_crs,
              dst_transform=d_tr, dst_crs=d_crs, resampling=resampling)
    return out


def load_month(month, params):
    nat, n_tr, n_crs = io_utils.read_full(site.nicfi_path(month))
    harm = harmonize.apply(nat, params[int(month[:4])])
    shape = harm.shape[1:]
    zs, sr_tr, sr_crs = load_npz(site.cache_dir / f"sr_{month}.npz")
    zm, s2_tr, s2_crs = load_npz(site.cache_dir / f"s2_mosaic_{month}.npz")
    sr_n = to_grid(zs["data"], sr_tr, sr_crs, shape, n_tr, n_crs, Resampling.average)
    base_n = to_grid(zm["data"], s2_tr, s2_crs, shape, n_tr, n_crs, Resampling.bilinear)
    sr_valid = fuse.reproject_valid_to_nicfi_grid(zm["valid"], s2_tr, s2_crs, shape, n_tr, n_crs)
    native = ~np.all(harm == 0, axis=0)
    eval_mask = ndimage.binary_erosion(sr_valid & native, iterations=EVAL_ERODE_PX)
    return dict(month=month, nat=nat, harm=harm, n_tr=n_tr, n_crs=n_crs, sr=zs["data"],
                sr_tr=sr_tr, sr_crs=sr_crs, s2=zm["data"], s2_valid=zm["valid"], s2_tr=s2_tr,
                s2_crs=s2_crs, sr_n=sr_n, base_n=base_n, sr_valid=sr_valid, eval=eval_mask)


def highpass(x):
    return x - ndimage.gaussian_filter(x, HP_SIGMA)


def metrics(ref, est, mask):
    a, b = ref[mask], est[mask]
    hp_r = np.corrcoef(highpass(ref)[mask], highpass(est)[mask])[0, 1]
    return dict(r=np.corrcoef(a, b)[0, 1], rmse=float(np.sqrt(np.mean((a - b) ** 2))),
                bias=float(np.mean(b - a)), hp_r=hp_r)


def stretch(rgb, lims):
    return np.clip((rgb - lims[:, 0]) / np.maximum(lims[:, 1] - lims[:, 0], 1e-6), 0, 1)


def crop_on_display_grid(src, s_tr, s_crs, bounds, res=2.5):
    """Nearest-neighbour onto a fine EPSG:3857 grid so every source shows its
    true pixel size side by side."""
    l, b, r, t = bounds
    w, h = int(round((r - l) / res)), int(round((t - b) / res))
    d_tr = rasterio.transform.from_bounds(l, b, r, t, w, h)
    return to_grid(src, s_tr, s_crs, (h, w), d_tr, rasterio.crs.CRS.from_epsg(3857), Resampling.nearest)


def pick_textured_windows(d, n=3, win=128):
    nir = d["harm"][3]
    m = ndimage.uniform_filter(nir, win)
    std = np.sqrt(np.clip(ndimage.uniform_filter(nir ** 2, win) - m ** 2, 0, None))
    ok = ndimage.uniform_filter(d["eval"].astype(np.float32), win) > 0.999
    score = np.where(ok, std, -1)
    picks = []
    for _ in range(n):
        r, c = np.unravel_index(np.argmax(score), score.shape)
        if score[r, c] <= 0:
            break
        picks.append((r - win // 2, c - win // 2))
        score[max(0, r - win):r + win, max(0, c - win):c + win] = -1
    return picks


def fig_visual_crops(d, out):
    win = 128
    picks = pick_textured_windows(d, win=win)
    cols = ["NICFI 4.77m (harmonized)", "Sentinel-2 10m", "SR Sentinel-2 2.5m",
            "SR SWIR1/NIR/Red (new bands)"]
    fig, axes = plt.subplots(len(picks), 4, figsize=(18, 4.6 * len(picks)))
    axes = np.atleast_2d(axes)
    rgb_s2 = [cfg.S2_INDEX[b] for b in ("B4", "B3", "B2")]
    fc_s2 = [cfg.S2_INDEX[b] for b in ("B11", "B8", "B4")]
    for i, (r0, c0) in enumerate(picks):
        win_tr = rasterio.windows.transform(rasterio.windows.Window(c0, r0, win, win), d["n_tr"])
        bounds = rasterio.transform.array_bounds(win, win, win_tr)  # (west, south, east, north)
        nicfi = crop_on_display_grid(d["harm"][[2, 1, 0]], d["n_tr"], d["n_crs"], bounds)
        s2 = crop_on_display_grid(d["s2"][rgb_s2], d["s2_tr"], d["s2_crs"], bounds)
        sr = crop_on_display_grid(d["sr"][rgb_s2], d["sr_tr"], d["sr_crs"], bounds)
        fc = crop_on_display_grid(d["sr"][fc_s2], d["sr_tr"], d["sr_crs"], bounds)
        # one shared stretch (from the SR crop) for the three RGB panels, so
        # radiometric differences stay visible instead of being normalised away
        lims = np.percentile(sr.reshape(3, -1), [2, 98], axis=1).T
        fc_lims = np.percentile(fc.reshape(3, -1), [2, 98], axis=1).T
        for j, img in enumerate([stretch(np.moveaxis(nicfi, 0, -1), lims),
                                 stretch(np.moveaxis(s2, 0, -1), lims),
                                 stretch(np.moveaxis(sr, 0, -1), lims),
                                 stretch(np.moveaxis(fc, 0, -1), fc_lims)]):
            ax = axes[i, j]
            ax.imshow(img, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(cols[j], fontsize=11, color=INK)
        axes[i, 0].set_ylabel(f"crop {i+1}: NICFI px r{r0} c{c0}\n(~{win*4.77:.0f} m wide)",
                              fontsize=9, color=INK)
    fig.suptitle(f"D02 {d['month']}: same ground, three resolutions (shared RGB stretch per row)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def fig_hard_constraint(months_data, out):
    show = ["B2", "B4", "B8", "B11"]
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    rng = np.random.default_rng(0)
    rows = []
    for k, band in enumerate(show):
        j = cfg.S2_INDEX[band]
        xs, ys = [], []
        for d in months_data:
            h, w = d["s2"].shape[1:]
            down = d["sr"][j, :h * 4, :w * 4].reshape(h, 4, w, 4).mean(axis=(1, 3))
            v = ndimage.binary_erosion(d["s2_valid"], iterations=3)
            xs.append(d["s2"][j][v]); ys.append(down[v])
        x, y = np.concatenate(xs), np.concatenate(ys)
        r = np.corrcoef(x, y)[0, 1]
        rel_bias = np.mean(y - x) / np.mean(x)
        rel_mae = np.mean(np.abs(y - x)) / np.mean(x)
        rows.append((band, r, rel_bias, rel_mae))
        idx = rng.choice(x.size, min(x.size, 400_000), replace=False)
        lim = [np.percentile(x, 0.5), np.percentile(x, 99.5)]
        axes[k].hexbin(x[idx], y[idx], gridsize=120, extent=lim + lim, bins="log", cmap="Blues",
                       mincnt=1)
        axes[k].plot(lim, lim, color=INK, lw=1, ls="--")
        axes[k].set_xlim(lim); axes[k].set_ylim(lim)
        axes[k].set_xlabel(f"real S2 {band} (10m mosaic)")
        axes[k].set_ylabel(f"SR {band}, block-averaged to 10m")
        axes[k].set_title(f"{band}: r={r:.3f}, bias={rel_bias:+.1%}, MAE={rel_mae:.1%}", fontsize=11)
    fig.suptitle(f"Hard-constraint check, {len(months_data)} months pooled "
                 "(SR should reproduce its own 10m input; dashed = y=x)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return rows


def fig_vs_nicfi(months_data, out):
    res = {m: {} for m in ("sr", "base")}
    for d in months_data:
        for band, s2b in BANDS:
            i, j = cfg.NICFI_BAND_NAMES.index(band), cfg.S2_INDEX[s2b]
            res["sr"].setdefault(band, []).append(metrics(d["harm"][i], d["sr_n"][j], d["eval"]))
            res["base"].setdefault(band, []).append(metrics(d["harm"][i], d["base_n"][j], d["eval"]))
    months = [d["month"][5:] for d in months_data]
    rows = [("r", "correlation r (higher = better)"),
            ("rmse", "RMSE, DN (lower = better)"),
            ("hp_r", f"high-pass r, σ={HP_SIGMA:g}px\n(fine-detail agreement)")]
    fig, axes = plt.subplots(3, 4, figsize=(20, 11), sharex=True)
    x = np.arange(len(months))
    for c, (band, s2b) in enumerate(BANDS):
        for rr, (key, label) in enumerate(rows):
            ax = axes[rr, c]
            for name, col, lab in (("sr", C_SR, "SR (2.5m → NICFI grid)"),
                                   ("base", C_BASE, "bilinear 10m S2 (baseline)")):
                vals = [m[key] for m in res[name][band]]
                ax.plot(x, vals, color=col, lw=2, marker="o", ms=8, label=lab)
                ax.annotate(f"{vals[-1]:.2f}" if key != "rmse" else f"{vals[-1]:.0f}",
                            (x[-1], vals[-1]), xytext=(6, 0), textcoords="offset points",
                            va="center", fontsize=9, color=INK)
            ax.grid(axis="y", color="#e6e5e0", lw=0.8)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            if rr == 0:
                ax.set_title(f"NICFI {band} vs S2 {s2b}", fontsize=12)
            if c == 0:
                ax.set_ylabel(label, fontsize=10, color=INK)
            if rr == 2:
                ax.set_xticks(x); ax.set_xticklabels(months)
                ax.set_xlabel("2022 month")
    axes[0, 0].legend(fontsize=9, frameon=False, loc="lower left")
    n_px = int(np.mean([d["eval"].sum() for d in months_data]))
    fig.suptitle("Does SR agree with NICFI better than plain interpolation? "
                 f"(~{n_px:,} clear-in-both px/month, harmonized NICFI as reference)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return res


def fig_contamination(months_data, out, z_thresh=4.0, min_blob=20, n_show=4):
    j_b2 = cfg.S2_INDEX["B2"]
    counts, blobs = [], []
    for d in months_data:
        diff = d["harm"][0] - d["sr_n"][j_b2]
        m = d["eval"]
        med = np.median(diff[m])
        mad = 1.4826 * np.median(np.abs(diff[m] - med))
        z = np.where(m, (diff - med) / mad, 0.0)
        pos, neg = z > z_thresh, z < -z_thresh
        lab, n = ndimage.label(pos)
        sizes = np.bincount(lab.ravel())[1:] if n else np.array([])
        big = np.nonzero(sizes >= min_blob)[0] + 1
        objs = ndimage.find_objects(lab)
        counts.append((d["month"], pos.mean(), neg.mean(), len(big)))
        for b in big:
            sl = objs[b - 1]
            blobs.append((int(sizes[b - 1]), d, (sl[0].start + sl[0].stop) // 2,
                          (sl[1].start + sl[1].stop) // 2, z))
    blobs.sort(key=lambda t: -t[0])
    picked = []
    for blob in blobs:  # one per month first so the figure isn't 4x the same month
        if all(p[1]["month"] != blob[1]["month"] for p in picked):
            picked.append(blob)
        if len(picked) == n_show:
            break

    n_rows = len(picked) + 1
    fig = plt.figure(figsize=(16, 4.4 * n_rows))
    gs = fig.add_gridspec(n_rows, 3)
    ax = fig.add_subplot(gs[0, :])
    x = np.arange(len(counts))
    ax.plot(x, [c[1] * 100 for c in counts], color=C_SR, lw=2, marker="o", ms=8,
            label=f"NICFI brighter than SR (z>{z_thresh:g}): NICFI haze/cloud candidates")
    ax.plot(x, [c[2] * 100 for c in counts], color=C_BASE, lw=2, marker="o", ms=8,
            label=f"SR brighter than NICFI (z<-{z_thresh:g}): S2 contamination candidates")
    for xi, c in zip(x, counts):
        ax.annotate(f"{c[3]} blobs", (xi, c[1] * 100), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=9, color=INK)
    ax.set_xticks(x); ax.set_xticklabels([c[0] for c in counts])
    ax.set_ylabel("% of clear-in-both px", color=INK)
    ax.grid(axis="y", color="#e6e5e0", lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=9, frameon=False)
    ax.set_title(f"Blue-band disagreement, robust z of (NICFI − SR); blobs = ≥{min_blob} px connected",
                 fontsize=12)

    half = 64
    for k, (size, d, r, c, z) in enumerate(picked, start=1):
        r0, c0 = max(0, r - half), max(0, c - half)
        sl = (slice(r0, r0 + 2 * half), slice(c0, c0 + 2 * half))
        nic = np.moveaxis(d["harm"][[2, 1, 0]][:, sl[0], sl[1]], 0, -1)
        sr = np.moveaxis(d["sr_n"][[cfg.S2_INDEX[b] for b in ("B4", "B3", "B2")]][:, sl[0], sl[1]], 0, -1)
        lims = np.percentile(sr.reshape(-1, 3), [2, 98], axis=0).T
        lims[:, 1] = np.maximum(lims[:, 1], np.percentile(nic.reshape(-1, 3), 98, axis=0))
        for jj, (img, title) in enumerate([(stretch(nic, lims), "NICFI (harmonized)"),
                                           (stretch(sr, lims), "SR Sentinel-2 on NICFI grid")]):
            a = fig.add_subplot(gs[k, jj]); a.imshow(img, interpolation="nearest")
            a.set_xticks([]); a.set_yticks([])
            a.set_title(f"{d['month']} blob {size} px — {title}", fontsize=10)
        a = fig.add_subplot(gs[k, 2])
        im = a.imshow(z[sl], cmap="RdBu_r", vmin=-10, vmax=10, interpolation="nearest")
        a.set_xticks([]); a.set_yticks([]); a.set_title("blue z (red = NICFI brighter)", fontsize=10)
        fig.colorbar(im, ax=a, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return counts


def main():
    t0 = time.time()
    params = load_params()
    months = [m for m in site.months if (site.cache_dir / f"sr_{m}.npz").exists()]
    if not months:
        print("No SR cache yet -- run 03_superresolve.py first.")
        return
    print(f"SR cached for {len(months)} months: {months}")
    months_data = []
    for m in months:
        months_data.append(load_month(m, params))
        print(f"  loaded {m}: eval px={months_data[-1]['eval'].sum():,}")

    best = max(months_data, key=lambda d: d["eval"].sum())
    fig_visual_crops(best, site.fig_dir / "perf_01_visual_crops.png")
    print(f"-> perf_01_visual_crops.png ({best['month']})")

    for band, r, b, mae in fig_hard_constraint(months_data, site.fig_dir / "perf_02_hard_constraint.png"):
        print(f"   hard constraint {band}: r={r:.3f} bias={b:+.2%} MAE={mae:.2%}")
    print("-> perf_02_hard_constraint.png")

    res = fig_vs_nicfi(months_data, site.fig_dir / "perf_03_vs_nicfi.png")
    print("-> perf_03_vs_nicfi.png   (mean over months: SR | bilinear)")
    for band, _ in BANDS:
        s = {k: np.mean([m[k] for m in res["sr"][band]]) for k in ("r", "rmse", "bias", "hp_r")}
        b = {k: np.mean([m[k] for m in res["base"][band]]) for k in ("r", "rmse", "bias", "hp_r")}
        print(f"   {band:5s} r {s['r']:.3f} | {b['r']:.3f}   rmse {s['rmse']:6.0f} | {b['rmse']:6.0f}"
              f"   bias {s['bias']:+6.0f} | {b['bias']:+6.0f}   hp_r {s['hp_r']:.3f} | {b['hp_r']:.3f}")

    for m, pos, neg, nb in fig_contamination(months_data, site.fig_dir / "perf_04_contamination.png"):
        print(f"   {m}: NICFI-brighter {pos:.3%} ({nb} blobs), SR-brighter {neg:.3%}")
    print("-> perf_04_contamination.png")
    print(f"DONE in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
