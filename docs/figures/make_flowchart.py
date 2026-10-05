"""Draws docs/figures/pipeline_flowchart.png (the README's pipeline overview).
Run: python docs/figures/make_flowchart.py"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

INK, MUTED, BG = "#0b0b0b", "#52514e", "#ffffff"
STAGE = {"in": "#52514e", "pre": "#2a78d6", "rec": "#1baf7a", "comp": "#eb6834"}
TINT = {"in": "#f1f0ec", "pre": "#e6f0fb", "rec": "#e3f5ee", "comp": "#fcebe3"}

fig, ax = plt.subplots(figsize=(15, 8.0))
fig.patch.set_facecolor(BG)
ax.set_xlim(0, 15)
ax.set_ylim(0, 8.0)
ax.axis("off")


def box(x, y, title, sub, key, w=2.55, h=0.95):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=TINT[key], ec=STAGE[key], lw=1.6))
    ax.text(x, y + 0.15, title, ha="center", va="center", fontsize=12.5, weight="bold", color=INK)
    ax.text(x, y - 0.2, sub, ha="center", va="center", fontsize=10, color=MUTED)
    return (x, y, w, h)


def arrow(a, b, rad=0.0, dashed=False, label=None, lpos=0.5, loff=(0, 0.14)):
    (x0, y0), (x1, y1) = a, b
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14,
                                 connectionstyle=f"arc3,rad={rad}", color=MUTED, lw=1.4,
                                 linestyle=(0, (4, 3)) if dashed else "-"))
    if label:
        ax.text(x0 + (x1 - x0) * lpos + loff[0], y0 + (y1 - y0) * lpos + loff[1], label,
                ha="center", va="bottom", fontsize=9, color=MUTED, style="italic")


def header(x, num, name, cmd, key):
    ax.text(x, 7.6, f"{num}  {name}" if num else name, ha="center", fontsize=14.5, weight="bold",
            color=STAGE[key])
    if cmd:
        ax.text(x, 7.25, cmd, ha="center", fontsize=9.5, color=MUTED, family="monospace")


R = lambda b: (b[0] + b[2] / 2, b[1])   # right edge middle
L = lambda b: (b[0] - b[2] / 2, b[1])   # left edge middle
T = lambda b: (b[0], b[1] + b[3] / 2)
B = lambda b: (b[0], b[1] - b[3] / 2)

# columns
xi, xp, xr, xc = 1.55, 5.05, 8.95, 12.85
header(xi, "0", "Download", "download", "in")
header(xp, "1", "Preprocess", "preprocess", "pre")
header(xr, "2", "Reconstruct", "reconstruct --method", "rec")
header(xc, "3", "Composite", "composite --type", "comp")

nicfi = box(xi, 4.9, "NICFI monthly", "GEE · 4 bands · 4.77 m", "in")
sen2 = box(xi, 2.9, "Sentinel-2 L2A", "GEE · optional · 10 m", "in")

mask = box(xp, 5.6, "Cloud mask", "OmniCloudMask", "pre")
post = box(xp, 4.0, "Post-check", "temporal + speckle", "pre")
s2c = box(xp, 2.4, "S2 composite", "clear frame / median", "pre")

r_mask = box(xr, 5.6, "mask", "clouds → nodata", "rec")
r_s2 = box(xr, 4.0, "s2fill", "harmonized S2 (+6 bands)", "rec")
r_ph = box(xr, 2.4, "phenology", "harmonic + DL refiner", "rec")

c_typ = box(xc, 5.6, "typical year", "12 months, all years", "comp")
c_ann = box(xc, 3.4, "annual", "one image per year", "comp")

# flows
arrow(R(nicfi), L(mask), rad=-0.15)
arrow(R(sen2), L(mask), rad=-0.12, dashed=True)
arrow(B(mask), T(post))
arrow(B(post), T(s2c), dashed=True)
arrow(R(post), L(r_mask), rad=-0.15)
arrow(R(post), L(r_ph), rad=0.15)
arrow(R(s2c), L(r_s2), rad=0.2, dashed=True)
for r in (r_mask, r_s2, r_ph):
    arrow(R(r), L(c_ann), rad=0.0)
arrow(T(mask), T(c_typ), rad=-0.2, label="NICFI + masks", lpos=0.5, loff=(0, 0.3))

# product band
ax.add_patch(FancyBboxPatch((6.9, 0.55), 7.95, 0.75, boxstyle="round,pad=0.02,rounding_size=0.12",
                            fc="#f7f7f5", ec="#c3c2b7", lw=1.0))
ax.text(10.875, 1.04, "Every product: GeoTIFF + quality layer", ha="center", fontsize=11.5,
        weight="bold", color=INK)
ax.text(10.875, 0.73, "source · cloud class · flags · score (0–100)", ha="center", fontsize=10,
        color=MUTED)
ax.text(0.3, 0.75, "- - -  Sentinel-2 path (optional)", fontsize=9.5, color=MUTED)
ax.text(0.3, 0.42, "python -m src.nicfirecon <stage> --tile <ID>", fontsize=9.5, color=MUTED,
        family="monospace")

out = Path(__file__).with_name("pipeline_flowchart.png")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
print(out)
