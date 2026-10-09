"""
Tile config + output layout for the NICFI reconstruction pipeline
(`python -m src.nicfirecon ...`, see `src/nicfirecon/cli.py`). Nothing is
specific to one tile: every command takes `--tile <ID>` (plus optional
directory overrides), and the month list is discovered from the NICFI
files actually on disk.

Expected input layout (defaults; override with --nicfi-dir / --s2-dir) --
exactly what the `download` stage writes:
    <data-root>/<TILE>/        one NICFI monthly mosaic per month, any
                               filename ending in YYYY-MM.tif
    <data-root>/<TILE>_S2/     optional: raw single-date Sentinel-2 frames
                               named YYYY-MM-DD.tif, all on one 10m grid,
                               with band descriptions (B2, B3, ...)

Output layout, outputs/nicfirecon/<TILE>/:
    cache/                     masks, S2 composites, fits, stats (regenerable)
    reconstructed/<method>/    monthly products: mask | s2fill | phenology
    composites/annual/<stat>/<source>/        one image per calendar year
    composites/typical_year/<stat>/<source>/  12 monthly images from all years
    figures/<stage>/           QA figures (tracked in git)
"""
import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = Path("/mnt/warehouse/amazon")
PIPELINE_NAME = "nicfirecon"

# ---- NICFI: Blue/Green/Red/NIR, same convention as src/sites.py ----
NICFI_BAND_NAMES = ["blue", "green", "red", "nir"]
REFLECTANCE_SCALE = 10000.0  # both sensors deliver DN = reflectance * 1e4

# Spectral correspondence used for harmonization + substitution. S2 band
# *positions* are never hardcoded -- they're read from each file's band
# descriptions (`masking.band_index`).
NICFI_TO_S2_BAND = {"blue": "B2", "green": "B3", "red": "B4", "nir": "B8"}
# S2 bands NICFI has no counterpart for (optional extra output bands)
S2_EXTRA_BANDS = ["B5", "B6", "B7", "B8A", "B11", "B12"]

_MONTH_RE = re.compile(r"(\d{4}-\d{2})\.tif$")
_DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})\.tif$")


@dataclass
class Tile:
    tile_id: str
    nicfi_dir: Path
    s2_dir: Path
    month_filter: Optional[List[str]] = None  # prefixes, e.g. ["2023", "2024-05"]
    out_base: Path = PROJECT_DIR / "outputs" / PIPELINE_NAME

    @property
    def months(self) -> List[str]:
        found = sorted({m.group(1) for p in self.nicfi_dir.glob("*.tif")
                        if (m := _MONTH_RE.search(p.name))})
        if self.month_filter:
            found = [m for m in found if any(m.startswith(f) for f in self.month_filter)]
        return found

    @property
    def all_months(self) -> List[str]:
        """Every NICFI month on disk, ignoring --months (for steps that need
        the whole series, e.g. the temporal post-check and composites)."""
        return Tile(self.tile_id, self.nicfi_dir, self.s2_dir, None, self.out_base).months

    @property
    def years(self) -> List[str]:
        return sorted({m[:4] for m in self.months})

    def nicfi_path(self, month: str) -> Path:
        hits = sorted(self.nicfi_dir.glob(f"*{month}.tif"))
        if len(hits) != 1:
            raise FileNotFoundError(f"expected exactly one NICFI file for {month} in "
                                    f"{self.nicfi_dir}, found {[h.name for h in hits]}")
        return hits[0]

    @property
    def has_s2(self) -> bool:
        return self.s2_dir.is_dir() and any(self.s2_frames())

    def s2_frames(self, month: Optional[str] = None) -> List[Path]:
        if not self.s2_dir.is_dir():
            return []
        pat = f"{month}-*.tif" if month else "*.tif"
        return sorted(p for p in self.s2_dir.glob(pat) if _DATE_RE.match(p.name))

    @property
    def out_root(self) -> Path:
        return self.out_base / self.tile_id

    def _mkdir(self, *sub: str) -> Path:
        d = self.out_root.joinpath(*sub)
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def cache_dir(self) -> Path:
        return self._mkdir("cache")

    def recon_dir(self, method: str) -> Path:
        return self._mkdir("reconstructed", method)

    def composite_dir(self, kind: str, source: str, stat: str) -> Path:
        return self._mkdir("composites", kind, stat, source)

    def fig_dir(self, stage: str) -> Path:
        return self._mkdir("figures", stage)


def add_tile_args(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    ap.add_argument("--tile", required=True, help="tile ID, e.g. D17")
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--nicfi-dir", type=Path, default=None,
                    help="default: <data-root>/<tile>")
    ap.add_argument("--s2-dir", type=Path, default=None,
                    help="default: <data-root>/<tile>_S2")
    ap.add_argument("--months", nargs="*", default=None,
                    help="only months starting with these prefixes, e.g. 2023 2024-05")
    ap.add_argument("--out-base", type=Path, default=PROJECT_DIR / "outputs" / PIPELINE_NAME)
    return ap


def tile_from_args(args) -> Tile:
    return Tile(
        tile_id=args.tile,
        nicfi_dir=args.nicfi_dir or args.data_root / args.tile,
        s2_dir=args.s2_dir or args.data_root / f"{args.tile}_S2",
        month_filter=args.months,
        out_base=args.out_base,
    )
