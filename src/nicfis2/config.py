"""
Tile config for the NICFI + Sentinel-2 reconstruction pipeline
(`scripts/amazon_nicfis2/`). Nothing here is specific to one tile: every
script takes `--tile <ID>` (plus optional directory overrides), and the
month list is discovered from the NICFI files actually on disk.

Expected layout (the defaults; override with --nicfi-dir / --s2-dir):
    <data-root>/<TILE>/        one NICFI monthly mosaic per month, any
                               filename ending in YYYY-MM.tif
    <data-root>/<TILE>_S2/     raw single-date Sentinel-2 frames named
                               YYYY-MM-DD.tif, all on one 10m grid, with
                               band descriptions (B2, B3, ... as delivered)

Kept separate from `src/sites.py`'s `Site` (4-band NICFI-only, shared by
the vietnam/amazon temporal-reconstruction pipeline): this pipeline is
genuinely dual-sensor.
"""
import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = Path("/mnt/warehouse/amazon")
PIPELINE_NAME = "amazon_nicfis2"

# ---- NICFI: Blue/Green/Red/NIR, same convention as src/sites.py ----
NICFI_BAND_NAMES = ["blue", "green", "red", "nir"]
REFLECTANCE_SCALE = 10000.0  # both sensors deliver DN = reflectance * 1e4

# Spectral correspondence used for harmonization + substitution. S2 band
# *positions* are not hardcoded -- they're read from each file's band
# descriptions (`s2_composite.band_index`).
NICFI_TO_S2_BAND = {"blue": "B2", "green": "B3", "red": "B4", "nir": "B8"}

_MONTH_RE = re.compile(r"(\d{4}-\d{2})\.tif$")
_DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})\.tif$")


@dataclass
class Tile:
    tile_id: str
    nicfi_dir: Path
    s2_dir: Path
    month_filter: Optional[List[str]] = None  # prefixes, e.g. ["2023", "2024-05"]

    @property
    def months(self) -> List[str]:
        found = sorted({m.group(1) for p in self.nicfi_dir.glob("*.tif")
                        if (m := _MONTH_RE.search(p.name))})
        if self.month_filter:
            found = [m for m in found if any(m.startswith(f) for f in self.month_filter)]
        return found

    def nicfi_path(self, month: str) -> Path:
        hits = sorted(self.nicfi_dir.glob(f"*{month}.tif"))
        if len(hits) != 1:
            raise FileNotFoundError(f"expected exactly one NICFI file for {month} in "
                                    f"{self.nicfi_dir}, found {[h.name for h in hits]}")
        return hits[0]

    def s2_frames(self, month: Optional[str] = None) -> List[Path]:
        pat = f"{month}-*.tif" if month else "*.tif"
        return sorted(p for p in self.s2_dir.glob(pat) if _DATE_RE.match(p.name))

    @property
    def out_root(self) -> Path:
        return PROJECT_DIR / "outputs" / PIPELINE_NAME / self.tile_id

    def _mkdir(self, sub: str) -> Path:
        d = self.out_root / sub
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def cache_dir(self) -> Path:
        return self._mkdir("cache")

    @property
    def fig_dir(self) -> Path:
        return self._mkdir("figures")

    @property
    def recon_dir(self) -> Path:
        return self._mkdir("reconstructed")


def add_tile_args(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    ap.add_argument("--tile", required=True, help="tile ID, e.g. D17")
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--nicfi-dir", type=Path, default=None,
                    help="default: <data-root>/<tile>")
    ap.add_argument("--s2-dir", type=Path, default=None,
                    help="default: <data-root>/<tile>_S2")
    ap.add_argument("--months", nargs="*", default=None,
                    help="only months starting with these prefixes, e.g. 2023 2024-05")
    return ap


def tile_from_args(args) -> Tile:
    return Tile(
        tile_id=args.tile,
        nicfi_dir=args.nicfi_dir or args.data_root / args.tile,
        s2_dir=args.s2_dir or args.data_root / f"{args.tile}_S2",
        month_filter=args.months,
    )
