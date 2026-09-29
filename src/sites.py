"""
Site configs for the NICFI monthly cloud-free reconstruction pipeline.

Everything else in `src/` (cloud_mask, phenology, robust_mask, inpaint,
compose, annual_composite) is site-agnostic -- it operates on plain numpy
arrays and knows nothing about file paths or tile IDs. This module is the
one place that knows *where a test site's data lives and how it's named*.
Adding a third test site means adding one `Site` entry here, nothing else.

The band-layout/reflectance-scale constants below are declared here too,
but they're deliberately *not* per-site fields: both current test sites
happen to use the same 4-band Blue/Green/Red/NIR NICFI analytic product at
the same ~10000x reflectance scaling. A genuinely different sensor would
need its own constants module, not just a new `Site` entry.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

# ---- Universal NICFI band layout + reflectance scale (see gdalinfo on
# either site's source tiles: band order is always 1=Blue,2=Green,3=Red,4=NIR) ----
BAND_NAMES = ["blue", "green", "red", "nir"]
BLUE, GREEN, RED, NIR = 0, 1, 2, 3
REFLECTANCE_SCALE = 10000.0

PROJECT_DIR = Path("/mnt/super/code/nicfi-reconstruct")


@dataclass
class Site:
    name: str                            # slug; also the outputs/<name>/ folder name
    data_dir: Path                       # where the source monthly GeoTIFFs live
    tile_id: str
    months: List[str]                    # date labels, exactly as used in filenames
    filename_fn: Callable[[str], str]    # month label -> source filename (not full path)
    aoi_row_off: int = 0
    aoi_col_off: int = 0
    aoi_size: Optional[int] = None       # None -> whole tile (see io_utils.read_aoi_stack)
    description: str = ""

    def tile_path(self, month: str) -> Path:
        return self.data_dir / self.filename_fn(month)

    @property
    def out_root(self) -> Path:
        return PROJECT_DIR / "outputs" / self.name

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

    @property
    def annual_dir(self) -> Path:
        return self._mkdir("annual")


def _vietnam_filename(month: str) -> str:
    return f"nicfi_{month}_00115_00023.tif"


def _amazon_filename(month: str) -> str:
    return f"D01_{month}.tif"


VIETNAM = Site(
    name="vietnam",
    data_dir=Path("/nfs/Planet/NICFI/vietnam"),
    tile_id="00115_00023",
    months=[f"2025-{m:02d}-01" for m in range(1, 13)],
    filename_fn=_vietnam_filename,
    aoi_row_off=4200, aoi_col_off=8200, aoi_size=2048,
    description=(
        "Northern Vietnam (Son La / Yen Bai mountains), a 2048x2048px AOI "
        "cropped from a much larger (20000x20015px) tile, 12 months of 2025. "
        "Mixed forest + cropland with real wet-season cloud/haze "
        "(Jun/Jul/Sep/Oct)."
    ),
)

AMAZON = Site(
    name="amazon",
    data_dir=Path("/mnt/warehouse/amazon/D01"),
    tile_id="D01",
    months=[f"{y}-{m:02d}" for y in range(2021, 2026) for m in range(1, 13)],
    filename_fn=_amazon_filename,
    aoi_row_off=0, aoi_col_off=0, aoi_size=None,  # whole tile, already ~AOI scale
    description=(
        "Amazon basin (D01 tile, ~63°W 0.5°N), the whole 2096x2109px tile "
        "(no cropping needed -- it's already close to AOI scale), 60 months "
        "(2021-2025). Rainforest with persistent, more year-to-year-variable "
        "cloud/haze than Vietnam."
    ),
)

SITES = {"vietnam": VIETNAM, "amazon": AMAZON}


def get_site(name: str) -> Site:
    try:
        return SITES[name]
    except KeyError:
        raise ValueError(f"Unknown site {name!r}, expected one of {list(SITES)}")
