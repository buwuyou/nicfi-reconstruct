"""
Site config for the D02 NICFI + Sentinel-2 fusion pipeline.

Unlike `src/sites.py`'s `Site` (4-band NICFI-only, shared by the vietnam/
amazon temporal-reconstruction pipeline), this pipeline pairs NICFI with a
second sensor and produces a 10-band output -- deliberately a separate
config rather than bolting dual-sensor fields onto `Site`, per that
module's own docstring ("A genuinely different sensor would need its own
constants module, not just a new `Site` entry").
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

PROJECT_DIR = Path("/mnt/super/code/nicfi-reconstruct")

# ---- NICFI: Blue/Green/Red/NIR, same convention as src/sites.py ----
NICFI_BAND_NAMES = ["blue", "green", "red", "nir"]
NICFI_REFLECTANCE_SCALE = 10000.0

# ---- Sentinel-2: band order as delivered in D02_S2 (verified with gdalinfo:
# Band 1..10 Description = B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12; already resampled
# to one common 10m grid, no B1/B9/B10, no SCL/cloud-mask band) ----
S2_BAND_NAMES = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
S2_REFLECTANCE_SCALE = 10000.0
S2_INDEX = {b: i for i, b in enumerate(S2_BAND_NAMES)}

# Spectral correspondence used for harmonization + gap-fill substitution.
NICFI_TO_S2_BAND = {"blue": "B2", "green": "B3", "red": "B4", "nir": "B8"}


def _amazon_d02_months() -> List[str]:
    return [f"{y}-{m:02d}" for y in range(2021, 2026) for m in range(1, 13)]


# The person who prepared D02_S2 masked it to forest land only (not just
# cloud/shadow) -- that's what the ~18.5% of the tile that's invalid in
# almost every month turns out to be, confirmed directly by checking that
# the invalid fraction is nearly constant across months with wildly
# different frame counts, which a cloud mask alone wouldn't produce. A
# reprocessed, unmasked extract is expected eventually; until then, restrict
# to one full year for testing rather than fighting every month's coverage
# quirks. 2022 was picked because every one of its 12 months sits at a
# consistent ~80-82% valid (no missing or near-empty months, unlike every
# other year in this archive -- see docs/amazon_d02_s2fusion.md).
TEST_YEAR = 2022


def _test_year_months() -> List[str]:
    return [m for m in _amazon_d02_months() if m.startswith(str(TEST_YEAR))]


@dataclass
class D02Site:
    name: str = "amazon_d02"
    nicfi_dir: Path = Path("/mnt/warehouse/amazon/D02")
    s2_dir: Path = Path("/mnt/warehouse/amazon/D02_S2")
    tile_id: str = "D02"
    months: List[str] = field(default_factory=_amazon_d02_months)

    def nicfi_path(self, month: str) -> Path:
        return self.nicfi_dir / f"D02_L1_{month}.tif"

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


D02 = D02Site()
# Same name/output dir as D02 (not a separate site) -- just fewer months, so
# cache built for these months is reused if `months` is widened back later.
D02_TEST_YEAR = D02Site(months=_test_year_months())
