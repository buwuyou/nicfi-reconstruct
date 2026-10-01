"""
Windowed I/O for NICFI monthly mosaics.

Source tiles range from ~2100px (Amazon, already AOI-scale) to ~20000px
(Vietnam, cropped to an AOI). Every read goes through rasterio's windowed
read against the on-disk COG, so peak memory is bounded by the AOI size,
not the source file size -- this is what lets the same code handle a
2048x2048 crop and a 2096x2109 whole-tile read the same way, and what
`iter_tiling_windows` documents scaling further to production tiling of a
still-larger tile without ever holding it all in RAM.
"""
from dataclasses import dataclass

import numpy as np
import rasterio
from rasterio.windows import Window

from . import sites


@dataclass
class AOIStack:
    """Monthly stack for one site's AOI: shape (T, C, H, W), plus georeferencing."""
    data: np.ndarray          # (T, 4, H, W) float32, raw DN (not yet scaled)
    months: list
    transform: "rasterio.Affine"
    crs: object

    @property
    def n_months(self):
        return self.data.shape[0]


def read_aoi_stack(site: sites.Site) -> AOIStack:
    """Read every monthly tile for `site`, windowed to its AOI.

    `site.aoi_size is None` means "read the whole tile" (used for Amazon,
    whose tiles are already close to AOI scale) -- offsets still apply, so
    a whole-tile site can still use a nonzero row/col_off if ever needed.
    """
    arrs = []
    transform = None
    crs = None
    for month in site.months:
        path = site.tile_path(month)
        with rasterio.open(path) as src:
            if site.aoi_size is None:
                win = Window(site.aoi_col_off, site.aoi_row_off,
                             src.width - site.aoi_col_off, src.height - site.aoi_row_off)
            else:
                win = Window(site.aoi_col_off, site.aoi_row_off, site.aoi_size, site.aoi_size)
            arr = src.read(window=win).astype(np.float32)  # (4,H,W)
            if transform is None:
                transform = src.window_transform(win)
                crs = src.crs
        arrs.append(arr)
    data = np.stack(arrs, axis=0)  # (T,4,H,W)
    return AOIStack(data=data, months=list(site.months), transform=transform, crs=crs)


def read_full(path):
    """Read an entire GeoTIFF (no AOI windowing, no site config -- used by the
    D02 NICFI+Sentinel-2 fusion pipeline, which reads one tile/mosaic at a
    time rather than a whole site stack). Returns (data (C,H,W) float32,
    transform, crs)."""
    with rasterio.open(path) as src:
        return src.read().astype(np.float32), src.transform, src.crs


def iter_tiling_windows(width, height, tile=1024, overlap=64):
    """Yield overlapping (row_off, col_off, h, w) windows covering a full
    tile in bounded-memory chunks. Not needed for either current test site
    (both are already AOI-scale or a manageable crop), but this is how the
    pipeline would scale to a much larger production tile without ever
    holding it all in RAM: process one chunk at a time, write results
    incrementally with rasterio in 'r+' windowed mode, and feather-blend
    the overlap regions.
    """
    step = tile - overlap
    for row_off in range(0, height, step):
        h = min(tile, height - row_off)
        for col_off in range(0, width, step):
            w = min(tile, width - col_off)
            yield row_off, col_off, h, w


def to_reflectance(data: np.ndarray) -> np.ndarray:
    return data / sites.REFLECTANCE_SCALE


def save_geotiff(path, arr, transform, crs, dtype="int16", nodata=None):
    """arr: (C,H,W)"""
    import rasterio as rio
    c, h, w = arr.shape
    with rio.open(
        path, "w", driver="GTiff", height=h, width=w, count=c,
        dtype=dtype, crs=crs, transform=transform, compress="lzw",
        nodata=nodata,
    ) as dst:
        dst.write(arr.astype(dtype))