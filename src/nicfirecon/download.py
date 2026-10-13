"""
Stage 0, "download": fetch a tile's inputs from Google Earth Engine, in
exactly the layout the rest of the pipeline reads (`config.py`):

    <data-root>/<TILE>/<TILE>_YYYY-MM.tif   NICFI monthly basemap, 4 bands B,G,R,N
    <data-root>/<TILE>_S2/YYYY-MM-DD.tif    raw Sentinel-2 L2A frames (optional)

- NICFI: `projects/planet-nicfi/assets/basemaps/{americas,africa,asia}` (the
  region is picked from the AOI's longitude unless given), one mosaic per
  month, on NICFI's native grid -- EPSG:3857 at 4.777 m, snapped to the
  global basemap grid -- so nothing is resampled.
- Sentinel-2: `COPERNICUS/S2_SR_HARMONIZED` (no +1000 offset), every
  acquisition day intersecting the AOI, same-day granules mosaicked, bands
  B2..B12 (10 bands) as uint16 on an EPSG:4326 grid of 0.0000898 deg
  (~10 m), *unmasked* -- the pipeline does its own cloud masking.

Core download logic adapted from /mnt/super/code/nicfi_download.py (chunked,
parallel `getDownloadURL` with retries, merge, skip what's already on
disk). Differences: chunks are pixel blocks on the exact output grid (fixed
`crs_transform` + dimensions), so the merge is a plain array copy; outputs
are named for the pipeline; masked pixels are written as 0 (= nodata).

Tiles from a grid file (`--grid FILE --grid-id "33,0"`) can be written with
the existing NICFI archive naming, nicfi_<YYYY-MM>-01_<col:05d>_<row:05d>
(`--naming archive`), and converted to JPEG2000 (`--format jp2`, lossless by
default and verified before the GeoTIFF is deleted).

Needs an authenticated Earth Engine account: run `earthengine authenticate`
once, and pass `--ee-project <cloud project>`.
"""
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import rasterio
import requests
from rasterio.transform import Affine
from rasterio.warp import transform_bounds

from . import config as cfg

NICFI_COLLECTION = "projects/planet-nicfi/assets/basemaps/{region}"
NICFI_BANDS = ["B", "G", "R", "N"]
NICFI_CRS = "EPSG:3857"
NICFI_RES = 4.777314267159966          # native basemap pixel size, m
WEB_MERCATOR_ORIGIN = -20037508.342789244

S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
S2_CRS = "EPSG:4326"
S2_RES = 0.000089831528412             # deg, = 10 m at the equator

CHUNK_PX = 1024                        # 1024^2 x 10 bands x 2 B = 21 MB < GEE's 32 MB limit
MAX_WORKERS = 8
MAX_RETRIES = 3


def init_ee(project: Optional[str]):
    """Initialize Earth Engine. If the stored login is rejected because the
    library's default OAuth scopes don't match what the account granted
    ("invalid_scope" -- seen after re-authenticating with another account),
    fall back to the same stored refresh token without requesting scopes."""
    import ee
    try:
        ee.Initialize(project=project) if project else ee.Initialize()
        return ee
    except Exception as first:
        try:
            import google.oauth2.credentials
            stored = json.loads((Path.home() / ".config/earthengine/credentials").read_text())
            creds = google.oauth2.credentials.Credentials(
                None, refresh_token=stored["refresh_token"], client_id=stored.get("client_id"),
                client_secret=stored.get("client_secret"), token_uri="https://oauth2.googleapis.com/token")
            ee.Initialize(credentials=creds, project=project)
            return ee
        except Exception:
            raise RuntimeError(f"Earth Engine init failed ({first}). Run `earthengine authenticate` "
                               "once and pass --ee-project <your cloud project>.") from first


# ------------------------------------------------------------------- AOI
def aoi_bounds(bbox: Optional[Sequence[float]] = None, aoi: Optional[Path] = None,
               aoi_id_field: Optional[str] = None, aoi_id=None) -> Tuple[float, float, float, float]:
    """(min_lon, min_lat, max_lon, max_lat) from --bbox, or from a polygon
    file (one feature, or the one whose `aoi_id_field` == `aoi_id`)."""
    if bbox:
        return tuple(float(v) for v in bbox)
    if aoi is None:
        raise ValueError("give --bbox MIN_LON MIN_LAT MAX_LON MAX_LAT or --aoi FILE")
    import geopandas as gpd
    gdf = gpd.read_file(aoi).to_crs("EPSG:4326")
    if aoi_id_field:
        gdf = gdf[gdf[aoi_id_field].astype(str) == str(aoi_id)]
    if len(gdf) != 1:
        raise ValueError(f"{aoi}: expected exactly one AOI feature, found {len(gdf)} "
                         f"(use --aoi-id-field to pick one; the value is the tile ID)")
    return tuple(gdf.geometry.iloc[0].bounds)


def grid_tile(grid: Path, grid_id: str, id_field: str = "id"):
    """(bounds lon/lat, (col, row)) of the grid feature whose `id_field` ==
    grid_id, e.g. "33,0" in /nfs/Planet/grids_lc/all_grids_1deg_lc.geojson."""
    bounds = aoi_bounds(aoi=grid, aoi_id_field=id_field, aoi_id=grid_id)
    col, row = (int(v) for v in str(grid_id).split(","))
    return bounds, (col, row)


def grid_nicfi_pattern(col: int, row: int, ext: str = "jp2") -> str:
    """Naming of the existing NICFI archives, e.g. nicfi_2023-05-01_00033_00000.jp2
    (negative ids keep their sign: _-0002)."""
    return f"nicfi_{{month}}-01_{col:05d}_{row:05d}.{ext}"


def nicfi_region(bounds) -> str:
    lon = (bounds[0] + bounds[2]) / 2
    return "americas" if lon < -30 else "africa" if lon < 60 else "asia"


def snapped_grid(bounds_lonlat, crs: str, res: float, origin_x=0.0, origin_y=0.0):
    """(Affine, width, height) of a north-up grid with pixel size `res`
    covering the AOI, its corners snapped to multiples of `res` from the
    given origin (the global grid the source data lives on)."""
    x0, y0, x1, y1 = (transform_bounds("EPSG:4326", crs, *bounds_lonlat) if crs != "EPSG:4326"
                      else bounds_lonlat)
    eps = 1e-6  # px; an AOI edge exactly on a pixel boundary must not round outward
    c0 = np.floor((x0 - origin_x) / res + eps)
    c1 = np.ceil((x1 - origin_x) / res - eps)
    r0 = np.floor((origin_y - y1) / res + eps)   # rows count down from the origin
    r1 = np.ceil((origin_y - y0) / res - eps)
    transform = Affine(res, 0, origin_x + c0 * res, 0, -res, origin_y - r0 * res)
    return transform, int(c1 - c0), int(r1 - r0)


def nicfi_grid(bounds):
    return snapped_grid(bounds, NICFI_CRS, NICFI_RES, WEB_MERCATOR_ORIGIN, -WEB_MERCATOR_ORIGIN)


def s2_grid(bounds):
    return snapped_grid(bounds, S2_CRS, S2_RES)


# ------------------------------------------------------------- download core
def _fetch(image, crs, chunk_tr, w, h, n_retries=MAX_RETRIES) -> np.ndarray:
    params = {"crs": crs, "crs_transform": list(chunk_tr)[:6], "dimensions": f"{w}x{h}",
              "format": "GEO_TIFF", "filePerBand": False}
    for attempt in range(1, n_retries + 2):
        if attempt > 1:
            time.sleep(min(60, 10 * (attempt - 1)))
        try:
            url = image.getDownloadURL(params)
            resp = requests.get(url, timeout=300)
        except Exception as exc:  # GEE or network error: retry
            last = exc
            continue
        if resp.ok:
            with rasterio.MemoryFile(resp.content) as mf, mf.open() as src:
                return src.read()
        last = f"HTTP {resp.status_code}: {resp.content[:200]!r}"
        if resp.status_code in {401, 403, 404}:
            break
    raise RuntimeError(f"chunk failed after {attempt} attempts: {last}")


def download_image(image, crs: str, transform: Affine, width: int, height: int,
                   out_path: Path, band_names: List[str], dtype="uint16", workers=MAX_WORKERS,
                   log=None):
    """Download `image` onto the exact grid (crs, transform, width, height)
    in CHUNK_PX blocks, in parallel, writing each block straight into one
    tiled GeoTIFF on disk (a 1-degree NICFI tile is ~3.5 GB per month --
    never held in memory). Resumable: finished blocks are listed in a
    sidecar file, and an interrupted month continues where it stopped. Only
    a complete file is renamed to `out_path`."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".part.tif")
    done_file = out_path.with_name(out_path.name + ".part.json")
    blocks = [(r, c, min(CHUNK_PX, height - r), min(CHUNK_PX, width - c))
              for r in range(0, height, CHUNK_PX) for c in range(0, width, CHUNK_PX)]
    done = set()
    if tmp.exists() and done_file.exists():
        done = {tuple(b) for b in json.loads(done_file.read_text())}
    else:
        with rasterio.open(tmp, "w", driver="GTiff", height=height, width=width,
                           count=len(band_names), dtype=dtype, crs=crs, transform=transform,
                           compress="deflate", predictor=2, tiled=True, blockxsize=512,
                           blockysize=512, BIGTIFF="IF_SAFER") as dst:
            dst.descriptions = tuple(band_names)
    todo = [b for b in blocks if tuple(b) not in done]

    def job(b):
        r, c, h, w = b
        return b, _fetch(image, crs, transform * Affine.translation(c, r), w, h)

    with rasterio.open(tmp, "r+") as dst, ThreadPoolExecutor(max_workers=workers) as ex:
        for i, fut in enumerate(as_completed([ex.submit(job, b) for b in todo]), start=1):
            (r, c, h, w), arr = fut.result()
            dst.write(arr[:, :h, :w].astype(dtype), window=rasterio.windows.Window(c, r, w, h))
            done.add((r, c, h, w))
            if i % 25 == 0 or i == len(todo):
                done_file.write_text(json.dumps(sorted(done)))
                if log:
                    log(f"      {len(done)}/{len(blocks)} blocks")
    tmp.rename(out_path)
    done_file.unlink(missing_ok=True)


def to_jp2(tif: Path, jp2: Path, mode: str = "lossless", log=None):
    """GeoTIFF -> JPEG2000 (GDAL JP2OpenJPEG via rasterio), then delete the GeoTIFF.
    lossless: reversible 5/3 wavelet, verified pixel-for-pixel on sample
    windows before the GeoTIFF is removed. lossy: irreversible, QUALITY=80,
    the settings of the existing archive conversions -- smaller, not exact.
    Both with 6 resolution levels (built-in overviews)."""
    from rasterio.shutil import copy as rio_copy
    tmp = jp2.with_name(jp2.stem + ".part.jp2")  # must end in .jp2, see below
    # lossy = the settings of the existing archive conversions
    # (/nfs/Planet/NICFI/convert_jp2_tree_explorer.py): QUALITY 80, irreversible
    co = dict(REVERSIBLE="YES", QUALITY="100", YCBCR420="NO") if mode == "lossless" else \
        dict(REVERSIBLE="NO", QUALITY="80", YCBCR420="NO")
    # rasterio's bundled GDAL has the OpenJPEG driver (the conda gdal_translate here doesn't)
    rio_copy(str(tif), str(tmp), driver="JP2OpenJPEG", BLOCKXSIZE="1024", BLOCKYSIZE="1024",
             RESOLUTIONS="6", NUM_THREADS="ALL_CPUS", **co)
    with rasterio.open(tif) as a, rasterio.open(tmp) as b:
        if (a.width, a.height, a.count) != (b.width, b.height, b.count):
            raise RuntimeError(f"{jp2.name}: JP2 shape differs from the GeoTIFF -- kept both")
        if b.crs != a.crs or not np.allclose(tuple(b.transform)[:6], tuple(a.transform)[:6]):
            raise RuntimeError(f"{jp2.name}: JP2 lost the georeferencing -- kept both")
    if mode == "lossless":
        with rasterio.open(tif) as a, rasterio.open(tmp) as b:
            rng = np.random.default_rng(0)
            for _ in range(8):
                r, c = rng.integers(0, max(a.height - 512, 1)), rng.integers(0, max(a.width - 512, 1))
                win = rasterio.windows.Window(c, r, min(512, a.width), min(512, a.height))
                if not np.array_equal(a.read(window=win), b.read(window=win)):
                    raise RuntimeError(f"{jp2.name}: lossless JP2 differs from the GeoTIFF -- kept both")
    # GDAL may keep part of the georeferencing in a PAM sidecar (<file>.aux.xml);
    # it must move with the file (a first version lost the CRS on rename)
    aux = tmp.with_name(tmp.name + ".aux.xml")
    tmp.rename(jp2)
    if aux.exists():
        aux.rename(jp2.with_name(jp2.name + ".aux.xml"))
    if log:
        log(f"      -> {jp2.name}: {tif.stat().st_size / 1e9:.2f} GB tif -> {jp2.stat().st_size / 1e9:.2f} GB jp2")
    tif.unlink()


# ------------------------------------------------------------------ sensors
def _month_range(start: str, end: str) -> List[str]:
    y, m = map(int, start.split("-"))
    ye, me = map(int, end.split("-"))
    out = []
    while (y, m) <= (ye, me):
        out.append(f"{y}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def _next_month(month: str) -> str:
    y, m = map(int, month.split("-"))
    return f"{y + 1}-01" if m == 12 else f"{y}-{m + 1:02d}"


def run_download_nicfi(ee, tile: cfg.Tile, bounds, start: str, end: str,
                       region: Optional[str] = None, out_format: str = "tif",
                       jp2_mode: str = "lossless", log=print):
    """Files are named by tile.nicfi_pattern ({month} = YYYY-MM) if it has no
    wildcard, else <tile>_<YYYY-MM>.tif; out_format "jp2" converts each
    finished month to JPEG2000 and removes the GeoTIFF."""
    region = region or nicfi_region(bounds)
    col = ee.ImageCollection(NICFI_COLLECTION.format(region=region))
    transform, width, height = nicfi_grid(bounds)
    pattern = tile.nicfi_pattern if "*" not in tile.nicfi_pattern else f"{tile.tile_id}_{{month}}.tif"
    log(f"  NICFI ({region}): {width}x{height} px at {NICFI_RES:.2f} m (EPSG:3857) -> "
        f"{tile.nicfi_dir}/{pattern}")
    n_new = n_skip = n_missing = 0
    for month in _month_range(start, end):
        final = tile.nicfi_dir / pattern.format(month=month)
        if final.exists():
            n_skip += 1
            continue
        tif = final if final.suffix == ".tif" else final.with_suffix(".tif")
        if not tif.exists():
            monthly = col.filterDate(f"{month}-01", f"{_next_month(month)}-01")
            if monthly.size().getInfo() == 0:
                n_missing += 1
                log(f"    {month}: no NICFI mosaic on GEE (pre-2020-09 basemaps are biannual)")
                continue
            log(f"    {month}: downloading")
            image = monthly.mosaic().select(NICFI_BANDS).unmask(0).toUint16()
            download_image(image, NICFI_CRS, transform, width, height, tif, NICFI_BANDS, log=log)
        if final.suffix == ".jp2":
            to_jp2(tif, final, jp2_mode, log)
        n_new += 1
    log(f"  NICFI: {n_new} downloaded, {n_skip} already on disk, {n_missing} not available")


def run_download_s2(ee, tile: cfg.Tile, bounds, start: str, end: str,
                    max_cloud: float = 100.0, log=print):
    rect = ee.Geometry.Rectangle(list(bounds), "EPSG:4326", False)
    col = (ee.ImageCollection(S2_COLLECTION).filterBounds(rect)
           .filterDate(f"{start}-01", f"{_next_month(end)}-01"))
    if max_cloud < 100:
        col = col.filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", max_cloud))
    days = sorted(set(col.aggregate_array("system:time_start")
                      .map(lambda t: ee.Date(t).format("YYYY-MM-dd")).getInfo()))
    transform, width, height = s2_grid(bounds)
    log(f"  Sentinel-2: {len(days)} acquisition days, {width}x{height} px at ~10 m -> {tile.s2_dir}")
    n_new = 0
    for day in days:
        out = tile.s2_dir / f"{day}.tif"
        if out.exists():
            continue
        d = ee.Date(day)
        image = (col.filterDate(d, d.advance(1, "day")).mosaic().select(S2_BANDS)
                 .unmask(0).toUint16())
        download_image(image, S2_CRS, transform, width, height, out, S2_BANDS)
        n_new += 1
        if n_new % 20 == 0:
            log(f"    {n_new} frames downloaded (latest {day})")
    log(f"  Sentinel-2: {n_new} downloaded, {len(days) - n_new} already on disk")


def run_download(tile: cfg.Tile, start: str, end: str, sensors=("nicfi", "s2"),
                 bbox=None, aoi=None, aoi_id_field=None, ee_project=None, nicfi_region_name=None,
                 s2_max_cloud: float = 100.0, grid=None, grid_id=None, grid_id_field="id",
                 out_format: str = "tif", jp2_mode: str = "lossless", log=print):
    if grid is not None:
        bounds, _ = grid_tile(grid, grid_id, grid_id_field)
    else:
        bounds = aoi_bounds(bbox, aoi, aoi_id_field, tile.tile_id)
    log(f"  AOI (lon/lat): {', '.join(f'{v:.5f}' for v in bounds)}")
    ee = init_ee(ee_project)
    if "nicfi" in sensors:
        run_download_nicfi(ee, tile, bounds, start, end, nicfi_region_name, out_format, jp2_mode, log)
    if "s2" in sensors:
        run_download_s2(ee, tile, bounds, start, end, s2_max_cloud, log)
    (tile.cache_dir / "download.json").write_text(json.dumps(
        {"bounds_lonlat": bounds, "start": start, "end": end, "sensors": list(sensors)}, indent=1))
