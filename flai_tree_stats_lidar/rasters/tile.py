"""Process a single LAZ + DEM pair into per-parameter 10 m rasters.

One ``process_tile()`` call:
  - reads the LAZ, drops everything but ground + vegetation classes,
  - reads the matching DEM,
  - normalises point Z by subtracting DEM elevation,
  - bins points into 10 m square cells (the cell IS the plot),
  - iterates cells (optionally with an internal thread pool) and fills
    pre-allocated parameter arrays,
  - writes one GeoTIFF per parameter to ``<save_dir>/<param>/<basename>.tif``.

Each LAZ tile is processed in isolation: no neighbour reads, no buffer.
"""
import logging
import os
from concurrent.futures import ThreadPoolExecutor

import laspy
import numpy as np
import rasterio
from rasterio.transform import rowcol  # noqa: F401  (kept handy)

from ..metrices.canopy import metrics_canopy_density, metrics_interval, metrics_lad
from ..metrices.height import (
    metrics_basic, metrics_dispersion, metrics_lmoments,
    metrics_percabove, metrics_percentiles,
)
from ..metrices.kde import metrics_kde
from ..metrices.voxel import metrics_rumple

from .binning import bin_points, build_grid
from .dem_utils import (
    compute_idw_dem, dem_outputs_at_grid, normalize_points_z, read_dem,
)
from .keys import DEM_KEYS, SUPPORTED_GROUPS, collect_keys
from .voxel_fast import metrics_voxels_fast as metrics_voxels

logger = logging.getLogger('flai')


# ----- per-cell metric assembly -------------------------------------------

def _cell_metrics(x, y, z, groups):
    """Compute all requested metrics for the points of a single cell.

    Returns a dict keyed by parameter name.
    """
    out = {}
    if 'height' in groups:
        out.update(metrics_basic(z))
        out.update(metrics_percentiles(z))
        out.update(metrics_percabove(z))
        out.update(metrics_dispersion(z))
        out.update(metrics_lmoments(z))
    if 'canopy' in groups:
        out.update(metrics_canopy_density(z))
        out.update(metrics_interval(z))
        out.update(metrics_lad(z))
    if 'kde' in groups:
        out.update(metrics_kde(z))
    if 'voxel' in groups:
        out.update(metrics_rumple(x, y, z))
        out.update(metrics_voxels(x, y, z))
    return out


def _process_cell_range(cell_ids, n_cols, cell_starts, cell_counts,
                        xyz_sorted, point_groups, min_points,
                        param_arrays):
    """Compute metrics for a slice of cell linear indices.

    Writes results in-place into the per-parameter (n_rows, n_cols) arrays.
    """
    for lin in cell_ids:
        n = int(cell_counts[lin])
        if n < min_points:
            continue
        start = int(cell_starts[lin])
        end = start + n
        x = xyz_sorted[start:end, 0]
        y = xyz_sorted[start:end, 1]
        z = xyz_sorted[start:end, 2]

        # drop NaNs introduced by DEM lookup misses
        if np.isnan(z).any():
            mask = ~np.isnan(z)
            x, y, z = x[mask], y[mask], z[mask]
            if z.size < min_points:
                continue

        try:
            metrics = _cell_metrics(x, y, z, point_groups)
        except Exception as exc:
            logger.debug('cell %d metric failure: %s', lin, exc)
            continue

        row = lin // n_cols
        col = lin % n_cols
        for k, v in metrics.items():
            arr = param_arrays.get(k)
            if arr is None:
                continue
            try:
                arr[row, col] = v
            except (TypeError, ValueError):
                arr[row, col] = np.nan


# ----- output writing -----------------------------------------------------

def _output_paths(save_dir, group_to_keys, basename):
    paths = {}
    for keys in group_to_keys.values():
        for k in keys:
            paths[k] = os.path.join(save_dir, k, f'{basename}.tif')
    return paths


def _all_outputs_exist(paths):
    return all(os.path.isfile(p) for p in paths.values())


def _write_raster(path, arr, transform, crs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    profile = {
        'driver': 'GTiff',
        'height': arr.shape[0],
        'width': arr.shape[1],
        'count': 1,
        'dtype': 'float32',
        'transform': transform,
        'crs': crs,
        'nodata': np.float32(np.nan),
        'compress': 'deflate',
        'predictor': 3,
        'tiled': True,
        'blockxsize': 256,
        'blockysize': 256,
    }
    # narrow tiles: drop tiling if dimensions < blocksize
    if arr.shape[1] < 256 or arr.shape[0] < 256:
        profile.pop('tiled')
        profile.pop('blockxsize')
        profile.pop('blockysize')
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(arr.astype(np.float32, copy=False), 1)


# ----- main entry point ---------------------------------------------------

def process_tile(
    laz_path: str,
    dem_path=None,
    save_dir: str = None,
    pixel_size: float = 10.0,
    extract_groups=('dem', 'height', 'canopy', 'kde', 'voxel'),
    ground_class: int = 2,
    vegetation_classes=(3, 4, 5),
    min_points: int = 4,
    cell_workers: int = 4,
    skip_existing: bool = True,
    dem_pixel_size: float = 0.5,
    idw_power: float = 2.0,
    idw_k: int = 12,
    crs_override=None,
):
    """Compute all requested parameter rasters for a single LAZ tile.

    If ``dem_path`` is ``None`` an on-the-fly DEM is derived from the LAZ's
    ground-class points using k-NN inverse-distance weighting at
    ``dem_pixel_size`` (default 0.5 m); the LAZ header CRS is propagated to
    the outputs.

    ``crs_override`` (anything ``rasterio.crs.CRS.from_user_input`` accepts:
    EPSG int/string, WKT, proj4) wins over both the LAZ and DEM CRS; pass
    this when the LAZ has no CRS metadata.

    Returns a small status dict.
    """
    basename = os.path.splitext(os.path.basename(laz_path))[0]
    groups = [g for g in extract_groups if g in SUPPORTED_GROUPS]
    if not groups:
        raise ValueError(f'No valid metric groups in {extract_groups!r}')

    group_to_keys = collect_keys(groups)
    paths = _output_paths(save_dir, group_to_keys, basename)

    if skip_existing and _all_outputs_exist(paths):
        logger.info('[%s] all outputs exist, skipping', basename)
        return {'tile': basename, 'status': 'skipped'}

    logger.info('[%s] start (%d groups, %d params)',
                basename, len(groups), sum(len(v) for v in group_to_keys.values()))

    # --- read LAZ
    classes_keep = list(set([ground_class, *vegetation_classes]))
    veg_set = set(int(c) for c in vegetation_classes)

    las = laspy.read(laz_path)
    x_all = np.asarray(las.x, dtype=np.float64)
    y_all = np.asarray(las.y, dtype=np.float64)
    z_all = np.asarray(las.z, dtype=np.float64)
    cls_all = np.asarray(las.classification, dtype=np.int16)
    keep = np.isin(cls_all, classes_keep)
    x_all = x_all[keep]
    y_all = y_all[keep]
    z_all = z_all[keep]
    cls_all = cls_all[keep]
    # propagate the LAZ CRS to outputs when no DEM TIF is supplied
    _laz_crs = None
    if crs_override is not None:
        from rasterio.crs import CRS as RioCRS
        try:
            _laz_crs = RioCRS.from_user_input(crs_override)
        except Exception as exc:
            logger.warning('[%s] crs_override=%r could not be parsed: %s',
                           basename, crs_override, exc)
    if _laz_crs is None:
        try:
            pyproj_crs = las.header.parse_crs()
            if pyproj_crs is not None:
                from rasterio.crs import CRS as RioCRS
                _laz_crs = RioCRS.from_wkt(pyproj_crs.to_wkt())
        except Exception:
            _laz_crs = None
    del las

    # tile bbox from LAZ header (already filtered: use point extents)
    if x_all.size == 0:
        logger.warning('[%s] no usable points after class filter', basename)
        return {'tile': basename, 'status': 'empty'}

    bbox = (float(np.min(x_all)), float(np.min(y_all)),
            float(np.max(x_all)), float(np.max(y_all)))
    n_rows, n_cols, x_min, y_max, transform = build_grid(bbox, pixel_size)

    # --- obtain a DEM either from disk or by IDW'ing the ground returns
    if dem_path is not None:
        dem_arr, dem_transform, dem_crs, _ = read_dem(dem_path)
        if crs_override is not None and _laz_crs is not None:
            # explicit override takes precedence over the DEM TIF's CRS
            dem_crs = _laz_crs
    else:
        ground_mask = cls_all == ground_class
        if not np.any(ground_mask):
            logger.error('[%s] no ground points (class %d) found, cannot '
                         'build on-the-fly DEM', basename, ground_class)
            return {'tile': basename, 'status': 'error',
                    'error': 'no ground points'}

        xyz_ground = np.column_stack((
            x_all[ground_mask], y_all[ground_mask], z_all[ground_mask],
        ))
        # Make the DEM cover exactly the output grid (so reprojection has no
        # missing edge contribution). DEM grid origin matches the 10 m grid.
        dem_bbox = (x_min, y_max - n_rows * pixel_size,
                    x_min + n_cols * pixel_size, y_max)
        logger.info('[%s] computing on-the-fly DEM at %.2f m from %d '
                    'ground points', basename, dem_pixel_size, xyz_ground.shape[0])
        dem_arr, dem_transform = compute_idw_dem(
            xyz_ground, dem_bbox,
            pixel_size=dem_pixel_size,
            power=idw_power, k=idw_k,
            workers=max(1, int(cell_workers)),
        )
        dem_crs = _laz_crs
        del xyz_ground

    # --- normalise Z for vegetation
    veg_mask = np.isin(cls_all, list(veg_set))
    xyz_veg = np.column_stack((x_all[veg_mask], y_all[veg_mask], z_all[veg_mask]))
    del x_all, y_all, z_all, cls_all

    point_groups = [g for g in groups if g != 'dem']
    has_point_metrics = bool(point_groups)

    if has_point_metrics and xyz_veg.shape[0] > 0:
        xyz_veg = normalize_points_z(xyz_veg, dem_arr, dem_transform)

    # --- dem group: derive slope/aspect/eastness/northness at native resolution,
    #     then average-resample onto the output grid (recover aspect from the
    #     averaged east/north). Done now so the native DEM can be released
    #     before the per-cell loop begins.
    if 'dem' in groups:
        dem_outputs = dem_outputs_at_grid(
            dem_arr, dem_transform, dem_crs,
            transform, (n_rows, n_cols),
        )
    else:
        dem_outputs = {}

    del dem_arr  # native-res DEM no longer needed

    # --- bin vegetation points into 10 m cells
    if has_point_metrics and xyz_veg.shape[0] > 0:
        order, cell_starts, cell_counts = bin_points(
            xyz_veg[:, :2], n_rows, n_cols, x_min, y_max, pixel_size,
        )
        xyz_sorted = xyz_veg[order]
    else:
        order = np.empty(0, dtype=np.int64)
        cell_starts = np.full(n_rows * n_cols, -1, dtype=np.int64)
        cell_counts = np.zeros(n_rows * n_cols, dtype=np.int64)
        xyz_sorted = np.empty((0, 3), dtype=np.float64)

    # --- pre-allocate per-parameter result arrays (point-based groups)
    param_arrays = {}
    for g, keys in group_to_keys.items():
        if g == 'dem':
            continue
        for k in keys:
            param_arrays[k] = np.full((n_rows, n_cols), np.nan, dtype=np.float32)

    # --- per-cell metric loop (only non-empty cells)
    if has_point_metrics:
        non_empty = np.where(cell_counts >= min_points)[0]
        logger.info('[%s] %d / %d cells with >= %d points',
                    basename, non_empty.size, n_rows * n_cols, min_points)

        if non_empty.size > 0:
            cw = max(1, int(cell_workers))
            if cw == 1 or non_empty.size < cw * 16:
                _process_cell_range(
                    non_empty, n_cols, cell_starts, cell_counts,
                    xyz_sorted, point_groups, min_points, param_arrays,
                )
            else:
                chunks = np.array_split(non_empty, cw)
                with ThreadPoolExecutor(max_workers=cw) as ex:
                    futs = [
                        ex.submit(
                            _process_cell_range,
                            chunk, n_cols, cell_starts, cell_counts,
                            xyz_sorted, point_groups, min_points, param_arrays,
                        )
                        for chunk in chunks if chunk.size > 0
                    ]
                    for fut in futs:
                        fut.result()

    # --- choose the CRS for output rasters: explicit override > DEM CRS > LAZ CRS
    crs_out = dem_crs if dem_crs is not None else _laz_crs
    if crs_out is None:
        logger.warning('[%s] no CRS could be determined (LAZ has no SRS VLR '
                       'and no --crs override); output rasters will lack a CRS',
                       basename)

    for k in DEM_KEYS:
        if k in dem_outputs:
            _write_raster(paths[k], dem_outputs[k], transform, crs_out)

    for k, arr in param_arrays.items():
        _write_raster(paths[k], arr, transform, crs_out)

    logger.info('[%s] done', basename)
    return {'tile': basename, 'status': 'ok',
            'n_cells': int(n_rows * n_cols)}
