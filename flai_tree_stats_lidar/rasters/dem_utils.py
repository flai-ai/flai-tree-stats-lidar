"""DEM read and aggregation helpers for the raster pipeline.

The input DEM is at native resolution (e.g. 0.5 m) covering the same tile as
the LAZ file. We:
  1. read it as-is for point Z normalisation,
  2. compute slope / aspect / eastness / northness at native resolution
     (preserves sub-pixel terrain complexity),
  3. resample those native fields onto the output grid (e.g. 10 m) using
     average resampling; aspect is recovered from the averaged eastness and
     northness so circular averaging is handled correctly.

Why we don't average the DEM first and then derive slope: averaging the DEM
removes sub-pixel variation, so the derived 10 m slope severely under-
estimates the true cell-mean slope on rough terrain. See the synthetic
ground-truth comparison in the README.

Slope / aspect sign convention matches the existing repo code in
utils.raster_data (aspect is the direction the slope faces: 0=N, 90=E,
180=S, 270=W), but uses ``np.gradient`` (one-sided diffs at edges) instead of
``np.roll`` to avoid wrap-around artifacts.
"""
import numpy as np
import rasterio
from rasterio.transform import from_origin, rowcol


def compute_idw_dem(xyz_ground, bbox, pixel_size=0.5, power=2.0, k=12,
                    workers=1):
    """Build a DEM at ``pixel_size`` from ground-class LiDAR returns using
    k-nearest-neighbour inverse-distance weighting.

    Parameters
    ----------
    xyz_ground : (N, 3) array
        Ground point cloud (columns x, y, z).
    bbox : tuple
        ``(x_min, y_min, x_max, y_max)`` covering the desired DEM extent.
        Snapped outward to a ``pixel_size``-aligned grid.
    pixel_size : float
        Output DEM cell size in input units.
    power : float
        IDW exponent (commonly 2.0).
    k : int
        Number of nearest neighbours used per cell.
    workers : int
        Threads for the cKDTree query; -1 = all CPUs. Capped at 1 for very
        small queries.

    Returns
    -------
    (dem_array, transform)
        ``dem_array`` is float32, shape ``(n_rows, n_cols)``. ``transform``
        is upper-left origin, ``+x`` east, ``-y`` south.
    """
    from scipy.spatial import cKDTree

    if xyz_ground.shape[0] == 0:
        raise ValueError("No ground points provided for IDW DEM creation")

    x0 = float(np.floor(bbox[0] / pixel_size) * pixel_size)
    y0 = float(np.floor(bbox[1] / pixel_size) * pixel_size)
    x1 = float(np.ceil(bbox[2] / pixel_size) * pixel_size)
    y1 = float(np.ceil(bbox[3] / pixel_size) * pixel_size)
    n_cols = int(round((x1 - x0) / pixel_size))
    n_rows = int(round((y1 - y0) / pixel_size))

    xs = x0 + (np.arange(n_cols) + 0.5) * pixel_size
    ys = y1 - (np.arange(n_rows) + 0.5) * pixel_size
    gx, gy = np.meshgrid(xs, ys)
    query = np.column_stack((gx.ravel(), gy.ravel()))

    tree = cKDTree(xyz_ground[:, :2])
    k_eff = min(int(k), xyz_ground.shape[0])

    try:
        dists, idxs = tree.query(query, k=k_eff, workers=int(workers))
    except TypeError:
        # scipy < 1.6 -- no ``workers`` kwarg
        dists, idxs = tree.query(query, k=k_eff)

    if k_eff == 1:
        dists = dists[:, None]
        idxs = idxs[:, None]

    dists = np.where(dists < 1e-12, 1e-12, dists)
    weights = 1.0 / dists ** power
    values = (
        np.sum(weights * xyz_ground[:, 2][idxs], axis=1)
        / np.sum(weights, axis=1)
    )

    dem_arr = values.reshape(n_rows, n_cols).astype(np.float32)
    transform = from_origin(x0, y1, pixel_size, pixel_size)
    return dem_arr, transform


def read_dem(dem_path):
    """Return (array, transform, crs, nodata) for a single-band DEM."""
    with rasterio.open(dem_path) as src:
        arr = src.read(1).astype(np.float32, copy=False)
        nodata = src.nodata
        if nodata is not None:
            arr = np.where(arr == nodata, np.nan, arr)
        return arr, src.transform, src.crs, nodata


def _block_average(src, src_transform, dst_transform, dst_shape):
    """NaN-aware block-mean of ``src`` onto an aligned destination grid.

    Alignment requirements:
      * DEM pixel size divides the output pixel size evenly (integer factor),
      * the offset between the DEM upper-left and the output upper-left is
        an integer multiple of the DEM pixel size (so pixel boundaries
        coincide). The two grids do **not** need to share an origin and the
        DEM does **not** need to cover the full output extent.

    Output cells that fall partially or fully outside the DEM are filled
    from whatever sub-pixels do overlap; cells with zero overlap come out
    as NaN. CRS is irrelevant -- this is a coordinate-free block average.
    """
    src_px_x = float(src_transform[0])
    src_px_y = float(-src_transform[4])
    dst_px_x = float(dst_transform[0])
    dst_px_y = float(-dst_transform[4])

    # 1) integer scale factor
    fx_f = dst_px_x / src_px_x
    fy_f = dst_px_y / src_px_y
    if abs(fx_f - round(fx_f)) > 1e-9 or abs(fy_f - round(fy_f)) > 1e-9:
        raise ValueError(
            f'DEM pixel size does not evenly divide output pixel size: '
            f'src=({src_px_x}, {src_px_y}) dst=({dst_px_x}, {dst_px_y}); '
            f'factors=({fx_f}, {fy_f}). Use a DEM whose pixel size is an '
            f'integer divisor of --pixel-size, or omit -d to compute a '
            f'tile-aligned DEM on the fly.')
    fx, fy = int(round(fx_f)), int(round(fy_f))

    # 2) integer-pixel offset (pixel boundaries coincide)
    src_x0 = float(src_transform[2])
    src_y0 = float(src_transform[5])
    dst_x0 = float(dst_transform[2])
    dst_y0 = float(dst_transform[5])

    off_col_f = (dst_x0 - src_x0) / src_px_x   # output starts here in src cols
    off_row_f = (src_y0 - dst_y0) / src_px_y   # output starts here in src rows
    tol = 1e-6
    if abs(off_col_f - round(off_col_f)) > tol or abs(off_row_f - round(off_row_f)) > tol:
        raise ValueError(
            f'DEM grid is not pixel-aligned with the output grid: '
            f'origin offset in src pixels = ({off_col_f}, {off_row_f}) '
            f'(must be integer). '
            f'DEM upper-left = ({src_x0}, {src_y0}), '
            f'output upper-left = ({dst_x0}, {dst_y0}), '
            f'DEM pixel = ({src_px_x}, {src_px_y}).')
    off_col, off_row = int(round(off_col_f)), int(round(off_row_f))

    # 3) build a NaN-padded view of src that exactly covers the output extent
    dst_rows, dst_cols = dst_shape
    src_h, src_w = src.shape
    needed_h = dst_rows * fy
    needed_w = dst_cols * fx

    padded = np.full((needed_h, needed_w), np.nan, dtype=np.float32)
    # overlap region in src indices
    s_r0 = max(0, off_row)
    s_r1 = min(src_h, off_row + needed_h)
    s_c0 = max(0, off_col)
    s_c1 = min(src_w, off_col + needed_w)
    if s_r1 > s_r0 and s_c1 > s_c0:
        # corresponding indices in the padded buffer
        p_r0 = s_r0 - off_row
        p_r1 = s_r1 - off_row
        p_c0 = s_c0 - off_col
        p_c1 = s_c1 - off_col
        padded[p_r0:p_r1, p_c0:p_c1] = src[s_r0:s_r1, s_c0:s_c1]

    # 4) NaN-aware block mean via manual sum / count
    blocks = padded.reshape(dst_rows, fy, dst_cols, fx)
    mask = ~np.isnan(blocks)
    counts = mask.sum(axis=(1, 3))
    sums = np.where(mask, blocks, 0.0).sum(axis=(1, 3))
    out = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    return out.astype(np.float32)


def slope_aspect_arrays(dem_arr, pixel_size):
    """Slope (deg), aspect (deg, 0-360, NaN on flat), eastness, northness.

    Operates at the resolution of ``dem_arr``; sign convention matches
    ``utils.raster_data.aspect``. Uses ``np.gradient`` so edges use one-sided
    finite differences rather than wrapping.
    """
    # axis 0 is rows (y, row index increases southward),
    # axis 1 is cols (x, col index increases eastward).
    gy, gx = np.gradient(dem_arr, pixel_size)
    # dzdx in the original code is (west - east) / (2 dx) = -gx
    # dzdy is (south - north) / (2 dx) = gy (rows increase south)
    dzdx = -gx
    dzdy = gy

    slope_deg = np.rad2deg(np.arctan(np.sqrt(dzdx ** 2 + dzdy ** 2)))

    aspect_rad = np.arctan2(dzdx, np.where(dzdy == 0, 1e-9, dzdy))
    aspect_deg = np.rad2deg(aspect_rad)
    aspect_deg = np.where(aspect_deg < 0, aspect_deg + 360.0, aspect_deg)

    eastness = np.sin(2 * np.pi * (aspect_deg / 360.0))
    northness = np.cos(2 * np.pi * (aspect_deg / 360.0))

    flat_mask = (dzdx == 0) & (dzdy == 0)
    if np.any(flat_mask):
        aspect_deg = np.where(flat_mask, np.nan, aspect_deg)
        eastness = np.where(flat_mask, 0.0, eastness)
        northness = np.where(flat_mask, 0.0, northness)

    return (slope_deg.astype(np.float32),
            aspect_deg.astype(np.float32),
            eastness.astype(np.float32),
            northness.astype(np.float32))


def dem_outputs_at_grid(dem_arr, dem_transform, dem_crs,
                       out_transform, out_shape):
    """Compute the 5 DEM-group rasters on the output grid using native-resolution
    derivatives.

    Returns a dict keyed by parameter name (elevation, slope, aspect, eastness,
    northness), each a float32 ndarray of shape ``out_shape``.

    Note: ``dem_crs`` is accepted for API symmetry but not used for the math
    (block averaging is a coordinate-free operation); the caller is
    responsible for tagging the output rasters with the right CRS.
    """
    del dem_crs  # unused: kept in the signature for backward compatibility

    native_px = float(dem_transform[0])
    slope_nat, _aspect_nat, east_nat, north_nat = slope_aspect_arrays(
        dem_arr, native_px,
    )

    elevation_out = _block_average(dem_arr, dem_transform,
                                   out_transform, out_shape)
    slope_out = _block_average(slope_nat, dem_transform,
                               out_transform, out_shape)
    east_out = _block_average(east_nat, dem_transform,
                              out_transform, out_shape)
    north_out = _block_average(north_nat, dem_transform,
                               out_transform, out_shape)

    # recover aspect from the average east/north vector (handles circular mean)
    aspect_out = np.rad2deg(np.arctan2(east_out, north_out))
    aspect_out = np.where(aspect_out < 0, aspect_out + 360.0, aspect_out)

    # cells with no orientation (flat or all-NaN inputs) -> NaN aspect
    mag = np.hypot(east_out, north_out)
    aspect_out = np.where(mag < 1e-6, np.nan, aspect_out)
    aspect_out = np.where(np.isnan(east_out) | np.isnan(north_out),
                          np.nan, aspect_out)

    return {
        'elevation': elevation_out.astype(np.float32),
        'slope': slope_out.astype(np.float32),
        'aspect': aspect_out.astype(np.float32),
        'eastness': east_out.astype(np.float32),
        'northness': north_out.astype(np.float32),
    }


def normalize_points_z(xyz, dem_arr, dem_transform):
    """In-place subtract DEM elevation at each point from z.

    Points falling outside the DEM, or on NaN cells, get NaN z values; the
    caller should drop those before computing metrics.
    """
    rows, cols = rowcol(dem_transform, xyz[:, 0], xyz[:, 1])
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)

    h, w = dem_arr.shape
    in_bounds = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)

    z_dem = np.full(xyz.shape[0], np.nan, dtype=np.float32)
    if np.any(in_bounds):
        z_dem[in_bounds] = dem_arr[rows[in_bounds], cols[in_bounds]]

    xyz[:, 2] = xyz[:, 2] - z_dem
    return xyz
