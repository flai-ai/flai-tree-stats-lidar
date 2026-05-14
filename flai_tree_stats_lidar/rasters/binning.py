"""Assign LiDAR points to 10 m (or other) raster cells and produce a sorted
index so that per-cell point slices are O(1) to look up.

The output grid is upper-left origin (row 0 is northern-most), matching the
rasterio convention.
"""
import numpy as np


class CellIndex:
    """Holds the per-tile spatial binning state.

    Attributes
    ----------
    n_rows, n_cols : int
        Output raster shape.
    transform : affine.Affine
        Output raster transform (upper-left origin, +x east / -y south).
    pixel_size : float
    x_min, y_max : float
        Output grid origin (upper-left corner).
    sorted_order : np.ndarray
        Permutation that sorts the input points by linear cell index.
        Use this on the original xyz arrays.
    cell_lin_idx : np.ndarray
        Length n_rows * n_cols. For each cell, the start offset into the
        sorted point arrays. -1 if the cell is empty.
    cell_counts : np.ndarray
        Length n_rows * n_cols. Number of points falling in each cell.
    """
    __slots__ = ('n_rows', 'n_cols', 'transform', 'pixel_size',
                 'x_min', 'y_max', 'sorted_order',
                 'cell_starts', 'cell_counts')


def _snap_outward(x_min, y_min, x_max, y_max, pixel_size):
    """Snap a bbox outward to a pixel_size-aligned grid."""
    x0 = np.floor(x_min / pixel_size) * pixel_size
    y0 = np.floor(y_min / pixel_size) * pixel_size
    x1 = np.ceil(x_max / pixel_size) * pixel_size
    y1 = np.ceil(y_max / pixel_size) * pixel_size
    return float(x0), float(y0), float(x1), float(y1)


def build_grid(bbox, pixel_size):
    """Return (n_rows, n_cols, x_min, y_max, affine_transform).

    bbox = (x_min, y_min, x_max, y_max). Snapped outward to the grid.
    """
    from rasterio.transform import from_origin

    x_min, y_min, x_max, y_max = _snap_outward(*bbox, pixel_size=pixel_size)
    n_cols = int(round((x_max - x_min) / pixel_size))
    n_rows = int(round((y_max - y_min) / pixel_size))
    transform = from_origin(x_min, y_max, pixel_size, pixel_size)
    return n_rows, n_cols, x_min, y_max, transform


def bin_points(xy, n_rows, n_cols, x_min, y_max, pixel_size):
    """Assign points to cells and return a sorted-by-cell layout.

    Parameters
    ----------
    xy : (N, 2) float array
    n_rows, n_cols : int
    x_min, y_max : float
        Upper-left corner of the output grid.
    pixel_size : float

    Returns
    -------
    sorted_order : (M,) int array
        Permutation mapping into the input arrays so that
        ``points[sorted_order]`` is grouped contiguously by linear cell idx.
        Only points falling inside the grid are kept (M <= N).
    cell_starts : (n_rows * n_cols,) int array
        Start offset of each cell in the sorted layout. -1 for empty cells.
    cell_counts : (n_rows * n_cols,) int array
        Number of points per cell.
    """
    n_cells = n_rows * n_cols

    col = np.floor((xy[:, 0] - x_min) / pixel_size).astype(np.int64)
    row = np.floor((y_max - xy[:, 1]) / pixel_size).astype(np.int64)

    valid = (col >= 0) & (col < n_cols) & (row >= 0) & (row < n_rows)
    if not np.all(valid):
        row = row[valid]
        col = col[valid]
        # need the original-index permutation only over the valid subset
        valid_idx = np.where(valid)[0]
    else:
        valid_idx = np.arange(xy.shape[0])

    lin = row * n_cols + col

    order_within_valid = np.argsort(lin, kind='stable')
    sorted_order = valid_idx[order_within_valid]
    lin_sorted = lin[order_within_valid]

    cell_counts = np.bincount(lin_sorted, minlength=n_cells)
    cell_starts = np.full(n_cells, -1, dtype=np.int64)
    if lin_sorted.size > 0:
        # first occurrence of each cell index in the sorted layout
        first = np.concatenate(([True], np.diff(lin_sorted) != 0))
        cell_starts[lin_sorted[first]] = np.where(first)[0]

    return sorted_order, cell_starts, cell_counts
