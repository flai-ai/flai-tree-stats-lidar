"""Vectorised version of ``flai_tree_stats_lidar.metrices.voxel.metrics_voxels``.

Produces **bit-identical** outputs to the original implementation but replaces
the per-column Python gap-analysis loop with a single lexsort + reduceat pass.
The original is kept untouched for the plot-based workflow; this fast variant
is wired into the raster pipeline because the per-cell cost matters there.

Algorithmic equivalence (the original ranges ``vz_i`` from ``col_z_min`` to
``col_z_max`` inclusive; ``col_z_max`` is always filled by construction, so
every empty ``vz_i`` in the range has at least one filled voxel above it).
Therefore in the original:

    open_gap   = 0   for every column
    closed_gap = (col_z_max - col_z_min + 1) - n_filled_per_column

The fast version computes the same expression directly.
"""
import numpy as np


_KEYS = ('vn', 'vFRall', 'vFRcanopy', 'vzrumple', 'vzsd', 'vzcv',
        'OpenGapSpace', 'ClosedGapSpace', 'Euphotic', 'Oligophotic')


def _nan_result():
    return {k: np.nan for k in _KEYS}


def metrics_voxels_fast(x, y, z, vox_size=1.0):
    if len(x) < 2:
        return _nan_result()

    vx = np.floor(np.asarray(x) / vox_size).astype(np.int64)
    vy = np.floor(np.asarray(y) / vox_size).astype(np.int64)
    vz = np.floor(np.asarray(z) / vox_size).astype(np.int64)

    # --- unique filled voxels, sorted lexicographically by (vx, vy, vz)
    order = np.lexsort((vz, vy, vx))
    vx_s = vx[order]
    vy_s = vy[order]
    vz_s = vz[order]

    if vx_s.size == 1:
        unique_mask = np.array([True])
    else:
        diff = ((vx_s[1:] != vx_s[:-1])
                | (vy_s[1:] != vy_s[:-1])
                | (vz_s[1:] != vz_s[:-1]))
        unique_mask = np.concatenate(([True], diff))

    uvx = vx_s[unique_mask]
    uvy = vy_s[unique_mask]
    uvz = vz_s[unique_mask]
    vn = int(uvx.size)

    # --- bbox in voxel space
    vx_range = int(uvx.max() - uvx.min() + 1)
    vy_range = int(uvy.max() - uvy.min() + 1)
    vz_min = int(uvz.min())
    vz_max = int(uvz.max())
    vz_range = vz_max - vz_min + 1

    total_voxels = vx_range * vy_range * vz_range
    vFRall = float(vn / total_voxels) if total_voxels > 0 else np.nan
    # The original sets canopy_total to the same product, so vFRcanopy == vFRall.
    vFRcanopy = vFRall

    # Vertical rumple: unique Z layers / max possible layers
    if vz_range > 0:
        # number of distinct vz values among filled voxels
        uniq_z = np.unique(uvz).size
        vzrumple = float(uniq_z / vz_range)
    else:
        vzrumple = np.nan

    # --- voxel Z statistics (use float promotion that matches np.std default)
    voxel_z = uvz.astype(np.float64) * vox_size
    if vn > 1:
        vzsd = float(np.std(voxel_z, ddof=1))
    else:
        vzsd = 0.0
    vzmean = float(np.mean(voxel_z))
    vzcv = float(vzsd / vzmean) if vzmean != 0 else np.nan

    # --- gap analysis: per (vx, vy) column, count cells in [z_min, z_max] not
    # filled. In the original loop open_gap == 0 (proven above), so the result
    # is total_empty per column.
    # uvx/uvy/uvz are already sorted by (vx, vy, vz) so we just find group
    # boundaries on (vx, vy).
    if vn == 1:
        empty_per_col = np.array([0])
    else:
        xy_diff = (uvx[1:] != uvx[:-1]) | (uvy[1:] != uvy[:-1])
        group_starts = np.concatenate(([0], np.where(xy_diff)[0] + 1))
        group_ends = np.concatenate((group_starts[1:], [vn]))
        n_filled = group_ends - group_starts
        z_min_per_col = uvz[group_starts]
        z_max_per_col = uvz[group_ends - 1]  # last z in sorted group
        empty_per_col = (z_max_per_col - z_min_per_col + 1) - n_filled

    total_empty = int(empty_per_col.sum())
    open_gap = 0  # mirrors the original behaviour exactly
    closed_gap = total_empty

    gap_total = open_gap + closed_gap + vn
    OpenGapSpace = float(open_gap / gap_total) if gap_total > 0 else 0.0
    ClosedGapSpace = float(closed_gap / gap_total) if gap_total > 0 else 0.0

    # --- euphotic / oligophotic
    z_threshold = vz_min + 0.65 * (vz_max - vz_min)
    euphotic_count = int(np.sum(uvz >= z_threshold))
    oligophotic_count = int(np.sum(uvz < z_threshold))
    Euphotic = float(euphotic_count / vn) if vn > 0 else np.nan
    Oligophotic = float(oligophotic_count / vn) if vn > 0 else np.nan

    return {
        'vn': vn,
        'vFRall': vFRall,
        'vFRcanopy': vFRcanopy,
        'vzrumple': vzrumple,
        'vzsd': vzsd,
        'vzcv': vzcv,
        'OpenGapSpace': OpenGapSpace,
        'ClosedGapSpace': ClosedGapSpace,
        'Euphotic': Euphotic,
        'Oligophotic': Oligophotic,
    }
