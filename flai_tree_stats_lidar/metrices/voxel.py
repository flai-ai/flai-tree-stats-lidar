import numpy as np
from scipy.spatial import Delaunay


def metrics_rumple(x, y, z, pixel_size=1.0):
    """Surface rumple index: ratio of 3D Delaunay surface area to 2D projected area."""
    if len(x) < 3:
        return {'rumple': np.nan}

    points_2d = np.column_stack((x, y))

    try:
        tri = Delaunay(points_2d)
    except Exception:
        return {'rumple': np.nan}

    simplices = tri.simplices
    p0 = np.column_stack((x[simplices[:, 0]], y[simplices[:, 0]], z[simplices[:, 0]]))
    p1 = np.column_stack((x[simplices[:, 1]], y[simplices[:, 1]], z[simplices[:, 1]]))
    p2 = np.column_stack((x[simplices[:, 2]], y[simplices[:, 2]], z[simplices[:, 2]]))

    # 3D triangle areas via cross product
    cross = np.cross(p1 - p0, p2 - p0)
    area_3d = 0.5 * np.sqrt(np.sum(cross ** 2, axis=1))
    total_3d = np.sum(area_3d)

    # 2D projected areas (z=0)
    p0_2d = points_2d[simplices[:, 0]]
    p1_2d = points_2d[simplices[:, 1]]
    p2_2d = points_2d[simplices[:, 2]]
    area_2d = 0.5 * np.abs(
        (p1_2d[:, 0] - p0_2d[:, 0]) * (p2_2d[:, 1] - p0_2d[:, 1]) -
        (p2_2d[:, 0] - p0_2d[:, 0]) * (p1_2d[:, 1] - p0_2d[:, 1])
    )
    total_2d = np.sum(area_2d)

    if total_2d == 0:
        return {'rumple': np.nan}

    return {'rumple': float(total_3d / total_2d)}


def metrics_voxels(x, y, z, vox_size=1.0):
    """Voxel-based canopy structure metrics.

    Returns
    -------
    dict with keys:
        vn              : number of filled voxels
        vFRall          : filled ratio (all voxels in bounding box)
        vFRcanopy       : filled ratio (voxels above min occupied height)
        vzrumple        : voxel-based vertical rumple (unique Z layers / max possible)
        vzsd            : std dev of voxel Z coordinates
        vzcv            : coefficient of variation of voxel Z coordinates
        OpenGapSpace    : proportion of empty voxels below the canopy
        ClosedGapSpace  : proportion of empty voxels with filled voxels above
        Euphotic        : proportion of filled voxels in upper 65% of height range
        Oligophotic     : proportion of filled voxels in lower 35% of height range
    """
    keys = ['vn', 'vFRall', 'vFRcanopy', 'vzrumple', 'vzsd', 'vzcv',
            'OpenGapSpace', 'ClosedGapSpace', 'Euphotic', 'Oligophotic']

    if len(x) < 2:
        return {k: np.nan for k in keys}

    # Voxelize: assign each point to a voxel
    vx = np.floor(x / vox_size).astype(int)
    vy = np.floor(y / vox_size).astype(int)
    vz = np.floor(z / vox_size).astype(int)

    # Unique filled voxels
    voxels = np.unique(np.column_stack((vx, vy, vz)), axis=0)
    vn = len(voxels)

    # Bounding box in voxel space
    vx_range = voxels[:, 0].max() - voxels[:, 0].min() + 1
    vy_range = voxels[:, 1].max() - voxels[:, 1].min() + 1
    vz_range = voxels[:, 2].max() - voxels[:, 2].min() + 1

    total_voxels = vx_range * vy_range * vz_range
    vFRall = float(vn / total_voxels) if total_voxels > 0 else np.nan

    # Canopy voxels: above minimum occupied height
    vz_min = voxels[:, 2].min()
    vz_max = voxels[:, 2].max()
    canopy_total = vx_range * vy_range * vz_range
    vFRcanopy = float(vn / canopy_total) if canopy_total > 0 else np.nan

    # Vertical rumple: unique Z layers / max possible layers
    unique_z_layers = len(np.unique(voxels[:, 2]))
    vzrumple = float(unique_z_layers / vz_range) if vz_range > 0 else np.nan

    # Voxel Z statistics
    voxel_z = voxels[:, 2].astype(float) * vox_size
    vzsd = float(np.std(voxel_z, ddof=1)) if vn > 1 else 0.0
    vzmean = np.mean(voxel_z)
    vzcv = float(vzsd / vzmean) if vzmean != 0 else np.nan

    # Gap analysis: build a set for fast lookup
    filled_set = set(map(tuple, voxels))
    xy_columns = np.unique(voxels[:, :2], axis=0)

    open_gap = 0
    closed_gap = 0
    total_empty = 0

    for col in xy_columns:
        col_x, col_y = col
        # Get min and max Z for this column
        col_mask = (voxels[:, 0] == col_x) & (voxels[:, 1] == col_y)
        col_z_vals = voxels[col_mask, 2]
        col_z_min, col_z_max = col_z_vals.min(), col_z_vals.max()

        for vz_i in range(col_z_min, col_z_max + 1):
            if (col_x, col_y, vz_i) not in filled_set:
                total_empty += 1
                # Check if there's a filled voxel above
                has_above = any((col_x, col_y, vz_j) in filled_set
                                for vz_j in range(vz_i + 1, col_z_max + 1))
                if has_above:
                    closed_gap += 1
                else:
                    open_gap += 1

    gap_total = open_gap + closed_gap + vn
    OpenGapSpace = float(open_gap / gap_total) if gap_total > 0 else 0.0
    ClosedGapSpace = float(closed_gap / gap_total) if gap_total > 0 else 0.0

    # Euphotic / Oligophotic
    z_threshold = vz_min + 0.65 * (vz_max - vz_min)
    euphotic_count = np.sum(voxels[:, 2] >= z_threshold)
    oligophotic_count = np.sum(voxels[:, 2] < z_threshold)
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
