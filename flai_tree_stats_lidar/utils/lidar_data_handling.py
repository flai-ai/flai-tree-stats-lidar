import glob
import os
import laspy
import numpy as np


def get_lidar_files_and_extents(lidar_path: str):
    lidar_files = sorted(glob.glob(os.path.join(lidar_path, "*.la?")))

    lidar_extents = np.empty((len(lidar_files), 4))
    for i, filepath in enumerate(lidar_files):
        with laspy.open(filepath) as f:
            lidar_extents[i] = np.hstack((f.header.mins[:2], f.header.maxs[:2]))

    return lidar_files, lidar_extents

def get_matching_files_by_bbox(lidar_files: list, lidar_extents: np.ndarray, plot_bbox: np.ndarray):

    # Check for intersection between plot_bbox and each lidar_extent
    intersects = (
        (lidar_extents[:, 0] < plot_bbox[2]) & (lidar_extents[:, 2] > plot_bbox[0]) &
        (lidar_extents[:, 1] < plot_bbox[3]) & (lidar_extents[:, 3] > plot_bbox[1])
    )

    return np.array(lidar_files)[intersects].tolist()

def get_lidar_data_in_bbox(lidar_files: list, plot_bbox: np.ndarray, class_filter: list = None):
    """Read LAS points within a bounding box.

    Parameters
    ----------

    """
    xyz_list = []
    class_list = []

    for filepath in lidar_files:
        las = laspy.read(filepath)

        # Clip to bbox
        mask = (
            (las.x >= plot_bbox[0]) & (las.x <= plot_bbox[2]) &
            (las.y >= plot_bbox[1]) & (las.y <= plot_bbox[3])
        )

        # Filter by classification
        if class_filter is not None:
            mask &= np.isin(las.classification, class_filter)

        if np.any(mask):
            xyz_list.append(las.xyz[mask])
            class_list.append(np.array(las.classification)[mask])

    if xyz_list:
        stacked_xyz = np.vstack(xyz_list)
        stacked_classes = np.hstack(class_list)
    else:
        stacked_xyz = np.empty((0, 3))
        stacked_classes = np.empty(0)

    return stacked_xyz, stacked_classes


def is_inside_clip_indices(xyz_data: np.ndarray, clip_xy_center: list, clip_radius: float, clip_shape: str = 'circle', return_indices: bool = False):

    dx = xyz_data[:, 0] - clip_xy_center[0]
    dy = xyz_data[:, 1] - clip_xy_center[1]

    if clip_shape == 'circle':
        idx_clip = dx ** 2 + dy ** 2 <= clip_radius ** 2
    elif clip_shape == 'square':
        idx_clip = (np.abs(dx) <= clip_radius) & (np.abs(dy) <= clip_radius)
    else:
        raise ValueError(f"Unsupported clip_shape: '{clip_shape}'. Use 'circle' or 'square'.")
    del dx, dy

    if return_indices:
        return np.where(idx_clip)[0]

    return idx_clip
