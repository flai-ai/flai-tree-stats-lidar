import numpy as np
import logging
from rasterio.transform import from_origin, rowcol
from scipy.spatial import KDTree

logger = logging.getLogger('flai')


def _dzdxdy(dem_data: np.ndarray, d_xy: float):
    dzdx = ((np.roll(dem_data, 1, axis=1) - np.roll(dem_data, -1, axis=1)) / 2) / d_xy
    dzdy = ((np.roll(dem_data, -1, axis=0) - np.roll(dem_data, 1, axis=0)) / 2) / d_xy
    return dzdx, dzdy


def slope(dem_data: np.ndarray, d_xy: float, to_degrees: bool = False):
    dzdx, dzdy = _dzdxdy(dem_data, d_xy)
    angle = np.arctan(np.sqrt(dzdx ** 2 + dzdy ** 2))
    return np.rad2deg(angle) if to_degrees else angle


def aspect(dem_data: np.ndarray, d_xy: float, to_degrees: bool = False):
    dzdx, dzdy = _dzdxdy(dem_data, d_xy)
    dzdy[dzdy == 0] = 10e-9
    angle = np.arctan2(dzdx, dzdy)
    return np.rad2deg(angle) if to_degrees else angle


def create_idw_z_raster(xyz_ground_data: np.ndarray, bbox: list, pixel_size: float = 0.5, power: float = 2.0, k: int = 12):

    x = np.arange(bbox[0] + pixel_size / 2, bbox[2], pixel_size)
    y = np.arange(bbox[3] - pixel_size / 2, bbox[1], -pixel_size)

    # Upper-left corner origin transform
    raster_transform = from_origin(bbox[0], bbox[3], pixel_size, pixel_size)

    if xyz_ground_data.shape[0] == 0:
        del x,y
        return np.zeros((len(y), len(x)), dtype=np.float32), raster_transform

    grid_x, grid_y = np.meshgrid(x, y)
    query_points = np.vstack((grid_x.flatten(), grid_y.flatten())).T
    del grid_x, grid_y

    # IDW interpolation using k nearest neighbours
    tree = KDTree(xyz_ground_data[:, :2])
    distances, indices = tree.query(query_points, k=min(k, xyz_ground_data.shape[0]))
    del tree, query_points

    # Replace zero distances to avoid division by zero
    distances[distances == 0] = 1e-10

    weights = 1.0 / distances ** power
    raster_values = np.sum(weights * xyz_ground_data[:, 2][indices], axis=1) / np.sum(weights, axis=1)
    raster_array = raster_values.reshape(len(y), len(x))
    del weights, distances, indices, x,y

    return raster_array, raster_transform

def normalize_data(xyz_data: np.ndarray, dem: np.ndarray = None, dem_transform = None):

    if dem is not None and dem_transform is not None:
        # Get row/col indices for each point
        rows, cols = rowcol(
            dem_transform,
            xyz_data[:, 0],
            xyz_data[:, 1]
        )

        # Clip indices to valid range
        rows = np.clip(rows, 0, dem.shape[0] - 1)
        cols = np.clip(cols, 0, dem.shape[1] - 1)

        # Subtract DEM elevation from point Z values
        xyz_data[:, 2] -= dem[rows, cols]

    return xyz_data
