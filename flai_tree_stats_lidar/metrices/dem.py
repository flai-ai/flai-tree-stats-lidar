import logging
import numpy as np
from rasterio.transform import rowcol
from ..utils.raster_data import slope, aspect


def dem_params_at_location(dem: np.ndarray, dem_transform, location: list):

    row, col = rowcol(
        dem_transform,
        location[0],
        location[1]
    )

    if row < 0 or row >= dem.shape[0] or col < 0 or col >= dem.shape[1]:
        logger.warning(f"Location {location} is outside the DEM bounds.")
        return {}

    slope_deg = slope(dem, dem_transform[0], to_degrees=True)
    aspect_deg = aspect(dem, dem_transform[0], to_degrees=True)

    # TBD: do we need a bit more smoothed value from nearby pixels?
    dem_params = {
        #'x_loc': location[0],
        #'y_loc': location[1],
        'elevation': np.mean(dem[row, col]),
        'slope': np.mean(slope_deg[row-1: row+2, col-1: col+2]),
        'aspect': np.mean(aspect_deg[row-1: row+2, col-1: col+2]),
    }

    # Izračun eastness in northness
    aspect_mean = dem_params['aspect']
    if aspect_mean == -1:
        dem_params['eastness'] = np.nan
        dem_params['northness'] = np.nan
    else:
        dem_params['eastness'] = np.round(np.sin(2 * np.pi * (aspect_mean / 360)), 3)
        dem_params['northness'] = np.round(np.cos(2 * np.pi * (aspect_mean / 360)), 3)

    return dem_params
