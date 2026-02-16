import os
import logging
from datetime import datetime
import numpy as np
from pandas import DataFrame
from ..utils.text_data import load_plots_file
from ..utils.lidar_data_handling import get_lidar_files_and_extents, get_lidar_data_in_bbox, is_inside_clip_indices
from ..utils.raster_data import create_idw_z_raster, normalize_data

loger = logging.getLogger('flai')


def extract_and_combine_parameter_type(
        xyz_data: np.ndarray, xy_center: list,
        dem: np.ndarray, dem_transform,
        extract_params: list,
) -> dict:

    from ..metrices.dem import dem_params_at_location
    from ..metrices.height import metrics_basic, metrics_percentiles, metrics_percabove, metrics_dispersion, metrics_lmoments
    from ..metrices.canopy import metrics_canopy_density, metrics_interval, metrics_lad
    from ..metrices.kde import metrics_kde
    from ..metrices.voxel import metrics_rumple, metrics_voxels

    combined_params = {}

    if 'dem' in extract_params:
        combined_params.update(dem_params_at_location(
            dem,
            dem_transform,
            xy_center,
        ))

    z = xyz_data[:, 2]

    if 'height' in extract_params:
        combined_params.update(metrics_basic(z))
        combined_params.update(metrics_percentiles(z))
        combined_params.update(metrics_percabove(z))
        combined_params.update(metrics_dispersion(z))
        combined_params.update(metrics_lmoments(z))

    if 'canopy' in extract_params:
        combined_params.update(metrics_canopy_density(z))
        combined_params.update(metrics_interval(z))
        combined_params.update(metrics_lad(z))

    if 'kde' in extract_params:
        combined_params.update(metrics_kde(z))

    if 'voxel' in extract_params:
        x = xyz_data[:, 0]
        y = xyz_data[:, 1]
        combined_params.update(metrics_rumple(x, y, z))
        combined_params.update(metrics_voxels(x, y, z))

    return combined_params


def extract_and_save_plot_params(
        plot_radius: float, lidar_dir: str, plots_filepath: str, save_dir: str,
        extract_params: list = None,
        plot_shape: str = 'circle', ground_class=2, vegetation_classes: list = [3, 4, 5],
        output_format: str = 'csv',
) -> bool:

    start_time = datetime.now()

    loger.info('Reading extents of lidar files.')
    lidar_files, lidar_extents = get_lidar_files_and_extents(lidar_dir)
    loger.info(f' -> {len(lidar_files)} files loaded.')

    loger.info('Reading plot locations.')
    plot_locations = load_plots_file(plots_filepath)
    loger.info(f' -> {len(plot_locations)} plots loaded.')

    loger.info('Extracting plot parameters.')

    # use a bit large data query to make data extraction more robust and to avoid edge effects in the extracted data
    radius_factor = 1.1
    plot_radius_use = plot_radius * radius_factor

    data_to_save = []

    for i_pc, plot_center in enumerate(plot_locations):
        loger.info(f' -> Extracting plot ({i_pc+1} of {len(plot_locations)}).')

        plot_bbox = np.array([
            plot_center[0] - plot_radius_use,
            plot_center[1] - plot_radius_use,
            plot_center[0] + plot_radius_use,
            plot_center[1] + plot_radius_use,
        ])


        points_xyz, points_classes = get_lidar_data_in_bbox(
            lidar_files,
            plot_bbox,
            class_filter=[ground_class] + vegetation_classes,
        )

        if points_xyz.shape[0] == 0:
            loger.warning(f'No points found for plot. Skipping.')
            data_to_save.append({})
            continue

        dem, dem_transform = create_idw_z_raster(
            points_xyz[points_classes == ground_class],
            plot_bbox,
        )

        if (dem == 0).all():
            loger.warning(f'No valid dem produced for, continue processing.')

        idx_in_plot = is_inside_clip_indices(points_xyz, plot_center, plot_radius, plot_shape)

        veg_mask = np.logical_and(idx_in_plot, points_classes != ground_class)

        vegetation_xyz_data = points_xyz[veg_mask]

        if vegetation_xyz_data.shape[0] == 0:
            loger.warning(f'No vegetation points to analyse. Skipping.')
            data_to_save.append({})
            continue

        vegetation_xyz_data = normalize_data(
            vegetation_xyz_data,
            dem=dem,
            dem_transform=dem_transform,
        )

        data_to_save.append(extract_and_combine_parameter_type(
                vegetation_xyz_data,
                plot_center,
                dem,
                dem_transform,
                extract_params,
            )
        )

    for i, plot_center in enumerate(plot_locations):
        data_to_save[i]['plot_x'] = plot_center[0]
        data_to_save[i]['plot_y'] = plot_center[1]

    loger.info('Exporting results')
    df = DataFrame(data_to_save)
    out_file_path = os.path.join(save_dir, f'plot_parameters_{start_time.strftime("%Y_%m_%d_%H_%M_%S")}.{output_format}')

    if output_format in ('shp', 'gpkg'):
        import geopandas as gpd
        from shapely.geometry import Point

        geometry = [Point(xy) for xy in zip(df['plot_x'], df['plot_y'])]
        gdf = gpd.GeoDataFrame(df, geometry=geometry)
        gdf.to_file(out_file_path)

    elif output_format == 'xml':
        df.to_xml(out_file_path, index=False)

    else:
        df.to_csv(out_file_path, index=False)

    return True
