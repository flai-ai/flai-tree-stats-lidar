import os
import click
import logging
from ..plots.params import extract_and_save_plot_params
from ..rasters.orchestrator import run as run_rasters
from ..rasters.keys import SUPPORTED_GROUPS

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
loger = logging.getLogger('flai')

@click.group()
def cli():
    """Main CLI group for forest stats."""
    pass

@cli.group()
def plots():
    """Commands related to plots."""
    pass

@cli.group()
def rasters():
    """Commands related to per-tile raster computation."""
    pass

@cli.group()
def trees():
    """Commands related to trees."""
    click.echo("Not implemented yet.")
    pass

@plots.command(no_args_is_help=True)
@click.option("-r", "--radius", default=None, required=True,
              help="Radius of the extracted plots.")
@click.option("-l", "--lidar", default=None, required=True,
              help="Folder containing lidar data.")
@click.option("-p", "--plots", default=None, required=True,
              help="Textual or vector file with the plot centers coordinates (csv, gpkg, shp..).")
@click.option("-s", "--save", default=None, required=True,
              help="Output folder to which plots report will be saved.")
@click.option("-e", "--extract", default=None, required=True,
              help="List of parameters type groups to extract, separated by comma. Supported parameter groups are: dem, height, canopy, kde, voxel.")
@click.option("-f", "--format", "output_format", default="csv",
              type=click.Choice(["csv", "shp", "gpkg", "xml"], case_sensitive=False),
              help="Output format for the results file (csv, shp, gpkg, xml).")
def extract_params(radius: str, lidar: str, plots: str, save: str, extract: str, output_format: str):

    if not os.path.exists(lidar) or not os.path.isdir(lidar):
        raise AttributeError(f"Invalid lidar directory: {lidar}")

    if not os.path.exists(plots) or not os.path.isfile(plots):
        raise AttributeError(f"Invalid plot locations file: {plots}")

    radius = float(radius)
    assert radius > 0, "Radius must be a positive number."

    os.makedirs(save, exist_ok=True)

    extract_and_save_plot_params(
        radius,
        lidar,
        plots,
        save,
        extract_params=[e_p.strip() for e_p in extract.replace('"', '').replace("'", "").split(',')],
        output_format=output_format,
    )

    return True


@rasters.command("extract-params", no_args_is_help=True)
@click.option("-l", "--lidar", required=True,
              help="Folder containing LAZ/LAS tiles.")
@click.option("-d", "--dem", default=None, required=False,
              help="(Optional) Folder containing matching DEM TIF files (same "
                   "basename as LAZ). If omitted, a DEM is built on the fly "
                   "from each LAZ's ground returns at --dem-pixel-size.")
@click.option("-s", "--save", required=True,
              help="Output folder. One subfolder is created per parameter; "
                   "each contains one TIF per input tile.")
@click.option("-e", "--extract", default=",".join(SUPPORTED_GROUPS),
              show_default=True,
              help=f"Comma-separated metric groups. Supported: {','.join(SUPPORTED_GROUPS)}. "
                   "Note: 'voxel' is significantly slower than the others.")
@click.option("--pixel-size", default=10.0, show_default=True, type=float,
              help="Output raster pixel size (and plot size) in CRS units.")
@click.option("--dem-pixel-size", default=0.5, show_default=True, type=float,
              help="DEM pixel size used when --dem is not provided. The "
                   "on-the-fly DEM is built at this resolution from ground "
                   "returns and is also the scale at which slope/aspect are "
                   "derived.")
@click.option("--idw-power", default=2.0, show_default=True, type=float,
              help="IDW power exponent for the on-the-fly DEM.")
@click.option("--idw-k", default=12, show_default=True, type=int,
              help="Number of nearest neighbours used per cell in the IDW DEM.")
@click.option("-w", "--workers", default=4, show_default=True, type=int,
              help="Number of LAZ tiles processed in parallel (process pool).")
@click.option("-t", "--cell-workers", default=4, show_default=True, type=int,
              help="Threads used for per-cell metric computation inside a "
                   "tile and for the IDW kNN query.")
@click.option("--blas-threads", default=1, show_default=True, type=int,
              help="Cap BLAS / OpenMP / MKL threads per worker process. "
                   "Default 1 keeps total core usage at workers x cell_workers; "
                   "raise this when scipy.gaussian_kde / cKDTree should use "
                   "more cores internally.")
@click.option("--ground-class", default=2, show_default=True, type=int,
              help="LAS classification value treated as ground.")
@click.option("--vegetation-classes", default="3,4,5", show_default=True,
              help="Comma-separated LAS classification values treated as vegetation.")
@click.option("--min-points", default=4, show_default=True, type=int,
              help="Minimum vegetation point count per cell to compute metrics.")
@click.option("--skip-existing/--overwrite", default=True, show_default=True,
              help="Skip a tile if every parameter TIF already exists.")
@click.option("--progress/--no-progress", default=True, show_default=True,
              help="Show a tqdm progress bar with ETA. Disable for plain log output.")
@click.option("--crs", "crs_override", default=None,
              help="Override the CRS for output rasters (and the resampling "
                   "math) when the LAZ has no SRS VLR. Accepts anything "
                   "rasterio.crs.CRS.from_user_input() handles, e.g. "
                   "'EPSG:3794'.")
def extract_params_rasters(lidar, dem, save, extract, pixel_size,
                           dem_pixel_size, idw_power, idw_k,
                           workers, cell_workers, blas_threads,
                           ground_class, vegetation_classes,
                           min_points, skip_existing,
                           progress, crs_override):
    """Compute per-tile forestry parameter rasters at a fixed pixel size.

    Pair each LAZ tile with a DEM (provided via -d, or computed on the fly
    when -d is omitted), normalize point heights, bin points into output
    cells, and write one float32 GeoTIFF per parameter per tile.
    """
    groups = [g.strip() for g in extract.replace('"', '').replace("'", "").split(',') if g.strip()]
    unknown = [g for g in groups if g not in SUPPORTED_GROUPS]
    if unknown:
        raise click.BadParameter(
            f"Unknown metric groups: {unknown}. Supported: {SUPPORTED_GROUPS}",
            param_hint="--extract")

    veg = [int(c) for c in vegetation_classes.split(',') if c.strip()]
    if not veg:
        raise click.BadParameter("Provide at least one vegetation class.",
                                 param_hint="--vegetation-classes")

    summary = run_rasters(
        lidar_dir=lidar,
        dem_dir=dem,
        save_dir=save,
        extract_groups=tuple(groups),
        pixel_size=float(pixel_size),
        workers=int(workers),
        cell_workers=int(cell_workers),
        ground_class=int(ground_class),
        vegetation_classes=tuple(veg),
        min_points=int(min_points),
        skip_existing=bool(skip_existing),
        dem_pixel_size=float(dem_pixel_size),
        idw_power=float(idw_power),
        idw_k=int(idw_k),
        progress=bool(progress),
        crs_override=crs_override,
        blas_threads=int(blas_threads),
    )
    click.echo(
        f"Pairs={summary['pairs']} ok={summary['ok']} "
        f"skipped={summary['skipped']} failed={summary['failed']} "
        f"elapsed={summary['elapsed_s']:.1f}s"
    )
    if summary['missing_dem']:
        click.echo(f"WARNING: {len(summary['missing_dem'])} LAZ files lacked a DEM.")
    if summary['failed']:
        click.echo(
            f"WARNING: {summary['failed']} tiles failed. "
            f"See {summary['failed_txt']} (retry list) and "
            f"{summary['failed_csv']} (with error messages)."
        )
    return True


if __name__ == "__main__":
    cli()
