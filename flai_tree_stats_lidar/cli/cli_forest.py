import os
import click
import logging
from ..plots.params import extract_and_save_plot_params

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

if __name__ == "__main__":
    cli()
