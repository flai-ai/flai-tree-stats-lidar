import os
import fiona


def load_plots_file(plots_file: str):

    ext = os.path.splitext(plots_file)[1].lower()
    if ext in (".csv", ".txt"):
        return _load_csv(plots_file)

    return _load_vector(plots_file)


def _load_csv(plots_file: str):

    with open(plots_file, "r") as f:
        content = f.read()
    lines = content.strip().replace("\r\n", "\n").split("\n")
    coordinates = []

    for line in lines:
        if not line.strip():
            continue
        x, y = line.split(",")
        coordinates.append((float(x.strip()), float(y.strip())))

    return coordinates


def _load_vector(plots_file: str):

    coordinates = []
    with fiona.open(plots_file) as src:

        for feature in src:
            geom = feature["geometry"]

            if "Point" not in geom["type"]:
                raise ValueError(f"Expected Point geometry, got: {geom['type']}")
            x, y = geom["coordinates"][:2]
            coordinates.append((float(x), float(y)))

    return coordinates