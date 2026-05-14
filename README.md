# ALS - NFI Plot Forest Metrics Calculator

Calculate forest structure metrics from **Airborne Laser Scanning (ALS)** point clouds for circular sampling plots of selected sizes.
**Programming Language**: Python

---
# **Features**

Extract tree and forestry statistics from LiDAR point clouds using the following command-line options:

| Option | Description                                                                                |
| ------ | ------------------------------------------------------------------------------------------ |
| `-r`   | Radius of the circular plot (in meters).                                                   |
| `-p`   | Folder containing plot data (extensions: `.csv`, `.gpkg`, `.shp`).                         |
| `-l`   | Folder containing ALS data.                                                                |
| `-s`   | Folder where results will be saved.                                                        |
| `-e`   | Metrics to extract from the point cloud (e.g., `dem`, `height`, `canopy`, `kde`, `voxel`). |

---
## **Metrics Overview**
### **1. Terrain Metrics (`dem`)**

Basic terrain parameters at a given location:
| Parameter | Description                                                                              | Units         |
| --------- | ---------------------------------------------------------------------------------------- | ------------- |
| elevation | Elevation at the center cell of the plot.                                                | DEM units     |
| slope     | Terrain slope in degrees.                                                                | Degrees       |
| aspect    | Terrain aspect in degrees.                                                               | Degrees       |
| eastness  | Orientation of the surface relative to east (calculated as `sin(2π × (aspect / 360))`).  | Dimensionless |
| northness | Orientation of the surface relative to north (calculated as `cos(2π × (aspect / 360))`). | Dimensionless |

---
### **2. Height Metrics (`height`)**
Height-related parameters computed for each plot location:

#### **Basic Statistics**
| Parameter | Description                                                        | Units   |
| --------- | ------------------------------------------------------------------ | ------- |
| n         | Number of LiDAR returns within the plot.                           | Count   |
| zmax      | Maximum height value observed among LiDAR returns within the plot. | Meters  |
| zmin      | Minimum height value observed among LiDAR returns within the plot. | Meters  |
| zmean     | Mean height of all LiDAR returns within the plot.                  | Meters  |
| zvar      | Variance of height values within the plot.                         | Meters² |
| zsd       | Standard deviation of height values within the plot.               | Meters  |
| zcv       | Coefficient of variation of height values.                         | Ratio   |
| zskew     | Skewness of the height distribution.                               | -       |
| zkurt     | Kurtosis of the height distribution.                               | -       |

#### **Height Percentiles**
| Parameter | Description                       | Units  |
| --------- | --------------------------------- | ------ |
| zq1       | 1st percentile of height values.  | Meters |
| zq5       | 5th percentile of height values.  | Meters |
| ...       | ...                               | ...    |
| zq99      | 99th percentile of height values. | Meters |

#### **Canopy Cover and Above-Threshold Metrics**
| Parameter   | Description                                                 | Units |
| ----------- | ----------------------------------------------------------- | ----- |
| pzabovemean | Proportion of LiDAR returns above mean height.              | Ratio |
| pzabove2    | Proportion of LiDAR returns above 2 meters.                 | Ratio |
| pzabove5    | Proportion of LiDAR returns above 5 meters.                 | Ratio |

#### **Robust Statistics and Complexity Metrics**
| Parameter  | Description                                                              | Units  |
| ---------- | ------------------------------------------------------------------------ | ------ |
| ziqr       | Interquartile range of height values.                                    | Meters |
| zMADmean   | Mean Median Absolute Deviation of height values.                         | Meters |
| zMADmedian | Median Absolute Deviation of height values.                              | Meters |
| CRR        | Canopy Relief Ratio, a measure of vertical complexity.                   | Ratio  |
| zentropy   | Entropy of the height distribution, a measure of disorder or randomness. | -      |

#### **L-Moments for Height Distribution**
| Parameter | Description                                                           | Units  |
| --------- | --------------------------------------------------------------------- | ------ |
| L1        | First L-moment (equivalent to the mean).                              | Meters |
| L2        | Second L-moment (related to the dispersion).                          | Meters |
| L3        | Third L-moment (related to skewness).                                 | Meters |
| L4        | Fourth L-moment (related to kurtosis).                                | Meters |
| Lskew     | L-skewness, a robust measure of asymmetry.                            | -      |
| Lkurt     | L-kurtosis, a robust measure of "tailedness".                         | -      |
| Lcoefvar  | L-coefficient of variation, a robust measure of relative variability. | Ratio  |


---
### **3. Canopy Metrics (`canopy`)**
 |Parameter             | Description                                                           | Units   |
 |----------------------|-----------------------------------------------------------------------|---------|
 | zpcum1               | Cumulative proportion of points below the first breakpoint.           | %       |
 | zpcum2               | Cumulative proportion of points below the second breakpoint.          | %       |
 | zpcum3               | Cumulative proportion of points below the third breakpoint.           | %       |
 | zpcum9               | Cumulative proportion of points below the ninth breakpoint.           | %       |
 | pInterval_0_0.15     | Proportion of points in the height interval from 0 to 0.15 meters.    | %       |
 | pInterval_0.15_2     | Proportion of points in the height interval from 0.15 to 2 meters.    | %       |
 | pInterval_2_5        | Proportion of points in the height interval from 2 to 5 meters.       | %       |
 | pInterval_above_30   | Proportion of points above 30 meters.                                 | %       |
 | lad_min              | Minimum Leaf Area Density (LAD) within height intervals.              | m²/m³   |
 | lad_max              | Maximum Leaf Area Density (LAD) within height intervals.              | m²/m³   |
 | lad_mean             | Mean Leaf Area Density (LAD) within height intervals.                 | m²/m³   |
 | lad_cv               | Coefficient of variation of LAD within height intervals.              | -       |
 | lad_sum              | Sum of Leaf Area Density (LAD) within height intervals.               | m²/m³   |


---
### **4. Kernel Density Estimation (`kde`)**
Kernel Density Estimation (KDE) is a non-parametric way to estimate the probability density function of a random variable. In the context of your data, KDE is likely used to identify the number and characteristics of "peaks" in the elevation distribution of points (e.g., from LiDAR or other topographic data).

 |Parameter             | Description                                                           | Units   |
 |----------------------|-----------------------------------------------------------------------|---------|
 | kde_peaks_count      | Number of peaks detected in the elevation distribution.               | Count   |
 | kde_peak1_elev       | Elevation of the first peak.                                          | Meters  |
 | kde_peak1_value      | Value (height or density) at the first peak.                          | Unitless (density or probability)  |
 | kde_peak1_diff       | Difference or additional metric related to the first peak.            | Depends on context                 |
 | kde_peak2_elev       | Elevation of the second peak (if present).                            | Meters  |
 | kde_peak2_value      | Value at the second peak.                                             | Unitless|
 | kde_peak2_diff       | Difference or additional metric related to the second peak.           | Depends on context                 |
 | kde_peak3_elev       | Elevation of the third peak (if present).                             | Meters  |
 ...

---
### **5. Voxel Metrics (`voxel`)**
This module derives 3D canopy structure metrics from point clouds (e.g., ALS/LiDAR) using two approaches:
1. Surface rumple via Delaunay triangulation of (x, y) with 3D areas from (x, y, z).
2. Voxelization of the point cloud into a 3D grid to quantify occupancy, vertical layering, and canopy gaps.

| Parameter      | Description                                                                                    | Units               |
| -------------- | ---------------------------------------------------------------------------------------------- | --------------------|
| rumple         | Surface rumple index: sum of 3D triangle areas divided by sum of 2D projected triangle areas   | > 1 rougher surface |
| vn             | Number of filled voxels (unique voxel cells occupied by at least one point).                   | Count               |
| vFRall         | Filled ratio (all): filled voxels divided by total voxels in the 3D bounding box               | Ratio               |
| vFRcanopy      | Filled ratio (canopy): same computation as vFRall in the current implementation                | Ratio               |
| vzrumple       | Vertical rumple: fraction of occupied Z layers over the total possible Z layers                | Ratio               |
| vzsd           | Standard deviation of occupied voxel Z centers (vertical dispersion).                          | Units of Z (m)      |
| vzcv           | Coefficient of variation of Z: vzsd divided by the mean Z of occupied voxel centers.           | Ratio               |
| OpenGapSpace   | Proportion of empty voxels with no filled voxel above them in the same (x, y) column           | Ratio               |
| ClosedGapSpace | Proportion of empty voxels that do have at least one filled voxel above them in the same (x, y)| Ratio               |
| Euphotic       | Fraction of filled voxels in the upper 65% of the height range                                 | Ratio               |
| Oligophotic    | Fraction of filled voxels in the lower 35% of the height range                                 | Ratio               |

---
## Installation

** download repository or zip file
** install the required packages
List of requirelents: numpy, laspy[lazrs], rasterio, scipy, pandas, fiona, geopandas, shapely

 ```bash
 cd flai-tree-stats-lidar
 pip install -r requirements.txt
 pip install .
```
---
## Quick Start
```bash
flai_forest plots extract-params `
    -r "10" `
    -l ./lidar data ` 
    -p ./vector data `
    -s ./result folder `
    -e "dem,height,canopy,kde,voxel" ` 
    -f "csv"
```

## **Usage Example**

```bash
cd c:/git/flai-tree-stats-lidar
pip install .
pip install -e.

flai_forest plots extract-params --help

flai_forest plots extract-params `
   -r "7.82" `
   -l "D:\flai-tree-stats-lidar\laz" `
   -p "D:\flai-tree-stats-lidar\shp\koordinate_3794.shp" `
   -s "D:\flai-tree-stats-lidar" `
   -e "dem" `
   -f "csv"
 ```

---
## **Per-tile raster mode (`rasters extract-params`)**

For wall-to-wall raster outputs computed directly from LAZ tiles with an
already-computed DEM, use the `rasters` group. Each output raster pixel is
treated as a square plot (default 10 m). One GeoTIFF is written per LAZ tile
per parameter, organized in subfolders by parameter name.

### Inputs
- A folder of LAZ/LAS tiles (e.g. 1 km × 1 km each, classified with ground
  and vegetation classes).
- *(Optional)* A folder of DEM GeoTIFFs whose filenames match the LAZ
  filenames (only the extension differs, e.g. `tile_001.laz` ↔
  `tile_001.tif`). When omitted, a DEM is built **on the fly** from each
  tile's ground-class returns at `--dem-pixel-size` (default 0.5 m) using
  multi-threaded k-NN inverse-distance weighting.

### Outputs
- `<save>/<param>/<laz_basename>.tif` — one float32 GeoTIFF per parameter
  per tile, snapped to a `pixel_size`-aligned grid, NaN where insufficient
  points were available.

### Quick start
```bash
# with a pre-built DEM folder
flai_forest rasters extract-params \
    -l ./laz \
    -d ./dem \
    -s ./out \
    -e "dem,height,canopy" \
    --pixel-size 10 \
    --workers 4 \
    --cell-workers 4

# without a DEM folder -- DEM is interpolated from ground returns at 0.5 m
flai_forest rasters extract-params \
    -l ./laz \
    -s ./out \
    -e "dem,height,canopy" \
    --pixel-size 10 \
    --dem-pixel-size 0.5
```

### Options
| Flag | Default | Description |
|------|---------|-------------|
| `-l, --lidar` | required | Folder of LAZ/LAS tiles. |
| `-d, --dem` | *(none)* | Folder of DEM TIFs (matching basenames). If omitted, a DEM is built on the fly per tile. |
| `-s, --save` | required | Output folder; gets one subfolder per parameter. |
| `-e, --extract` | `dem,height,canopy,kde,voxel` | Metric groups to compute. |
| `--pixel-size` | `10.0` | Output raster pixel size (in CRS units). Each pixel is treated as a square plot of this size. |
| `--dem-pixel-size` | `0.5` | DEM pixel size used when `--dem` is not provided. Also the scale at which slope/aspect are derived. |
| `--idw-power` | `2.0` | IDW power exponent for the on-the-fly DEM. |
| `--idw-k` | `12` | Number of nearest neighbours used per cell in the IDW DEM. |
| `-w, --workers` | `4` | Number of LAZ tiles processed in parallel (process pool). Keep low if storage I/O is the bottleneck. |
| `-t, --cell-workers` | `4` | Threads used for per-cell metric computation inside one tile and for the IDW kNN query. |
| `--blas-threads` | `1` | Cap BLAS / OpenMP / MKL thread pools per worker process. Default 1 gives predictable total core usage of `workers x cell_workers`. |
| `--ground-class` | `2` | LAS classification value for ground. |
| `--vegetation-classes` | `3,4,5` | Comma-separated LAS classification values for vegetation. |
| `--min-points` | `4` | Minimum vegetation point count per cell to compute metrics; cells below this are written as NaN. |
| `--skip-existing / --overwrite` | `skip-existing` | Skip a tile when every parameter TIF already exists. |
| `--progress / --no-progress` | `progress` | Show a tqdm progress bar with ETA. |
| `--crs` | *(none)* | Override the CRS for outputs when the LAZ has no SRS VLR. Accepts anything `rasterio.crs.CRS.from_user_input()` handles (e.g. `EPSG:3794`). |

### Parallelism layers

There are five layers of concurrency in play at runtime. Only the first two
are direct CLI knobs; the others are internal but worth knowing about so you
don't oversubscribe a machine.

| Layer | Controlled by | Notes |
|---|---|---|
| **Tile-level (process pool)** | `--workers N` | One whole LAZ tile per Python process. True CPU parallelism, separate memory per process. Main I/O contention point — keep low on spinning disks, bump up on NVMe. |
| **Cell-level (thread pool)** | `--cell-workers M` | Inside a tile, non-empty cells are split into chunks processed by M threads. NumPy-heavy metrics (`height`, `canopy`, `kde`) release the GIL and scale well; the `voxel` gap-analysis loop is GIL-bound and benefits less. |
| **IDW kNN query** | `--cell-workers M` (reused) | `scipy.cKDTree.query` uses M threads when building an on-the-fly DEM. |
| **NumPy / SciPy BLAS** | `--blas-threads N` (uses `threadpoolctl` + env vars in the worker) | Used by `scipy.stats.gaussian_kde`, `cKDTree`, and similar matrix operations. **Capped to 1 by default** so total core usage stays at `workers × cell_workers`. Raise it if BLAS-heavy metrics benefit from intra-tile multi-core matmul. |

DEM aggregation no longer goes through `rasterio.warp.reproject` (it now uses
a pure-numpy block average); supplied DEMs must be aligned with the output
grid (same upper-left origin, integer pixel-size ratio) or a clear
`ValueError` is raised.

Effective worst-case concurrency is roughly **`workers × cell_workers × BLAS_threads`**. With the defaults (4 × 4) plus uncapped BLAS that can saturate a 16-core machine. Recommended starting points:

- **NVMe / fast local storage, 16+ cores**: `-w 8 -t 2`, set `OPENBLAS_NUM_THREADS=1` to prevent BLAS oversubscription.
- **HDD or networked storage**: `-w 2 -t 4` — keep concurrent tile reads low.
- **Voxel-heavy runs** (GIL-bound metric): `-w 8 -t 1` — more processes, no inner threading needed.

### Resuming after failures
Each run writes two files at the top of `--save` if any tile fails:
- `failed_tiles.txt` — one LAZ path per line; feed it back into a re-run with
  `xargs` or by pointing `--lidar` at a directory of symlinks.
- `failed_tiles.csv` — `tile,laz_path,error` for diagnostics.

These files are cleared at the start of a clean run. A live `tqdm` progress
bar shows ok/skip/fail counts and ETA; use `--no-progress` to disable.

### Notes
- The **voxel** metric uses a vectorised implementation in
  `rasters/voxel_fast.py` that produces bit-identical results to
  `metrices/voxel.py` but is ~20× faster per cell. The plot-based workflow
  still uses the original.
- Tiles are processed **self-contained** (no neighbour reads). Expect minor
  edge effects on the outermost row/column of each tile for slope/aspect.
- Vegetation Z is normalised by subtracting the DEM at native resolution at
  every point's `(x, y)`.
- Slope / aspect / eastness / northness are derived **at the DEM's native
  resolution** (e.g. 50 cm) and then averaged onto the 10 m output grid
  (aspect is recovered via `atan2(mean(eastness), mean(northness))` so the
  circular average is correct). On rough terrain this is order-of-magnitude
  closer to the true cell-mean slope than the alternative of averaging the
  DEM first and deriving on the coarse grid.
- When the DEM is computed on the fly, slope/aspect carry the noise of the
  IDW reconstruction (≈ `point_noise / dem_pixel_size` per cell). Use a
  cleaner supplied DEM if precise topography is needed.
- `voxel` and to a lesser extent `kde` are much slower than `dem/height/canopy`.
  Start with `-e dem,height,canopy` for a fast first pass and add the heavy
  groups only when needed.


## **Notes**

- Ensure that input data paths are correct and accessible.
- For more details on specific metrics, refer to the respective sections above.
- If you encounter any issues, check the logs for detailed error messages.
