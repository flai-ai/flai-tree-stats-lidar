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


## **Notes**

- Ensure that input data paths are correct and accessible.
- For more details on specific metrics, refer to the respective sections above.
- If you encounter any issues, check the logs for detailed error messages.
