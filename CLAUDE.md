# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a 3D object detection and point cloud processing pipeline for autonomous driving datasets, specifically designed for the nuScenes dataset. The project provides tools for:
- Loading nuScenes camera and LiDAR data
- Projecting 2D bounding box pixels to 3D point clouds
- Ground plane removal using RANSAC
- Point cloud clustering with DBSCAN
- Visualization of point clouds, rays, and clusters

## Key Architecture

### Core Pipeline Components

The pipeline follows this flow: **2D Image → Bounding Boxes → Pixel Extraction → Ray Projection → 3D Point Cloud → Ground Removal → Clustering**

1. **Data Loading** ([nuscenes_dataset_loader.py](nuscenes_dataset_loader.py))
   - `NuScenesDatasetLoader`: Loads nuScenes dataset and extracts camera/LiDAR data
   - Computes transformation matrices: camera intrinsic (K), camera extrinsic (T_cam_to_ego), camera-to-LiDAR transform (T_cam_to_lidar)
   - Returns synchronized camera images with corresponding point clouds

2. **Bounding Box Extraction** ([bounding_boxes.py](bounding_boxes.py))
   - `BoundingBoxes`: Extracts 2D bounding boxes from nuScenes annotations
   - Projects 3D boxes to 2D image plane using camera intrinsics
   - `BoundingBox`: Represents individual boxes with 2D/3D coordinates and metadata
   - `get_all_bb_pixels()`: Samples pixels inside bounding boxes (default: 10% sampling ratio)

3. **2D-to-3D Projection** ([pointcloud_projection.py](pointcloud_projection.py))
   - `Projection2DTo3D`: Core projection class that transforms 2D pixels to 3D rays in LiDAR coordinate space
   - **Coordinate System Transforms**: Handles complex transforms between camera, ego vehicle, and LiDAR coordinate frames
   - `pixel_to_ray()`: Back-projects 2D pixels to 3D rays using inverse camera intrinsics
   - `find_closest_point_on_ray()`: Finds nearest point cloud points to each ray using distance thresholding
   - **Key Parameters**:
     - `max_distance`: Maximum ray extension (default: 100m)
     - `distance_threshold`: Max perpendicular distance to consider a point on the ray (default: 0.5m)

4. **Point Cloud Processing** ([pointcloud_projection.py](pointcloud_projection.py))
   - `PointCloud`: Manages point cloud data and operations
   - `remove_ground_plane_ransac()`: Uses Open3D's RANSAC to segment and remove ground plane
   - `cluster_with_dbscan()`: Clusters non-ground points with configurable eps and min_samples
   - `cluster_with_dbscan_adaptive()`: Automatically determines eps using k-nearest neighbor distances

5. **Visualization** ([pointcloud_projection.py](pointcloud_projection.py))
   - `PointCloudVisualizer`: Open3D-based 3D visualization
   - `visualize_point_cloud()`: Displays point clouds with color-coded clusters, rays, and coordinate systems
   - `visualize_image_with_pixels()`: 2D image visualization with marked projection pixels

### Coordinate System Handling

The project manages multiple coordinate frames:
- **LiDAR coordinates**: Origin at LiDAR sensor (typically vehicle center)
- **Camera coordinates**: X-right, Y-down, Z-forward (standard camera convention)
- **Ego vehicle coordinates**: Vehicle body frame

Transformation chain: `Pixel → Camera → Ego Vehicle → LiDAR`

## Dataset Structure

The project expects nuScenes dataset in this structure:
```
v1.0-mini/
  v1.0-mini/
    *.json (metadata files)
  samples/
    CAM_FRONT/, CAM_BACK/, etc. (camera images)
    LIDAR_TOP/ (point cloud data)
```

Both `v1.0-mini` and `v1.0-trainval_meta` datasets are present in the repository.

## Running the Pipeline

Execute the full pipeline with the example script:
```bash
python example_projection_usage.py
```

This script demonstrates:
1. Loading a nuScenes sample
2. Extracting bounding boxes
3. Projecting 2D pixels to 3D space
4. Ground plane removal
5. DBSCAN clustering
6. Visualization at each stage

## Key Parameters and Tuning

**RANSAC Ground Removal** ([pointcloud_projection.py:209](pointcloud_projection.py#L209)):
- `distance_threshold`: 0.3m (typical for flat ground)
- `num_iterations`: 1000 (increase for noisy data)

**DBSCAN Clustering** ([pointcloud_projection.py:261](pointcloud_projection.py#L261)):
- `eps`: Neighborhood radius (1.0m default; reduce for dense scenes, increase for sparse)
- `min_samples`: Minimum points per cluster (10 default)
- Note: Current implementation may produce too few clusters; consider using `cluster_with_dbscan_adaptive()` for automatic tuning

**Projection Parameters** ([pointcloud_projection.py:106](pointcloud_projection.py#L106)):
- `max_distance`: 100m (match to LiDAR range)
- `distance_threshold`: 0.5m (tighter threshold = stricter ray-point matching)

**Bounding Box Sampling** ([bounding_boxes.py:152](bounding_boxes.py#L152)):
- `sample_ratio`: 0.1 (10% of pixels per box; increase for denser sampling)

## Python Environment

The project uses a virtual environment (`.venv`) with key dependencies:
- `nuscenes-devkit`: nuScenes dataset API
- `open3d`: 3D visualization and RANSAC
- `scikit-learn`: DBSCAN clustering
- `ultralytics`: YOLO models (referenced but not actively used in core pipeline)
- `opencv-python`, `matplotlib`: Image processing and visualization
- `pyquaternion`: Rotation handling

Activate environment before development:
```bash
# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

## Known Issues and Limitations

1. **Clustering Performance**: Current DBSCAN parameters may produce too few clusters. The adaptive clustering method is recommended but needs validation.
2. **No Tests**: The repository has no test suite. Manual validation is required by inspecting visualizations.
3. **Single Sample Focus**: `example_projection_usage.py` processes only the first sample. Batch processing is not implemented.
4. **Ground Plane Assumption**: RANSAC assumes a single dominant ground plane, which may fail in complex terrain.

## Code Organization Notes

- All main functionality is in 4 Python files at the root level
- No modular package structure (no `src/` or module directories)
- Example usage is the primary entry point for understanding the pipeline
- Heavy use of numpy arrays with shape conventions: Nx2 for pixels, Nx3 for 3D points, 4x4 for transforms
