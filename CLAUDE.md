# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **3D Object Detection and Point Cloud Processing Pipeline** for autonomous driving datasets (NuScenes and KITTI). The system is built as a Streamlit web application that enables 2D-to-3D projection, ground plane removal, clustering, and 3D bounding box generation with evaluation capabilities.

## Running the Application

```bash
# Run the Streamlit application
streamlit run app.py

# Run with Docker
docker-compose -f docker-compose.streamlit.yml up

# Build Docker image
docker build -t 3d-detection-pipeline .
```

The app runs on port 8501 by default. Dataset paths are configured in the loaders:
- NuScenes: `dataset/nuscenes`
- KITTI: `dataset/kitti`

## Core Architecture

### Pipeline Flow
1. **Dataset Loading** → 2. **2D Bounding Box Extraction** → 3. **2D-to-3D Frustum Projection** → 4. **Ground Plane Removal (RANSAC)** → 5. **Point Cloud Clustering** → 6. **3D Cuboid Generation** → 7. **Evaluation & Visualization**

### Key Modules and Responsibilities

**Dataset Loaders:**
- `nuscenes_dataset_loader.py`: Loads NuScenes mini dataset, returns standardized sample data
- `kitti_dataset_loader.py`: Loads KITTI dataset, converts ground truth to LiDAR coordinates, provides 2D/3D bounding boxes

Both loaders return a common data format:
```python
{
    "sample_index": int,
    "image_path": str,
    "point_cloud": np.ndarray,  # Nx3
    "camera_intrinsic": np.ndarray,  # 3x3
    "camera_extrinsic": np.ndarray,  # 4x4
    "camera_to_lidar_transform": np.ndarray,  # 4x4
    "ground_truth_boxes": List[Dict]  # KITTI only
}
```

**Point Cloud Processing:**
- `pointcloud_projection.py`: Contains `Projection` class for 2D-to-3D ray projection, frustum plane computation, and point filtering
- `frustum_manager.py`: Manages frustum creation from 2D bboxes, handles per-frustum clustering workflow, provides `FrustumManager` and `Frustum` dataclasses

**Clustering:**
- `clustering_manager.py`: Unified interface for 5 clustering algorithms (DBSCAN, HDBSCAN, BIRCH, Agglomerative, OPTICS)
- Contains `KITTI_CUBOID_TEMPLATES` with median dimensions for each object category
- Generates 3D bounding boxes (cuboids) from clusters using template dimensions

**Evaluation:**
- `evaluation.py`: Provides `CuboidMatcher` for matching detected cuboids to ground truth
- Implements 3D IoU calculation (both axis-aligned and oriented BEV)
- Computes Precision, Recall, F1-Score, TP/FP/FN metrics

**Visualization:**
- `visualization_helper.py`: Plotly-based 3D scatter plots, cuboid meshes, frustum visualization
- `bounding_boxes.py`: Extracts and manages 2D/3D bounding boxes from NuScenes

**UI Application:**
- `app.py`: Main Streamlit application with multi-page interface
  - Session state manages: `point_cloud`, `clustering_results`, `frustums`, `per_frustum_clusters`
  - Automatically triggers frustum-based clustering when KITTI data with 2D bboxes is loaded

## Frustum-Based Clustering (Key Feature)

When KITTI dataset is loaded with 2D bounding boxes, the system **automatically** filters point clouds using 3D frustum projections before clustering. This ensures clustering happens only on points within each detected object's field of view.

**How It Works:**
1. Load KITTI sample with ground truth 2D bboxes
2. System computes frustums: camera origin + 4 base corners at specified depth
3. When any clustering button is clicked → automatic frustum filtering
4. Each frustum is clustered independently using HDBSCAN
5. Results stored in `st.session_state.per_frustum_clusters`

**Data Structures:**
```python
# st.session_state.frustums
[{
    'idx': int,
    'camera_origin': np.ndarray,  # 3D camera position
    'base_corners': np.ndarray,  # 4x3 frustum base corners
    'category': str,
    'bbox_2d': dict  # {left, top, right, bottom}
}, ...]

# st.session_state.per_frustum_clusters
[{
    'frustum_idx': int,
    'category': str,
    'points': np.ndarray,
    'labels': np.ndarray,
    'n_points': int,
    'n_clusters': int,
    'status': str  # 'success', 'too_few_points', or error
}, ...]
```

**Relevant Functions:**
- `compute_frustums_from_kitti(sample_data, depth)` in app.py - Creates frustums from 2D bboxes
- `run_frustum_clustering(points, frustums, ...)` in app.py - Runs per-frustum HDBSCAN
- `filter_points_in_frustum(points, camera_origin, base_corners)` in pointcloud_projection.py

## Important Implementation Details

### Coordinate Systems
- **KITTI**: Camera coordinates → LiDAR coordinates transformation via `camera_to_lidar_transform`
- **NuScenes**: Uses quaternions and ego vehicle frame for transformations
- All visualizations and clustering happen in **LiDAR coordinates**

### Ground Truth Handling
- KITTI ground truth boxes are stored in camera coordinates
- `_load_ground_truth_cuboids()` transforms 8-corner boxes to LiDAR frame
- Each cuboid dict includes: `category`, `corners` (8x3), `bbox_2d`, and min/max bounds

### Clustering Result Storage
Clustering results include an `is_frustum_based` flag:
```python
st.session_state.clustering_results['dbscan'] = {
    'labels': None,  # None for frustum-based
    'is_frustum_based': True,
    'per_frustum_clusters': [...],
    'bbox_results': [...],
    'params': {...},
    'runtime': float
}
```

### Evaluation Workflow
1. `CuboidMatcher.match_cuboids()` matches detected to GT based on center distance + category
2. Computes 3D IoU for matched pairs
3. `need_review` flag set when IoU < threshold (default 0.25)
4. Per-category metrics displayed in UI

## Configuration Files

- `.streamlit/config.toml`: Streamlit server settings (port 8501, headless mode)
- `requirements.txt`: Core dependencies including nuscenes-devkit, open3d, scikit-learn, hdbscan
- `Dockerfile`: Multi-stage build for deployment (Python 3.10-slim base)
- `docker-compose.streamlit.yml`: Local Docker deployment configuration

## Dataset Structure

**KITTI Expected Layout:**
```
dataset/kitti/
├── training/
│   ├── image_2/      # PNG images (000000.png, ...)
│   ├── velodyne/     # BIN point clouds
│   ├── calib/        # TXT calibration files
│   └── label_2/      # TXT annotation files
```

**NuScenes Expected Layout:**
```
dataset/nuscenes/
├── v1.0-mini/        # Version metadata
├── samples/          # Sensor data
└── sweeps/           # Intermediate frames
```

## Common Workflows

**Adding a New Clustering Algorithm:**
1. Add to `ClusteringManager.DEFAULT_PARAMS` in clustering_manager.py
2. Implement in `run_clustering()` method
3. Add UI controls in app.py clustering page
4. Update `_generate_cuboids_from_labels()` if needed

**Modifying Cuboid Templates:**
- Edit `KITTI_CUBOID_TEMPLATES` in clustering_manager.py
- Templates use format: `{'length': X, 'width': Y, 'height': Z}` in meters
- To analyze KITTI dimensions: use `analyze_kitti_dimensions.py`

**Adding New Visualization:**
- Primitive mesh creators in visualization_helper.py: `cuboid_from_minmax()`, `frustum_mesh()`
- Higher-level functions: `create_3d_scatter_plot()`, `add_cuboids_to_figure()`
- All use Plotly graph_objects for 3D rendering
