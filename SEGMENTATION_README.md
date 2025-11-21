# Segmentation-based 3D Object Detection

This document explains how to use the segmentation-based approach for 3D object detection in point clouds, which can be used as an alternative to the bounding box-based approach.

## Overview

The segmentation-based approach uses semantic segmentation models like SAM2 or DeepLabv3 to identify objects in 2D images, then projects these segmentation masks onto the 3D LiDAR point cloud. This can provide more precise object boundaries compared to bounding boxes.

## Key Components

1. **SegmentationDetector** (`segmentation_detection.py`)
   - Supports SAM2 and DeepLabv3 models
   - Generates segmentation masks from 2D images
   - Extracts pixel coordinates from masks

2. **SegmentationToPointCloud** (`segmentation_detection.py`)
   - Projects 2D segmentation masks to 3D point clouds
   - Handles multiple masks in a single segmentation
   - Provides methods for individual and batch projection

3. **NuScenesAnnotationLoader** (`nuscenes_annotation_loader.py`)
   - Loads ground truth annotations from nuScenes dataset
   - Transforms annotations to LiDAR coordinate frame
   - Provides visualization tools for annotations

4. **PointCloud Extensions** (`pointcloud_projection.py`)
   - `cluster_with_segmentation_masks()`: Creates clusters based on segmentation masks
   - `add_segmentation_projected_points()`: Adds projected points to the point cloud

## Why DBSCAN Produces Few Clusters

The DBSCAN algorithm in your current implementation might produce few clusters due to several reasons:

1. **Parameter Settings**:
   - `eps` (epsilon) might be too small, causing points to be considered noise
   - `min_samples` might be too high, requiring more points to form a cluster

2. **Point Cloud Density**:
   - After ground plane removal, the remaining points might be sparse
   - The distance between points might exceed the `eps` threshold

3. **Projection Issues**:
   - The 2D-to-3D projection might not be accurate enough
   - Points from the same object might be projected too far apart

## Solutions

1. **Use Adaptive DBSCAN**:
   ```python
   pointcloud.cluster_with_dbscan_adaptive(k=5, min_samples=10, percentile=50.0)
   ```

2. **Adjust Parameters**:
   ```python
   pointcloud.cluster_with_dbscan(eps=1.0, min_samples=5)  # Increase eps, decrease min_samples
   ```

3. **Use Segmentation-based Approach**:
   - Provides more precise object boundaries
   - Can handle complex shapes better than bounding boxes
   - Less sensitive to parameter tuning

## Installation

### For SAM2
```bash
pip install git+https://github.com/facebookresearch/segment-anything-2.git
```

### For DeepLabv3
```bash
pip install torchvision
```

## Usage Example

```python
# Initialize segmentation detector
detector = SegmentationDetector(model_type="sam2")  # or "deeplabv3"

# Get segmentation mask
if detector.model_type == "sam2":
    # For SAM2, provide prompts
    prompts = {
        "points": np.array([[x1, y1], [x2, y2]]),  # Point coordinates
        "point_labels": np.array([1, 1])  # 1 for positive points
    }
    mask = detector.get_segmentation_mask(image, prompts)
else:
    # For DeepLabv3, no prompts needed
    mask = detector.get_segmentation_mask(image)

# Project mask to 3D
seg_to_3d = SegmentationToPointCloud(projection)
mask_points = seg_to_3d.project_all_masks(mask)

# Add to point cloud and cluster
pointcloud.add_segmentation_projected_points(mask_points)
clusters = pointcloud.cluster_with_segmentation_masks(mask_points)
```

## Comparison with Bounding Box Approach

| Aspect | Bounding Box Approach | Segmentation Approach |
|--------|----------------------|-----------------------|
| **Precision** | Rectangular boundaries | Precise object boundaries |
| **Implementation** | Simple | More complex |
| **Performance** | Faster | Slower (due to segmentation) |
| **Parameter Sensitivity** | High (DBSCAN parameters) | Lower (uses segmentation masks) |
| **Handling Complex Shapes** | Poor | Good |

## Getting LiDAR Annotations from nuScenes

The `NuScenesAnnotationLoader` class provides functionality to load and visualize ground truth annotations:

```python
# Initialize annotation loader
ann_loader = NuScenesAnnotationLoader(nusc)

# Get annotations for a sample
annotations = ann_loader.get_lidar_annotations(sample_token)

# Get annotation points for visualization
annotation_points = ann_loader.get_annotation_points(annotations)

# Visualize annotations
visualizer = AnnotationVisualizer(point_cloud)
visualizer.visualize_with_annotations(annotations, annotation_points)
```

## Visualizing Annotations with Existing Visualization

You can add ground truth annotations to your existing visualization using the `AnnotationVisualizer` class:

```python
# Compare annotations with detected clusters
ann_visualizer.visualize_annotation_comparison(
    annotation_points,
    clusters,
    title="Annotations vs Detected Clusters"
)
```

This will show:
- Ground truth annotation points in red
- Detected clusters in different colors
- The original point cloud in gray

## Running the Example

```bash
python example_segmentation_usage.py
```

This will:
1. Load a nuScenes sample
2. Get ground truth annotations
3. Generate segmentation masks
4. Project masks to 3D
5. Cluster based on masks
6. Visualize results and compare with ground truth

## Tips and Best Practices

1. **For SAM2**:
   - Provide good prompts (points inside objects)
   - Use multiple points for large objects
   - Consider using automatic mask generation for better results

2. **For DeepLabv3**:
   - Focus on specific classes (e.g., cars, pedestrians)
   - Filter masks by area to remove small noise
   - Consider using class-specific colors for visualization

3. **For Clustering**:
   - Adjust `min_points_per_cluster` based on object size
   - Consider using a distance threshold when matching projected points to the point cloud
   - Filter clusters by size to remove noise

4. **For Evaluation**:
   - Compare with ground truth annotations
   - Use metrics like IoU (Intersection over Union) for 3D boxes
   - Consider both precision and recall for evaluation
