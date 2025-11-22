# SAM Integration with 3D Point Cloud Pipeline

This document explains how to use the SAM (Segment Anything Model) integration with the 3D point cloud pipeline.

## Overview

The SAM integration provides a more advanced approach to object segmentation and 3D projection compared to traditional bounding boxes. It supports multiple SAM models and flexible prompting methods.

## Key Components

### 1. SAM Integration Module (`sam_integration.py`)

#### SAMModelManager
- Manages different SAM models (SAM2-t, SAM2-b, SAM2-l, SAM-B, MobileSAM)
- Handles model loading and initialization
- Provides methods for prediction with different prompt types

#### BoundingBoxToSAM
- Converts bounding boxes to SAM prompts
- Extracts center points from bounding boxes for use as prompts

### 2. Updated Point Cloud Projection (`pointcloud_projection.py`)

#### New Methods Added:
- `cluster_with_sam_masks()`: Creates clusters based on SAM segmentation masks
- Integrates with existing projection pipeline

### 3. Streamlit Apps

#### Main App (`app.py`)
- Integrated SAM model selection and segmentation
- Supports both traditional clustering and SAM-based approaches

#### SAM App (`sam_app.py`)
- Dedicated app for SAM-based segmentation and projection
- More focused on SAM features and workflows

## Usage Examples

### Basic SAM Segmentation Workflow

```python
# Initialize SAM model manager
sam_manager = SAMModelManager(model_type="sam2_t")

# Load dataset
dataset_loader = NuScenesDatasetLoader(dataroot='v1.0-mini')
dataset_loader.load_dataset()
sample_token = dataset_loader.nusc.sample[0]['token']
sample_data = dataset_loader.load_nuscenes_data(sample_token)

# Load image
image = cv2.imread(sample_data['image_path'])
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Initialize point cloud
point_cloud = PointCloud()
point_cloud.set_points(sample_data['point_cloud'])
point_cloud.remove_ground_plane_ransac()
point_cloud.remove_ego_vehicle_points()

# Initialize projection
projection = Projection2DTo3D(
    camera_intrinsic=sample_data['camera_intrinsic'],
    camera_extrinsic=sample_data['camera_extrinsic'],
    camera_to_lidar_transform=sample_data['camera_to_lidar_transform'],
    point_cloud=sample_data['point_cloud'],
    image=image
)

# Segment with bounding boxes
bbox_extractor = BoundingBoxes(nusc=dataset_loader.nusc)
bboxes = bbox_extractor.get_boxes_for_sample(sample_token, "CAM_FRONT")

# Convert to SAM prompts
bbox_to_sam = BoundingBoxToSAM(sam_manager)
mask = bbox_to_sam.segment_from_bboxes(image, bboxes)

# Project to 3D
seg_to_3d = SegmentationToPointCloud(projection)
mask_points = seg_to_3d.project_all_masks(mask)

# Add to point cloud and cluster
point_cloud.add_segmentation_projected_points(mask_points)
clusters = point_cloud.cluster_with_sam_masks(sam_manager, image, bboxes=bboxes)
```

### Point-based SAM Segmentation

```python
# Initialize SAM model manager
sam_manager = SAMModelManager(model_type="sam2_t")

# Load image
image = cv2.imread("image.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Segment with point prompts
points = [[100, 100], [200, 200]]  # Example points
labels = [1, 1]  # Positive points
mask = sam_manager.predict_from_points(image, points, labels)
```

### Automatic SAM Segmentation

```python
# Initialize SAM model manager
sam_manager = SAMModelManager(model_type="sam2_t")

# Load image
image = cv2.imread("image.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Automatic segmentation
results = sam_manager.predict(image)
mask = sam_manager.get_segmentation_masks(results)
```

## Running the Apps

### Main App (with DBSCAN and SAM)
```bash
streamlit run app.py
```

### SAM-focused App
```bash
streamlit run sam_app.py
```

## Model Comparison

| Model | Description | Use Case | Performance |
|--------|-------------|---------|-----------|
| SAM2-t | Highest accuracy, supports memory | Best |
| SAM2-b | Balanced performance | Good |
| SAM2-l | Faster inference, slightly lower accuracy | Good |
| SAM-B | Standard SAM model | Fair |
| MobileSAM | Optimized for mobile/edge devices | Fastest |

## Integration with Existing Pipeline

The SAM integration is designed to work seamlessly with the existing pipeline:

1. **Data Loading**: Uses the same `NuScenesDatasetLoader` as before
2. **Point Cloud Processing**: Uses the same `PointCloud` class with ground removal
3. **Projection**: Uses the same `Projection2DTo3D` class for 2D-to-3D projection
4. **Clustering**: Adds SAM-based clustering to complement existing DBSCAN methods

## Advantages of SAM Integration

1. **Precise Segmentation**: More accurate object boundaries compared to rectangular bounding boxes
2. **Flexible Prompts**: Support for bounding boxes, points, or automatic segmentation
3. **Memory Updates**: SAM2 models support memory for consistent object tracking
4. **Better 3D Projection**: More accurate 2D-to-3D projection with precise masks

## Tips and Best Practices

1. **Model Selection**:
   - Use SAM2-t for highest accuracy
   - Use SAM2-b for balanced performance
   - Use MobileSAM for faster inference on edge devices

2. **Segmentation Method**:
   - Use bounding boxes for known object locations
   - Use point prompts for interactive segmentation
   - Use automatic for general scene understanding

3. **Parameter Tuning**:
   - Adjust distance threshold in projection (0.3-0.7m) based on segmentation quality
   - Adjust minimum points per cluster based on object size

4. **Performance Optimization**:
   - Use smaller images for faster processing
   - Consider using GPU acceleration if available
   - Cache model predictions for repeated use

## Troubleshooting

### Common Issues

1. **Model Loading Errors**:
   - Ensure ultralytics is installed: `pip install ultralytics`
   - Check model file availability

2. **Projection Errors**:
   - Verify camera intrinsics and extrinsics
   - Check coordinate system transformations

3. **Clustering Issues**:
   - Adjust minimum points per cluster
   - Check distance threshold for point matching
   - Verify ground plane removal

### Debug Mode

Enable debug mode by setting environment variable:
```bash
export SAM_DEBUG=True
```

This will provide additional logging and visualization for troubleshooting.
