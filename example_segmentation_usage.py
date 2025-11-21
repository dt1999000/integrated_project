"""
Example usage of segmentation-based object detection with 3D point cloud projection.
This script demonstrates how to use SAM2 or DeepLabv3 to get segmentation masks
and project them onto 3D LiDAR scenes, with comparison to ground truth annotations.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Optional

# Import our custom modules
from nuscenes_dataset_loader import NuScenesDatasetLoader
from pointcloud_projection import Projection2DTo3D, PointCloud, PointCloudVisualizer
from segmentation_detection import SegmentationDetector, SegmentationToPointCloud
from nuscenes_annotation_loader import NuScenesAnnotationLoader, AnnotationVisualizer

def main():
    """Main function demonstrating segmentation-based object detection."""
    
    print("="*70)
    print("SEGMENTATION-BASED OBJECT DETECTION WITH 3D POINT CLOUD PROJECTION")
    print("="*70)
    
    # Initialize dataset loader
    print("\n1. Loading nuScenes dataset...")
    loader = NuScenesDatasetLoader(dataroot="v1.0-mini", version="v1.0-mini")
    loader.load_dataset()
    
    # Get a sample
    sample_token = loader.nusc.sample[0]['token']
    print(f"Using sample: {sample_token}")
    
    # Get synchronized camera and LiDAR data
    data = loader.load_nuscenes_data(sample_token)
    
    image = cv2.imread(data['image_path'])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    point_cloud = data['point_cloud']    
   
    # Initialize 2D to 3D projection
    print("\n2. Initializing 2D to 3D projection...")
    projection = Projection2DTo3D(
        camera_intrinsic=data['camera_intrinsic'],
        camera_extrinsic=data['camera_extrinsic'],
        camera_to_lidar_transform=data['camera_to_lidar_transform'],
        point_cloud=point_cloud
    )
    
    # Initialize point cloud processor
    pointcloud = PointCloud(point_cloud)
    
    # Remove ground plane
    print("\n3. Removing ground plane...")
    pointcloud.remove_ground_plane_ransac(distance_threshold=0.3, num_iterations=1000)
    
    # Load ground truth annotations
    print("\n4. Loading ground truth annotations...")
    ann_loader = NuScenesAnnotationLoader(loader.nusc)
    annotations = ann_loader.get_lidar_annotations(sample_token)
    print(f"Found {len(annotations)} annotations")
    
    # Get annotation points for visualization
    annotation_points = ann_loader.get_annotation_points(annotations, num_points_per_box=50)
    print(f"Generated points for {len(annotation_points)} annotations")
    
    # Initialize segmentation detector
    print("\n5. Initializing segmentation detector...")
    try:
        # Try SAM2 first
        detector = SegmentationDetector(model_type="sam2")
        print("Using SAM2 for segmentation")
    except (ValueError, ImportError):
        try:
            # Fall back to DeepLabv3
            detector = SegmentationDetector(model_type="deeplabv3")
            print("Using DeepLabv3 for segmentation")
        except (ValueError, ImportError):
            print("Neither SAM2 nor DeepLabv3 is available. Please install one of them.")
            return
    
    # Get segmentation mask
    print("\n6. Getting segmentation mask...")
    try:
        if detector.model_type == "sam2":
            # For SAM2, we need prompts. Let's use a simple grid of points as prompts
            h, w = image.shape[:2]
            grid_points = []
            grid_labels = []
            
            # Create a grid of points
            grid_size = 10
            for i in range(grid_size):
                for j in range(grid_size):
                    x = int((i + 0.5) * w / grid_size)
                    y = int((j + 0.5) * h / grid_size)
                    grid_points.append([x, y])
                    grid_labels.append(1)  # Positive point
            
            prompts = {
                "points": np.array(grid_points),
                "point_labels": np.array(grid_labels)
            }
            
            mask = detector.get_segmentation_mask(image, prompts)
        else:
            # For DeepLabv3, we don't need prompts
            mask = detector.get_segmentation_mask(image)
        
        print(f"Segmentation mask shape: {mask.shape}")
        print(f"Unique mask values: {np.unique(mask)}")
        
        # Extract pixels from mask
        # For DeepLabv3, we might want to focus on specific classes (e.g., cars, pedestrians)
        # For simplicity, we'll use all non-zero pixels
        mask_pixels = detector.get_mask_pixels(mask, min_area=100)
        print(f"Extracted {len(mask_pixels)} pixels from segmentation mask")
        
    except Exception as e:
        print(f"Error getting segmentation mask: {e}")
        return
    
    # Initialize segmentation to point cloud projection
    print("\n7. Projecting segmentation mask to 3D...")
    seg_to_3d = SegmentationToPointCloud(projection)
    
    # Project all masks to 3D
    mask_points = seg_to_3d.project_all_masks(mask)
    print(f"Projected {len(mask_points)} masks to 3D")
    
    for mask_id, points in mask_points.items():
        print(f"  Mask {mask_id}: {len(points)} points")
    
    # Add projected points to point cloud
    print("\n8. Adding projected points to point cloud...")
    pointcloud.add_segmentation_projected_points(mask_points)
    
    # Cluster based on segmentation masks
    print("\n9. Clustering based on segmentation masks...")
    clusters = pointcloud.cluster_with_segmentation_masks(mask_points, min_points_per_cluster=10)
    
    # Visualize results
    print("\n10. Visualizing results...")
    
    # First, visualize ground truth annotations
    print("Visualizing ground truth annotations...")
    ann_visualizer = AnnotationVisualizer(pointcloud.original_point_cloud)
    ann_visualizer.visualize_with_annotations(
        annotations, 
        annotation_points,
        title="Ground Truth Annotations"
    )
    
    # Visualize with clusters
    print("Visualizing segmentation-based detection...")
    visualizer = PointCloudVisualizer(point_cloud=pointcloud)
    visualizer.visualize_point_cloud(
        points=None,
        rays=None,
        clusters=clusters,
        title="Segmentation-based 3D Object Detection"
    )
    
    # Compare annotations with detected clusters
    print("Comparing annotations with detected clusters...")
    ann_visualizer.visualize_annotation_comparison(
        annotation_points,
        clusters,
        title="Annotations vs Detected Clusters"
    )
    
    # Also visualize the segmentation mask
    from segmentation_detection import visualize_segmentation_with_projection
    visualize_segmentation_with_projection(
        image=image,
        mask=mask,
        projected_points=mask_points,
        save_path="segmentation_with_projection.png"
    )
    
    print("\n" + "="*70)
    print("SEGMENTATION-BASED OBJECT DETECTION COMPLETED")
    print("="*70)


if __name__ == "__main__":
    main()
