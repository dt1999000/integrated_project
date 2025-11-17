"""
Example script showing how to use the 2D to 3D projection with nuScenes dataset.
"""

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes_dataset_loader import NuScenesDatasetLoader
from pointcloud_projection import Projection2DTo3D, PointCloudVisualizer
from bounding_boxes import BoundingBoxes
from typing import Optional



def get_bounding_box_corners(bbox_extractor: BoundingBoxes, sample_token: str, 
                             camera_channel: str = "CAM_FRONT",
                             max_boxes: Optional[int] = None) -> np.ndarray:
    """
    Get corner pixels of bounding boxes using BoundingBoxes extractor.
    
    Args:
        bbox_extractor: BoundingBoxes object for extracting bounding boxes
        sample_token: Token of the sample
        camera_channel: Camera channel name
        max_boxes: Maximum number of bounding boxes to process (None for all)
        
    Returns:
        Nx2 array of pixel coordinates (corners of all bounding boxes)
    """
    bb_pixels = bbox_extractor.get_all_bb_pixels(
        sample_token=sample_token,
        camera_channel=camera_channel,
        max_boxes=max_boxes
    )
    
    if len(bb_pixels) == 0:
        print("Warning: No valid bounding boxes found. Using default pixels.")
        return np.array([
            [800, 450],   # Center
            [100, 100],   # Top-left
            [1500, 100],  # Top-right
            [100, 800],   # Bottom-left
            [1500, 800]   # Bottom-right
        ])
    
    return bb_pixels


def main():
    """Main example function."""
    
    # Initialize nuScenes
    dataroot = "v1.0-mini"
    version = "v1.0-mini"
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)
    
    # Get first sample
    sample_token = nusc.sample[0]['token']
    
    print("Loading nuScenes data...")
    dataset_loader = NuScenesDatasetLoader(
        dataroot=dataroot,
        version=version,
        verbose=True
    )
    dataset_loader.load_dataset()
    data = dataset_loader.load_nuscenes_data(sample_token, camera_channel="CAM_FRONT")
    
    # Create visualizer first to remove ground plane
    print("Creating point cloud visualizer and removing ground plane...")
    coord_systems_temp = None  # Will get from projector later
    visualizer_temp = PointCloudVisualizer(
        point_cloud=data['point_cloud'],
        coordinate_systems=coord_systems_temp,
        remove_ground=True  # Remove ground using RANSAC
    )
    
    # Get ground-removed point cloud
    filtered_point_cloud = visualizer_temp.point_cloud
    
    # Create projection object with ground-removed point cloud
    print("Creating projection object with ground-removed point cloud...")
    projector = Projection2DTo3D(
        camera_intrinsic=data['camera_intrinsic'],
        camera_extrinsic=data['camera_extrinsic'],
        camera_to_lidar_transform=data['camera_to_lidar_transform'],
        point_cloud=filtered_point_cloud
    )
    
    # Create bounding box extractor
    print("Creating bounding box extractor...")
    bbox_extractor = BoundingBoxes(nusc=data['nusc'], data_format="nuscenes")
    
    # Get corner pixels from bounding boxes
    print("Extracting bounding box corner pixels from nuScenes annotations...")
    pixels = get_bounding_box_corners(
        bbox_extractor=bbox_extractor,
        sample_token=sample_token,
        camera_channel="CAM_FRONT",
        max_boxes=1
    )
    
    print(f"Found {len(pixels)} corner pixels from bounding boxes")
    
    print("Projecting pixels to 3D...")
    result = projector.project_pixels_to_3d(pixels, max_distance=100.0, distance_threshold=1.0)
    
    print(f"Projected {len(result['projected_points'])} points")
    print("Projected 3D points:")
    for i, point in enumerate(result['projected_points']):
        print(f"  Pixel {pixels[i]}: 3D point {point}")
    
    # Get coordinate systems from projector and update visualizer
    print("Finalizing visualization...")
    coord_systems = projector.get_coordinate_systems()
    visualizer = PointCloudVisualizer(
        point_cloud=filtered_point_cloud,
        coordinate_systems=coord_systems,
        remove_ground=False  # Already removed
    )
    
    # Visualize image with marked pixels
    print("Visualizing image with projected pixels...")
    visualizer.visualize_image_with_pixels(
        image=data['image_path'],
        pixel_coords=pixels,
        save_path="image_with_pixels.png",
        show=False
    )
    
    # Optional: Perform DBSCAN clustering
    print("Performing DBSCAN clustering...")
    clusters = visualizer.cluster_with_dbscan(eps=1.0, min_samples=10)
    
    # Visualize with projected points, coordinate systems, and clusters
    visualizer.visualize_with_projected_points(
        result['projected_points'],
        rays=result['rays'],
        clusters=clusters,
        title="nuScenes: 2D to 3D Projection with Coordinate Systems",
        draw_coordinate_systems=True,
        axis_length=3.0
    )
    
    # Visualize with clusters only (using the same function)
    visualizer.visualize_with_projected_points(
        projected_points=None,  # No projected points, just clusters
        rays=None,
        clusters=clusters,
        title="nuScenes: Point Cloud Clusters",
        draw_coordinate_systems=True,
        axis_length=3.0
    )
    
    print("Done!")


if __name__ == "__main__":
    main()

