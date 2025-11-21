"""
Example script showing how to use the 2D to 3D projection with nuScenes dataset.
"""

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes_dataset_loader import NuScenesDatasetLoader
from pointcloud_projection import Projection2DTo3D, PointCloudVisualizer, PointCloud, visualize_image_with_pixels
from bounding_boxes import BoundingBoxes
from typing import Optional



def get_bounding_box_pixels(bbox_extractor: BoundingBoxes, sample_token: str, 
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
    bbox_extractor.get_boxes_for_sample(sample_token, camera_channel, max_boxes)
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
    print("Sample token:")
    print(sample_token)
    print("Loading nuScenes data...")
    dataset_loader = NuScenesDatasetLoader(
        dataroot=dataroot,
        version=version,
        verbose=True
    )
    dataset_loader.load_dataset()
    data = dataset_loader.load_nuscenes_data(sample_token)
    print(data['image_path'])
    
    # Create visualizer first to remove ground plane
    print("Creating point cloud visualizer and removing ground plane...")
    coord_systems_temp = None  # Will get from projector later
    pointcloud = PointCloud(data['point_cloud'], coordinate_systems=coord_systems_temp)

    visualizer = PointCloudVisualizer(point_cloud=pointcloud)
    visualizer.visualize_point_cloud(points=None, rays=None, clusters=None, title="Point Cloud original")
    pointcloud.remove_ground_plane_ransac()
    visualizer_without_ground = PointCloudVisualizer(point_cloud=pointcloud)
    visualizer_without_ground.visualize_point_cloud(points=None, rays=None, clusters=None, title="Point Cloud ground removed")
    # Create projection object with ground-removed point cloud
    print("Creating projection object with ground-removed point cloud...")
    projector = Projection2DTo3D(
        camera_intrinsic=data['camera_intrinsic'],
        camera_extrinsic=data['camera_extrinsic'],
        camera_to_lidar_transform=data['camera_to_lidar_transform'],
        point_cloud=pointcloud.point_cloud_plane_removed
    )
    
    # Create bounding box extractor
    print("Creating bounding box extractor...")
    bbox_extractor = BoundingBoxes(nusc=data['nusc'], data_format="nuscenes")
    
    # Get corner pixels from bounding boxes
    print("Extracting bounding box corner pixels from nuScenes annotations...")
    pixels = get_bounding_box_pixels(
        bbox_extractor=bbox_extractor,
        sample_token=sample_token,
        camera_channel="CAM_FRONT",
        max_boxes=2
    )
    
    print(f"Found {len(pixels)} pixels from bounding boxes")

    print("Projecting pixels to 3D...")
    result_all_pixels = projector.project_pixels_to_3d(pixels, max_distance=100.0, distance_threshold=1.0)
    rays = projector.pixel_to_ray(pixels)
    
    print(f"Projected {len(result_all_pixels['projected_points'])} points")    
    
    # Visualize image with marked pixels
    print("Visualizing image with projected pixels...")
    visualize_image_with_pixels(
        image=data['image_path'],
        pixel_coords=pixels,
        save_path="image_with_pixels.png",
        show=False
    )
    
    bbox_extractor.get_boxes_for_sample(sample_token, camera_channel="CAM_FRONT")
    
    # Visualize with projected points, coordinate systems, and clusters
    visualizer_without_ground.visualize_point_cloud(
        result_all_pixels['projected_points'],
        rays=result_all_pixels['rays'],
        clusters=None,
        title="nuScenes: 2D to 3D Projection with Coordinate Systems"
    )
    # Run visualizations for every 5th box instead of every box
    print(f"Found {len(bbox_extractor.boxes)} total boxes. Running visualizations for every 5th box...")

    for idx in range(43, len(bbox_extractor.boxes), 5):  # Increment by 5
        pixels = np.array([]).reshape(0, 2)
        for i in range(5):
            box = bbox_extractor.get_box_from_idx(sample_token, idx + i)
            print(f"Box {idx + i}:")
            pixels = np.concatenate([pixels, box.get_corners()])
            result = projector.project_pixels_to_3d(pixels, max_distance=100.0, distance_threshold=1.0)
        visualize_image_with_pixels(
            image=data['image_path'],
            pixel_coords=pixels,
            save_path=f"image_with_pixels_idx_{idx + i}.png",
            show=True
        )
        visualizer_without_ground.visualize_point_cloud(
            result['projected_points'],
            rays=result['rays'],
            clusters=None,
            title="nuScenes: 2D to 3D Projection with Coordinate Systems"
        )

    visualizer_without_ground.visualize_point_cloud(
        rays=rays,
        title="nuScenes: 2D to 3D Projection with Coordinate Systems"
    )
    pointcloud.add_projected_points(result_all_pixels['projected_points'])
    pointcloud.cluster_with_dbscan(eps=0.50, min_samples=5)
    # Visualize with clusters only (using the same function)
    visualizer_without_ground = PointCloudVisualizer(point_cloud=pointcloud)
    visualizer_without_ground.visualize_point_cloud(
        points=None,  # No projected points, just clusters
        rays=None,
        clusters=pointcloud.clusters,
        title="nuScenes: Point Cloud Clusters"
    )
    
  
    # # ================================================================================
    # # NEW: Demonstrate Multiple Clustering Algorithms Comparison
    # # ================================================================================

    # print("\n" + "="*70)
    # print("DEMONSTRATING MULTIPLE CLUSTERING ALGORITHMS")
    # print("="*70)

    # # Reset clustering and run multiple algorithms comparison
    # pointcloud.clusters = []  # Clear previous clustering results

    # print("Running multiple clustering algorithms with automatic parameter optimization...")
    # print("This will test different algorithms and select the best one based on multiple metrics.")

    # # Run clustering comparison (testing fewer algorithms for faster demo)
    # clustering_results = pointcloud.cluster_with_multiple_algorithms(
    #     algorithms=['dbscan', 'optics', 'birch'],
    #     max_combinations_per_algorithm=15  # Reduced for faster execution
    # )

    # # Print summary of results
    # if hasattr(pointcloud, 'best_algorithm_name'):
    #     print(f"\n🏆 Best algorithm: {pointcloud.best_algorithm_name}")
    #     print(f"   Composite score: {pointcloud.best_algorithm_result['best_evaluation']['composite_score']:.4f}")
    #     print(f"   Number of clusters: {pointcloud.best_algorithm_result['best_evaluation']['num_clusters']}")
    #     print(f"   Noise ratio: {pointcloud.best_algorithm_result['best_evaluation']['noise_ratio']:.3f}")
    #     print(f"   Computation time: {pointcloud.best_algorithm_result['computation_time']:.2f}s")

    #     # Visualize best clustering result
    #     print("\nVisualizing best clustering result...")
    #     visualizer_best = PointCloudVisualizer(point_cloud=pointcloud)
    #     visualizer_best.visualize_point_cloud(
    #         points=result_all_pixels['projected_points'],
    #         rays=result_all_pixels['rays'],
    #         clusters=pointcloud.best_clusters,
    #         title=f"Best Clustering: {pointcloud.best_algorithm_name} (Score: {pointcloud.best_algorithm_result['best_evaluation']['composite_score']:.4f})"
    #     )

    #     # Get clustering summary
    #     summary = pointcloud.get_clustering_summary()
    #     print(f"\n📊 Algorithms tested: {summary['total_algorithms_tested']}")
    #     print("Algorithm ranking:")

    #     for algo_name, result in summary['algorithm_results'].items():
    #         if result['status'] == 'success':
    #             print(f"   • {algo_name:15s}: Score={result['composite_score']:.4f}, "
    #                   f"Clusters={result['num_clusters']}, Time={result['computation_time']:.2f}s")

    # print("\n" + "="*70)
    # print("NEW FEATURES SUMMARY:")
    # print("✓ Multiple clustering algorithms automatically compared")
    # print("✓ Parameter optimization via grid search")
    # print("✓ Comprehensive evaluation metrics")
    # print("✓ Automatic best algorithm selection")
    # print("✓ Performance ranking and comparison")
    # print("="*70)

    print("Done!")


if __name__ == "__main__":
    main()

