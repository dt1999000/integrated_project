"""
Shared utilities for page modules.
Contains common imports and helper functions used across pages.
"""
import streamlit as st
import sys
import os
import numpy as np
import cv2

# Add the components directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from components.dataset_loaders.kitti_dataset_loader import KITTIDatasetLoader
from components.core.pointcloud_projection import PointCloud, Projection
from components.core.depth_estimation import compute_sparse_depth_map


def load_dataset_sample(sample_index: int = 0, distance_threshold: float = 0.3, 
                       ransac_n: int = 3, num_iterations: int = 1000, 
                       dataset: str = "kitti", filter_forward_only: bool = True):
    """
    Load a sample from KITTI dataset.

    Args:
        sample_index: Index of the sample to load
        distance_threshold: RANSAC distance threshold for ground plane removal
        ransac_n: RANSAC number of points
        num_iterations: RANSAC number of iterations
        dataset: 'kitti'
        filter_forward_only: Whether to keep only forward-facing points (x > 0)

    Returns:
        Tuple of (sample_data dict, PointCloud object with ground removed)
    """
    if dataset == "kitti":
        # Load KITTI data
        dataset_loader = KITTIDatasetLoader(dataroot='dataset/kitti', split='training')
        dataset_loader.load_dataset()

        # Load synchronized camera, LiDAR, and ground truth data
        sample_data = dataset_loader.load_kitti_data(sample_index)

    else:
        st.error(f"Unknown dataset: {dataset}")
        return None, None

    if sample_data is None:
        st.error(f"Failed to load sample {sample_index}")
        return None, None

    # Create sparse depth map
    sparse_depth_map = None
    if 'image_path' in sample_data and sample_data['image_path']:
        img = cv2.imread(sample_data['image_path'])
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img_rgb.shape[:2]
            camera_intrinsic = sample_data.get('camera_intrinsic')
            camera_to_lidar_transform = sample_data.get('camera_to_lidar_transform')

            if camera_intrinsic is not None and camera_to_lidar_transform is not None:
                sparse_depth_map = compute_sparse_depth_map(
                    point_cloud=sample_data['point_cloud'],
                    image_shape=(h, w),
                    camera_intrinsic=camera_intrinsic,
                    camera_to_lidar_transform=camera_to_lidar_transform
                )
                st.session_state.sparse_depth_map = sparse_depth_map

                projection = Projection(
                    camera_intrinsic=camera_intrinsic,
                    camera_extrinsic=sample_data.get('camera_extrinsic', np.eye(4)),
                    camera_to_lidar_transform=camera_to_lidar_transform,
                    point_cloud=sample_data['point_cloud']
                )
                colored_points, colors = projection.backproject_sparse_depth_map_with_colors(
                    sparse_depth_map=sparse_depth_map,
                    image=img_rgb
                )
                st.session_state.colored_sparse_points = colored_points
                st.session_state.colored_sparse_colors = colors
                print(f"Computed colored sparse depth backprojection: {len(colored_points)} points")

    # Load point cloud and remove ground plane
    point_cloud = PointCloud(sample_data['point_cloud'])
    point_cloud.remove_ground_plane_ransac(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
        filter_forward_only=filter_forward_only
    )

    # Store ground_z at origin in session state for template cuboids
    ground_z = point_cloud.get_ground_z(x=0.0, y=0.0)
    st.session_state.ground_z = ground_z
    st.session_state.ground_plane_model = point_cloud.ground_plane_model
    if ground_z is not None:
        print(f"Ground plane z at origin: {ground_z:.3f}m")

    return sample_data, point_cloud


def get_bbox_from_mask(mask: np.ndarray) -> list:
    """
    Get bounding box coordinates from a binary mask.
    
    Args:
        mask: Binary mask as numpy array (H, W)
    
    Returns:
        Bounding box as [x1, y1, x2, y2]
    """
    # Find all non-zero pixels
    coords = np.column_stack(np.where(mask > 0))
    
    if len(coords) == 0:
        return [0, 0, 0, 0]
    
    # Get min and max coordinates
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    return [int(x_min), int(y_min), int(x_max), int(y_max)]


def calculate_iou(bbox1: list, bbox2: list) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        bbox1: First bounding box [x1, y1, x2, y2]
        bbox2: Second bounding box [x1, y1, x2, y2]
    
    Returns:
        IoU value between 0 and 1
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union
