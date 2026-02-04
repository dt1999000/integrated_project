"""
Shared utilities for page modules.
Contains common imports and helper functions used across pages.
"""
import streamlit as st
import sys
import os

# Add the current directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kitti_dataset_loader import KITTIDatasetLoader
from pointcloud_projection import PointCloud


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


