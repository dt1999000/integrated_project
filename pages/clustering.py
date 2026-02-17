"""
Clustering Page
Shows DBSCAN clustering results on reprojected points from each mask.
"""
import streamlit as st
import numpy as np
import cv2
import pandas as pd
import plotly.graph_objects as go
from typing import List, Dict, Optional, Tuple

from components.utils.visualization_helper import (
    draw_2d_boxes_on_image,
    create_3d_scatter_plot,
    generate_distinct_colors,
    overlay_masks_on_image,
)
from components.core.pointcloud_projection import Projection
from components.core.sam_integration import assign_points_to_masks
from components.core.pose_estimation import estimate_pose_l_shape, cuboid_from_pose
from components.core.constants import KITTI_CUBOID_TEMPLATES
from components.core.clustering_manager import ClusteringManager


def get_center_region_points(
    mask_points: np.ndarray,
    mask: np.ndarray,
    projection: Projection,
    image_shape: Tuple[int, int],
    sample_ratio: float = 0.1
) -> np.ndarray:
    """
    Sample points from the center region of a mask.
    
    Args:
        mask_points: Nx3 array of 3D points assigned to this mask
        mask: HxW binary mask
        projection: Projection object for 3D to 2D mapping
        image_shape: (height, width) of the image
        sample_ratio: Ratio of points to sample (default 0.1 = 10%)
    
    Returns:
        Sampled points from center region of mask
    """
    if len(mask_points) == 0:
        return np.array([]).reshape(0, 3)
    
    h, w = image_shape
    
    # Find center of mask
    mask_coords = np.column_stack(np.where(mask > 0))
    if len(mask_coords) == 0:
        return np.array([]).reshape(0, 3)
    
    center_y, center_x = mask_coords.mean(axis=0)
    
    # Project all mask points to 2D
    pixels, valid_mask = projection.point_to_pixel(mask_points)
    
    # Calculate distance from each point's projection to mask center
    valid_pixels = pixels[valid_mask]
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_pixels) == 0:
        return np.array([]).reshape(0, 3)
    
    # Calculate distances from center
    distances = np.sqrt((valid_pixels[:, 0] - center_x)**2 + (valid_pixels[:, 1] - center_y)**2)
    
    # Sort by distance and take closest points (10% of total)
    n_sample = max(1, int(len(mask_points) * sample_ratio))
    closest_indices = np.argsort(distances)[:n_sample]
    sampled_point_indices = valid_indices[closest_indices]
    
    return mask_points[sampled_point_indices]


def clustering_page():
    """Clustering page showing DBSCAN results on reprojected points"""
    st.header("🔍 Clustering Analysis")
    
    # Check if data is loaded
    if 'sample_data' not in st.session_state or st.session_state.sample_data is None:
        st.info("👈 Load a sample from the sidebar to see clustering results")
        return
    
    sample_data = st.session_state.sample_data
    point_cloud_obj = st.session_state.get('point_cloud')
    
    if point_cloud_obj is None:
        st.warning("⚠️ No point cloud available. Please load a sample first.")
        return
    
    # Check if SAM masks are available
    sam_masks = st.session_state.get('sam_masks')
    if sam_masks is None or len(sam_masks) == 0:
        st.warning("⚠️ No SAM masks available. Please generate masks first.")
        return
    
    # Get ground truth boxes for category mapping
    ground_truth_boxes = sample_data.get('ground_truth_boxes', [])
    
    # Load image
    try:
        img = cv2.imread(sample_data['image_path'])
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
    except Exception as e:
        st.error(f"Could not load image: {str(e)}")
        return
    
    # Get backprojected sparse depth points
    has_colored_sparse = (
        st.session_state.get('colored_sparse_points') is not None and 
        len(st.session_state.get('colored_sparse_points', [])) > 0
    )
    
    if not has_colored_sparse:
        st.warning("⚠️ No backprojected sparse depth points available.")
        return
    
    backprojected_points = st.session_state.colored_sparse_points
    
    # Sidebar controls
    st.sidebar.markdown("### DBSCAN Parameters")
    dbscan_eps = st.sidebar.slider(
        "Eps (max distance)",
        min_value=0.1,
        max_value=2.0,
        value=0.5,
        step=0.1,
        key="clustering_dbscan_eps",
        help="Maximum distance between points in the same cluster"
    )
    dbscan_min_samples = st.sidebar.slider(
        "Min Samples",
        min_value=3,
        max_value=20,
        value=5,
        step=1,
        key="clustering_dbscan_min_samples",
        help="Minimum number of points to form a cluster"
    )
    
    # Create projection object
    projection = Projection(
        camera_intrinsic=sample_data['camera_intrinsic'],
        camera_extrinsic=sample_data.get('camera_extrinsic', np.eye(4)),
        camera_to_lidar_transform=sample_data['camera_to_lidar_transform'],
        point_cloud=backprojected_points,
    )
    
    # Assign backprojected points to masks
    with st.spinner("Assigning points to masks..."):
        mask_assignments = assign_points_to_masks(
            backprojected_points, sam_masks, projection, (h, w)
        )
    
    # Match masks to ground truth boxes for category
    mask_to_bbox_map = {}
    if ground_truth_boxes:
        for mask_idx, mask in enumerate(sam_masks):
            if mask is None:
                continue
            mask_bbox = _get_bbox_from_mask(mask)
            
            best_iou = 0.0
            best_bbox_idx = -1
            for bbox_idx, gt_box in enumerate(ground_truth_boxes):
                bbox_2d = gt_box.get('bbox_2d')
                if bbox_2d is None:
                    continue
                gt_bbox = [bbox_2d['left'], bbox_2d['top'], bbox_2d['right'], bbox_2d['bottom']]
                iou = _calculate_iou(mask_bbox, gt_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_bbox_idx = bbox_idx
            
            if best_iou > 0.3:
                mask_to_bbox_map[mask_idx] = best_bbox_idx
    
    # Process each mask with clustering
    clustering_results = []
    mask_cluster_labels: Dict[int, np.ndarray] = {}

    ground_plane_model = st.session_state.get('ground_plane_model')
    ground_z = st.session_state.get('ground_z')
    
    for mask_idx, mask in enumerate(sam_masks):
        if mask is None:
            continue
        
        # Get points assigned to this mask
        mask_points = backprojected_points[mask_assignments == mask_idx]
        
        if len(mask_points) < dbscan_min_samples:
            continue
        
        # Get category
        category = 'Unknown'
        if mask_idx in mask_to_bbox_map:
            bbox_idx = mask_to_bbox_map[mask_idx]
            if bbox_idx < len(ground_truth_boxes):
                category = ground_truth_boxes[bbox_idx].get('category', 'Unknown')
        
        # Cluster mask points with DBSCAN
        clustering_manager = ClusteringManager(mask_points)
        cluster_labels = clustering_manager.run_dbscan(
            eps=dbscan_eps, min_samples=dbscan_min_samples
        )

        mask_cluster_labels[mask_idx] = cluster_labels

        unique_clusters = np.unique(cluster_labels)
        n_clusters = np.sum(unique_clusters >= 0)  # exclude noise (-1)
        n_cluster_points = np.sum(cluster_labels >= 0)

        clustering_results.append({
            'Mask ID': mask_idx + 1,
            'Category': category,
            'Total Points': len(mask_points),
            'Clusters Found': int(n_clusters),
            'Clustered Points': int(n_cluster_points),
        })
    
    # Display results
    st.subheader("📊 Clustering Results")
    
    if clustering_results:
        df = pd.DataFrame(clustering_results)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No clusters found. Try adjusting DBSCAN parameters.")
        return
    
    # Display image with masks
    st.subheader("📷 Image with Masks")
    colors = generate_distinct_colors(len(sam_masks))
    img_with_masks = overlay_masks_on_image(img_rgb, sam_masks, colors, alpha=0.5)
    st.image(img_with_masks, use_container_width=True)
    
    # 3D Visualization
    st.subheader("🎯 3D Clustering Visualization")
    
    # Create figure with LiDAR background
    fig = create_3d_scatter_plot(
        points=point_cloud_obj,
        labels=None,
        mask_points=None,
        cuboids=None,
        rays=None,
        points_in_frustums=None,
        reconstructed_points=None,
        show_lidar=True,
        show_reconstructed=False,
        color_by_depth=False,
        title="Clustered Points and Fitted Cuboids"
    )
    
    # Add cluster points with same color per mask, different opacity per cluster
    for mask_idx, mask in enumerate(sam_masks):
        if mask_idx not in mask_cluster_labels:
            continue

        cluster_labels = mask_cluster_labels[mask_idx]
        mask_points = backprojected_points[mask_assignments == mask_idx]

        if len(mask_points) == 0:
            continue

        base_color = colors[mask_idx]
        base_color_str = f"rgb({int(base_color[0]*255)}, {int(base_color[1]*255)}, {int(base_color[2]*255)})"

        unique_clusters = np.unique(cluster_labels)

        for cluster_id in unique_clusters:
            cluster_points = mask_points[cluster_labels == cluster_id]
            if len(cluster_points) == 0:
                continue

            # Use opacity to distinguish clusters for the same mask
            if cluster_id == -1:
                opacity = 0.2  # noise
            else:
                # Map cluster_id to an opacity in [0.4, 0.9]
                opacity = 0.7

            fig.add_trace(go.Scatter3d(
                x=cluster_points[:, 0],
                y=cluster_points[:, 1],
                z=cluster_points[:, 2],
                mode='markers',
                marker=dict(size=3, color=base_color_str, opacity=opacity),
                name=f"Mask {mask_idx+1} Cluster {cluster_id}"
            ))
    
    st.plotly_chart(fig, use_container_width=True)


def _get_bbox_from_mask(mask: np.ndarray) -> List[float]:
    """Get bounding box from mask"""
    coords = np.column_stack(np.where(mask > 0))
    if len(coords) == 0:
        return [0, 0, 0, 0]
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    return [int(x_min), int(y_min), int(x_max), int(y_max)]


def _calculate_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """Calculate IoU between two bounding boxes"""
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union

