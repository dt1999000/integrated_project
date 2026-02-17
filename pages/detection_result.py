"""
Detection Result Page
Shows detection results using SAM masks, reprojection, and pose estimation.
"""
import streamlit as st
import numpy as np
import cv2
import pandas as pd
from typing import List, Dict, Optional

from components.utils.visualization_helper import (
    draw_2d_boxes_on_image,
    add_cuboids_to_figure,
    create_3d_scatter_plot,
)
from components.core.pointcloud_projection import Projection
from components.core.sam_integration import assign_points_to_masks
from components.core.pose_estimation import estimate_pose_l_shape, cuboid_from_pose
from components.core.constants import KITTI_CUBOID_TEMPLATES
from components.core.clustering_manager import ClusteringManager


def _select_best_cluster_points(
    mask_points: np.ndarray,
    mask: np.ndarray,
    projection: Projection,
    image_shape: tuple,
    dbscan_eps: float = 0.5,
    dbscan_min_samples: int = 5,
) -> Optional[np.ndarray]:
    """
    Cluster mask points with DBSCAN and select the cluster closest to the mask center.

    Returns the points of the selected cluster, or None if no valid cluster is found.
    """
    if len(mask_points) < dbscan_min_samples:
        return None

    h, w = image_shape

    # Find center of mask in image space
    mask_coords = np.column_stack(np.where(mask > 0))
    if len(mask_coords) == 0:
        return None

    center_y, center_x = mask_coords.mean(axis=0)

    # Project all mask points to 2D
    pixels, valid_mask = projection.point_to_pixel(mask_points)
    valid_pixels = pixels[valid_mask]
    valid_indices = np.where(valid_mask)[0]

    if len(valid_pixels) == 0:
        return None

    # Compute distances from projected points to mask center
    distances = np.sqrt(
        (valid_pixels[:, 0] - center_x) ** 2 + (valid_pixels[:, 1] - center_y) ** 2
    )

    # Use 10% of closest points to approximate center region in 3D
    n_sample = max(1, int(len(mask_points) * 0.1))
    closest_indices = np.argsort(distances)[:n_sample]
    sampled_point_indices = valid_indices[closest_indices]
    center_points = mask_points[sampled_point_indices]

    if len(center_points) == 0:
        return None

    center_centroid = np.mean(center_points, axis=0)

    # Run DBSCAN on all mask points
    clustering_manager = ClusteringManager(mask_points)
    cluster_labels = clustering_manager.run_dbscan(
        eps=dbscan_eps, min_samples=dbscan_min_samples
    )

    unique_labels = np.unique(cluster_labels)
    unique_labels = unique_labels[unique_labels >= 0]  # remove noise (-1)
    if len(unique_labels) == 0:
        return None

    best_cluster_id = -1
    min_distance = float("inf")

    for cluster_id in unique_labels:
        cluster_points = mask_points[cluster_labels == cluster_id]
        if len(cluster_points) < 5:
            continue

        cluster_centroid = np.mean(cluster_points, axis=0)
        distance = np.linalg.norm(cluster_centroid - center_centroid)
        if distance < min_distance:
            min_distance = distance
            best_cluster_id = cluster_id

    if best_cluster_id == -1:
        return None
    print(f"Best cluster ID: {best_cluster_id}")
    return mask_points[cluster_labels == best_cluster_id]


def detection_result_page():
    """Detection Result page using SAM masks, reprojection, and pose estimation"""
    st.header("🎯 Detection Results")
    
    # Check if data is loaded
    if 'sample_data' not in st.session_state or st.session_state.sample_data is None:
        st.info("👈 Load a sample from the sidebar to see detection results")
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
    
    # Display image with original bounding boxes
    st.subheader("📷 Image with Original Bounding Boxes")
    if ground_truth_boxes:
        img_with_boxes = draw_2d_boxes_on_image(img_rgb.copy(), ground_truth_boxes)
        st.image(img_with_boxes, use_container_width=True)
    else:
        st.image(img_rgb, use_container_width=True)
    
    # Get backprojected sparse depth points
    has_colored_sparse = (
        st.session_state.get('colored_sparse_points') is not None and 
        len(st.session_state.get('colored_sparse_points', [])) > 0
    )
    
    if not has_colored_sparse:
        st.warning("⚠️ No backprojected sparse depth points available.")
        return
    
    backprojected_points = st.session_state.colored_sparse_points
    
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
    
    # For each mask, get points and estimate pose
    detected_cuboids = []
    mask_to_bbox_map = {}  # Map mask index to bbox index for category lookup
    
    # Match masks to ground truth boxes (if available)
    if ground_truth_boxes:
        for mask_idx, mask in enumerate(sam_masks):
            if mask is None:
                continue
            # Find best matching bbox by IoU
            best_iou = 0.0
            best_bbox_idx = -1
            mask_bbox = _get_bbox_from_mask(mask)
            
            for bbox_idx, gt_box in enumerate(ground_truth_boxes):
                bbox_2d = gt_box.get('bbox_2d')
                if bbox_2d is None:
                    continue
                gt_bbox = [bbox_2d['left'], bbox_2d['top'], bbox_2d['right'], bbox_2d['bottom']]
                iou = _calculate_iou(mask_bbox, gt_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_bbox_idx = bbox_idx
            
            if best_iou > 0.3:  # Threshold for matching
                mask_to_bbox_map[mask_idx] = best_bbox_idx
    
    # Process each mask
    for mask_idx, mask in enumerate(sam_masks):
        if mask is None:
            continue
        
        # Get points assigned to this mask
        mask_points = backprojected_points[mask_assignments == mask_idx]
        
        if len(mask_points) < 5:  # Need minimum points for clustering
            continue
        
        print(f"Mask index: {mask_idx}")
        # Refine to best cluster using DBSCAN around the center of the mask
        best_cluster_points = _select_best_cluster_points(
            mask_points=mask_points,
            mask=mask,
            projection=projection,
            image_shape=(h, w),
        )

        if best_cluster_points is None or len(best_cluster_points) < 5:
            continue
        
        # Get category from ground truth box if available
        category = 'Unknown'
        if mask_idx in mask_to_bbox_map:
            bbox_idx = mask_to_bbox_map[mask_idx]
            if bbox_idx < len(ground_truth_boxes):
                category = ground_truth_boxes[bbox_idx].get('category', 'Unknown')
        
        # Estimate pose using L-shape fitting on the selected cluster
        ground_plane_model = st.session_state.get('ground_plane_model')
        pose_result = estimate_pose_l_shape(
            best_cluster_points,
            category=category,
            ground_plane_model=ground_plane_model
        )
        
        # Get template dimensions for this category
        template_dims = KITTI_CUBOID_TEMPLATES.get(category, KITTI_CUBOID_TEMPLATES['Unknown'])
        ground_z = st.session_state.get('ground_z')
        
        # Create cuboid from pose
        cuboid = cuboid_from_pose(
            pose_result,
            category=category,
            template_dims=template_dims,
            ground_z=ground_z
        )
        
        # Add metadata
        cuboid['source_bbox_idx'] = mask_to_bbox_map.get(mask_idx, None)
        cuboid['mask_idx'] = mask_idx
        cuboid['n_points'] = len(best_cluster_points)
        
        detected_cuboids.append(cuboid)
    
    # Store detected cuboids
    st.session_state.cuboids = detected_cuboids
    
    # Display detection statistics
    st.subheader("📊 Detection Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Detected Objects", len(detected_cuboids))
    with col2:
        st.metric("SAM Masks", len(sam_masks))
    with col3:
        total_points = len(backprojected_points)
        assigned_points = np.sum(mask_assignments >= 0)
        st.metric("Assigned Points", f"{assigned_points:,}/{total_points:,}")
    
    # Display detected cuboids table
    if detected_cuboids:
        st.subheader("📦 Detected Objects")
        cuboid_data = []
        for i, cuboid in enumerate(detected_cuboids):
            cuboid_data.append({
                'ID': i + 1,
                'Category': cuboid.get('category', 'Unknown'),
                'Center X': f"{cuboid['center'][0]:.2f}",
                'Center Y': f"{cuboid['center'][1]:.2f}",
                'Center Z': f"{cuboid['center'][2]:.2f}",
                'Yaw (deg)': f"{np.degrees(cuboid['yaw']):.1f}",
                'Length': f"{cuboid['length']:.2f}",
                'Width': f"{cuboid['width']:.2f}",
                'Height': f"{cuboid['height']:.2f}",
                'Points': cuboid.get('n_points', 0),
            })
        df = pd.DataFrame(cuboid_data)
        st.dataframe(df, use_container_width=True)
    
    # 3D Visualization
    st.subheader("🎯 3D Visualization")
    fig = create_3d_scatter_plot(
        points=point_cloud_obj,
        labels=None,
        mask_points=None,
        cuboids=detected_cuboids,
        rays=None,
        points_in_frustums=None,
        reconstructed_points=None,
        show_lidar=True,
        show_reconstructed=False,
        color_by_depth=False,
        title="Detected Objects"
    )
    
    # Add cuboids to figure
    if detected_cuboids:
        add_cuboids_to_figure(fig, detected_cuboids, color='red', opacity=0.3, name_prefix="Detected: ")
    
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

