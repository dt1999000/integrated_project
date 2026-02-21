"""
3D Object Detection Pipeline
Unified detection pipeline with step-by-step execution and full pipeline mode.
"""
import streamlit as st
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import time
from typing import List, Dict, Optional, Tuple

from components.core.pointcloud_projection import PointCloud, Projection
from components.core.depth_estimation import compute_sparse_depth_map
from components.core.sam_integration import SAMIntegration, assign_points_to_masks
from components.core.pose_estimation import fit_cuboid_to_points, get_dimensions_from_class
from components.core.clustering_manager import ClusteringManager
from components.utils.visualization_helper import (
    draw_2d_boxes_on_image,
    add_cuboids_to_figure,
    create_3d_scatter_plot,
    generate_distinct_colors,
    overlay_masks_on_image,
)


# ============================================================================
# Helper Functions
# ============================================================================

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
    
    return mask_points[cluster_labels == best_cluster_id]


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


# ============================================================================
# Pipeline Step Functions
# ============================================================================

def step_1_ground_plane_removal(
    point_cloud: np.ndarray,
    distance_threshold: float = 0.3,
    ransac_n: int = 3,
    num_iterations: int = 1000,
    filter_forward_only: bool = True
) -> Dict:
    """
    Step 1: Remove ground plane from point cloud using RANSAC.
    
    Args:
        point_cloud: Nx3 array of 3D points
        distance_threshold: RANSAC distance threshold
        ransac_n: RANSAC number of points
        num_iterations: RANSAC number of iterations
        filter_forward_only: Whether to keep only forward-facing points (x > 0)
    
    Returns:
        Dict with 'point_cloud_obj', 'ground_plane_model', 'ground_z'
    """
    start_time = time.time()
    
    # Create PointCloud object
    point_cloud_obj = PointCloud(point_cloud)
    
    # Remove ground plane
    point_cloud_obj.remove_ground_plane_ransac(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
        filter_forward_only=filter_forward_only
    )
    
    # Get ground_z at origin
    ground_z = point_cloud_obj.get_ground_z(x=0.0, y=0.0)
    
    elapsed_time = time.time() - start_time
    
    return {
        'point_cloud_obj': point_cloud_obj,
        'ground_plane_model': point_cloud_obj.ground_plane_model,
        'ground_z': ground_z,
        'points_remaining': len(point_cloud_obj.point_cloud_plane_removed),
        'time': elapsed_time
    }


def step_2_sparse_depth_backprojection(
    sample_meta_data: Dict,
    image: np.ndarray,
    point_cloud: np.ndarray
) -> Dict:
    """
    Step 2: Backproject LiDAR points to 2D image to create sparse depth map.
    
    Args:
        sample_meta_data: Sample metadata with camera parameters
        image: HxWx3 RGB image
        point_cloud: Nx3 array of 3D points (after ground removal)
    
    Returns:
        Dict with 'sparse_depth_map', 'colored_sparse_points', 'colored_sparse_colors'
    """
    start_time = time.time()
    
    h, w = image.shape[:2]
    camera_intrinsic = sample_meta_data['camera_intrinsic']
    camera_to_lidar_transform = sample_meta_data['camera_to_lidar_transform']
    
    # Create sparse depth map
    sparse_depth_map = compute_sparse_depth_map(
        point_cloud=point_cloud,
        image_shape=(h, w),
        camera_intrinsic=camera_intrinsic,
        camera_to_lidar_transform=camera_to_lidar_transform
    )
    
    # Backproject sparse depth map to 3D with colors
    projection = Projection(
        camera_intrinsic=camera_intrinsic,
        camera_extrinsic=sample_meta_data.get('camera_extrinsic', np.eye(4)),
        camera_to_lidar_transform=camera_to_lidar_transform,
        point_cloud=point_cloud
    )
    
    colored_sparse_points, colored_sparse_colors = projection.backproject_sparse_depth_map_with_colors(
        sparse_depth_map=sparse_depth_map,
        image=image
    )
    
    elapsed_time = time.time() - start_time
    
    return {
        'sparse_depth_map': sparse_depth_map,
        'colored_sparse_points': colored_sparse_points,
        'colored_sparse_colors': colored_sparse_colors,
        'n_points': len(colored_sparse_points),
        'time': elapsed_time
    }


def step_3_sam_segmentation(
    sample_meta_data: Dict,
    image: np.ndarray,
    sparse_points: np.ndarray,
    sam_model_type: str = 'sam2_t'
) -> Dict:
    """
    Step 3: Generate SAM masks and assign original LiDAR points to masks.
    
    Args:
        sample_meta_data: Sample metadata
        image: HxWx3 RGB image
        sparse_points: Nx3 array of backprojected sparse depth points
        sam_model_type: 'sam2_t' or 'sam3'
    
    Returns:
        Dict with 'sam_masks', 'mask_assignments', 'mask_bboxes' (if SAM3)
    """
    start_time = time.time()
    
    # Initialize SAM integration if needed
    if 'sam_integration' not in st.session_state or st.session_state.sam_integration is None:
        try:
            st.session_state.sam_integration = SAMIntegration(model_type=sam_model_type)
            st.session_state.sam_initialized_model_type = sam_model_type
        except Exception as e:
            return {
                'error': f"SAM initialization failed: {str(e)}",
                'sam_masks': None,
                'mask_assignments': None,
                'time': time.time() - start_time
            }
    
    sam_integration = st.session_state.sam_integration
    h, w = image.shape[:2]
    ground_truth_boxes = sample_meta_data.get('ground_truth_boxes', [])
    
    sam_masks = []
    mask_bboxes = []
    
    if sam_model_type.startswith('sam2'):
        # SAM2: Generate masks from bounding boxes
        for gt_box in ground_truth_boxes:
            bbox_2d = gt_box.get('bbox_2d')
            if bbox_2d is not None:
                bbox_list = [
                    bbox_2d['left'],
                    bbox_2d['top'],
                    bbox_2d['right'],
                    bbox_2d['bottom'],
                ]
                mask = sam_integration.get_mask_from_bbox(image, bbox_list)
                sam_masks.append(mask)
    
    elif sam_model_type == 'sam3':
        # SAM3: Generate masks from class names
        class_names = list(set([box.get('category', 'unknown') for box in ground_truth_boxes]))
        class_names = [c for c in class_names if c != 'unknown' and c != 'DontCare']
        
        if class_names:
            segment_results = sam_integration.segment_by_classes(image, class_names)
            all_masks = segment_results['masks']
            
            # Match masks to bounding boxes
            bboxes_list = []
            for gt_box in ground_truth_boxes:
                bbox_2d = gt_box.get('bbox_2d')
                if bbox_2d is not None:
                    bboxes_list.append([
                        bbox_2d['left'],
                        bbox_2d['top'],
                        bbox_2d['right'],
                        bbox_2d['bottom'],
                    ])
            
            if bboxes_list:
                matches = sam_integration.match_instances_to_bboxes(
                    all_masks, bboxes_list, iou_threshold=0.3
                )
                
                masks = [None] * len(bboxes_list)
                for mask_idx, bbox_idx in matches.items():
                    masks[bbox_idx] = all_masks[mask_idx]
                
                sam_masks = [m for m in masks if m is not None]
                
                # Extract minimal bounding boxes from masks (for SAM3)
                for mask in sam_masks:
                    if mask is not None:
                        mask_bbox = _get_bbox_from_mask(mask)
                        mask_bboxes.append(mask_bbox)
            else:
                sam_masks = all_masks
                for mask in sam_masks:
                    if mask is not None:
                        mask_bbox = _get_bbox_from_mask(mask)
                        mask_bboxes.append(mask_bbox)
    
    # Assign original LiDAR points to masks based on sparse depth map and mask overlap
    mask_assignments = None
    if sam_masks and len(sam_masks) > 0:
        # Use the sparse points (already backprojected) and assign to masks
        projection = Projection(
            camera_intrinsic=sample_meta_data['camera_intrinsic'],
            camera_extrinsic=sample_meta_data.get('camera_extrinsic', np.eye(4)),
            camera_to_lidar_transform=sample_meta_data['camera_to_lidar_transform'],
            point_cloud=sparse_points
        )
        
        mask_assignments = assign_points_to_masks(
            sparse_points, sam_masks, projection, (h, w)
        )
    
    elapsed_time = time.time() - start_time
    
    return {
        'sam_masks': sam_masks,
        'mask_assignments': mask_assignments,
        'mask_bboxes': mask_bboxes if sam_model_type == 'sam3' else None,
        'n_masks': len(sam_masks),
        'time': elapsed_time
    }


def step_4_clustering(
    sample_meta_data: Dict,
    sparse_points: np.ndarray,
    sam_masks: List[np.ndarray],
    mask_assignments: np.ndarray,
    dbscan_eps: float = 0.5,
    dbscan_min_samples: int = 5
) -> Dict:
    """
    Step 4: Run DBSCAN clustering on points assigned to each mask.
    
    Args:
        sample_meta_data: Sample metadata
        sparse_points: Nx3 array of backprojected sparse depth points
        sam_masks: List of binary masks
        mask_assignments: N array assigning each point to a mask index
        dbscan_eps: DBSCAN eps parameter
        dbscan_min_samples: DBSCAN min_samples parameter
    
    Returns:
        Dict with 'mask_cluster_labels', 'clustering_results', 'best_cluster_points'
    """
    start_time = time.time()
    
    # Get image shape
    if 'image' in st.session_state.sample:
        h, w = st.session_state.sample['image'].shape[:2]
    else:
        h, w = sample_meta_data.get('image_shape', (375, 1242))  # Default KITTI size
    
    projection = Projection(
        camera_intrinsic=sample_meta_data['camera_intrinsic'],
        camera_extrinsic=sample_meta_data.get('camera_extrinsic', np.eye(4)),
        camera_to_lidar_transform=sample_meta_data['camera_to_lidar_transform'],
        point_cloud=sparse_points
    )
    
    mask_cluster_labels = {}
    clustering_results = []
    best_cluster_points_dict = {}
    
    for mask_idx, mask in enumerate(sam_masks):
        if mask is None:
            continue
        
        # Get points assigned to this mask
        mask_points = sparse_points[mask_assignments == mask_idx]
        
        if len(mask_points) < dbscan_min_samples:
            continue
        
        # Run DBSCAN clustering
        clustering_manager = ClusteringManager(mask_points)
        cluster_labels = clustering_manager.run_dbscan(
            eps=dbscan_eps, min_samples=dbscan_min_samples
        )
        
        mask_cluster_labels[mask_idx] = cluster_labels
        
        # Select best cluster
        best_cluster_points = _select_best_cluster_points(
            mask_points=mask_points,
            mask=mask,
            projection=projection,
            image_shape=(h, w),
            dbscan_eps=dbscan_eps,
            dbscan_min_samples=dbscan_min_samples
        )
        
        if best_cluster_points is not None:
            best_cluster_points_dict[mask_idx] = best_cluster_points
        
        # Statistics
        unique_clusters = np.unique(cluster_labels)
        n_clusters = np.sum(unique_clusters >= 0)
        n_cluster_points = np.sum(cluster_labels >= 0)
        
        clustering_results.append({
            'Mask ID': mask_idx + 1,
            'Total Points': len(mask_points),
            'Clusters Found': int(n_clusters),
            'Clustered Points': int(n_cluster_points),
            'Best Cluster Points': len(best_cluster_points) if best_cluster_points is not None else 0
        })
    
    elapsed_time = time.time() - start_time
    
    return {
        'mask_cluster_labels': mask_cluster_labels,
        'clustering_results': clustering_results,
        'best_cluster_points': best_cluster_points_dict,
        'time': elapsed_time
    }


def step_5_detection_pose_estimation(
    sample_meta_data: Dict,
    best_cluster_points: Dict[int, np.ndarray],
    sam_masks: List[np.ndarray],
    ground_z: float,
    cuboid_params: Dict
) -> Dict:
    """
    Step 5: Fit cuboids to best cluster points using scoring-based method.
    
    Args:
        sample_meta_data: Sample metadata
        best_cluster_points: Dict mapping mask_idx to cluster points
        sam_masks: List of binary masks
        ground_z: Ground plane z value
        cuboid_params: Cuboid fitting parameters
    
    Returns:
        Dict with 'detected_cuboids'
    """
    start_time = time.time()
    
    ground_truth_boxes = sample_meta_data.get('ground_truth_boxes', [])
    detected_cuboids = []
    mask_to_bbox_map = {}
    
    # Match masks to ground truth boxes for category
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
    
    # Fit cuboid to each mask's best cluster
    for mask_idx, cluster_points in best_cluster_points.items():
        if len(cluster_points) < 5:
            continue
        
        # Get category from ground truth
        category = 'Unknown'
        if mask_idx in mask_to_bbox_map:
            bbox_idx = mask_to_bbox_map[mask_idx]
            if bbox_idx < len(ground_truth_boxes):
                category = ground_truth_boxes[bbox_idx].get('category', 'Unknown')
        
        # Get dimensions for this category
        dimensions = get_dimensions_from_class(category)
        
        # Get cuboid fitting parameters
        score_weights = (
            cuboid_params['w_distance'],
            cuboid_params['w_geometric'],
            cuboid_params['w_outlier']
        )
        
        # Fit cuboid
        try:
            fit_result = fit_cuboid_to_points(
                points=cluster_points,
                dimensions=dimensions,
                step_center_search=cuboid_params['step_center_search'],
                max_step_center=cuboid_params['max_step_center'],
                d_theta=cuboid_params['d_theta'],
                normals=None,
                score_weights=score_weights
            )
            
            # Convert fit_result to cuboid format
            center = fit_result['center']
            yaw = fit_result['yaw']
            length = fit_result['length']
            width = fit_result['width']
            height = fit_result['height']
            
            # Create cuboid corners
            l_half = length / 2.0
            w_half = width / 2.0
            h_half = height / 2.0
            
            corners_local = np.array([
                [-l_half, -w_half, -h_half],
                [ l_half, -w_half, -h_half],
                [ l_half,  w_half, -h_half],
                [-l_half,  w_half, -h_half],
                [-l_half, -w_half,  h_half],
                [ l_half, -w_half,  h_half],
                [ l_half,  w_half,  h_half],
                [-l_half,  w_half,  h_half],
            ])
            
            # Adjust z to use ground_z
            base_z = ground_z if ground_z is not None else np.min(cluster_points[:, 2])
            corners_local[:, 2] += (base_z + h_half) - center[2]
            
            # Rotation matrix around Z-axis
            cos_yaw = np.cos(yaw)
            sin_yaw = np.sin(yaw)
            R_z = np.array([
                [cos_yaw, -sin_yaw, 0],
                [sin_yaw,  cos_yaw, 0],
                [0,        0,       1]
            ])
            
            # Rotate and translate
            corners_rotated = (R_z @ corners_local.T).T
            corners = corners_rotated + center
            
            cuboid = {
                'center': center,
                'yaw': yaw,
                'length': length,
                'width': width,
                'height': height,
                'category': category,
                'corners': corners,
                'min_x': float(np.min(corners[:, 0])),
                'max_x': float(np.max(corners[:, 0])),
                'min_y': float(np.min(corners[:, 1])),
                'max_y': float(np.max(corners[:, 1])),
                'min_z': float(np.min(corners[:, 2])),
                'max_z': float(np.max(corners[:, 2])),
                'format': 'kitti',
                'method': fit_result.get('method', 'cuboid_fit'),
                'score': fit_result.get('score', float('inf')),
                'source_bbox_idx': mask_to_bbox_map.get(mask_idx, None),
                'mask_idx': mask_idx,
                'n_points': len(cluster_points),
            }
            
            detected_cuboids.append(cuboid)
            
        except Exception as e:
            print(f"Error fitting cuboid for mask {mask_idx}: {str(e)}")
            continue
    
    elapsed_time = time.time() - start_time
    
    return {
        'detected_cuboids': detected_cuboids,
        'n_detected': len(detected_cuboids),
        'time': elapsed_time
    }


# ============================================================================
# Pipeline Orchestrator
# ============================================================================

def run_full_pipeline(params: Dict) -> Dict:
    """
    Run the full detection pipeline from start to finish.
    
    Args:
        params: Dictionary with all pipeline parameters
    
    Returns:
        Dict with results from all steps
    """
    results = {}
    
    # Step 1: Ground plane removal
    if 'step_1' not in results or not results['step_1'].get('completed', False):
        step_1_result = step_1_ground_plane_removal(
            point_cloud=st.session_state.sample['point_cloud'],
            **params['pipeline']
        )
        results['step_1'] = {'completed': True, 'result': step_1_result}
        st.session_state.pipeline_state['step_1'] = results['step_1']
    
    # Step 2: Sparse depth backprojection
    step_2_result = step_2_sparse_depth_backprojection(
        sample_meta_data=st.session_state.sample['sample_meta_data'],
        image=st.session_state.sample['image'],
        point_cloud=results['step_1']['result']['point_cloud_obj'].point_cloud_plane_removed
    )
    results['step_2'] = {'completed': True, 'result': step_2_result}
    st.session_state.pipeline_state['step_2'] = results['step_2']
    
    # Step 3: SAM segmentation
    step_3_result = step_3_sam_segmentation(
        sample_meta_data=st.session_state.sample['sample_meta_data'],
        image=st.session_state.sample['image'],
        sparse_points=step_2_result['colored_sparse_points'],
        sam_model_type=params.get('sam_model_type', 'sam2_t')
    )
    results['step_3'] = {'completed': True, 'result': step_3_result}
    st.session_state.pipeline_state['step_3'] = results['step_3']
    
    # Step 4: Clustering
    step_4_result = step_4_clustering(
        sample_meta_data=st.session_state.sample['sample_meta_data'],
        sparse_points=step_2_result['colored_sparse_points'],
        sam_masks=step_3_result['sam_masks'],
        mask_assignments=step_3_result['mask_assignments'],
        **params['clustering']
    )
    results['step_4'] = {'completed': True, 'result': step_4_result}
    st.session_state.pipeline_state['step_4'] = results['step_4']
    
    # Step 5: Detection & pose estimation
    step_5_result = step_5_detection_pose_estimation(
        sample_meta_data=st.session_state.sample['sample_meta_data'],
        best_cluster_points=step_4_result['best_cluster_points'],
        sam_masks=step_3_result['sam_masks'],
        ground_z=results['step_1']['result']['ground_z'],
        cuboid_params=params['cuboid_fitting']
    )
    results['step_5'] = {'completed': True, 'result': step_5_result}
    st.session_state.pipeline_state['step_5'] = results['step_5']
    
    return results


# ============================================================================
# Main Page Function
# ============================================================================

def main():
    """Main detection pipeline page"""
    st.set_page_config(
        page_title="3D Object Detection Pipeline",
        page_icon="🎯",
        layout="wide"
    )
    
    st.header("🎯 3D Object Detection Pipeline")
    st.markdown("""
    Run the complete detection pipeline step-by-step or all at once.
    Each step builds on the previous one, showing results and visualizations.
    """)
    
    # Initialize pipeline state
    if 'pipeline_state' not in st.session_state:
        st.session_state.pipeline_state = {
            'step_1': {'completed': False, 'result': None, 'time': None, 'error': None},
            'step_2': {'completed': False, 'result': None, 'time': None, 'error': None},
            'step_3': {'completed': False, 'result': None, 'time': None, 'error': None},
            'step_4': {'completed': False, 'result': None, 'time': None, 'error': None},
            'step_5': {'completed': False, 'result': None, 'time': None, 'error': None},
        }
    
    # Initialize parameters if not set
    if 'params' not in st.session_state:
        st.session_state.params = {
            'pipeline': {
                'distance_threshold': 0.3,
                'ransac_n': 3,
                'num_iterations': 1000,
                'filter_forward_only': True
            },
            'clustering': {
                'dbscan_eps': 0.5,
                'dbscan_min_samples': 5
            },
            'cuboid_fitting': {
                'w_distance': 1.0,
                'w_geometric': 0.5,
                'w_outlier': 2.0,
                'step_center_search': 0.2,
                'max_step_center': 10,
                'd_theta': 0.05
            },
            'sam_model_type': 'sam2_t'
        }
    
    # Check if sample is loaded
    if 'sample' not in st.session_state or st.session_state.sample is None:
        st.info("👈 Please load a sample from **1_Dataset_Extraction** page first.")
        return
    
    sample = st.session_state.sample
    sample_meta_data = sample['sample_meta_data']
    image = sample['image']
    point_cloud = sample['point_cloud']
    
    # Sidebar: Parameters
    st.sidebar.header("⚙️ Pipeline Parameters")
    
    with st.sidebar.expander("Ground Plane Removal", expanded=False):
        st.session_state.params['pipeline']['distance_threshold'] = st.slider(
            "Distance Threshold", 0.1, 1.0, 0.3, 0.01
        )
        st.session_state.params['pipeline']['ransac_n'] = st.slider(
            "RANSAC N", 3, 10, 3, 1
        )
        st.session_state.params['pipeline']['num_iterations'] = st.slider(
            "Iterations", 100, 1000, 1000, 100
        )
        st.session_state.params['pipeline']['filter_forward_only'] = st.checkbox(
            "Forward-Facing Only", True
        )
    
    with st.sidebar.expander("Clustering (Step 4)", expanded=False):
        st.session_state.params['clustering']['dbscan_eps'] = st.slider(
            "DBSCAN Eps", 0.1, 2.0, 0.5, 0.1
        )
        st.session_state.params['clustering']['dbscan_min_samples'] = st.slider(
            "Min Samples", 3, 20, 5, 1
        )
    
    with st.sidebar.expander("Cuboid Fitting (Step 5)", expanded=False):
        st.session_state.params['cuboid_fitting']['w_distance'] = st.slider(
            "Weight: Distance", 0.0, 5.0, 1.0, 0.1
        )
        st.session_state.params['cuboid_fitting']['w_geometric'] = st.slider(
            "Weight: Geometric", 0.0, 5.0, 0.5, 0.1
        )
        st.session_state.params['cuboid_fitting']['w_outlier'] = st.slider(
            "Weight: Outlier", 0.0, 10.0, 2.0, 0.1
        )
        st.session_state.params['cuboid_fitting']['step_center_search'] = st.slider(
            "Center Search Step", 0.05, 1.0, 0.2, 0.05
        )
        st.session_state.params['cuboid_fitting']['max_step_center'] = st.slider(
            "Max Center Steps", 1, 20, 10, 1
        )
        st.session_state.params['cuboid_fitting']['d_theta'] = st.slider(
            "Yaw Search Step", 0.01, 0.2, 0.05, 0.01
        )
    
    st.sidebar.markdown("### SAM Model")
    st.session_state.params['sam_model_type'] = st.sidebar.selectbox(
        "SAM Model Type",
        options=['sam2_t', 'sam3'],
        index=0,
        help="SAM2: Uses bounding boxes. SAM3: Uses text prompts."
    )
    
    # Main controls
    col1, col2 = st.columns(2)
    with col1:
        run_full = st.button("🚀 Run Full Pipeline", type="primary", use_container_width=True)
    with col2:
        reset_pipeline = st.button("🔄 Reset Pipeline", use_container_width=True)
    
    if reset_pipeline:
        st.session_state.pipeline_state = {
            'step_1': {'completed': False, 'result': None, 'time': None, 'error': None},
            'step_2': {'completed': False, 'result': None, 'time': None, 'error': None},
            'step_3': {'completed': False, 'result': None, 'time': None, 'error': None},
            'step_4': {'completed': False, 'result': None, 'time': None, 'error': None},
            'step_5': {'completed': False, 'result': None, 'time': None, 'error': None},
        }
        st.rerun()
    
    if run_full:
        with st.spinner("Running full pipeline..."):
            try:
                results = run_full_pipeline(st.session_state.params)
                st.success("✅ Pipeline completed successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Pipeline failed: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
    # Display sample info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Sample Info")
    st.sidebar.info(f"Dataset: {sample_meta_data.get('dataset_type', 'unknown').upper()}")
    st.sidebar.info(f"Sample: {sample_meta_data.get('sample_index', 'N/A')}")
    st.sidebar.info(f"Image: {image.shape}")
    st.sidebar.info(f"Point Cloud: {len(point_cloud):,} points")
    
    # Step-by-step execution
    st.markdown("---")
    st.subheader("Pipeline Steps")
    
    # Step 1: Ground Plane Removal
    with st.container():
        step_1_state = st.session_state.pipeline_state['step_1']
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            status_icon = "✅" if step_1_state['completed'] else "⏸️"
            st.markdown(f"### {status_icon} Step 1: Ground Plane Removal")
            st.caption("Remove ground plane from point cloud using RANSAC")
        
        with col2:
            step_1_enabled = True  # Always enabled
            if st.button("▶️ Run Step 1", key="run_step_1", disabled=not step_1_enabled):
                with st.spinner("Running Step 1..."):
                    try:
                        result = step_1_ground_plane_removal(
                            point_cloud=point_cloud,
                            **st.session_state.params['pipeline']
                        )
                        st.session_state.pipeline_state['step_1'] = {
                            'completed': True,
                            'result': result,
                            'time': result['time'],
                            'error': None
                        }
                        st.rerun()
                    except Exception as e:
                        st.session_state.pipeline_state['step_1']['error'] = str(e)
                        st.error(f"Step 1 failed: {str(e)}")
        
        with col3:
            if step_1_state['completed']:
                st.metric("Time", f"{step_1_state['time']:.2f}s")
        
        if step_1_state['completed']:
            result = step_1_state['result']
            st.success(f"✅ Completed: {result['points_remaining']:,} points remaining")
            
            with st.expander("View Step 1 Details", expanded=False):
                st.metric("Points Remaining", f"{result['points_remaining']:,}")
                st.metric("Ground Z", f"{result['ground_z']:.3f}m" if result['ground_z'] else "N/A")
        
        if step_1_state.get('error'):
            st.error(f"❌ Error: {step_1_state['error']}")
    
    st.markdown("---")
    
    # Step 2: Sparse Depth Backprojection
    with st.container():
        step_2_state = st.session_state.pipeline_state['step_2']
        step_1_completed = st.session_state.pipeline_state['step_1']['completed']
        
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            status_icon = "✅" if step_2_state['completed'] else "⏸️"
            st.markdown(f"### {status_icon} Step 2: Sparse Depth Backprojection")
            st.caption("Backproject LiDAR points to 2D image to create sparse depth map")
            if not step_1_completed:
                st.warning("⚠️ Requires Step 1")
        
        with col2:
            if st.button("▶️ Run Step 2", key="run_step_2", disabled=not step_1_completed):
                with st.spinner("Running Step 2..."):
                    try:
                        step_1_result = st.session_state.pipeline_state['step_1']['result']
                        result = step_2_sparse_depth_backprojection(
                            sample_meta_data=sample_meta_data,
                            image=image,
                            point_cloud=step_1_result['point_cloud_obj'].point_cloud_plane_removed
                        )
                        st.session_state.pipeline_state['step_2'] = {
                            'completed': True,
                            'result': result,
                            'time': result['time'],
                            'error': None
                        }
                        st.rerun()
                    except Exception as e:
                        st.session_state.pipeline_state['step_2']['error'] = str(e)
                        st.error(f"Step 2 failed: {str(e)}")
        
        with col3:
            if step_2_state['completed']:
                st.metric("Time", f"{step_2_state['time']:.2f}s")
        
        if step_2_state['completed']:
            result = step_2_state['result']
            st.success(f"✅ Completed: {result['n_points']:,} points backprojected")
            
            with st.expander("View Step 2 Details", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, caption="Original Image", use_container_width=True)
                with col2:
                    sparse_depth = result['sparse_depth_map']
                    fig, ax = plt.subplots(figsize=(8, 6))
                    vmax = sparse_depth[sparse_depth > 0].max() if np.sum(sparse_depth > 0) > 0 else 100.0
                    im = ax.imshow(sparse_depth, cmap='viridis', vmin=0, vmax=vmax)
                    ax.set_title("Sparse Depth Map")
                    ax.axis('off')
                    plt.colorbar(im, ax=ax, label="Depth (m)")
                    st.pyplot(fig)
                    plt.close()
        
        if step_2_state.get('error'):
            st.error(f"❌ Error: {step_2_state['error']}")
    
    st.markdown("---")
    
    # Step 3: SAM Segmentation
    with st.container():
        step_3_state = st.session_state.pipeline_state['step_3']
        step_2_completed = st.session_state.pipeline_state['step_2']['completed']
        
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            status_icon = "✅" if step_3_state['completed'] else "⏸️"
            st.markdown(f"### {status_icon} Step 3: SAM Segmentation")
            st.caption("Generate SAM masks and assign points to masks")
            if not step_2_completed:
                st.warning("⚠️ Requires Step 2")
        
        with col2:
            if st.button("▶️ Run Step 3", key="run_step_3", disabled=not step_2_completed):
                with st.spinner("Running Step 3..."):
                    try:
                        step_2_result = st.session_state.pipeline_state['step_2']['result']
                        result = step_3_sam_segmentation(
                            sample_meta_data=sample_meta_data,
                            image=image,
                            sparse_points=step_2_result['colored_sparse_points'],
                            sam_model_type=st.session_state.params['sam_model_type']
                        )
                        st.session_state.pipeline_state['step_3'] = {
                            'completed': True,
                            'result': result,
                            'time': result['time'],
                            'error': result.get('error')
                        }
                        st.rerun()
                    except Exception as e:
                        st.session_state.pipeline_state['step_3']['error'] = str(e)
                        st.error(f"Step 3 failed: {str(e)}")
        
        with col3:
            if step_3_state['completed']:
                st.metric("Time", f"{step_3_state['time']:.2f}s")
        
        if step_3_state['completed'] and not step_3_state.get('error'):
            result = step_3_state['result']
            st.success(f"✅ Completed: {result['n_masks']} masks generated")
            
            with st.expander("View Step 3 Details", expanded=False):
                sam_masks = result['sam_masks']
                colors = generate_distinct_colors(len(sam_masks))
                img_with_masks = overlay_masks_on_image(image, sam_masks, colors, alpha=0.5)
                st.image(img_with_masks, caption="Image with SAM Masks", use_container_width=True)
        
        if step_3_state.get('error'):
            st.error(f"❌ Error: {step_3_state['error']}")
    
    st.markdown("---")
    
    # Step 4: Clustering
    with st.container():
        step_4_state = st.session_state.pipeline_state['step_4']
        step_3_completed = st.session_state.pipeline_state['step_3']['completed'] and not st.session_state.pipeline_state['step_3'].get('error')
        
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            status_icon = "✅" if step_4_state['completed'] else "⏸️"
            st.markdown(f"### {status_icon} Step 4: Clustering")
            st.caption("Run DBSCAN clustering on points assigned to each mask")
            if not step_3_completed:
                st.warning("⚠️ Requires Step 3")
        
        with col2:
            if st.button("▶️ Run Step 4", key="run_step_4", disabled=not step_3_completed):
                with st.spinner("Running Step 4..."):
                    try:
                        step_2_result = st.session_state.pipeline_state['step_2']['result']
                        step_3_result = st.session_state.pipeline_state['step_3']['result']
                        result = step_4_clustering(
                            sample_meta_data=sample_meta_data,
                            sparse_points=step_2_result['colored_sparse_points'],
                            sam_masks=step_3_result['sam_masks'],
                            mask_assignments=step_3_result['mask_assignments'],
                            **st.session_state.params['clustering']
                        )
                        st.session_state.pipeline_state['step_4'] = {
                            'completed': True,
                            'result': result,
                            'time': result['time'],
                            'error': None
                        }
                        st.rerun()
                    except Exception as e:
                        st.session_state.pipeline_state['step_4']['error'] = str(e)
                        st.error(f"Step 4 failed: {str(e)}")
        
        with col3:
            if step_4_state['completed']:
                st.metric("Time", f"{step_4_state['time']:.2f}s")
        
        if step_4_state['completed']:
            result = step_4_state['result']
            total_clusters = sum(r['Clusters Found'] for r in result['clustering_results'])
            st.success(f"✅ Completed: {total_clusters} clusters found across {len(result['clustering_results'])} masks")
            
            with st.expander("View Step 4 Details", expanded=False):
                if result['clustering_results']:
                    df = pd.DataFrame(result['clustering_results'])
                    st.dataframe(df, use_container_width=True)
        
        if step_4_state.get('error'):
            st.error(f"❌ Error: {step_4_state['error']}")
    
    st.markdown("---")
    
    # Step 5: Detection & Pose Estimation
    with st.container():
        step_5_state = st.session_state.pipeline_state['step_5']
        step_4_completed = st.session_state.pipeline_state['step_4']['completed']
        
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            status_icon = "✅" if step_5_state['completed'] else "⏸️"
            st.markdown(f"### {status_icon} Step 5: Detection & Pose Estimation")
            st.caption("Fit cuboids to best cluster points using scoring-based method")
            if not step_4_completed:
                st.warning("⚠️ Requires Step 4")
        
        with col2:
            if st.button("▶️ Run Step 5", key="run_step_5", disabled=not step_4_completed):
                with st.spinner("Running Step 5..."):
                    try:
                        step_1_result = st.session_state.pipeline_state['step_1']['result']
                        step_3_result = st.session_state.pipeline_state['step_3']['result']
                        step_4_result = st.session_state.pipeline_state['step_4']['result']
                        result = step_5_detection_pose_estimation(
                            sample_meta_data=sample_meta_data,
                            best_cluster_points=step_4_result['best_cluster_points'],
                            sam_masks=step_3_result['sam_masks'],
                            ground_z=step_1_result['ground_z'],
                            cuboid_params=st.session_state.params['cuboid_fitting']
                        )
                        st.session_state.pipeline_state['step_5'] = {
                            'completed': True,
                            'result': result,
                            'time': result['time'],
                            'error': None
                        }
                        # Store detected cuboids in session state for evaluation
                        st.session_state.cuboids = result['detected_cuboids']
                        st.rerun()
                    except Exception as e:
                        st.session_state.pipeline_state['step_5']['error'] = str(e)
                        st.error(f"Step 5 failed: {str(e)}")
        
        with col3:
            if step_5_state['completed']:
                st.metric("Time", f"{step_5_state['time']:.2f}s")
        
        if step_5_state['completed']:
            result = step_5_state['result']
            st.success(f"✅ Completed: {result['n_detected']} cuboids detected")
            
            with st.expander("View Step 5 Details", expanded=False):
                if result['detected_cuboids']:
                    cuboid_data = []
                    for i, cuboid in enumerate(result['detected_cuboids']):
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
                    step_1_result = st.session_state.pipeline_state['step_1']['result']
                    point_cloud_obj = step_1_result['point_cloud_obj']
                    fig = create_3d_scatter_plot(
                        points=point_cloud_obj,
                        labels=None,
                        mask_points=None,
                        cuboids=result['detected_cuboids'],
                        rays=None,
                        points_in_frustums=None,
                        reconstructed_points=None,
                        show_lidar=True,
                        show_reconstructed=False,
                        color_by_depth=False,
                        title="Detected Objects"
                    )
                    if result['detected_cuboids']:
                        add_cuboids_to_figure(fig, result['detected_cuboids'], color='red', opacity=0.3, name_prefix="Detected: ")
                    st.plotly_chart(fig, use_container_width=True)
        
        if step_5_state.get('error'):
            st.error(f"❌ Error: {step_5_state['error']}")


if __name__ == "__main__":
    main()

