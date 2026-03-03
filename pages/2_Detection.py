"""
3D Object Detection Pipeline
Unified detection pipeline with step-by-step execution and full pipeline mode.
"""
import os
import time
from typing import List, Dict, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.core.pointcloud_projection import PointCloud, Projection
from components.core.depth_estimation import compute_sparse_depth_map
from components.core.sam_integration import SAMIntegration, assign_points_to_masks
from components.core.pose_estimation import fit_cuboid_to_points
from components.core.clustering_manager import ClusteringManager, select_best_cluster_points
from components.core.utils import get_bbox_from_mask, calculate_iou
from components.dataset_loaders.utils import load_dataset_sample
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
# Note: _get_bbox_from_mask and _calculate_iou are now imported from components.core.utils


def save_sample_to_hard_drive_after_processing(
    image: Optional[np.ndarray],
    point_cloud: Optional[np.ndarray],
    sample_meta_data: Dict,
) -> None:
    """
    Save processed sample (image + LiDAR) to disk under the output_root_dir specified
    on the Dataset Extraction page. Creates 'images' and 'lidar' subfolders and writes:
    - image as PNG
    - point cloud as PCD (XYZ)
    """
    output_root = st.session_state.get("output_root_dir", "")
    if not output_root:
        return

    root_path = Path(output_root).expanduser()
    images_dir = root_path / "images"
    lidar_dir = root_path / "lidar"
    images_dir.mkdir(parents=True, exist_ok=True)
    lidar_dir.mkdir(parents=True, exist_ok=True)

    dataset_type = (sample_meta_data or {}).get("dataset_type", "unknown")
    sample_index = (sample_meta_data or {}).get("sample_index", "unknown")

    base_name = f"{dataset_type}_{sample_index}"

    # Save image
    if image is not None:
        img_path = images_dir / f"{base_name}.png"
        try:
            # Ensure RGB → BGR for OpenCV writing
            img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(img_path), img_bgr)
        except Exception:
            pass

    # Save point cloud as PCD (XYZ)
    if point_cloud is not None and len(point_cloud) > 0:
        pcd_path = lidar_dir / f"{base_name}.pcd"
        try:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
            o3d.io.write_point_cloud(str(pcd_path), pcd, write_ascii=True)
        except Exception:
            pass


# ============================================================================
# Pipeline Step Functions
# ============================================================================

def step_1_ground_plane_removal(
    point_cloud: np.ndarray,
    distance_threshold: float = 0.3,
    ransac_n: int = 3,
    num_iterations: int = 1000,
    filter_forward_only: bool = False
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
    print(f"Point cloud object created with {len(point_cloud_obj.original_point_cloud)} points")
    # Remove ground plane
    point_cloud_obj.remove_ground_plane_ransac(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
        filter_forward_only=filter_forward_only
    )
    print(f"Ground plane removed with {len(point_cloud_obj.point_cloud_plane_removed)} points remaining")
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
    class_names: List[str],
    sam_model_type: str = 'sam2_t',
    yolo_model_path: Optional[str] = None,
    conf_threshold: float = 0.25
) -> Dict:
    """
    Step 3: Generate SAM masks using open-vocabulary detection and assign original LiDAR points to masks.
    
    Args:
        sample_meta_data: Sample metadata
        image: HxWx3 RGB image
        sparse_points: Nx3 array of backprojected sparse depth points
        class_names: List of class names to detect (e.g., ["car", "person", "bicycle"])
        sam_model_type: 'sam2_t' or 'sam3'
        yolo_model_path: Optional path to YOLO-World model (only used for SAM2)
        conf_threshold: Confidence threshold for YOLO detections (default: 0.25)
    
    Returns:
        Dict with 'sam_masks', 'mask_assignments', 'mask_bboxes', 'class_names', 'confidences'
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
    
    # Check if model type changed, reinitialize if needed
    if 'sam_initialized_model_type' not in st.session_state:
        st.session_state.sam_initialized_model_type = sam_model_type
    
    if st.session_state.sam_initialized_model_type != sam_model_type:
        try:
            st.session_state.sam_integration = SAMIntegration(model_type=sam_model_type)
            st.session_state.sam_initialized_model_type = sam_model_type
        except Exception as e:
            return {
                'error': f"SAM reinitialization failed: {str(e)}",
                'sam_masks': None,
                'mask_assignments': None,
                'mask_bboxes': [],
                'class_names': [],
                'confidences': [],
                'time': time.time() - start_time
            }
    
    sam_integration = st.session_state.sam_integration
    h, w = image.shape[:2]
    
    # Use open-vocabulary detection pipeline
    if not class_names:
        return {
            'error': "No class names provided",
            'sam_masks': [],
            'mask_assignments': None,
            'mask_bboxes': [],
            'class_names': [],
            'confidences': [],
            'time': time.time() - start_time
        }
    
    try:
        # Use the unified segment_by_class_names method
        segment_results = sam_integration.segment_by_class_names(
            image=image,
            class_names=class_names,
            yolo_model_path=yolo_model_path,
            conf_threshold=conf_threshold
        )
        
        sam_masks = segment_results['masks']
        mask_bboxes = segment_results['bboxes']
        detected_class_names = segment_results['class_names']
        confidences = segment_results['confidences']
        
    except Exception as e:
        return {
            'error': f"Segmentation failed: {str(e)}",
            'sam_masks': [],
            'mask_assignments': None,
            'mask_bboxes': [],
            'class_names': [],
            'confidences': [],
            'time': time.time() - start_time
        }
    
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
        'mask_bboxes': mask_bboxes,
        'class_names': detected_class_names,
        'confidences': confidences,
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
        best_cluster_points = select_best_cluster_points(
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
    
    # Get detected class names from step 3 if available
    detected_class_names = None
    if 'step_3' in st.session_state.pipeline_state:
        step_3_result = st.session_state.pipeline_state['step_3'].get('result', {})
        detected_class_names = step_3_result.get('class_names', [])
    
    detected_cuboids = []
    
    # Fit cuboid to each mask's best cluster
    for mask_idx, cluster_points in best_cluster_points.items():
        if len(cluster_points) < 5:
            continue
        
        # Get category from detected class names (from open-vocab detection)
        category = 'Unknown'
        if detected_class_names and mask_idx < len(detected_class_names):
            category = detected_class_names[mask_idx]
        else:
            # Fallback to ground truth if available
            ground_truth_boxes = sample_meta_data.get('ground_truth_boxes', [])
            if ground_truth_boxes and mask_idx < len(sam_masks):
                mask_bbox = get_bbox_from_mask(sam_masks[mask_idx])
                best_iou = 0.0
                best_category = 'Unknown'
                for gt_box in ground_truth_boxes:
                    bbox_2d = gt_box.get('bbox_2d')
                    if bbox_2d is None:
                        continue
                    gt_bbox = [bbox_2d['left'], bbox_2d['top'], bbox_2d['right'], bbox_2d['bottom']]
                    iou = calculate_iou(mask_bbox, gt_bbox)
                    if iou > best_iou and iou > 0.3:
                        best_iou = iou
                        best_category = gt_box.get('category', 'Unknown')
                category = best_category
        
        # Get dimensions for this category from session_state (pre-computed when class names change)
        dimensions_by_class = st.session_state.params.get('dimensions_by_class', {})
        dimensions = dimensions_by_class.get(category)
        if dimensions is None:
            from components.core.constants import KITTI_CUBOID_TEMPLATES
            t = KITTI_CUBOID_TEMPLATES.get(category, KITTI_CUBOID_TEMPLATES['Unknown'])
            dimensions = (float(t['length']), float(t['width']), float(t['height']))
        
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
                score_weights=score_weights,
                ground_z=ground_z
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
                'source_bbox_idx': None,  # Not using ground truth matching anymore
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
    class_names = params.get('class_names', [])
    if not class_names:
        class_names = ['car', 'person', 'bicycle']  # Default classes
    
    step_3_result = step_3_sam_segmentation(
        sample_meta_data=st.session_state.sample['sample_meta_data'],
        image=st.session_state.sample['image'],
        sparse_points=step_2_result['colored_sparse_points'],
        class_names=class_names,
        sam_model_type=params.get('sam_model_type', 'sam2_t'),
        yolo_model_path=params.get('yolo_model_path', None),
        conf_threshold=params.get('yolo_conf_threshold', 0.25)
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
# Batch processing: run pipeline for one sample and return export_results
# ============================================================================

def _run_pipeline_for_batch_sample(
    dataset_path: str,
    dataset_type: str,
    sample_index,
) -> Optional[Dict]:
    """Load one sample, run full pipeline, return export_results for Export page."""
    meta, image, point_cloud = load_dataset_sample(
        dataset_path=dataset_path,
        sample_index=sample_index,
        dataset_type=dataset_type,
        filter_forward_only=False,
    )
    if meta is None or image is None or point_cloud is None:
        return None
    st.session_state.sample = {
        "sample_meta_data": meta,
        "image": image,
        "point_cloud": point_cloud,
    }
    st.session_state.pipeline_state = {
        "step_1": {"completed": False, "result": None, "time": None, "error": None},
        "step_2": {"completed": False, "result": None, "time": None, "error": None},
        "step_3": {"completed": False, "result": None, "time": None, "error": None},
        "step_4": {"completed": False, "result": None, "time": None, "error": None},
        "step_5": {"completed": False, "result": None, "time": None, "error": None},
    }
    try:
        results = run_full_pipeline(st.session_state.params)
    except Exception:
        return None
    step_5_result = results["step_5"]["result"]
    dataset_type_lower = meta.get("dataset_type", "unknown").lower()
    export_results = {
        "detected_cuboids": step_5_result["detected_cuboids"],
        "metadata": {
            "dataset_type": dataset_type_lower,
            "sample_index": meta.get("sample_index", "unknown"),
            "image_path": meta.get("image_path", "unknown"),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_detections": step_5_result["n_detected"],
            "pipeline_params": st.session_state.params.copy() if "params" in st.session_state else {},
        },
    }
    if dataset_type_lower == "kitti":
        ground_truth_boxes = meta.get("ground_truth_boxes", [])
        if ground_truth_boxes:
            ground_truth_cuboids = []
            for gt_box in ground_truth_boxes:
                gt_cuboid = {
                    "category": gt_box.get("category", "Unknown"),
                    "corners": gt_box.get("corners"),
                    "bbox_2d": gt_box.get("bbox_2d"),
                    "min_x": gt_box.get("min_x"),
                    "max_x": gt_box.get("max_x"),
                    "min_y": gt_box.get("min_y"),
                    "max_y": gt_box.get("max_y"),
                    "min_z": gt_box.get("min_z"),
                    "max_z": gt_box.get("max_z"),
                    "format": "kitti_gt",
                }
                ground_truth_cuboids.append(gt_cuboid)
            export_results["ground_truth_cuboids"] = ground_truth_cuboids
            export_results["metadata"]["n_ground_truth"] = len(ground_truth_cuboids)

    # Save each processed batch sample (image + LiDAR) to disk
    save_sample_to_hard_drive_after_processing(
        image=image,
        point_cloud=point_cloud,
        sample_meta_data=meta,
    )

    return export_results


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
                'filter_forward_only': False
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
            'sam_model_type': 'sam2_t',
            'class_names': ['car', 'person', 'bicycle'],
            'yolo_model_path': None,
            'yolo_conf_threshold': 0.25
        }
    
    # Batch mode: process entire batch only (no per-sample image view)
    batch_samples = st.session_state.get("batch_samples", [])
    process_all_samples = st.session_state.get("process_all_samples", False)
    if batch_samples and process_all_samples:
        st.subheader("📚 Batch Processing")
        total = len(batch_samples)
        st.info(f"Batch loaded: **{total}** samples. Process the entire batch to run detection on all samples.")
        if st.button("🚀 Process entire batch", type="primary", key="process_entire_batch"):
            results_list = []
            progress = st.progress(0.0)
            for i, sample_desc in enumerate(batch_samples):
                progress.progress((i + 1) / total)
                export_res = _run_pipeline_for_batch_sample(
                    dataset_path=sample_desc["dataset_path"],
                    dataset_type=sample_desc.get("dataset_type", "kitti"),
                    sample_index=sample_desc["sample_index"],
                )
                if export_res is not None:
                    results_list.append(export_res)
            st.session_state.batch_export_results = {"samples": results_list}
            st.session_state.process_all_samples = False
            st.success(f"✅ Processed **{len(results_list)}** / {total} samples. Go to **4_Export** to save (e.g. XML).")
            st.rerun()
        if st.session_state.get("batch_export_results"):
            br = st.session_state.batch_export_results
            n = len(br.get("samples", []))
            st.success(f"Last batch: **{n}** samples ready for export on **4_Export**.")
        return
    
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
            "Forward-Facing Only", False
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
        help="SAM2: Uses YOLOWorld for open vocab bounding boxes and segment with SAM2. SAM3: Uses text prompts, full open vocab capability."
    )
    
    # Class names input
    st.sidebar.markdown("### Open Vocabulary Detection")
    class_names_input = st.sidebar.text_input(
        "Class Names (comma-separated)",
        value="car, person, bicycle, bus, truck",
        help="Enter class names separated by commas (e.g., 'car, person, bicycle')"
    )
    
    # LLM Settings
    with st.sidebar.expander("LLM Settings", expanded=False):
        from components.core.llm_service import set_llm_temperature, get_llm_temperature
        
        # Initialize temperature in session state if not present
        if 'llm_temperature' not in st.session_state:
            st.session_state.llm_temperature = get_llm_temperature()
        
        # Temperature slider
        st.session_state.llm_temperature = st.slider(
            "LLM Temperature",
            0.0, 2.0, st.session_state.llm_temperature, 0.1,
            help="Temperature for LLM generation. Lower values (0.0-0.5) = more deterministic, Higher values (1.0-2.0) = more creative/random. Default: 0.3"
        )
        
        # Update LLM service temperature
        set_llm_temperature(st.session_state.llm_temperature)
        
        st.sidebar.caption("💡 LLM is used when semantic similarity doesn't find a match (similarity < 0.75)")

    # Parse class names
    if class_names_input:
        class_names = [name.strip() for name in class_names_input.split(',') if name.strip()]
        
        # Check if class names have changed
        previous_class_names = st.session_state.get('previous_class_names', [])
        if set(class_names) != set(previous_class_names):
            # Pre-compute dimensions for new class names and store in session_state
            from components.core.llm_service import query_llm_for_dimensions
            
            with st.sidebar.spinner("Pre-computing dimensions for class names..."):
                dims_by_class = {}
                for class_name in class_names:
                    length, width, height = query_llm_for_dimensions(class_name)
                    dims_by_class[class_name] = (length, width, height)
                
                st.session_state.params['dimensions_by_class'] = dims_by_class
                # template_dims format for frustum_manager / evaluation
                st.session_state.params['template_dims'] = {
                    k: {'length': v[0], 'width': v[1], 'height': v[2]}
                    for k, v in dims_by_class.items()
                }
            
            st.session_state.previous_class_names = class_names.copy()
        
        st.session_state.params['class_names'] = class_names
    else:
        st.session_state.params['class_names'] = []
    
    if not st.session_state.params['class_names']:
        st.sidebar.warning("⚠️ Please enter at least one class name")
    
    # YOLO confidence threshold (only for SAM2)
    if st.session_state.params['sam_model_type'].startswith('sam2'):
        st.session_state.params['yolo_conf_threshold'] = st.sidebar.slider(
            "YOLO Confidence Threshold",
            0.0, 1.0, 0.25, 0.05,
            help="Confidence threshold for YOLO-World detections (only used with SAM2)"
        )
        st.sidebar.info("💡 SAM2 uses YOLO-World for detection, then SAM2 for segmentation")
    else:
        st.sidebar.info("💡 SAM3 uses direct text prompts for open-vocabulary segmentation")
    
    st.sidebar.caption("ℹ️ Confidence scores are shown for reference only and not used in further processing")
    
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
                # Update session state so Export and Evaluation pages see the latest results
                step_5_result = results['step_5']['result']
                st.session_state.cuboids = step_5_result['detected_cuboids']
                sample_meta_data = st.session_state.sample['sample_meta_data']
                dataset_type = sample_meta_data.get('dataset_type', 'unknown').lower()
                export_results = {
                    'detected_cuboids': step_5_result['detected_cuboids'],
                    'metadata': {
                        'dataset_type': dataset_type,
                        'sample_index': sample_meta_data.get('sample_index', 'unknown'),
                        'image_path': sample_meta_data.get('image_path', 'unknown'),
                        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                        'n_detections': step_5_result['n_detected'],
                        'pipeline_params': st.session_state.params.copy() if 'params' in st.session_state else {}
                    }
                }
                if dataset_type == 'kitti':
                    ground_truth_boxes = sample_meta_data.get('ground_truth_boxes', [])
                    if ground_truth_boxes:
                        ground_truth_cuboids = []
                        for gt_box in ground_truth_boxes:
                            gt_cuboid = {
                                'category': gt_box.get('category', 'Unknown'),
                                'corners': gt_box.get('corners'),
                                'bbox_2d': gt_box.get('bbox_2d'),
                                'min_x': gt_box.get('min_x'),
                                'max_x': gt_box.get('max_x'),
                                'min_y': gt_box.get('min_y'),
                                'max_y': gt_box.get('max_y'),
                                'min_z': gt_box.get('min_z'),
                                'max_z': gt_box.get('max_z'),
                                'format': 'kitti_gt'
                            }
                            ground_truth_cuboids.append(gt_cuboid)
                        export_results['ground_truth_cuboids'] = ground_truth_cuboids
                        export_results['metadata']['n_ground_truth'] = len(ground_truth_cuboids)
                st.session_state.export_results = export_results

                # Save processed sample (single-sample mode) to disk
                current_image = st.session_state.sample.get("image")
                current_pc = st.session_state.sample.get("point_cloud")
                save_sample_to_hard_drive_after_processing(
                    image=current_image,
                    point_cloud=current_pc,
                    sample_meta_data=sample_meta_data,
                )

                # Explore and display what the pipeline returns
                st.success("✅ Pipeline completed successfully!")
                
                # Display pipeline results structure
                with st.expander("🔍 Pipeline Results Structure", expanded=True):
                    st.markdown("### Full Pipeline Return Value")
                    
                    # Show top-level structure
                    st.markdown("#### Top-level keys:")
                    st.write(list(results.keys()))
                    
                    # Show structure for each step
                    for step_key in ['step_1', 'step_2', 'step_3', 'step_4', 'step_5']:
                        if step_key in results:
                            st.markdown(f"#### {step_key.upper()}:")
                            step_data = results[step_key]
                            
                            # Show step metadata
                            st.write(f"**Completed:** {step_data.get('completed', False)}")
                            
                            # Show result keys
                            if 'result' in step_data and step_data['result'] is not None:
                                result = step_data['result']
                                st.write(f"**Result keys:** {list(result.keys())}")
                                
                                # Show summary for each step
                                if step_key == 'step_1':
                                    st.write(f"- Points remaining: {result.get('points_remaining', 'N/A')}")
                                    st.write(f"- Ground Z: {result.get('ground_z', 'N/A')}")
                                    st.write(f"- Time: {result.get('time', 'N/A'):.3f}s")
                                elif step_key == 'step_2':
                                    st.write(f"- N points: {result.get('n_points', 'N/A')}")
                                    st.write(f"- Time: {result.get('time', 'N/A'):.3f}s")
                                elif step_key == 'step_3':
                                    st.write(f"- N masks: {result.get('n_masks', 'N/A')}")
                                    st.write(f"- Time: {result.get('time', 'N/A'):.3f}s")
                                    if result.get('error'):
                                        st.error(f"- Error: {result['error']}")
                                elif step_key == 'step_4':
                                    st.write(f"- Clustering results: {len(result.get('clustering_results', []))} masks")
                                    st.write(f"- Time: {result.get('time', 'N/A'):.3f}s")
                                elif step_key == 'step_5':
                                    st.write(f"- N detected: {result.get('n_detected', 'N/A')}")
                                    st.write(f"- Time: {result.get('time', 'N/A'):.3f}s")
                                
                                # Show full result structure (simplified)
                                st.json({
                                    'keys': list(result.keys()),
                                    'types': {k: str(type(v)) for k, v in result.items() if not isinstance(v, (np.ndarray, list, dict))},
                                    'array_shapes': {k: v.shape if isinstance(v, np.ndarray) else None for k, v in result.items() if isinstance(v, np.ndarray)},
                                    'list_lengths': {k: len(v) if isinstance(v, list) else None for k, v in result.items() if isinstance(v, list)},
                                    'dict_keys': {k: list(v.keys()) if isinstance(v, dict) else None for k, v in result.items() if isinstance(v, dict)},
                                })
                    
                    # Show full results JSON (collapsed)
                    with st.expander("📋 Full Results JSON (for debugging)", expanded=False):
                        # Convert numpy arrays and objects to serializable format
                        def make_serializable(obj):
                            if isinstance(obj, np.ndarray):
                                return {
                                    'type': 'numpy.ndarray',
                                    'shape': obj.shape,
                                    'dtype': str(obj.dtype),
                                    'size': obj.size,
                                    'sample': obj.flatten()[:10].tolist() if obj.size > 0 else []
                                }
                            elif isinstance(obj, dict):
                                return {k: make_serializable(v) for k, v in obj.items()}
                            elif isinstance(obj, list):
                                return [make_serializable(item) for item in obj[:5]]  # Limit list size
                            elif hasattr(obj, '__dict__'):
                                return {'type': str(type(obj)), '__dict__': make_serializable(obj.__dict__)}
                            else:
                                return obj
                        
                        serializable_results = make_serializable(results)
                        st.json(serializable_results)
                
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
        col1, col2 = st.columns([3, 1])
        
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
                        prev = st.session_state.pipeline_state['step_1']
                        st.session_state.pipeline_state['step_1'] = {
                            'completed': prev.get('completed', False),
                            'result': prev.get('result'),
                            'time': prev.get('time'),
                            'error': str(e)
                        }
                        st.error(f"Step 1 failed: {str(e)}")
        
        if step_1_state['completed']:
            result = step_1_state['result']
            st.success(f"✅ Completed: {result['points_remaining']:,} points remaining")
            
            with st.expander("View Step 1 Details", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Original Points", f"{len(point_cloud):,}")
                st.metric("Points Remaining", f"{result['points_remaining']:,}")
                with col2:
                    reduction = (1 - result['points_remaining'] / len(point_cloud)) * 100
                    st.metric("Reduction", f"{reduction:.1f}%")
                st.metric("Ground Z", f"{result['ground_z']:.3f}m" if result['ground_z'] else "N/A")
                
                # 3D Visualization: Before and After
                st.markdown("#### 3D Point Cloud Visualization")
                point_cloud_obj = result['point_cloud_obj']
                
                # Show ground-removed point cloud
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
                    title="Point Cloud After Ground Removal"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        if step_1_state.get('error'):
            st.error(f"❌ Error: {step_1_state['error']}")
    
    st.markdown("---")
    
    # Step 2: Sparse Depth Backprojection
    with st.container():
        step_2_state = st.session_state.pipeline_state['step_2']
        step_1_completed = st.session_state.pipeline_state['step_1']['completed']
        
        col1, col2 = st.columns([3, 1])
        
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
                        prev = st.session_state.pipeline_state['step_2']
                        st.session_state.pipeline_state['step_2'] = {
                            'completed': prev.get('completed', False),
                            'result': prev.get('result'),
                            'time': prev.get('time'),
                            'error': str(e)
                        }
                        st.error(f"Step 2 failed: {str(e)}")
        
        if step_2_state['completed']:
            result = step_2_state['result']
            st.success(f"✅ Completed: {result['n_points']:,} points backprojected")
            
            with st.expander("View Step 2 Details", expanded=True):
                # 2D Visualizations
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
                
                # 3D Visualization: Colored sparse points
                st.markdown("#### 3D Colored Sparse Points")
                if len(result['colored_sparse_points']) > 0:
                    # Create visualization with colored points
                    fig = go.Figure()
                    
                    # Add colored points
                    colored_points = result['colored_sparse_points']
                    colors = result['colored_sparse_colors']
                    
                    # Sample points if too many for performance
                    max_points = 10000
                    if len(colored_points) > max_points:
                        indices = np.random.choice(len(colored_points), max_points, replace=False)
                        colored_points = colored_points[indices]
                        colors = colors[indices]
                    
                    # Colors are already in 0-255 format
                    fig.add_trace(go.Scatter3d(
                        x=colored_points[:, 0],
                        y=colored_points[:, 1],
                        z=colored_points[:, 2],
                        mode='markers',
                        marker=dict(
                            size=2,
                            color=[f'rgb({int(r)},{int(g)},{int(b)})' for r, g, b in colors],
                            opacity=0.8
                        ),
                        name='Colored Sparse Points'
                    ))
                    
                    fig.update_layout(
                        title="3D Colored Sparse Points (Backprojected from Image)",
                        scene=dict(
                            xaxis_title="X (m)",
                            yaxis_title="Y (m)",
                            zaxis_title="Z (m)",
                            aspectmode='data'
                        ),
                        height=600
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No colored sparse points to visualize")
        
        if step_2_state.get('error'):
            st.error(f"❌ Error: {step_2_state['error']}")
    
    st.markdown("---")
    
    # Step 3: SAM Segmentation
    with st.container():
        step_3_state = st.session_state.pipeline_state['step_3']
        step_2_completed = st.session_state.pipeline_state['step_2']['completed']
        
        col1, col2 = st.columns([3, 1])
        
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
                        class_names = st.session_state.params.get('class_names', ['car', 'person', 'bicycle'])
                        result = step_3_sam_segmentation(
                            sample_meta_data=sample_meta_data,
                            image=image,
                            sparse_points=step_2_result['colored_sparse_points'],
                            class_names=class_names,
                            sam_model_type=st.session_state.params['sam_model_type'],
                            yolo_model_path=st.session_state.params.get('yolo_model_path', None),
                            conf_threshold=st.session_state.params.get('yolo_conf_threshold', 0.25)
                        )
                        st.session_state.pipeline_state['step_3'] = {
                            'completed': True,
                            'result': result,
                            'time': result['time'],
                            'error': result.get('error')
                        }
                        st.rerun()
                    except Exception as e:
                        prev = st.session_state.pipeline_state['step_3']
                        st.session_state.pipeline_state['step_3'] = {
                            'completed': prev.get('completed', False),
                            'result': prev.get('result'),
                            'time': prev.get('time'),
                            'error': str(e)
                        }
                        st.error(f"Step 3 failed: {str(e)}")
        
        if step_3_state['completed'] and not step_3_state.get('error'):
            result = step_3_state['result']
            detected_classes = result.get('class_names', [])
            unique_classes = list(set(detected_classes)) if detected_classes else []
            n_masks = result.get('n_masks', len(result.get('sam_masks', [])))
            st.success(f"✅ Completed: {n_masks} masks generated ({len(unique_classes)} unique classes: {', '.join(unique_classes)})")
            
            with st.expander("View Step 3 Details", expanded=True):
                sam_masks = result['sam_masks']
                mask_bboxes = result.get('mask_bboxes', [])
                detected_class_names = result.get('class_names', [])
                confidences = result.get('confidences', [])
                colors = generate_distinct_colors(len(sam_masks))
                
                # 2D Visualization: Masks overlay with bounding boxes and labels
                st.markdown("#### 2D Mask Visualization")
                img_with_masks = overlay_masks_on_image(image, sam_masks, colors, alpha=0.5)
                
                # Draw bounding boxes and labels with confidence scores
                import matplotlib.patches as patches
                fig, ax = plt.subplots(1, 1, figsize=(12, 8))
                ax.imshow(img_with_masks)
                ax.axis('off')
                
                for i, (bbox, class_name, confidence) in enumerate(zip(mask_bboxes, detected_class_names, confidences)):
                    if bbox and len(bbox) == 4:
                        x1, y1, x2, y2 = bbox
                        rect = patches.Rectangle(
                            (x1, y1), x2 - x1, y2 - y1,
                            linewidth=2, edgecolor=colors[i], facecolor='none'
                        )
                        ax.add_patch(rect)
                        # Add label with confidence
                        label = f"{class_name}: {confidence:.2f}" if confidence is not None else class_name
                        ax.text(x1, y1 - 5, label, color=colors[i], fontsize=10, 
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
                
                ax.set_title("Detected Objects with Masks, Bounding Boxes, and Confidence Scores")
                st.pyplot(fig)
                plt.close()
                
                # Summary table of detected objects
                if detected_class_names:
                    st.markdown("#### Detected Objects Summary")
                    detection_data = []
                    for i, (class_name, confidence, bbox) in enumerate(zip(detected_class_names, confidences, mask_bboxes)):
                        detection_data.append({
                            'ID': i + 1,
                            'Class': class_name,
                            'Confidence': f"{confidence:.3f}" if confidence is not None else "N/A",
                            'BBox': f"[{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]" if bbox and len(bbox) == 4 else "N/A"
                        })
                    df_detections = pd.DataFrame(detection_data)
                    st.dataframe(df_detections, use_container_width=True)
                
                # 3D Visualization: Points assigned to masks
                st.markdown("#### 3D Point Assignment Visualization")
                if result['mask_assignments'] is not None and len(result['mask_assignments']) > 0:
                    step_2_result = st.session_state.pipeline_state['step_2']['result']
                    sparse_points = step_2_result['colored_sparse_points']
                    mask_assignments = result['mask_assignments']
                    
                    # Create visualization showing points colored by mask assignment
                    fig = go.Figure()
                    
                    # Add points for each mask
                    for mask_idx in range(len(sam_masks)):
                        mask_points = sparse_points[mask_assignments == mask_idx]
                        if len(mask_points) > 0:
                            # Sample if too many points
                            max_points = 5000
                            if len(mask_points) > max_points:
                                indices = np.random.choice(len(mask_points), max_points, replace=False)
                                mask_points = mask_points[indices]
                            
                            color = colors[mask_idx]
                            fig.add_trace(go.Scatter3d(
                                x=mask_points[:, 0],
                                y=mask_points[:, 1],
                                z=mask_points[:, 2],
                                mode='markers',
                                marker=dict(
                                    size=2,
                                    color=f'rgb({int(color[0]*255)},{int(color[1]*255)},{int(color[2]*255)})',
                                    opacity=0.7
                                ),
                                name=f'Mask {mask_idx + 1}'
                            ))
                    
                    # Add unassigned points (if any)
                    unassigned_points = sparse_points[mask_assignments == -1]
                    if len(unassigned_points) > 0:
                        max_points = 2000
                        if len(unassigned_points) > max_points:
                            indices = np.random.choice(len(unassigned_points), max_points, replace=False)
                            unassigned_points = unassigned_points[indices]
                        
                        fig.add_trace(go.Scatter3d(
                            x=unassigned_points[:, 0],
                            y=unassigned_points[:, 1],
                            z=unassigned_points[:, 2],
                            mode='markers',
                            marker=dict(
                                size=1,
                                color='gray',
                                opacity=0.3
                            ),
                            name='Unassigned'
                        ))
                    
                    fig.update_layout(
                        title="3D Points Colored by Mask Assignment",
                        scene=dict(
                            xaxis_title="X (m)",
                            yaxis_title="Y (m)",
                            zaxis_title="Z (m)",
                            aspectmode='data'
                        ),
                        height=600
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No mask assignments to visualize")
        
        if step_3_state.get('error'):
            st.error(f"❌ Error: {step_3_state['error']}")
    
    st.markdown("---")
    
    # Step 4: Clustering
    with st.container():
        step_4_state = st.session_state.pipeline_state['step_4']
        step_3_completed = st.session_state.pipeline_state['step_3']['completed'] and not st.session_state.pipeline_state['step_3'].get('error')
        
        col1, col2 = st.columns([3, 1])
        
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
                        prev = st.session_state.pipeline_state['step_4']
                        st.session_state.pipeline_state['step_4'] = {
                            'completed': prev.get('completed', False),
                            'result': prev.get('result'),
                            'time': prev.get('time'),
                            'error': str(e)
                        }
                        st.error(f"Step 4 failed: {str(e)}")
        
        if step_4_state['completed']:
            result = step_4_state['result']
            total_clusters = sum(r['Clusters Found'] for r in result['clustering_results'])
            st.success(f"✅ Completed: {total_clusters} clusters found across {len(result['clustering_results'])} masks")
            
            with st.expander("View Step 4 Details", expanded=True):
                if result['clustering_results']:
                    # Statistics table
                    st.markdown("#### Clustering Statistics")
                    df = pd.DataFrame(result['clustering_results'])
                    st.dataframe(df, use_container_width=True)
                    
                    # 3D Visualization: Clusters
                    st.markdown("#### 3D Cluster Visualization")
                    step_2_result = st.session_state.pipeline_state['step_2']['result']
                    sparse_points = step_2_result['colored_sparse_points']
                    step_3_result = st.session_state.pipeline_state['step_3']['result']
                    mask_assignments = step_3_result['mask_assignments']
                    mask_cluster_labels = result['mask_cluster_labels']
                    
                    # Create visualization showing clusters
                    fig = go.Figure()
                    
                    # Generate colors for clusters
                    max_clusters = 0
                    for mask_idx, cluster_labels in mask_cluster_labels.items():
                        unique_labels = np.unique(cluster_labels)
                        unique_labels = unique_labels[unique_labels >= 0]
                        max_clusters = max(max_clusters, len(unique_labels))
                    
                    cluster_colors = generate_distinct_colors(max_clusters * len(mask_cluster_labels))
                    color_idx = 0
                    
                    # Add points for each cluster
                    for mask_idx, cluster_labels in mask_cluster_labels.items():
                        mask_points = sparse_points[mask_assignments == mask_idx]
                        if len(mask_points) == 0:
                            continue
                        
                        unique_labels = np.unique(cluster_labels)
                        unique_labels = unique_labels[unique_labels >= 0]
                        
                        for cluster_id in unique_labels:
                            cluster_points = mask_points[cluster_labels == cluster_id]
                            if len(cluster_points) > 0:
                                # Sample if too many points
                                max_points = 3000
                                if len(cluster_points) > max_points:
                                    indices = np.random.choice(len(cluster_points), max_points, replace=False)
                                    cluster_points = cluster_points[indices]
                                
                                color = cluster_colors[color_idx % len(cluster_colors)]
                                fig.add_trace(go.Scatter3d(
                                    x=cluster_points[:, 0],
                                    y=cluster_points[:, 1],
                                    z=cluster_points[:, 2],
                                    mode='markers',
                                    marker=dict(
                                        size=2,
                                        color=f'rgb({int(color[0]*255)},{int(color[1]*255)},{int(color[2]*255)})',
                                        opacity=0.7
                                    ),
                                    name=f'Mask {mask_idx + 1}, Cluster {cluster_id}'
                                ))
                                color_idx += 1
                        
                        # Add noise points for this mask
                        noise_points = mask_points[cluster_labels == -1]
                        if len(noise_points) > 0:
                            max_points = 1000
                            if len(noise_points) > max_points:
                                indices = np.random.choice(len(noise_points), max_points, replace=False)
                                noise_points = noise_points[indices]
                            
                            fig.add_trace(go.Scatter3d(
                                x=noise_points[:, 0],
                                y=noise_points[:, 1],
                                z=noise_points[:, 2],
                                mode='markers',
                                marker=dict(
                                    size=1,
                                    color='gray',
                                    opacity=0.3
                                ),
                                name=f'Mask {mask_idx + 1}, Noise',
                                showlegend=False
                            ))
                    
                    # Highlight best clusters
                    best_cluster_points = result['best_cluster_points']
                    for mask_idx, best_points in best_cluster_points.items():
                        if len(best_points) > 0:
                            max_points = 2000
                            if len(best_points) > max_points:
                                indices = np.random.choice(len(best_points), max_points, replace=False)
                                best_points = best_points[indices]
                            
                            fig.add_trace(go.Scatter3d(
                                x=best_points[:, 0],
                                y=best_points[:, 1],
                                z=best_points[:, 2],
                                mode='markers',
                                marker=dict(
                                    size=3,
                                    color='red',
                                    opacity=0.9,
                                    line=dict(width=1, color='darkred')
                                ),
                                name=f'Best Cluster (Mask {mask_idx + 1})'
                            ))
                    
                    fig.update_layout(
                        title="3D Clusters (Colored by Cluster ID, Red = Best Clusters)",
                        scene=dict(
                            xaxis_title="X (m)",
                            yaxis_title="Y (m)",
                            zaxis_title="Z (m)",
                            aspectmode='data'
                        ),
                        height=600
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        if step_4_state.get('error'):
            st.error(f"❌ Error: {step_4_state['error']}")
    
    st.markdown("---")
    
    # Step 5: Detection & Pose Estimation
    with st.container():
        step_5_state = st.session_state.pipeline_state['step_5']
        step_4_completed = (
            st.session_state.pipeline_state['step_4']['completed']
            and not st.session_state.pipeline_state['step_4'].get('error')
        )
        
        col1, col2 = st.columns([3, 1])
        
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
                        
                        # Save comprehensive results in export-ready format
                        dataset_type = sample_meta_data.get('dataset_type', 'unknown').lower()
                        export_results = {
                            'detected_cuboids': result['detected_cuboids'],
                            'metadata': {
                                'dataset_type': dataset_type,
                                'sample_index': sample_meta_data.get('sample_index', 'unknown'),
                                'image_path': sample_meta_data.get('image_path', 'unknown'),
                                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                                'n_detections': result['n_detected'],
                                'pipeline_params': st.session_state.params.copy() if 'params' in st.session_state else {}
                            }
                        }
                        
                        # Add ground truth cuboids if available (for KITTI dataset)
                        if dataset_type == 'kitti':
                            ground_truth_boxes = sample_meta_data.get('ground_truth_boxes', [])
                            if ground_truth_boxes:
                                # Convert ground truth boxes to cuboid format for export/visualization
                                ground_truth_cuboids = []
                                for gt_box in ground_truth_boxes:
                                    # Ground truth boxes from KITTI already have corners and other fields
                                    # Ensure they're in the right format for export
                                    gt_cuboid = {
                                        'category': gt_box.get('category', 'Unknown'),
                                        'corners': gt_box.get('corners'),
                                        'bbox_2d': gt_box.get('bbox_2d'),
                                        'min_x': gt_box.get('min_x'),
                                        'max_x': gt_box.get('max_x'),
                                        'min_y': gt_box.get('min_y'),
                                        'max_y': gt_box.get('max_y'),
                                        'min_z': gt_box.get('min_z'),
                                        'max_z': gt_box.get('max_z'),
                                        'format': 'kitti_gt'
                                    }
                                    ground_truth_cuboids.append(gt_cuboid)
                                
                                export_results['ground_truth_cuboids'] = ground_truth_cuboids
                                export_results['metadata']['n_ground_truth'] = len(ground_truth_cuboids)
                        
                        # Store export-ready results
                        st.session_state.export_results = export_results
                        
                        st.rerun()
                    except Exception as e:
                        prev = st.session_state.pipeline_state['step_5']
                        st.session_state.pipeline_state['step_5'] = {
                            'completed': prev.get('completed', False),
                            'result': prev.get('result'),
                            'time': prev.get('time'),
                            'error': str(e)
                        }
                        st.error(f"Step 5 failed: {str(e)}")
        
        if step_5_state['completed']:
            result = step_5_state['result']
            st.success(f"✅ Completed: {result['n_detected']} cuboids detected")
            
            with st.expander("View Step 5 Details", expanded=True):
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

