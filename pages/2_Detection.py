"""
3D Object Detection Pipeline
Unified detection pipeline with step-by-step execution and full pipeline mode.
"""
import json
import os
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

import cv2
from matplotlib import patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch

from components.core.pointcloud_projection import PointCloud, Projection
from components.core.sam_integration import (
    SAMIntegration,
    assign_points_to_masks,
    get_available_models,
    get_bbox_from_mask,
    calculate_iou,
)
from components.core.pose_estimation import fit_cuboid_to_points_outdoor, fit_cuboid_to_points_indoor
from components.core.clustering_manager import (
    ClusteringManager,
    select_best_cluster_id,
    filter_clusters_by_max_volume,
    scene_is_indoor_from_point_cloud,
)
from components.core.tracking import ObjectTracker
from components.dataset_loaders.sunrgbd_dataset_loader import (
    SUNRGBDDatasetLoader,
    sunrgbd_keep_fraction_for_load,
)
from components.dataset_loaders.utils import load_dataset_sample
from components.utils.visualization_helper import (
    draw_2d_boxes_on_image,
    add_cuboids_to_figure,
    create_3d_scatter_plot,
    generate_distinct_colors,
    overlay_masks_on_image,
    render_point_cloud_plot as shared_render_point_cloud_plot,
)
from components.core.llm_service import (
    set_llm_temperature,
    get_llm_temperature,
    query_llm_for_dimensions,
    get_available_llm_models,
    get_current_llm_model_name,
    set_llm_model_name,
)
from components.core.constants import KITTI_CUBOID_TEMPLATES

# ============================================================================
# Helper Functions
# ============================================================================


def save_sample_to_hard_drive_after_processing(
    image: Optional[np.ndarray],
    point_cloud: Optional[np.ndarray],
    sample_meta_data: Dict,
    base_name: Optional[str] = None,
) -> None:
    """
    Save processed sample (image + LiDAR) to disk under the output_root_dir specified
    on the Dataset Extraction page. Creates 'images' and 'lidar' subfolders and writes:
    - image as PNG
    - point cloud as PCD (XYZ)
    """
    # If user disabled saving or a batch of raw samples (e.g., extracted from a ROS bag)
    # has already been saved, skip saving again here to avoid duplicate disk writes.
    if not st.session_state.get("save_processed_samples", True):
        return

    output_root = st.session_state.get("output_root_dir", "")
    if not output_root:
        return
    print(f'get sample {base_name}')
    root_path = Path(output_root).expanduser()
    images_dir = root_path / "images"
    lidar_dir = root_path / "lidar"
    images_dir.mkdir(parents=True, exist_ok=True)
    lidar_dir.mkdir(parents=True, exist_ok=True)

    dataset_type = (sample_meta_data or {}).get("dataset_type", "unknown")
    sample_index = (sample_meta_data or {}).get("sample_index", "unknown")

    if base_name is None:
        base_name = f"{dataset_type}_{sample_index}"

    # Save image
    if image is not None:
        img_path = images_dir / f"{base_name}.png"
        try:
            # Ensure RGB → BGR for OpenCV writing
            img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(img_path), img_bgr)
        except Exception:
            print(f'failed to save image with error: {Exception}')

    # Save point cloud as PCD (XYZ)
    if point_cloud is not None and len(point_cloud) > 0:
        pcd_path = lidar_dir / f"{base_name}.pcd"
        try:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
            o3d.io.write_point_cloud(str(pcd_path), pcd, write_ascii=True)
        except Exception:
            print(f'failed to save point cloud with error: {Exception}')


def _render_point_cloud_plot(
    fig: go.Figure,
    export_basename: str,
    use_container_width: bool = False,
    show_legend: bool = True,
) -> None:
    """Thin wrapper over shared point-cloud renderer."""
    shared_render_point_cloud_plot(
        fig=fig,
        export_basename=export_basename,
        use_container_width=use_container_width,
        show_legend=show_legend,
    )


# Step 3 mask overlay (matplotlib): tune class/confidence labels on each box.
STEP3_MASK_VIZ_LABEL_FONT_SIZE = 10 * 2.5
# Step 3 figure title / caption (single line above the image).
STEP3_CAPTION_FONT_SIZE = 10 * 2.5
STEP3_CAPTION_TEXT = "3D Mask on SIM dataset"
# Step 3 matplotlib overlay: toggle figure title and per-mask class/confidence text.
SHOW_CAPTION = False
SHOW_MASK_LABELS = False


def _step3_build_mask_overlay_figure(
    image: np.ndarray,
    sam_masks: List[np.ndarray],
    colors: List[Tuple[float, float, float]],
    mask_bboxes: List,
    detected_class_names: List[str],
    confidences: List,
    mask_alpha: float = 0.5,
) -> plt.Figure:
    """
    Matplotlib figure for Step 3 (mask tint + 2D boxes + optional class/confidence labels).
    Title: ``SHOW_CAPTION``, ``STEP3_CAPTION_TEXT``, ``STEP3_CAPTION_FONT_SIZE``.
    Labels: ``SHOW_MASK_LABELS``, ``STEP3_MASK_VIZ_LABEL_FONT_SIZE``.
    """
    img_with_masks = overlay_masks_on_image(image, sam_masks, colors, alpha=mask_alpha)
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(img_with_masks)
    ax.axis("off")
    fs = STEP3_MASK_VIZ_LABEL_FONT_SIZE
    text_y_pad = max(fs * 0.55, 18.0)
    for i, (bbox, class_name, confidence) in enumerate(
        zip(mask_bboxes, detected_class_names, confidences)
    ):
        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            rect = mpatches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor=colors[i],
                facecolor="none",
            )
            ax.add_patch(rect)
            if SHOW_MASK_LABELS:
                label = f"{class_name}: {confidence:.2f}" if confidence is not None else class_name
                ax.text(
                    x1,
                    y1 - text_y_pad,
                    label,
                    color=colors[i],
                    fontsize=fs,
                    bbox=dict(boxstyle="round,pad=0.45", facecolor="black", alpha=0.7),
                )
    if SHOW_CAPTION:
        ax.set_title(STEP3_CAPTION_TEXT, fontsize=STEP3_CAPTION_FONT_SIZE)
    return fig


def _update_mask_capacity_hint(step_3_result: Optional[Dict], sample_meta_data: Dict) -> None:
    """
    Persist an evaluation hint when detection produces fewer masks than GT boxes.
    """
    if not step_3_result:
        return
    gt_boxes = sample_meta_data.get("ground_truth_boxes", []) or []
    n_gt = len(gt_boxes)
    sam_masks = step_3_result.get("sam_masks", []) or []
    n_masks = len(sam_masks)
    if n_gt > 0 and n_masks < n_gt:
        prev_max = int(st.session_state.get("eval_mask_capacity_max", 0) or 0)
        st.session_state["eval_mask_capacity_max"] = max(prev_max, n_masks)
        st.session_state["eval_mask_capacity_hint_active"] = True


def default_detection_params() -> Dict:
    """Default pipeline/session parameters (single source of truth for merges)."""
    return {
        "pipeline": {
            "distance_threshold": 0.3,
            "ransac_n": 3,
            "num_iterations": 1000,
            "filter_forward_only": False,
        },
        "pipeline_indoor": {
            "distance_threshold": 0.12,
            "ransac_n": 3,
            "num_iterations": 1500,
            "filter_forward_only": False,
        },
        "clustering": {
            "clustering_algorithm": "adaptive_dbscan",
            "dbscan_eps": 0.5,
            "dbscan_min_samples": 5,
            "hdbscan_min_cluster_size": 5,
            "hdbscan_min_samples": 5,
            "adaptive_dbscan_base_eps": 0.35,
            "adaptive_dbscan_eps_growth_rate": 1.0,
            "adaptive_dbscan_reference_distance": 15.0,
            "adaptive_dbscan_min_scale": 0.7,
            "adaptive_dbscan_max_scale": 4.0,
            "volume_factor": 1.2,
        },
        "cuboid_fitting": {
            "w_distance": 1.0,
            "w_geometric": 0.5,
            "w_outlier": 2.0,
            "step_center_search": 0.2,
            "max_step_center": 10,
            "d_theta": 0.05,
        },
        "cuboid_fitting_indoor": {
            "margin": 0.02,
            "min_extent": 0.05,
            "d_theta": 0.02,
            "inlier_quantile": 0.99,
            "coverage_weight": 4.0,
        },
        "scene_from_pointcloud": {
            "max_horizontal_span_m": 52.0,
            "min_points_per_m3": 1.0,
            "min_points": 400,
            "max_aabb_volume_m3": 22000.0,
            "max_vertical_span_m": 14.0,
            "near_xy_radius_m": 14.0,
            "min_near_xy_fraction": 0.16,
            "density_relax_factor": 0.42,
        },
        "bag_freq_hz": 45.0,
        "class_max_speed_mps": {
            "car": 40.0,
            "truck": 35.0,
            "bus": 35.0,
            "bicycle": 15.0,
            "bike": 15.0,
            "motorcycle": 25.0,
            "person": 5.0,
            "pedestrian": 5.0,
            "default": 50.0,
        },
        "sam_model_type": "sam2_t",
        "image_track_mode": "appearance",
        "class_names": ["car", "person", "bicycle"],
        "open_vocab_detector": "yolo",
        "grounding_dino_model_id": "IDEA-Research/grounding-dino-base",
        "yolo_model_path": None,
        "yolo_conf_threshold": 0.25,
        "use_gpu": True,
        "sunrgbd_use_label_bboxes_step3": False,
        "use_gt_2d_bboxes_step3": False,
        # When Step 3 uses dataset GT 2D boxes: comma/newline-separated class names from the
        # label file. Empty means include every GT instance with a valid 2D box.
        "gt_step3_target_classes_text": "",
    }


def _copy_param_value(val):
    if isinstance(val, dict):
        return dict(val)
    if isinstance(val, list):
        return list(val)
    return val


def ensure_detection_params(params: Dict) -> None:
    """Fill missing keys from defaults (e.g. batch path created params = {})."""
    defaults = default_detection_params()
    for key, default_val in defaults.items():
        if key not in params:
            params[key] = _copy_param_value(default_val)
        elif isinstance(default_val, dict):
            if not isinstance(params[key], dict):
                params[key] = dict(default_val)
            else:
                for sk, sv in default_val.items():
                    if sk not in params[key]:
                        params[key][sk] = _copy_param_value(sv)


_SCENE_FROM_POINTCLOUD_KEYS = (
    "max_horizontal_span_m",
    "min_points_per_m3",
    "min_points",
    "max_aabb_volume_m3",
    "max_vertical_span_m",
    "near_xy_radius_m",
    "min_near_xy_fraction",
    "density_relax_factor",
)


def _pipeline_scene_is_indoor(params: Dict, point_cloud: np.ndarray) -> bool:
    """Indoor vs outdoor from LiDAR: AABB, density, ceiling height, near-field fraction."""
    raw = params.get("scene_from_pointcloud") or {}
    kw = {k: raw[k] for k in _SCENE_FROM_POINTCLOUD_KEYS if k in raw}
    return scene_is_indoor_from_point_cloud(point_cloud, **kw)


def _ground_removal_kwargs(params: Dict, point_cloud: np.ndarray) -> Dict:
    if _pipeline_scene_is_indoor(params, point_cloud):
        return dict(params["pipeline_indoor"])
    return dict(params["pipeline"])


def _default_use_ground_plane_removal(sample_meta_data: Optional[Dict[str, Any]]) -> bool:
    dataset_type = ((sample_meta_data or {}).get("dataset_type") or "").lower()
    return dataset_type != "sunrgbd"


# ============================================================================
# Pipeline Step Functions
# ============================================================================

def step_1_ground_plane_removal(
    point_cloud: np.ndarray,
    distance_threshold: float = 0.3,
    ransac_n: int = 3,
    num_iterations: int = 1000,
    filter_forward_only: bool = False,
    camera_to_lidar_transform: Optional[np.ndarray] = None,
    use_ground_plane_removal: bool = True,
) -> Dict:
    """
    Step 1: Remove ground plane from point cloud using RANSAC.
    
    Args:
        point_cloud: Nx3 array of 3D points
        distance_threshold: RANSAC distance threshold
        ransac_n: RANSAC number of points
        num_iterations: RANSAC number of iterations
        filter_forward_only: Whether to keep only forward-facing points. If
            camera_to_lidar_transform is provided, this uses the camera z-axis in
            LiDAR coordinates; otherwise it falls back to x > 0.
        camera_to_lidar_transform: Optional 4x4 camera-to-LiDAR transform used
            when filter_forward_only is True to define the forward direction.
        use_ground_plane_removal: If False, Step 1 is marked completed while
            keeping the original point cloud unchanged.
    
    Returns:
        Dict with 'point_cloud_obj', 'ground_plane_model', 'ground_z'
    """
    start_time = time.time()
    
    # Create PointCloud object
    point_cloud_obj = PointCloud(point_cloud)
    print(f"Point cloud object created with {len(point_cloud_obj.original_point_cloud)} points")
    if use_ground_plane_removal:
        # Remove ground plane
        point_cloud_obj.remove_ground_plane_ransac(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations,
            filter_forward_only=filter_forward_only,
            camera_to_lidar_transform=camera_to_lidar_transform,
        )
        print(f"Ground plane removed with {len(point_cloud_obj.point_cloud_plane_removed)} points remaining")
        # Get ground_z at origin
        ground_z = point_cloud_obj.get_ground_z(x=0.0, y=0.0)
        ground_plane_model = point_cloud_obj.ground_plane_model
    else:
        point_cloud_obj.ground_removed = True
        point_cloud_obj.point_cloud_plane_removed = point_cloud_obj.original_point_cloud.copy()
        print(f"Ground plane removal skipped with {len(point_cloud_obj.point_cloud_plane_removed)} points")
        ground_z = None
        ground_plane_model = None
    
    elapsed_time = time.time() - start_time
    
    return {
        'point_cloud_obj': point_cloud_obj,
        'ground_plane_model': ground_plane_model,
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
    camera_extrinsic = sample_meta_data.get('camera_extrinsic', np.eye(4))
    camera_to_lidar_transform = sample_meta_data['camera_to_lidar_transform']
    
    projection = Projection(
        camera_intrinsic=camera_intrinsic,
        camera_extrinsic=camera_extrinsic,
        camera_to_lidar_transform=camera_to_lidar_transform,
        point_cloud=point_cloud
    )
    
    # Create sparse depth map using Projection's camera-frame convention (z_cam > 0)
    sparse_depth_map = projection.compute_sparse_depth_map((h, w))
    
    # Backproject sparse depth map to 3D with colors
    
    colored_sparse_points, colored_sparse_colors = projection.backproject_sparse_depth_map_with_colors(
        sparse_depth_map=sparse_depth_map,
        image=image
    )
    
    elapsed_time = time.time() - start_time
    
    return {
        'projection': projection,
        'sparse_depth_map': sparse_depth_map,
        'colored_sparse_points': colored_sparse_points,
        'colored_sparse_colors': colored_sparse_colors,
        'n_points': len(colored_sparse_points),
        'time': elapsed_time
    }


def _ensure_projection(
    projection: Optional[Projection],
    sample_meta_data: Dict,
    sparse_points: np.ndarray,
) -> Projection:
    if projection is not None:
        return projection
    return Projection(
        camera_intrinsic=sample_meta_data['camera_intrinsic'],
        camera_extrinsic=sample_meta_data.get('camera_extrinsic', np.eye(4)),
        camera_to_lidar_transform=sample_meta_data['camera_to_lidar_transform'],
        point_cloud=sparse_points,
    )


def _get_current_sam_integration(
    sam_model_type: str,
    use_gpu: bool,
) -> Optional[SAMIntegration]:
    sam_integration = st.session_state.get("sam_integration")
    initialized_model_type = st.session_state.get("sam_initialized_model_type")
    initialized_use_gpu = st.session_state.get("sam_initialized_use_gpu")
    if (
        sam_integration is None
        or initialized_model_type != sam_model_type
        or initialized_use_gpu != use_gpu
    ):
        sam_integration = SAMIntegration(model_type=sam_model_type, use_gpu=use_gpu)
        st.session_state.sam_integration = sam_integration
        st.session_state.sam_initialized_model_type = sam_model_type
        st.session_state.sam_initialized_use_gpu = use_gpu
    return sam_integration


def _resolve_ground_truth_boxes_for_sample(
    sample_meta_data: Dict,
    image_shape: Tuple[int, ...],
) -> List[Dict[str, Any]]:
    """
    Same GT 2D box list as Step 3 uses: ``sample_meta_data['ground_truth_boxes']`` when present,
    else SUNRGBD lazy-load from ``annotation_path`` / scene label files.
    """
    dataset_type = str(sample_meta_data.get("dataset_type", "")).lower()
    gt_boxes = list(sample_meta_data.get("ground_truth_boxes", []) or [])
    if not gt_boxes and dataset_type == "sunrgbd":
        annotation_path = sample_meta_data.get("annotation_path")
        if not annotation_path:
            scene_root = sample_meta_data.get("scene_root")
            scene_id = sample_meta_data.get("scene_id")
            if scene_root and scene_id:
                base_root = Path(scene_root)
                label_v2 = base_root / "label" / f"{scene_id}.txt"
                label_v1 = base_root / "label_v1" / f"{scene_id}.txt"
                if label_v1.exists():
                    annotation_path = str(label_v1)
                elif label_v2.exists():
                    annotation_path = str(label_v2)
        if annotation_path:
            gt_boxes = SUNRGBDDatasetLoader._load_ground_truth_boxes(annotation_path, image_shape)
    return gt_boxes


def _parse_gt_step3_target_class_text(text: Optional[str]) -> List[str]:
    """Split comma/newline-separated class names for GT Step 3 filtering. Empty input → []."""
    if text is None:
        return []
    raw = str(text).strip()
    if not raw:
        return []
    names: List[str] = []
    for line in raw.replace("\r\n", "\n").split("\n"):
        for chunk in line.split(","):
            t = chunk.strip()
            if t:
                names.append(t)
    return names


def _class_names_for_step3_segmentation(params: Dict) -> List[str]:
    """
    Class list passed to ``step_3_sam_segmentation``.

    Open-vocabulary path uses sidebar ``class_names``. GT-2D-box path uses
    ``gt_step3_target_classes_text`` when set; an empty parsed list means no filter
    (all annotation boxes with valid 2D).
    """
    use_gt = bool(
        params.get("use_gt_2d_bboxes_step3", params.get("sunrgbd_use_label_bboxes_step3", False))
    )
    if use_gt:
        return _parse_gt_step3_target_class_text(params.get("gt_step3_target_classes_text"))
    return list(params.get("class_names") or [])


def _unique_gt_annotation_labels(gt_boxes: List[Dict[str, Any]]) -> List[str]:
    labels: set[str] = set()
    for gt in gt_boxes:
        bbox_2d = gt.get("bbox_2d")
        if not isinstance(bbox_2d, dict):
            continue
        left = int(bbox_2d.get("left", 0))
        top = int(bbox_2d.get("top", 0))
        right = int(bbox_2d.get("right", 0))
        bottom = int(bbox_2d.get("bottom", 0))
        if right <= left or bottom <= top:
            continue
        labels.add(str(gt.get("class", gt.get("category", "Unknown"))))
    return sorted(labels)


def _build_step3_from_bboxes(
    bbox_data: Dict,
    image: np.ndarray,
    sparse_points: np.ndarray,
    sample_meta_data: Dict,
    sam_integration: Optional[SAMIntegration] = None,
    projection: Optional[Projection] = None,
) -> Dict:
    """
    Build a Step 3-compatible result dict from an uploaded bbox annotation file.

    Accepts both ``bbox_only`` and ``bbox_tracking`` formats exported by **4_Export**.
    For the tracking variant the current frame's annotations are selected by *frame_index*
    stored in ``st.session_state``.

    Bounding boxes are used as SAM prompts when ``sam_integration`` is provided.
    If SAM is not available, rectangular masks are synthesised as fallback.
    """
    start_time = time.time()
    h, w = image.shape[:2]
    fmt = bbox_data.get("format", "bbox_only")

    annotations: List[Dict] = []
    if fmt == "bbox_tracking":
        frame_idx = st.session_state.get("_bbox_load_frame_index", 0)
        frames = bbox_data.get("frames", [])
        for f in frames:
            if int(f.get("frame_index", -1)) == frame_idx:
                annotations = f.get("annotations", [])
                break
        if not annotations and frames:
            annotations = frames[0].get("annotations", [])
    else:
        annotations = bbox_data.get("annotations", [])

    sam_masks: List[np.ndarray] = []
    mask_bboxes: List[List[int]] = []
    class_names_out: List[str] = []
    confidences_out: List[float] = []
    instance_ids: List[Optional[int]] = []

    for ann in annotations:
        bbox = ann.get("bbox")
        if bbox is None or len(bbox) != 4:
            continue
        if fmt == "bbox_tracking" and not ann.get("appears", True):
            continue
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        if x2 <= x1 or y2 <= y1:
            continue

        if sam_integration is not None:
            sam_mask = sam_integration.get_mask_from_bbox(
                image=image,
                bbox=[x1, y1, x2, y2],
            )
            sam_masks.append((sam_mask > 0).astype(np.uint8))
        else:
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[y1:y2, x1:x2] = 1
            sam_masks.append(mask)
        mask_bboxes.append([x1, y1, x2, y2])
        class_names_out.append(str(ann.get("class_name", "Unknown")))
        conf = ann.get("confidence")
        confidences_out.append(float(conf) if conf is not None else 1.0)
        instance_ids.append(ann.get("instance_id"))

    mask_assignments = None
    if sam_masks:
        proj = _ensure_projection(
            projection=projection,
            sample_meta_data=sample_meta_data,
            sparse_points=sparse_points,
        )
        mask_assignments = assign_points_to_masks(sparse_points, sam_masks, proj, (h, w))

    has_tracking = any(iid is not None for iid in instance_ids)

    result: Dict = {
        "sam_masks": sam_masks,
        "mask_assignments": mask_assignments,
        "mask_bboxes": mask_bboxes,
        "class_names": class_names_out,
        "confidences": confidences_out,
        "segmentation_debug": {
            "source": "loaded_bbox_annotation",
            "format": fmt,
            "sam_refined": sam_integration is not None,
        },
        "n_masks": len(sam_masks),
        "time": time.time() - start_time,
    }
    if has_tracking:
        result["loaded_instance_ids"] = instance_ids
    return result


def _build_step3_from_gt_2d_bboxes(
    sample_meta_data: Dict,
    image: np.ndarray,
    sparse_points: np.ndarray,
    class_names: Optional[List[str]] = None,
    sam_integration: Optional[SAMIntegration] = None,
    projection: Optional[Projection] = None,
) -> Dict:
    """Build Step 3 masks from dataset GT 2D boxes (e.g. KITTI/SUNRGBD), refined by SAM."""
    normalized_targets = {
        str(name).strip().casefold()
        for name in (class_names or [])
        if str(name).strip()
    }

    dataset_type = str(sample_meta_data.get("dataset_type", "")).lower()
    gt_boxes = _resolve_ground_truth_boxes_for_sample(sample_meta_data, image.shape)

    annotations: List[Dict[str, Any]] = []
    sam_masks: List[np.ndarray] = []
    for gt in gt_boxes:
        bbox_2d = gt.get("bbox_2d")
        if not isinstance(bbox_2d, dict):
            continue
        left = int(bbox_2d.get("left", 0))
        top = int(bbox_2d.get("top", 0))
        right = int(bbox_2d.get("right", 0))
        bottom = int(bbox_2d.get("bottom", 0))
        if right <= left or bottom <= top:
            continue
        class_name = str(gt.get("class", gt.get("category", "Unknown")))
        if normalized_targets and class_name.strip().casefold() not in normalized_targets:
            continue
        if sam_integration is None:
            continue
        sam_mask = sam_integration.get_mask_from_bbox(
            image=image,
            bbox=[left, top, right, bottom],
        )
        annotations.append(
            {
                "bbox": [left, top, right, bottom],
                "class_name": class_name,
                "confidence": 1.0,
            }
        )
        sam_masks.append((sam_mask > 0).astype(np.uint8))

    h, w = image.shape[:2]
    mask_bboxes = [ann["bbox"] for ann in annotations]
    class_names_out = [ann["class_name"] for ann in annotations]
    confidences_out = [float(ann.get("confidence", 1.0)) for ann in annotations]
    mask_assignments = None
    if sam_masks:
        proj = _ensure_projection(
            projection=projection,
            sample_meta_data=sample_meta_data,
            sparse_points=sparse_points,
        )
        mask_assignments = assign_points_to_masks(sparse_points, sam_masks, proj, (h, w))

    result = {
        "sam_masks": sam_masks,
        "mask_assignments": mask_assignments,
        "mask_bboxes": mask_bboxes,
        "class_names": class_names_out,
        "confidences": confidences_out,
        "n_masks": len(sam_masks),
    }
    seg_debug = dict(result.get("segmentation_debug", {}))
    seg_debug["source"] = "dataset_gt_bbox_2d"
    seg_debug["format"] = "dataset_gt_bbox_2d"
    seg_debug["dataset_type"] = dataset_type
    seg_debug["used_targets"] = sorted(normalized_targets)
    seg_debug["sam_refined"] = sam_integration is not None
    result["segmentation_debug"] = seg_debug
    return result


def step_3_sam_segmentation(
    sample_meta_data: Dict,
    image: np.ndarray,
    sparse_points: np.ndarray,
    class_names: List[str],
    sam_model_type: str = 'sam2_t',
    yolo_model_path: Optional[str] = None,
    conf_threshold: float = 0.25,
    open_vocab_detector: str = "yolo",
    grounding_dino_model_id: Optional[str] = None,
    use_gpu: bool = True,
    projection: Optional[Projection] = None,
    use_dataset_gt_2d_bboxes: bool = False,
) -> Dict:
    """
    Step 3: Generate SAM masks using open-vocabulary detection and assign original LiDAR points to masks.
    
    Args:
        sample_meta_data: Sample metadata
        image: HxWx3 RGB image
        sparse_points: Nx3 array of backprojected sparse depth points
        class_names: List of class names for open-vocabulary segmentation, or for the
            dataset-GT-2D path a filter list parsed from ``gt_step3_target_classes_text``;
            when that list is empty, every GT instance with a valid 2D box is used.
        sam_model_type: 'sam2_t' or 'sam3'
        yolo_model_path: Optional path to YOLO-World model (SAM2 + YOLO detector)
        conf_threshold: Confidence threshold for open-vocab detections (default: 0.25)
        open_vocab_detector: ``yolo`` or ``grounding_dino`` (SAM2 only)
        grounding_dino_model_id: Hugging Face model id when using Grounding DINO
        use_gpu: Whether to use GPU for inference (default: True)
    
    Returns:
        Dict with 'sam_masks', 'mask_assignments', 'mask_bboxes', 'class_names', 'confidences'
    """
    start_time = time.time()
    dataset_type = str(sample_meta_data.get("dataset_type", "")).lower()

    # Initialize SAM integration if needed
    if 'sam_integration' not in st.session_state or st.session_state.sam_integration is None:
        try:
            st.session_state.sam_integration = SAMIntegration(model_type=sam_model_type, use_gpu=use_gpu)
            st.session_state.sam_initialized_model_type = sam_model_type
            st.session_state.sam_initialized_use_gpu = use_gpu
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
    
    if st.session_state.sam_initialized_model_type != sam_model_type or st.session_state.sam_initialized_use_gpu != use_gpu:
        try:
            st.session_state.sam_integration = SAMIntegration(model_type=sam_model_type, use_gpu=use_gpu)
            st.session_state.sam_initialized_model_type = sam_model_type
            st.session_state.sam_initialized_use_gpu = use_gpu
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
    if use_dataset_gt_2d_bboxes and dataset_type in {"sunrgbd", "kitti"}:
        result = _build_step3_from_gt_2d_bboxes(
            sample_meta_data=sample_meta_data,
            image=image,
            sparse_points=sparse_points,
            class_names=class_names,
            sam_integration=sam_integration,
            projection=projection,
        )
        result["time"] = time.time() - start_time
        return result
    
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
        # Always run per-frame segmentation here; cross-frame association is done in ObjectTracker.
        segment_results = sam_integration.segment_by_class_names(
            image=image,
            class_names=class_names,
            yolo_model_path=yolo_model_path,
            conf_threshold=conf_threshold,
            open_vocab_detector=open_vocab_detector,
            grounding_dino_model_id=grounding_dino_model_id,
        )
        
        sam_masks = segment_results['masks']
        mask_bboxes = segment_results['bboxes']
        detected_class_names = segment_results['class_names']
        confidences = segment_results['confidences']
        segmentation_debug = segment_results.get('debug', {})
        
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
        projection = _ensure_projection(
            projection=projection,
            sample_meta_data=sample_meta_data,
            sparse_points=sparse_points,
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
        'segmentation_debug': segmentation_debug,
        'n_masks': len(sam_masks),
        'time': elapsed_time
    }


def step_4_clustering(
    sample_meta_data: Dict,
    sparse_points: np.ndarray,
    sam_masks: List[np.ndarray],
    mask_assignments: np.ndarray,
    clustering_algorithm: str = "adaptive_dbscan",
    dbscan_eps: float = 0.5,
    dbscan_min_samples: int = 5,
    hdbscan_min_cluster_size: int = 5,
    hdbscan_min_samples: int = 5,
    adaptive_dbscan_base_eps: float = 0.35,
    adaptive_dbscan_eps_growth_rate: float = 1.0,
    adaptive_dbscan_reference_distance: float = 15.0,
    adaptive_dbscan_min_scale: float = 0.7,
    adaptive_dbscan_max_scale: float = 4.0,
    sparse_depth_map: Optional[np.ndarray] = None,
    volume_factor: float = 1.2,
    projection: Optional[Projection] = None,
) -> Dict:
    """
    Step 4: Run clustering on points assigned to each mask.
    
    Args:
        sample_meta_data: Sample metadata
        sparse_points: Nx3 array of backprojected sparse depth points
        sam_masks: List of binary masks
        mask_assignments: N array assigning each point to a mask index
        clustering_algorithm: One of {'hdbscan', 'dbscan', 'adaptive_dbscan'}
        dbscan_eps: DBSCAN eps parameter
        dbscan_min_samples: DBSCAN/adaptive-DBSCAN min_samples parameter
        sparse_depth_map: Optional HxW sparse depth map from step 2. When provided,
                          best cluster per mask is selected by highest 2D IoU with the mask.

    Outdoor scans (from point-cloud span + density): clusters larger than LLM/template volume are
    filtered. Indoor scans skip that filter.

    Returns:
        Dict with ``mask_cluster_labels``, ``clustering_results``,
        ``filtered_mask_sparse_indices``, and ``best_cluster_sparse_indices``
        (global row indices into ``sparse_points``; avoids duplicate point arrays).
    """
    start_time = time.time()

    raw_pc = st.session_state.sample.get("point_cloud")
    if raw_pc is not None and len(raw_pc) > 0:
        filter_by_template_volume = not _pipeline_scene_is_indoor(
            st.session_state.params, raw_pc
        )
    else:
        filter_by_template_volume = True

    # Get per-class dimensions estimated via LLM or templates (stored in session_state)
    dimensions_by_class = st.session_state.params.get('dimensions_by_class', {})

    # Get detected class names from step 3 (mask-wise categories)
    detected_class_names = []
    if 'step_3' in st.session_state.pipeline_state:
        step_3_result = st.session_state.pipeline_state['step_3'].get('result', {})
        detected_class_names = step_3_result.get('class_names', [])

    # Get image shape
    if 'image' in st.session_state.sample:
        h, w = st.session_state.sample['image'].shape[:2]
    else:
        h, w = sample_meta_data.get('image_shape', (375, 1242))  # Default KITTI size
    
    if projection is None:
        projection = Projection(
            camera_intrinsic=sample_meta_data['camera_intrinsic'],
            camera_extrinsic=sample_meta_data.get('camera_extrinsic', np.eye(4)),
            camera_to_lidar_transform=sample_meta_data['camera_to_lidar_transform'],
            point_cloud=sparse_points
        )
    
    mask_cluster_labels: Dict[int, np.ndarray] = {}
    clustering_results: List[Dict] = []
    best_cluster_sparse_indices: Dict[int, np.ndarray] = {}
    filtered_mask_sparse_indices: Dict[int, np.ndarray] = {}

    for mask_idx, mask in enumerate(sam_masks):
        if mask is None:
            continue

        global_idx = np.flatnonzero(mask_assignments == mask_idx)
        if global_idx.size == 0:
            continue
        mask_points = sparse_points[global_idx]

        min_required_points = dbscan_min_samples
        if clustering_algorithm == "hdbscan":
            min_required_points = max(hdbscan_min_samples, hdbscan_min_cluster_size)
        if len(mask_points) < min_required_points:
            continue
        
        # Derive expected object volume for this mask from LLM/template dimensions
        category = 'Unknown'
        if mask_idx < len(detected_class_names):
            category = detected_class_names[mask_idx] or 'Unknown'

        dims = dimensions_by_class.get(category)
        if dims is None:
            t = KITTI_CUBOID_TEMPLATES.get(category, KITTI_CUBOID_TEMPLATES['Unknown'])
            dims = (float(t['length']), float(t['width']), float(t['height']))

        length, width, height = float(dims[0]), float(dims[1]), float(dims[2])
        reference_volume = length * width * height

        # Run configured clustering algorithm
        clustering_manager = ClusteringManager(mask_points)
        if clustering_algorithm == "hdbscan":
            cluster_labels = clustering_manager.run_hdbscan(
                min_cluster_size=hdbscan_min_cluster_size,
                min_samples=hdbscan_min_samples,
            )
        elif clustering_algorithm == "adaptive_dbscan":
            cluster_labels = clustering_manager.run_adaptive_dbscan(
                base_eps=adaptive_dbscan_base_eps,
                min_samples=dbscan_min_samples,
                eps_growth_rate=adaptive_dbscan_eps_growth_rate,
                reference_distance=adaptive_dbscan_reference_distance,
                min_scale=adaptive_dbscan_min_scale,
                max_scale=adaptive_dbscan_max_scale,
            )
        else:
            cluster_labels = clustering_manager.run_dbscan(
                eps=dbscan_eps, min_samples=dbscan_min_samples
            )

        if filter_by_template_volume:
            keep_mask = filter_clusters_by_max_volume(
                points=mask_points,
                labels=cluster_labels,
                template_volume=reference_volume,
                volume_factor=volume_factor,
            )
            mask_points_filtered = mask_points[keep_mask]
            cluster_labels = cluster_labels[keep_mask]
            idx_f = global_idx[keep_mask]
        else:
            mask_points_filtered = mask_points
            cluster_labels = cluster_labels.copy()
            idx_f = global_idx

        filtered_mask_sparse_indices[mask_idx] = idx_f.astype(np.int64, copy=False)

        if len(mask_points_filtered) < min_required_points:
            # Not enough points left for clustering/selection; use all remaining points
            best_cluster_sparse_indices[mask_idx] = idx_f.astype(np.int64, copy=False)
            unique_clusters = np.unique(cluster_labels)
            n_clusters = np.sum(unique_clusters >= 0)
            n_cluster_points = np.sum(cluster_labels >= 0)
            clustering_results.append({
                'Mask ID': mask_idx + 1,
                'Total Points': len(mask_points),
                'Clusters Found': int(n_clusters),
                'Clustered Points': int(n_cluster_points),
                'Best Cluster Points': len(mask_points),
            })
            continue

        mask_cluster_labels[mask_idx] = cluster_labels
        print(f'mask_id: {mask_idx}')
        best_id = select_best_cluster_id(
            mask_points=mask_points_filtered,
            mask=mask,
            projection=projection,
            image_shape=(h, w),
            cluster_labels=cluster_labels,
        )
        if best_id is not None:
            sel = cluster_labels == best_id
            best_cluster_sparse_indices[mask_idx] = idx_f[sel].astype(np.int64, copy=False)

        unique_clusters = np.unique(cluster_labels)
        n_clusters = np.sum(unique_clusters >= 0)
        n_cluster_points = np.sum(cluster_labels >= 0)
        n_best = int(np.sum(cluster_labels == best_id)) if best_id is not None else 0

        clustering_results.append({
            'Mask ID': mask_idx + 1,
            'Total Points': len(mask_points),
            'Clusters Found': int(n_clusters),
            'Clustered Points': int(n_cluster_points),
            'Best Cluster Points': n_best,
        })
    
    elapsed_time = time.time() - start_time
    
    return {
        'mask_cluster_labels': mask_cluster_labels,
        'clustering_results': clustering_results,
        'best_cluster_sparse_indices': best_cluster_sparse_indices,
        'filtered_mask_sparse_indices': filtered_mask_sparse_indices,
        'time': elapsed_time
    }


def step_5_detection_pose_estimation(
    sample_meta_data: Dict,
    sparse_points: np.ndarray,
    best_cluster_sparse_indices: Dict[int, np.ndarray],
    filtered_mask_sparse_indices: Optional[Dict[int, np.ndarray]],
    sam_masks: List[np.ndarray],
    mask_bboxes: Optional[List[List[int]]],
    mask_confidences: Optional[List[float]],
    projection: Optional[Projection],
    ground_z: float,
    cuboid_params: Dict
) -> Dict:
    """
    Step 5: Fit cuboids to best cluster points using scoring-based method.

    Args:
        sample_meta_data: Sample metadata
        sparse_points: Nx3 array (same reference frame as Step 2 ``colored_sparse_points``)
        best_cluster_sparse_indices: Per-mask global row indices into ``sparse_points``
        filtered_mask_sparse_indices: Per-mask row indices after filtering in Step 4
        sam_masks: List of binary masks
        mask_bboxes: Step 3 2D mask bboxes ``[x1, y1, x2, y2]``
        mask_confidences: Step 3 detector confidences aligned with mask index
        projection: Projection helper for 3D cuboid reprojection
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
    _pc = st.session_state.sample.get("point_cloud")
    indoor_scene = (
        _pipeline_scene_is_indoor(st.session_state.params, _pc)
        if _pc is not None and len(_pc) > 0
        else False
    )
    indoor_params = st.session_state.params.get("cuboid_fitting_indoor", {})

    def _unit_interval(x: float) -> float:
        return float(np.clip(x, 0.0, 1.0))

    def _template_consistency(cat: str, lwh: Tuple[float, float, float]) -> float:
        if cat not in KITTI_CUBOID_TEMPLATES or cat == "Unknown":
            return 1.0
        t = KITTI_CUBOID_TEMPLATES[cat]
        tpl = np.array([float(t["length"]), float(t["width"]), float(t["height"])], dtype=np.float64)
        pred = np.array([float(lwh[0]), float(lwh[1]), float(lwh[2])], dtype=np.float64)
        err_direct = np.mean(np.abs((pred - tpl) / np.maximum(tpl, 1e-6)))
        pred_swapped = np.array([pred[1], pred[0], pred[2]], dtype=np.float64)
        err_swap = np.mean(np.abs((pred_swapped - tpl) / np.maximum(tpl, 1e-6)))
        rel_err = float(min(err_direct, err_swap))
        return float(np.exp(-rel_err))

    # Fit cuboid to each mask's best cluster (materialize from sparse index map)
    for mask_idx, row_ix in best_cluster_sparse_indices.items():
        if row_ix is None or len(row_ix) < 5:
            continue
        cluster_points = sparse_points[row_ix]
        
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
        
        if indoor_scene:
            d_theta_in = float(indoor_params.get("d_theta", cuboid_params["d_theta"]))
            fit_result = fit_cuboid_to_points_indoor(
                points=cluster_points,
                d_theta=d_theta_in,
                margin=float(indoor_params.get("margin", 0.02)),
                min_extent=float(indoor_params.get("min_extent", 0.05)),
                inlier_quantile=float(indoor_params.get("inlier_quantile", 0.99)),
                coverage_weight=float(indoor_params.get("coverage_weight", 4.0)),
            )
        else:
            dimensions_by_class = st.session_state.params.get('dimensions_by_class', {})
            dimensions = dimensions_by_class.get(category)
            if dimensions is None:
                t = KITTI_CUBOID_TEMPLATES.get(category, KITTI_CUBOID_TEMPLATES['Unknown'])
                dimensions = (float(t['length']), float(t['width']), float(t['height']))
            score_weights = (
                cuboid_params['w_distance'],
                cuboid_params['w_geometric'],
                cuboid_params['w_outlier']
            )
            fit_result = fit_cuboid_to_points_outdoor(
                points=cluster_points,
                dimensions=dimensions,
                step_center_search=cuboid_params['step_center_search'],
                max_step_center=cuboid_params['max_step_center'],
                d_theta=cuboid_params['d_theta'],
                normals=None,
                score_weights=score_weights,
                ground_z=ground_z
            )

        center = fit_result['center']
        yaw = fit_result['yaw']
        length = fit_result['length']
        width = fit_result['width']
        height = fit_result['height']

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

        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        R_z = np.array([
            [cos_yaw, -sin_yaw, 0],
            [sin_yaw,  cos_yaw, 0],
            [0,        0,       1]
        ])

        if indoor_scene:
            corners_rotated = (R_z @ corners_local.T).T
            corners = corners_rotated + center
        else:
            base_z = ground_z if ground_z is not None else float(np.min(cluster_points[:, 2]))
            corners_local[:, 2] += (base_z + h_half) - center[2]
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
            'source_bbox_idx': None,
            'mask_idx': mask_idx,
            'n_points': len(cluster_points),
        }

        mask_conf = 1.0
        if mask_confidences is not None and mask_idx < len(mask_confidences):
            mask_conf = _unit_interval(float(mask_confidences[mask_idx]))

        reproj_agree = 0.5
        if projection is not None and mask_bboxes is not None and mask_idx < len(mask_bboxes):
            proj_payload = projection.cuboid_to_2d(cuboid)
            if proj_payload is not None:
                bbox_2d = proj_payload.get("bbox_2d")
                if bbox_2d is not None:
                    reproj_bbox = [
                        float(bbox_2d["left"]),
                        float(bbox_2d["top"]),
                        float(bbox_2d["right"]),
                        float(bbox_2d["bottom"]),
                    ]
                    reproj_agree = _unit_interval(
                        float(calculate_iou(mask_bboxes[mask_idx], reproj_bbox))
                    )
                    cuboid["projected_bbox_2d"] = bbox_2d

        n_points = int(len(cluster_points))
        n_ref = float(st.session_state.params.get("confidence_points_ref", 80.0))
        fit_lambda = float(st.session_state.params.get("confidence_fit_lambda", 1.0))
        n_ref = max(1.0, n_ref)
        fit_lambda = max(0.0, fit_lambda)

        c_points = min(1.0, n_points / n_ref)
        fit_score = float(fit_result.get("score", 1e9))
        fit_energy = max(0.0, fit_score)
        c_fit = float(np.exp(-fit_lambda * fit_energy))
        c_template = _template_consistency(category, (length, width, height))
        c_fit_total = _unit_interval(c_fit * c_template)

        conf_components = {
            "c_2d": mask_conf,
            "c_points": _unit_interval(c_points),
            "c_fit": c_fit_total,
            "c_reproj": reproj_agree,
            "fit_energy": fit_energy,
            "fit_lambda": fit_lambda,
            "n_points": n_points,
            "n_ref": n_ref,
            "template_consistency": c_template,
        }
        confidence = (
            conf_components["c_2d"]
            * conf_components["c_points"]
            * conf_components["c_fit"]
            * conf_components["c_reproj"]
        )
        cuboid["confidence"] = _unit_interval(confidence)
        cuboid["confidence_components"] = conf_components

        detected_cuboids.append(cuboid)

    elapsed_time = time.time() - start_time
    
    return {
        'detected_cuboids': detected_cuboids,
        'n_detected': len(detected_cuboids),
        'time': elapsed_time
    }


# ============================================================================
# Pipeline Orchestrator
# ============================================================================

def run_full_pipeline(params: Dict, preloaded_bbox_data: Optional[Dict] = None) -> Dict:
    """
    Run the full detection pipeline from start to finish.
    
    Args:
        params: Dictionary with all pipeline parameters
        preloaded_bbox_data: Raw bbox annotation dict (as exported by 4_Export).
            When provided, Step 3 builds masks from these bboxes instead of
            running SAM.  Built *after* Step 2 so that mask_assignments use
            the correct sparse-depth points and projection.
    
    Returns:
        Dict with results from all steps
    """
    results = {}
    raw_pc = st.session_state.sample["point_cloud"]
    ground_kw = _ground_removal_kwargs(params, raw_pc)
    use_ground_plane_removal = bool(st.session_state.get("use_ground_plane_removal", True))

    # Step 1: Ground plane removal
    if 'step_1' not in results or not results['step_1'].get('completed', False):
        step_1_result = step_1_ground_plane_removal(
            point_cloud=st.session_state.sample['point_cloud'],
            use_ground_plane_removal=use_ground_plane_removal,
            **ground_kw
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
    
    # Step 3: SAM segmentation (or preloaded bbox annotations)
    if preloaded_bbox_data is not None:
        sam_integration = _get_current_sam_integration(
            sam_model_type=params.get('sam_model_type', 'sam2_t'),
            use_gpu=params.get('use_gpu', True),
        )
        step_3_result = _build_step3_from_bboxes(
            bbox_data=preloaded_bbox_data,
            image=st.session_state.sample['image'],
            sparse_points=step_2_result['colored_sparse_points'],
            sample_meta_data=st.session_state.sample['sample_meta_data'],
            sam_integration=sam_integration,
            projection=step_2_result['projection'],
        )
    else:
        class_names = _class_names_for_step3_segmentation(params)
        print(f'class_names: {class_names}')
        step_3_result = step_3_sam_segmentation(
            sample_meta_data=st.session_state.sample['sample_meta_data'],
            image=st.session_state.sample['image'],
            sparse_points=step_2_result['colored_sparse_points'],
            class_names=class_names,
            sam_model_type=params.get('sam_model_type', 'sam2_t'),
            yolo_model_path=params.get('yolo_model_path', None),
            conf_threshold=params.get('yolo_conf_threshold', 0.25),
            open_vocab_detector=params.get('open_vocab_detector', 'yolo'),
            grounding_dino_model_id=params.get('grounding_dino_model_id'),
            use_gpu=params.get('use_gpu', True),
            projection=step_2_result['projection'],
            use_dataset_gt_2d_bboxes=params.get(
                "use_gt_2d_bboxes_step3",
                params.get("sunrgbd_use_label_bboxes_step3", False),
            ),
        )
    results['step_3'] = {'completed': True, 'result': step_3_result}
    st.session_state.pipeline_state['step_3'] = results['step_3']
    
    # Step 4: Clustering
    step_4_result = step_4_clustering(
        sample_meta_data=st.session_state.sample['sample_meta_data'],
        sparse_points=step_2_result['colored_sparse_points'],
        sam_masks=step_3_result['sam_masks'],
        mask_assignments=step_3_result['mask_assignments'],
        sparse_depth_map=step_2_result['sparse_depth_map'],
        projection=step_2_result['projection'],
        **params['clustering']
    )
    results['step_4'] = {'completed': True, 'result': step_4_result}
    st.session_state.pipeline_state['step_4'] = results['step_4']
    
    # Step 5: Detection & pose estimation
    step_5_result = step_5_detection_pose_estimation(
        sample_meta_data=st.session_state.sample['sample_meta_data'],
        sparse_points=step_2_result['colored_sparse_points'],
        best_cluster_sparse_indices=step_4_result['best_cluster_sparse_indices'],
        filtered_mask_sparse_indices=step_4_result.get('filtered_mask_sparse_indices'),
        sam_masks=step_3_result['sam_masks'],
        mask_bboxes=step_3_result.get('mask_bboxes'),
        mask_confidences=step_3_result.get('confidences'),
        projection=step_2_result.get('projection'),
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
    tracker: Optional[ObjectTracker],
    frame_index: int,
    prev_image: Optional[np.ndarray] = None,
    preloaded_bbox_data: Optional[Dict] = None,
    saved_image_path: str = "",
    saved_point_cloud_path: str = "",
) -> Optional[Dict]:
    """Load one sample, run full pipeline, update tracker, return export_results for Export page."""
    print(
        f"[batch] start sample "
        f"dataset_path={dataset_path}, dataset_type={dataset_type}, "
        f"sample_index={sample_index}, frame_index={frame_index}"
    )
    # SUNRGBD is always loaded directly from the dataset (depth .mat + label .txt).
    # We no longer rely on pre-saved PCD scenes for batch mode.
    use_saved_media_paths = False
    print(f'st.session_state.get("batch_samples_saved"): {st.session_state.get("batch_samples_saved")}')
    print(f'dataset_type.lower(): {dataset_type.lower()}')
    print(f'bool(st.session_state.get("batch_samples_saved")): {bool(st.session_state.get("batch_samples_saved"))}')
    print(f"use_saved_media_paths: {use_saved_media_paths}")
    meta, image, point_cloud = load_dataset_sample(
        dataset_path=dataset_path,
        sample_index=sample_index,
        dataset_type=dataset_type,
        filter_forward_only=False,
        use_saved_media_paths=use_saved_media_paths,
        saved_image_path=saved_image_path,
        saved_point_cloud_path=saved_point_cloud_path,
        sunrgbd_keep_fraction=sunrgbd_keep_fraction_for_load(),
    )
    if meta is None or image is None or point_cloud is None:
        print(
            f"[batch] load_dataset_sample returned None "
            f"(meta is None: {meta is None}, "
            f"image is None: {image is None}, "
            f"point_cloud is None: {point_cloud is None})"
        )
        return None

    if use_saved_media_paths:
        print(
            f"[batch] requested saved SUNRGBD media "
            f"(image={saved_image_path}, pcd={saved_point_cloud_path})"
        )

    print(
        f"[batch] loaded sample: "
        f"image_shape={image.shape if hasattr(image, 'shape') else 'n/a'}, "
        f"num_points={len(point_cloud)}"
    )
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
    # If rosbag metadata exposes a measured frequency, keep bag_freq_hz in sync.
    if dataset_type.lower() == "rosbag":
        if meta is not None:
            detected_freq = meta.get("bag_freq_hz")
            if detected_freq is None:
                detected_freq = meta.get("bag_frequency_hz")
            if detected_freq is not None:
                if "params" not in st.session_state:
                    st.session_state.params = default_detection_params()
                else:
                    ensure_detection_params(st.session_state.params)
                st.session_state.params["bag_freq_hz"] = float(detected_freq)
    if preloaded_bbox_data is not None:
        st.session_state["_bbox_load_frame_index"] = frame_index
    try:
        print("[batch] running full pipeline")
        results = run_full_pipeline(
            st.session_state.params,
            preloaded_bbox_data=preloaded_bbox_data,
        )
        print("[batch] pipeline finished")
    except Exception as e:
        print(f"[batch] run_full_pipeline raised exception: {e}")
        import traceback

        traceback.print_exc()
        save_sample_to_hard_drive_after_processing(
            image=image,
            point_cloud=point_cloud,
            sample_meta_data=meta,
            base_name=f"frame_{frame_index:06d}",
        )
        return None
    step_5_result = results["step_5"]["result"]
    step_3_result = results["step_3"]["result"]
    _update_mask_capacity_hint(step_3_result, meta)
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
    if dataset_type_lower in ("kitti", "sim", "sunrgbd"):
        # Always write ground_truth_cuboids (even empty list) for annotated dataset types.
        # An absent key means "no GT available"; an empty list means "GT available, empty scene".
        # This ensures filtered frames with zero annotations still contribute FPs to AP.
        ground_truth_boxes = meta.get("ground_truth_boxes", [])
        ground_truth_cuboids = []
        for gt_box in ground_truth_boxes:
            gt_cuboid = {
                "category": gt_box.get("category", gt_box.get("class", "Unknown")),
                "corners": gt_box.get("corners"),
                "bbox_2d": gt_box.get("bbox_2d"),
                "truncation": gt_box.get("truncation"),
                "occlusion": gt_box.get("occlusion", gt_box.get("occluded")),
                "alpha": gt_box.get("alpha"),
                "bbox_height_px": gt_box.get("bbox_height_px"),
                "min_x": gt_box.get("min_x"),
                "max_x": gt_box.get("max_x"),
                "min_y": gt_box.get("min_y"),
                "max_y": gt_box.get("max_y"),
                "min_z": gt_box.get("min_z"),
                "max_z": gt_box.get("max_z"),
                "format": f"{dataset_type_lower}_gt",
            }
            ground_truth_cuboids.append(gt_cuboid)
        export_results["ground_truth_cuboids"] = ground_truth_cuboids
        export_results["metadata"]["n_ground_truth"] = len(ground_truth_cuboids)

    if tracker is not None:
        print(
            f"[batch] updating tracker for frame_index={frame_index}, "
            f"n_detections={step_5_result.get('n_detected', 0)}"
        )
        if step_3_result is None:
            print("[batch] step_3_result is None, skipping tracker update")
        else:
            sam_masks = step_3_result.get("sam_masks", [])
            class_names = step_3_result.get("class_names", [])
            print(
                f"[batch] step_3_result: "
                f"n_masks={len(sam_masks)}, n_class_names={len(class_names)}"
            )
            loaded_ids = step_3_result.get("loaded_instance_ids")
            if loaded_ids and any(iid is not None for iid in loaded_ids):
                mask_to_track: Dict[int, int] = {
                    i: int(iid) for i, iid in enumerate(loaded_ids) if iid is not None
                }
                print(f"[batch] using loaded instance IDs as mask_to_track: {mask_to_track}")
                tracker.apply_external_image_tracks(
                    frame_index=frame_index,
                    image=image,
                    masks=sam_masks,
                    class_names=class_names,
                    meta=meta,
                    mask_to_track=mask_to_track,
                )
            else:
                image_track_mode = st.session_state.params.get("image_track_mode", "appearance")
                if image_track_mode == "deepsort":
                    mask_to_track = tracker.track_on_image_deepsort(
                        frame_index=frame_index,
                        image=image,
                        masks=sam_masks,
                        class_names=class_names,
                        meta=meta,
                    )
                elif image_track_mode == "bytetrack":
                    mask_to_track = tracker.track_on_image_bytetrack(
                        frame_index=frame_index,
                        image=image,
                        masks=sam_masks,
                        class_names=class_names,
                        meta=meta,
                    )
                else:
                    mask_to_track = tracker.track_on_image(
                        frame_index=frame_index,
                        image=image,
                        masks=sam_masks,
                        class_names=class_names,
                        meta=meta,
                    )
            tracker.match_tracks_with_3d_detections(
                frame_index=frame_index,
                detected_cuboids=step_5_result["detected_cuboids"],
                masks=sam_masks,
                class_names=class_names,
                mask_to_track=mask_to_track,
                meta=meta,
            )

    save_sample_to_hard_drive_after_processing(
        image=image,
        point_cloud=point_cloud,
        sample_meta_data=meta,
        base_name=f"frame_{frame_index:06d}",
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
    
    # Initialize parameters; merge defaults if session has partial params (e.g. from batch).
    if "params" not in st.session_state:
        st.session_state.params = default_detection_params()
    else:
        ensure_detection_params(st.session_state.params)

    # Discover available models (cheap; safe on rerun)
    available = get_available_models()
    sam2_models = available.get("sam2", [])
    sam3_models = available.get("sam3", [])
    yolo_models = available.get("yolo", [])
    st.session_state.available_models = available

    # Sidebar: Parameters
    st.sidebar.header("⚙️ Pipeline Parameters")

    st.session_state["save_processed_samples"] = st.sidebar.checkbox(
        "Save processed sample to disk",
        value=st.session_state.get("save_processed_samples", False),
    )
    sidebar_sample_meta = (st.session_state.get("sample") or {}).get("sample_meta_data", {})
    sidebar_dataset_type = (sidebar_sample_meta.get("dataset_type") or "").lower()
    batch_samples_for_sidebar = st.session_state.get("batch_samples", []) or []
    batch_dataset_types = {
        str(s.get("dataset_type", "")).lower()
        for s in batch_samples_for_sidebar
        if str(s.get("dataset_type", "")).strip()
    }
    if not sidebar_dataset_type:
        if batch_samples_for_sidebar:
            if len(batch_dataset_types) == 1:
                sidebar_dataset_type = next(iter(batch_dataset_types))
    if "ground_plane_toggle_dataset_type" not in st.session_state:
        st.session_state["ground_plane_toggle_dataset_type"] = sidebar_dataset_type
        st.session_state["use_ground_plane_removal"] = _default_use_ground_plane_removal(sidebar_sample_meta)
    elif st.session_state["ground_plane_toggle_dataset_type"] != sidebar_dataset_type:
        st.session_state["ground_plane_toggle_dataset_type"] = sidebar_dataset_type
        st.session_state["use_ground_plane_removal"] = _default_use_ground_plane_removal(sidebar_sample_meta)

    st.session_state["use_ground_plane_removal"] = st.sidebar.checkbox(
        "Use ground plane removal (RANSAC)",
        value=st.session_state.get("use_ground_plane_removal", True),
        help="Disable this for datasets/scenes where keeping all points is preferred (e.g., SUNRGBD).",
    )
    
    with st.sidebar.expander("Ground Plane Removal (outdoor)", expanded=False):
        st.session_state.params['pipeline']['distance_threshold'] = st.slider(
            "Distance Threshold", 0.1, 1.0, 0.3, 0.01, key="pipe_out_dt"
        )
        st.session_state.params['pipeline']['ransac_n'] = st.slider(
            "RANSAC N", 3, 10, 3, 1, key="pipe_out_rn"
        )
        st.session_state.params['pipeline']['num_iterations'] = st.slider(
            "Iterations", 100, 2000, 1000, 100, key="pipe_out_ni"
        )
        st.session_state.params['pipeline']['filter_forward_only'] = st.checkbox(
            "Forward-Facing Only", False, key="pipe_out_ff"
        )

    with st.sidebar.expander("Ground Plane Removal (indoor)", expanded=False):
        st.session_state.params['pipeline_indoor']['distance_threshold'] = st.slider(
            "Distance Threshold", 0.05, 0.5, 0.12, 0.01, key="pipe_in_dt"
        )
        st.session_state.params['pipeline_indoor']['ransac_n'] = st.slider(
            "RANSAC N", 3, 10, 3, 1, key="pipe_in_rn"
        )
        st.session_state.params['pipeline_indoor']['num_iterations'] = st.slider(
            "Iterations", 100, 3000, 1500, 100, key="pipe_in_ni"
        )
        st.session_state.params['pipeline_indoor']['filter_forward_only'] = st.checkbox(
            "Forward-Facing Only", False, key="pipe_in_ff"
        )

    with st.sidebar.expander("Scene from LiDAR (indoor vs outdoor)", expanded=False):
        st.caption(
            "Outdoor: huge AABB volume or wide horizontal span. Indoor: strong N/V density, "
            "or low ceiling with relaxed density, or many points within near_xy_radius (walls/room)."
        )
        _s = st.session_state.params["scene_from_pointcloud"]
        _s["max_aabb_volume_m3"] = st.slider(
            "Max AABB volume (m³); above → outdoor",
            1000.0,
            80000.0,
            float(_s.get("max_aabb_volume_m3", 22000.0)),
            500.0,
            key="sc_pc_vol",
        )
        _s["max_horizontal_span_m"] = st.slider(
            "Max horizontal span (m) for indoor",
            10.0,
            120.0,
            float(_s.get("max_horizontal_span_m", 52.0)),
            1.0,
            key="sc_pc_span",
        )
        _s["min_points_per_m3"] = st.slider(
            "Min points / m³ (primary indoor rule)",
            0.2,
            30.0,
            float(_s.get("min_points_per_m3", 1.0)),
            0.1,
            key="sc_pc_den",
        )
        _s["min_points"] = int(
            st.number_input(
                "Min points to classify",
                min_value=100,
                max_value=50000,
                value=int(_s.get("min_points", 400)),
                step=50,
                key="sc_pc_min",
            )
        )
        _s["max_vertical_span_m"] = st.slider(
            "Max Δz (m) for low-ceiling indoor path",
            4.0,
            30.0,
            float(_s.get("max_vertical_span_m", 14.0)),
            0.5,
            key="sc_pc_dz",
        )
        _s["density_relax_factor"] = st.slider(
            "Density relax (× min pts/m³) for low-ceiling path",
            0.15,
            0.95,
            float(_s.get("density_relax_factor", 0.42)),
            0.01,
            key="sc_pc_relax",
        )
        _s["near_xy_radius_m"] = st.slider(
            "Near-field xy radius (m)",
            4.0,
            40.0,
            float(_s.get("near_xy_radius_m", 14.0)),
            0.5,
            key="sc_pc_near_r",
        )
        _s["min_near_xy_fraction"] = st.slider(
            "Min fraction of points inside near cylinder",
            0.05,
            0.55,
            float(_s.get("min_near_xy_fraction", 0.16)),
            0.01,
            key="sc_pc_near_f",
        )
    
    with st.sidebar.expander("Clustering (Step 4)", expanded=False):
        _clustering = st.session_state.params['clustering']
        _algo_options = ["adaptive_dbscan", "hdbscan", "dbscan"]
        _algo_labels = {
            "adaptive_dbscan": "Adaptive DBSCAN (distance-aware eps)",
            "hdbscan": "HDBSCAN",
            "dbscan": "DBSCAN",
        }
        _sidebar_point_cloud = (st.session_state.get("sample") or {}).get("point_cloud")
        _default_algo = "adaptive_dbscan"
        if isinstance(_sidebar_point_cloud, np.ndarray) and _sidebar_point_cloud.size > 0:
            if _pipeline_scene_is_indoor(st.session_state.params, _sidebar_point_cloud):
                _default_algo = "dbscan"

        _current_algo = _clustering.get("clustering_algorithm", _default_algo)
        if _current_algo not in _algo_options:
            _current_algo = _default_algo
        _algo_index = _algo_options.index(_current_algo)
        _selected_algo_label = st.selectbox(
            "Clustering Algorithm",
            options=[_algo_labels[a] for a in _algo_options],
            index=_algo_index,
            help="Use adaptive DBSCAN for near/far objects with different local densities."
        )
        _selected_algo = _algo_options[[_algo_labels[a] for a in _algo_options].index(_selected_algo_label)]
        _clustering["clustering_algorithm"] = _selected_algo

        st.session_state.params['clustering']['dbscan_eps'] = st.slider(
            "DBSCAN Eps", 0.1, 2.0, 0.2, 0.1
        )
        st.session_state.params['clustering']['dbscan_min_samples'] = st.slider(
            "Min Samples", 3, 20, 5, 1
        )
        if _selected_algo == "hdbscan":
            _clustering["hdbscan_min_cluster_size"] = st.slider(
                "HDBSCAN Min Cluster Size",
                3,
                50,
                int(_clustering.get("hdbscan_min_cluster_size", 5)),
                1
            )
            _clustering["hdbscan_min_samples"] = st.slider(
                "HDBSCAN Min Samples",
                1,
                30,
                int(_clustering.get("hdbscan_min_samples", 5)),
                1
            )
        elif _selected_algo == "adaptive_dbscan":
            _clustering["adaptive_dbscan_base_eps"] = st.slider(
                "Adaptive DBSCAN Base Eps",
                0.05,
                1.5,
                float(_clustering.get("adaptive_dbscan_base_eps", 0.35)),
                0.05
            )
            _clustering["adaptive_dbscan_eps_growth_rate"] = st.slider(
                "Adaptive Eps Growth Rate",
                0.0,
                3.0,
                float(_clustering.get("adaptive_dbscan_eps_growth_rate", 1.0)),
                0.1,
                help=(
                    "Scales how much per-point neighborhood size varies with 3D range "
                    "relative to the median range of *this* point set. Larger values "
                    "spread scale within a mask (e.g. roof vs base), so clustering "
                    "reacts more to this slider. Uses 3D distance so vertical gaps "
                    "shrink like horizontal ones after warping."
                ),
            )
            _clustering["adaptive_dbscan_reference_distance"] = st.slider(
                "Adaptive Reference Distance (m)",
                5.0,
                60.0,
                float(_clustering.get("adaptive_dbscan_reference_distance", 15.0)),
                1.0,
                help=(
                    "Denominator for growth: same (r - median r) in meters produces "
                    "smaller scale changes when this is larger. Lower it if the "
                    "growth slider feels too weak."
                ),
            )
            _clustering["adaptive_dbscan_min_scale"] = st.slider(
                "Adaptive Min Scale",
                0.4,
                2.0,
                float(_clustering.get("adaptive_dbscan_min_scale", 0.7)),
                0.1
            )
            _clustering["adaptive_dbscan_max_scale"] = st.slider(
                "Adaptive Max Scale",
                1.0,
                8.0,
                float(_clustering.get("adaptive_dbscan_max_scale", 4.0)),
                0.1
            )
        st.session_state.params['clustering']['volume_factor'] = st.slider(
            "Max Cluster Volume Factor",
            0.5, 5.0, 1.1, 0.1,
            help="Multiplier on the template volume used to discard clusters that are too large."
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

    with st.sidebar.expander("Indoor cuboid (Step 5)", expanded=False):
        st.session_state.params['cuboid_fitting_indoor']['margin'] = st.slider(
            "Padding (m)", 0.01, 0.3, 0.02, 0.01, key="cub_in_margin"
        )
        st.session_state.params['cuboid_fitting_indoor']['min_extent'] = st.slider(
            "Min L/W/H (m)", 0.02, 0.3, 0.05, 0.01, key="cub_in_minex"
        )
        st.session_state.params['cuboid_fitting_indoor']['d_theta'] = st.slider(
            "Yaw step (rad)", 0.02, 0.2, 0.02, 0.01, key="cub_in_dth"
        )
        st.session_state.params['cuboid_fitting_indoor']['inlier_quantile'] = st.slider(
            "Inlier quantile", 0.85, 1.0, 0.99, 0.01, key="cub_in_iq",
            help="Higher keeps more points in the fit; lower trims sparse outliers for tighter boxes."
        )
        st.session_state.params['cuboid_fitting_indoor']['coverage_weight'] = st.slider(
            "Coverage weight", 0.0, 10.0, 4.0, 0.5, key="cub_in_covw",
            help="Penalty for excluding points. Lower favors tighter boxes, higher favors coverage."
        )
    
    st.sidebar.markdown("### SAM Model")
    sam_options = sam2_models + sam3_models
    if not sam_options:
        sam_options = ['sam2_t', 'sam3']

    current_sam = st.session_state.params.get('sam_model_type')
    if current_sam in sam_options:
        default_sam_index = sam_options.index(current_sam)
    else:
        if sam3_models:
            default_sam = sam3_models[0]
        elif sam2_models:
            default_sam = sam2_models[0]
        else:
            default_sam = 'sam3' if 'sam3' in sam_options else sam_options[0]
        default_sam_index = sam_options.index(default_sam)

    st.session_state.params['sam_model_type'] = st.sidebar.selectbox(
        "SAM Model Type",
        options=sam_options,
        index=default_sam_index,
        help="Automatically discovered from models directory."
    )

    # Class names input
    st.sidebar.markdown("### Open Vocabulary Detection")
    class_names_input = st.sidebar.text_input(
        "Class Names (comma-separated)",
        value="person",
        help="Enter class names separated by commas (e.g., 'car, person, bicycle')"
    )

    # Parse class names from sidebar input for all dataset types.
    parsed_class_names: List[str] = []
    if class_names_input:
        parsed_class_names = [name.strip() for name in class_names_input.split(',') if name.strip()]
    st.session_state.params['class_names'] = parsed_class_names

    # For SUNRGBD (indoor-only), skip LLM initialization entirely.
    # Keep this robust for both single-sample and batch-only sessions.
    is_sunrgbd_dataset = (
        sidebar_dataset_type == "sunrgbd"
        or "sunrgbd" in batch_dataset_types
    )
    has_gt_2d_bbox_dataset = (
        sidebar_dataset_type in {"sunrgbd", "kitti"}
        or "sunrgbd" in batch_dataset_types
        or "kitti" in batch_dataset_types
    )

    if not is_sunrgbd_dataset:
        # LLM Settings
        with st.sidebar.expander("LLM Settings", expanded=False):

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

            # Local/remote model selection
            available_llm_models = get_available_llm_models()
            current_llm_model = get_current_llm_model_name()
            llm_model_options = available_llm_models.copy()
            if current_llm_model not in llm_model_options:
                llm_model_options.insert(0, current_llm_model)

            llm_model_labels = {}
            for m in llm_model_options:
                if os.path.isabs(m):
                    llm_model_labels[m] = f"{os.path.basename(m)} (local)"
                else:
                    llm_model_labels[m] = m

            llm_model_index = 0
            if current_llm_model in llm_model_options:
                llm_model_index = llm_model_options.index(current_llm_model)

            selected_llm_model = st.selectbox(
                "LLM Model",
                options=llm_model_options,
                index=llm_model_index,
                format_func=lambda m: llm_model_labels.get(m, m),
                help="Pick a stronger instruction model from ./llm for better dimension estimates."
            )
            set_llm_model_name(selected_llm_model)

            if available_llm_models:
                st.caption(f"Discovered {len(available_llm_models)} local model(s) in ./llm")
            else:
                st.caption("No local models found in ./llm; using LLM_MODEL_NAME or default model id.")

            # Toggle for enabling LLM model queries (may download a model on first use)
            if "llm_enable_model_query" not in st.session_state:
                st.session_state.llm_enable_model_query = True

            st.session_state.llm_enable_model_query = st.checkbox(
                "Enable LLM model (dimension lookup)",
                value=st.session_state.llm_enable_model_query,
                help="When enabled, a Hugging Face LLM is used for unseen classes. "
                     "May download a model the first time it runs."
            )

            if st.session_state.llm_enable_model_query:
                os.environ["LLM_ENABLE_MODEL_QUERY"] = "1"
                os.environ["LLM_ALLOW_DOWNLOAD"] = "1"
                st.sidebar.caption("LLM model queries enabled")
            else:
                os.environ["LLM_ENABLE_MODEL_QUERY"] = "0"
                st.sidebar.caption("LLM model queries disabled (using template defaults only)")

            st.sidebar.caption("💡 LLM is used when semantic similarity doesn't find a match (similarity < 0.75)")

        # Pre-compute dimensions via LLM for parsed class names.
        if parsed_class_names:
            # Check if class names have changed
            previous_class_names = st.session_state.get('previous_class_names', [])
            if set(parsed_class_names) != set(previous_class_names):
                with st.sidebar.spinner("Pre-computing dimensions for class names..."):
                    dims_by_class = {}
                    for class_name in parsed_class_names:
                        length, width, height = query_llm_for_dimensions(class_name)
                        dims_by_class[class_name] = (length, width, height)

                    st.session_state.params['dimensions_by_class'] = dims_by_class
                    # template_dims format for frustum_manager / evaluation
                    st.session_state.params['template_dims'] = {
                        k: {'length': v[0], 'width': v[1], 'height': v[2]}
                        for k, v in dims_by_class.items()
                    }

                st.session_state.previous_class_names = parsed_class_names.copy()

        if not st.session_state.params['class_names']:
            st.sidebar.warning("⚠️ Please enter at least one class name")
    else:
        # For SUNRGBD, do not initialize or query any LLM components.
        st.sidebar.info("LLM-based dimension lookup is disabled for SUNRGBD dataset.")
        if not st.session_state.params['class_names']:
            st.sidebar.warning("⚠️ Please enter at least one class name")

    if has_gt_2d_bbox_dataset:
        _gt_bbox_override = st.sidebar.checkbox(
            "Step 3: use dataset GT 2D bboxes (KITTI/SUNRGBD)",
            value=st.session_state.params.get(
                "use_gt_2d_bboxes_step3",
                st.session_state.params.get("sunrgbd_use_label_bboxes_step3", False),
            ),
            help=(
                "Use dataset ground-truth 2D boxes to build Step 3 masks instead of open-vocabulary "
                "2D detection. Supported for KITTI and SUNRGBD. When enabled, use the Step 3 text area "
                "to filter by label class names, or leave it empty for all GT boxes."
            ),
        )
        st.session_state.params["use_gt_2d_bboxes_step3"] = _gt_bbox_override
        st.session_state.params["sunrgbd_use_label_bboxes_step3"] = _gt_bbox_override
    
    is_sam2 = st.session_state.params['sam_model_type'].startswith('sam2')
    if is_sam2:
        st.sidebar.markdown("### Open-vocabulary detection (SAM2)")
        det_options = {"YOLO-World": "yolo", "Grounding DINO (HF)": "grounding_dino"}
        det_labels = list(det_options.keys())
        current_det = st.session_state.params.get('open_vocab_detector', 'yolo')
        det_index = 0
        for i, lbl in enumerate(det_labels):
            if det_options[lbl] == current_det:
                det_index = i
                break
        picked = st.sidebar.radio(
            "Detector",
            options=det_labels,
            index=det_index,
            help="Bounding boxes for SAM2: YOLO-World from local weights, or Grounding DINO via Hugging Face.",
        )
        st.session_state.params['open_vocab_detector'] = det_options[picked]

        if st.session_state.params['open_vocab_detector'] == 'grounding_dino':
            st.session_state.params['grounding_dino_model_id'] = st.sidebar.text_input(
                "Grounding DINO model id",
                value=st.session_state.params.get(
                    'grounding_dino_model_id', 'IDEA-Research/grounding-dino-base'
                ),
                help="Hugging Face model id for transformers zero-shot-object-detection.",
            )
            st.session_state.params['yolo_model_path'] = None
        else:
            st.sidebar.markdown("#### YOLO-World weights")
            if yolo_models:
                current_yolo = st.session_state.params.get('yolo_model_path')
                yolo_index = yolo_models.index(current_yolo) if current_yolo in yolo_models else 0
                st.session_state.params['yolo_model_path'] = st.sidebar.selectbox(
                    "YOLO Model",
                    options=yolo_models,
                    index=yolo_index,
                    help="Discovered from models directory."
                )
            else:
                st.session_state.params['yolo_model_path'] = None
                st.sidebar.warning("⚠️ No YOLO models found in models directory.")

        st.session_state.params['yolo_conf_threshold'] = st.sidebar.slider(
            "Detector confidence threshold",
            0.0, 1.0, 0.25, 0.05,
            help="Confidence threshold for open-vocabulary detections (SAM2)"
        )
        st.sidebar.info("💡 SAM2 uses the chosen detector for boxes, then SAM2 for masks")

    else:
        st.session_state.params['yolo_model_path'] = None
        st.session_state.params['open_vocab_detector'] = 'yolo'
        st.sidebar.info("💡 SAM3 uses direct text prompts for open-vocabulary segmentation")
    # Keep the preference in session state; Step 3 applies it for datasets with GT 2D boxes.
    
    # Compute Device
    st.sidebar.markdown("### Compute Device")
    gpu_available = torch.cuda.is_available()
    st.session_state.params['use_gpu'] = st.sidebar.checkbox(
        "Use GPU (CUDA)",
        value=st.session_state.params.get('use_gpu', True) and gpu_available,
        disabled=not gpu_available,
        help="Enable CUDA acceleration for SAM/YOLO when available."
    )
    # Expose CUDA preference and availability to the LLM service as well
    use_cuda_flag = gpu_available and st.session_state.params['use_gpu']
    os.environ["LLM_USE_CUDA"] = "1" if use_cuda_flag else "0"
    if gpu_available:
        if use_cuda_flag:
            st.sidebar.caption("CUDA available and enabled")
        else:
            st.sidebar.caption("CUDA available but disabled in settings")
    else:
        st.sidebar.caption("CUDA not available, using CPU")
    
    #batch processing
    batch_samples = st.session_state.get("batch_samples", [])
    process_all_samples = st.session_state.get("process_all_samples", False)
    if batch_samples:
        _itm = st.session_state.params.get("image_track_mode", "appearance")
        _itm_options = ["appearance", "deepsort", "bytetrack"]
        _itm_idx = _itm_options.index(_itm) if _itm in _itm_options else 0
        st.session_state.params["image_track_mode"] = st.sidebar.radio(
            "2D tracking (batch / cross-frame)",
            options=_itm_options,
            index=_itm_idx,
            format_func=lambda m: (
                "Appearance (patch histogram + cosine)"
                if m == "appearance"
                else ("DeepSORT (Kalman + ReID)" if m == "deepsort" else "ByteTrack (two-stage assoc)")
            ),
            help=(
                "Appearance matches masks by patch similarity. "
                "DeepSORT combines Kalman motion gating + ReID embedding distance. "
                "ByteTrack uses two-stage high/low confidence association."
            ),
            key="sidebar_image_track_mode",
        )
        st.session_state.batch_process_enabled_tracking = st.sidebar.checkbox(
            "Run cross-frame tracking on the batch (2D/3D association; enables tracklet & Datumaro exports)",
            key="batch_process_enable_tracking",
            help=(
                "When enabled, an ObjectTracker runs across all frames and builds datumaro_tracking. "
                "Disable for detection-only batch runs (faster; Export page hides tracking exports)."
            ),
        )
        
    if batch_samples and process_all_samples:
        st.subheader("📚 Batch Processing")
        total = len(batch_samples)
        st.info(f"Batch loaded: **{total}** samples. Process the entire batch to run detection on all samples.")

        with st.expander("📂 Load 2D bounding-box annotations for batch (Step 3 override)", expanded=False):
            st.caption(
                "Upload a bbox_only or bbox_tracking JSON file exported from **4_Export**. "
                "When provided, Step 3 (SAM segmentation) is replaced by the loaded bounding boxes for each frame. "
                "If the file contains tracking instance IDs, they are used directly for cross-frame tracking."
            )
            batch_bbox_file = st.file_uploader(
                "Upload bbox annotation JSON for batch",
                type=["json"],
                key="step3_bbox_upload_batch",
            )
            batch_bbox_data: Optional[Dict] = None
            if batch_bbox_file is not None:
                batch_bbox_data = json.loads(batch_bbox_file.read())
                st.session_state["_batch_bbox_data"] = batch_bbox_data
                bbox_fmt = batch_bbox_data.get("format", "bbox_only")
                if bbox_fmt == "bbox_tracking":
                    n_frames_file = len(batch_bbox_data.get("frames", []))
                    st.info(
                        f"Tracking file loaded with **{n_frames_file}** frames. "
                        f"Batch has **{total}** samples — annotations will be matched by frame index."
                    )
                else:
                    n_ann = len(batch_bbox_data.get("annotations", []))
                    st.info(
                        f"Single-frame bbox file loaded with **{n_ann}** annotations. "
                        "The same annotations will be applied to every frame in the batch."
                    )
            else:
                batch_bbox_data = st.session_state.get("_batch_bbox_data")

        if st.button("🚀 Process entire batch", type="primary", key="process_entire_batch"):
            do_batch_tracking = st.session_state.get("batch_process_enable_tracking", True)
            st.session_state["eval_mask_capacity_max"] = 0
            st.session_state["eval_mask_capacity_hint_active"] = False
            print(
                f"[batch] starting batch processing for {total} samples "
                f"(tracking={'on' if do_batch_tracking else 'off'})"
            )
            results_list = []
            st.session_state.datumaro_tracking = None
            st.session_state.tracking_2d_history = None
            st.session_state.tracker = None
            tracker = None
            if do_batch_tracking:
                tracker = ObjectTracker(
                    bag_freq_hz=st.session_state.params.get('bag_freq_hz', 45.0),
                    class_max_speed_mps=st.session_state.params.get('class_max_speed_mps'),
                )
                st.session_state.tracker = tracker
            progress = st.progress(0.0)
            prev_image: Optional[np.ndarray] = None
            for i, sample_desc in enumerate(batch_samples):
                print(
                    f"[batch] processing index {i} / {total - 1}, "
                    f"dataset_type={sample_desc.get('dataset_type')}, "
                    f"sample_index={sample_desc.get('sample_index')}"
                )
                progress.progress((i + 1) / total)
                export_res = _run_pipeline_for_batch_sample(
                    dataset_path=sample_desc["dataset_path"],
                    dataset_type=sample_desc.get("dataset_type", "kitti"),
                    sample_index=sample_desc["sample_index"],
                    tracker=tracker,
                    frame_index=i,
                    prev_image=prev_image,
                    preloaded_bbox_data=batch_bbox_data,
                    saved_image_path=str(sample_desc.get("image_path") or ""),
                    saved_point_cloud_path=str(sample_desc.get("point_cloud_path") or ""),
                )
                sample_state = st.session_state.get("sample")
                prev_image = sample_state.get("image") if isinstance(sample_state, dict) else None
                if export_res is not None:
                    results_list.append(export_res)
                else:
                    print(f"[batch] sample {i} returned no export_res")
            st.session_state.batch_export_results = {
                "samples": results_list,
                "batch_tracking_enabled": do_batch_tracking,
            }
            if results_list:
                # Keep per-sample export consumers working after batch runs.
                st.session_state.export_results = results_list[-1]
            print(f"[batch] collected {len(results_list)} successful samples")
            if do_batch_tracking:
                datumaro_state = tracker.build_datumaro_state()
                tracking_2d_history = tracker.build_2d_tracking_history()
                print(
                    "[batch] built datumaro_state: "
                    f"n_items={len(datumaro_state.get('items', []))}, "
                    f"categories={list(datumaro_state.get('categories', {}).keys())}"
                )
                st.session_state.datumaro_tracking = datumaro_state
                st.session_state.tracking_2d_history = tracking_2d_history
            else:
                st.session_state.datumaro_tracking = None
                st.session_state.tracking_2d_history = None
            st.success(
                f"✅ Processed **{len(results_list)}** / {total} samples. "
                "Go to **4_Export** to save KITTI and Datumaro-style exports."
            )
            st.rerun()
        if st.session_state.get("tracker") is not None:
            st.caption("Tracker cached in session state for this run.")
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
    
    # Main controls
    col1, col2 = st.columns(2)
    with col1:
        run_full = st.button("🚀 Run Full Pipeline", type="primary")
    with col2:
        reset_pipeline = st.button("🔄 Reset Pipeline")
    
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
                step_3_result = results['step_3']['result']
                _update_mask_capacity_hint(step_3_result, st.session_state.sample['sample_meta_data'])
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
                if dataset_type in ('kitti', 'sim', 'sunrgbd'):
                    ground_truth_boxes = sample_meta_data.get('ground_truth_boxes', [])
                    ground_truth_cuboids = []
                    for gt_box in ground_truth_boxes:
                        gt_cuboid = {
                            'category': gt_box.get('category', gt_box.get('class', 'Unknown')),
                            'corners': gt_box.get('corners'),
                            'bbox_2d': gt_box.get('bbox_2d'),
                            'truncation': gt_box.get('truncation'),
                            'occlusion': gt_box.get('occlusion', gt_box.get('occluded')),
                            'alpha': gt_box.get('alpha'),
                            'bbox_height_px': gt_box.get('bbox_height_px'),
                            'min_x': gt_box.get('min_x'),
                            'max_x': gt_box.get('max_x'),
                            'min_y': gt_box.get('min_y'),
                            'max_y': gt_box.get('max_y'),
                            'min_z': gt_box.get('min_z'),
                            'max_z': gt_box.get('max_z'),
                            'format': f'{dataset_type}_gt'
                        }
                        ground_truth_cuboids.append(gt_cuboid)
                    export_results['ground_truth_cuboids'] = ground_truth_cuboids
                    export_results['metadata']['n_ground_truth'] = len(ground_truth_cuboids)
                st.session_state.export_results = export_results

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
    _lbl = (
        "indoor (LiDAR span + density)"
        if _pipeline_scene_is_indoor(st.session_state.params, point_cloud)
        else "outdoor (LiDAR span + density)"
    )
    st.sidebar.info(f"Scene: {_lbl}")
    st.sidebar.info(f"Image: {image.shape}")
    st.sidebar.info(f"Point Cloud: {len(point_cloud):,} points")
    
    # Step-by-step execution
    st.markdown("---")
    st.subheader("Pipeline Steps")
    
    # Step 1: Ground Plane Removal
    with st.container():
        step_1_state = st.session_state.pipeline_state['step_1']
        col1, col2 = st.columns([3, 1])
        use_ground_plane_removal = bool(st.session_state.get("use_ground_plane_removal", True))
        
        with col1:
            status_icon = "✅" if step_1_state['completed'] else "⏸️"
            st.markdown(f"### {status_icon} Step 1: Ground Plane Removal")
            if use_ground_plane_removal:
                st.caption("Remove ground plane from point cloud using RANSAC")
            else:
                st.caption("Ground plane removal disabled: using full point cloud")
        
        with col2:
            step_1_enabled = True  # Always enabled
            if st.button("▶️ Run Step 1", key="run_step_1", disabled=not step_1_enabled):
                with st.spinner("Running Step 1..."):
                    try:
                        result = step_1_ground_plane_removal(
                            point_cloud=point_cloud,
                            use_ground_plane_removal=use_ground_plane_removal,
                            **_ground_removal_kwargs(st.session_state.params, point_cloud)
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
            if not use_ground_plane_removal:
                st.info("Ground plane removal is disabled; points are unchanged from the input cloud.")
            
            with st.expander("View Step 1 Details", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Original Points", f"{len(point_cloud):,}")
                st.metric("Points Remaining", f"{result['points_remaining']:,}")
                with col2:
                    reduction = (1 - result['points_remaining'] / len(point_cloud)) * 100
                    st.metric("Reduction", f"{reduction:.1f}%")
                st.metric("Ground Z", f"{result['ground_z']:.3f}m" if result['ground_z'] else "N/A")
                
                # 3D Visualization: Ground-kept and removed points
                st.markdown("#### 3D Point Cloud Visualization")
                point_cloud_obj = result['point_cloud_obj']

                kept_points = point_cloud_obj.point_cloud_plane_removed
                removed_points = getattr(point_cloud_obj, "ground_inliers", None)

                fig = go.Figure()

                if kept_points is not None and len(kept_points) > 0:
                    fig.add_trace(
                        go.Scatter3d(
                            x=kept_points[:, 0],
                            y=kept_points[:, 1],
                            z=kept_points[:, 2],
                            mode="markers",
                            marker=dict(size=1.5, color="deepskyblue", opacity=0.45),
                            name="Points kept (non-ground)",
                        )
                    )

                if removed_points is not None and len(removed_points) > 0:
                    fig.add_trace(
                        go.Scatter3d(
                            x=removed_points[:, 0],
                            y=removed_points[:, 1],
                            z=removed_points[:, 2],
                            mode="markers",
                            marker=dict(size=2.0, color="orangered", opacity=0.65),
                            name="Removed ground points",
                        )
                    )

                fig.update_layout(
                    title="Ground Plane Removal: Kept vs Removed Points",
                    scene=dict(
                        xaxis=dict(title="X (m)"),
                        yaxis=dict(title="Y (m)"),
                        zaxis=dict(title="Z (m)"),
                        aspectmode="data",
                    ),
                    margin=dict(l=0, r=0, b=0, t=40),
                    height=600,
                    legend=dict(itemsizing="constant"),
                )
                _render_point_cloud_plot(fig, "step1_point_cloud_after_ground_removal")
        
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
                    st.image(image, caption="Original Image")
                with col2:
                    sparse_depth = result['sparse_depth_map']
                    fig, ax = plt.subplots(figsize=(8, 6))
                    vmax = sparse_depth[sparse_depth > 0].max() if np.sum(sparse_depth > 0) > 0 else 100.0
                    ax.imshow(sparse_depth, cmap='viridis', vmin=0, vmax=vmax)
                    ax.axis('off')
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
                    _render_point_cloud_plot(
                        fig,
                        "step2_colored_sparse_points",
                        use_container_width=True,
                    )
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
            else:
                _ds_gt = (sample_meta_data.get("dataset_type") or "").lower()
                _use_gt_2d = bool(
                    st.session_state.params.get(
                        "use_gt_2d_bboxes_step3",
                        st.session_state.params.get("sunrgbd_use_label_bboxes_step3", False),
                    )
                )
                if _use_gt_2d and _ds_gt in {"sunrgbd", "kitti"}:
                    st.markdown("##### GT Step 3: annotation class filter")
                    _gt_boxes_preview = _resolve_ground_truth_boxes_for_sample(
                        sample_meta_data, image.shape
                    )
                    _ann_labels = _unique_gt_annotation_labels(_gt_boxes_preview)
                    if _ann_labels:
                        st.caption(
                            "Classes with a valid 2D box in this sample: "
                            + ", ".join(_ann_labels)
                        )
                    else:
                        st.caption(
                            "No GT 2D boxes in sample metadata or SUNRGBD label files for this frame."
                        )
                    _ta_key = "gt_step3_target_classes_text_area"
                    if _ta_key not in st.session_state:
                        st.session_state[_ta_key] = st.session_state.params.get(
                            "gt_step3_target_classes_text", ""
                        )
                    _entered = st.text_area(
                        "Target classes (comma or newline; optional)",
                        height=88,
                        placeholder=(
                            "e.g. chair, table, bed\n"
                            "Leave empty to include every GT box from the annotation."
                        ),
                        help=(
                            "Names must match the dataset label (case-insensitive). "
                            "Empty field uses all ground-truth 2D boxes. "
                            "Sidebar “Class names” is not used for this Step 3 mode."
                        ),
                        key=_ta_key,
                    )
                    st.session_state.params["gt_step3_target_classes_text"] = _entered
        
        with col2:
            if st.button("▶️ Run Step 3", key="run_step_3", disabled=not step_2_completed):
                with st.spinner("Running Step 3..."):
                    try:
                        step_2_result = st.session_state.pipeline_state['step_2']['result']
                        class_names = _class_names_for_step3_segmentation(st.session_state.params)
                        result = step_3_sam_segmentation(
                            sample_meta_data=sample_meta_data,
                            image=image,
                            sparse_points=step_2_result['colored_sparse_points'],
                            class_names=class_names,
                            sam_model_type=st.session_state.params['sam_model_type'],
                            yolo_model_path=st.session_state.params.get('yolo_model_path', None),
                            conf_threshold=st.session_state.params.get('yolo_conf_threshold', 0.25),
                            open_vocab_detector=st.session_state.params.get('open_vocab_detector', 'yolo'),
                            grounding_dino_model_id=st.session_state.params.get('grounding_dino_model_id'),
                            use_gpu=st.session_state.params.get('use_gpu', True),
                            projection=step_2_result['projection'],
                            use_dataset_gt_2d_bboxes=st.session_state.params.get(
                                "use_gt_2d_bboxes_step3",
                                st.session_state.params.get("sunrgbd_use_label_bboxes_step3", False),
                            ),
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

        # -- Load 2D bbox annotation file as alternative to running SAM --
        with st.expander("📂 Load 2D bounding-box annotations from file", expanded=False):
            st.caption(
                "Upload a JSON file exported from **4_Export** (bbox_only or bbox_tracking format) "
                "to populate Step 3 without running SAM."
            )
            uploaded_bbox_file = st.file_uploader(
                "Upload bbox annotation JSON",
                type=["json"],
                key="step3_bbox_upload_single",
            )
            if uploaded_bbox_file is not None and step_2_completed:
                bbox_data = json.loads(uploaded_bbox_file.read())
                bbox_fmt = bbox_data.get("format", "bbox_only")
                frame_idx_for_load = 0
                if bbox_fmt == "bbox_tracking":
                    n_frames_avail = len(bbox_data.get("frames", []))
                    frame_idx_for_load = st.number_input(
                        "Frame index to load",
                        min_value=0,
                        max_value=max(0, n_frames_avail - 1),
                        value=0,
                        key="step3_bbox_frame_idx_single",
                    )
                    st.info(f"Tracking file detected with **{n_frames_avail}** frames.")
                if st.button("📥 Load annotations into Step 3", key="load_bbox_step3_single"):
                    st.session_state["_bbox_load_frame_index"] = int(frame_idx_for_load)
                    step_2_result = st.session_state.pipeline_state['step_2']['result']
                    sam_integration = _get_current_sam_integration(
                        sam_model_type=st.session_state.params.get('sam_model_type', 'sam2_t'),
                        use_gpu=st.session_state.params.get('use_gpu', True),
                    )
                    result = _build_step3_from_bboxes(
                        bbox_data=bbox_data,
                        image=image,
                        sparse_points=step_2_result['colored_sparse_points'],
                        sample_meta_data=sample_meta_data,
                        sam_integration=sam_integration,
                        projection=step_2_result['projection'],
                    )
                    st.session_state.pipeline_state['step_3'] = {
                        'completed': True,
                        'result': result,
                        'time': result['time'],
                        'error': result.get('error'),
                    }
                    loaded_ids = result.get("loaded_instance_ids")
                    if loaded_ids:
                        st.session_state["_loaded_bbox_instance_ids"] = loaded_ids
                    st.rerun()

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
                fig = _step3_build_mask_overlay_figure(
                    image=image,
                    sam_masks=sam_masks,
                    colors=colors,
                    mask_bboxes=mask_bboxes,
                    detected_class_names=detected_class_names,
                    confidences=confidences,
                    mask_alpha=0.5,
                )
                st.pyplot(fig)
                plt.close(fig)
                
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
                    st.dataframe(df_detections)
                
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
                    _render_point_cloud_plot(fig, "step3_point_assignment")
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
            st.caption("Run selected clustering algorithm on points assigned to each mask")
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
                            sparse_depth_map=step_2_result['sparse_depth_map'],
                            projection=step_2_result['projection'],
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
                    st.dataframe(df)
                    
                    # 3D Visualization: Clusters
                    st.markdown("#### 3D Cluster Visualization")
                    step_2_result = st.session_state.pipeline_state['step_2']['result']
                    sparse_points = step_2_result['colored_sparse_points']
                    step_3_result = st.session_state.pipeline_state['step_3']['result']
                    mask_assignments = step_3_result['mask_assignments']
                    mask_cluster_labels = result['mask_cluster_labels']
                    filtered_ix = result.get('filtered_mask_sparse_indices', {})
                    legacy_filtered = result.get('filtered_mask_points', {})

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
                        row_ix = filtered_ix.get(mask_idx)
                        if row_ix is not None and len(row_ix) > 0:
                            mask_points = sparse_points[row_ix]
                        else:
                            mask_points = legacy_filtered.get(mask_idx)
                        if mask_points is None:
                            continue
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
                    best_ix = result.get('best_cluster_sparse_indices', {})
                    legacy_best = result.get('best_cluster_points', {})
                    for mask_idx, ix in best_ix.items():
                        if ix is not None and len(ix) > 0:
                            best_points = sparse_points[ix]
                        else:
                            best_points = legacy_best.get(mask_idx)
                        if best_points is None or len(best_points) == 0:
                            continue
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
                    _render_point_cloud_plot(fig, "step4_clusters")

                    # 3D Visualization: best-cluster vs non-best points, colored by mask
                    st.markdown("#### 3D Best Cluster vs Noise (Mask Colors)")
                    fig_best_vs_noise = go.Figure()
                    step_3_masks = step_3_result.get("sam_masks", []) or []
                    mask_colors = generate_distinct_colors(len(step_3_masks))
                    best_ix = result.get('best_cluster_sparse_indices', {})

                    for mask_idx, cluster_labels in mask_cluster_labels.items():
                        row_ix = filtered_ix.get(mask_idx)
                        if row_ix is None or len(row_ix) == 0:
                            continue

                        mask_points = sparse_points[row_ix]
                        if mask_points is None or len(mask_points) == 0:
                            continue

                        best_row_ix = best_ix.get(mask_idx)
                        if best_row_ix is None or len(best_row_ix) == 0:
                            best_mask = np.zeros(len(row_ix), dtype=bool)
                        else:
                            best_mask = np.isin(row_ix, best_row_ix)

                        noise_mask = ~best_mask
                        color = mask_colors[mask_idx % len(mask_colors)] if mask_colors else (1.0, 0.0, 0.0)
                        color_rgb = f'rgb({int(color[0]*255)},{int(color[1]*255)},{int(color[2]*255)})'

                        best_points = mask_points[best_mask]
                        if len(best_points) > 0:
                            max_points = 3000
                            if len(best_points) > max_points:
                                indices = np.random.choice(len(best_points), max_points, replace=False)
                                best_points = best_points[indices]
                            fig_best_vs_noise.add_trace(go.Scatter3d(
                                x=best_points[:, 0],
                                y=best_points[:, 1],
                                z=best_points[:, 2],
                                mode='markers',
                                marker=dict(
                                    size=2,
                                    color=color_rgb,
                                    opacity=0.95
                                ),
                                name=f'Mask {mask_idx + 1} Best Cluster'
                            ))

                        noise_points = mask_points[noise_mask]
                        if len(noise_points) > 0:
                            max_points = 3000
                            if len(noise_points) > max_points:
                                indices = np.random.choice(len(noise_points), max_points, replace=False)
                                noise_points = noise_points[indices]
                            fig_best_vs_noise.add_trace(go.Scatter3d(
                                x=noise_points[:, 0],
                                y=noise_points[:, 1],
                                z=noise_points[:, 2],
                                mode='markers',
                                marker=dict(
                                    size=2,
                                    color=color_rgb,
                                    opacity=0.25
                                ),
                                name=f'Mask {mask_idx + 1} Noise (Non-best)'
                            ))

                    fig_best_vs_noise.update_layout(
                        title="3D Best Cluster vs Noise (Same Mask Color, Different Opacity)",
                        scene=dict(
                            xaxis_title="X (m)",
                            yaxis_title="Y (m)",
                            zaxis_title="Z (m)",
                            aspectmode='data'
                        ),
                        height=600
                    )
                    _render_point_cloud_plot(fig_best_vs_noise, "step4_best_vs_noise_by_mask")
        
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
            _cap = (
                "Indoor: deformable minimum-volume cuboid per cluster. "
                "Outdoor: fixed LLM/template dimensions + BEV scoring fit."
                if _pipeline_scene_is_indoor(st.session_state.params, point_cloud)
                else "Outdoor: fit fixed-size cuboids (LLM/template dims) with BEV scoring."
            )
            st.caption(_cap)
            if not step_4_completed:
                st.warning("⚠️ Requires Step 4")
        
        with col2:
            if st.button("▶️ Run Step 5", key="run_step_5", disabled=not step_4_completed):
                with st.spinner("Running Step 5..."):
                    try:
                        step_1_result = st.session_state.pipeline_state['step_1']['result']
                        step_2_result = st.session_state.pipeline_state['step_2']['result']
                        step_3_result = st.session_state.pipeline_state['step_3']['result']
                        step_4_result = st.session_state.pipeline_state['step_4']['result']
                        result = step_5_detection_pose_estimation(
                            sample_meta_data=sample_meta_data,
                            sparse_points=step_2_result['colored_sparse_points'],
                            best_cluster_sparse_indices=step_4_result['best_cluster_sparse_indices'],
                            filtered_mask_sparse_indices=step_4_result.get('filtered_mask_sparse_indices'),
                            sam_masks=step_3_result['sam_masks'],
                            mask_bboxes=step_3_result.get('mask_bboxes'),
                            mask_confidences=step_3_result.get('confidences'),
                            projection=step_2_result.get('projection'),
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
                        
                        # Add ground truth cuboids for annotated dataset types (KITTI and sim).
                        if dataset_type in ('kitti', 'sim', 'sunrgbd'):
                            ground_truth_boxes = sample_meta_data.get('ground_truth_boxes', [])
                            ground_truth_cuboids = []
                            for gt_box in ground_truth_boxes:
                                gt_cuboid = {
                                    'category': gt_box.get('category', gt_box.get('class', 'Unknown')),
                                    'corners': gt_box.get('corners'),
                                    'bbox_2d': gt_box.get('bbox_2d'),
                                    'truncation': gt_box.get('truncation'),
                                    'occlusion': gt_box.get('occlusion', gt_box.get('occluded')),
                                    'alpha': gt_box.get('alpha'),
                                    'bbox_height_px': gt_box.get('bbox_height_px'),
                                    'min_x': gt_box.get('min_x'),
                                    'max_x': gt_box.get('max_x'),
                                    'min_y': gt_box.get('min_y'),
                                    'max_y': gt_box.get('max_y'),
                                    'min_z': gt_box.get('min_z'),
                                    'max_z': gt_box.get('max_z'),
                                    'format': f'{dataset_type}_gt'
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
                    st.dataframe(df)
                    
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
                    _render_point_cloud_plot(fig, "step5_detected_objects")
        
        if step_5_state.get('error'):
            st.error(f"❌ Error: {step_5_state['error']}")
            
            
if __name__ == "__main__":
    main()