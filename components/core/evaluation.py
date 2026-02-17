"""
Evaluation Module

Provides utilities for evaluating object detection results by matching
detected cuboids to ground truth annotations.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
import numpy as np

from shapely.geometry import Polygon

from ..dataset_loaders.kitti_dataset_loader import KITTIDatasetLoader
from .pointcloud_projection import PointCloud
from .frustum_manager import FrustumManager
from .depth_estimation import DepthEstimator
import cv2


# =============================================================================
# 3D IoU Calculation Functions
# =============================================================================

def compute_3d_iou_axis_aligned(box1: Dict, box2: Dict) -> float:
    """
    Compute 3D Intersection over Union between two axis-aligned cuboids.

    Args:
        box1: First cuboid with min_x, max_x, min_y, max_y, min_z, max_z
        box2: Second cuboid with min_x, max_x, min_y, max_y, min_z, max_z

    Returns:
        IoU value between 0 and 1
    """
    # Get intersection box
    x_min = max(box1['min_x'], box2['min_x'])
    x_max = min(box1['max_x'], box2['max_x'])
    y_min = max(box1['min_y'], box2['min_y'])
    y_max = min(box1['max_y'], box2['max_y'])
    z_min = max(box1['min_z'], box2['min_z'])
    z_max = min(box1['max_z'], box2['max_z'])

    # Check if there is an intersection
    if x_max <= x_min or y_max <= y_min or z_max <= z_min:
        return 0.0

    # Intersection volume
    intersection = (x_max - x_min) * (y_max - y_min) * (z_max - z_min)

    # Individual volumes
    vol1 = (box1['max_x'] - box1['min_x']) * (box1['max_y'] - box1['min_y']) * (box1['max_z'] - box1['min_z'])
    vol2 = (box2['max_x'] - box2['min_x']) * (box2['max_y'] - box2['min_y']) * (box2['max_z'] - box2['min_z'])

    # Union volume
    union = vol1 + vol2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def corners_to_bev_polygon(corners: np.ndarray) -> np.ndarray:
    """
    Extract bird's eye view (BEV) polygon from 8 corner points.

    Args:
        corners: 8x3 array of corner points

    Returns:
        4x2 array of BEV polygon vertices (x, y)
    """
    # Take bottom 4 corners (assuming corners 0-3 are bottom face)
    # KITTI corner ordering: 0-3 bottom, 4-7 top
    bev_corners = corners[:4, :2]  # x, y only
    return bev_corners


def compute_bev_iou_shapely(corners1: np.ndarray, corners2: np.ndarray) -> float:
    """
    Compute BEV IoU using shapely polygon intersection.

    Args:
        corners1: 8x3 array of first box corners
        corners2: 8x3 array of second box corners

    Returns:
        BEV IoU value
    """

    try:
        # Get BEV polygons
        bev1 = corners_to_bev_polygon(corners1)
        bev2 = corners_to_bev_polygon(corners2)

        # Create shapely polygons
        poly1 = Polygon(bev1)
        poly2 = Polygon(bev2)

        if not poly1.is_valid:
            poly1 = poly1.buffer(0)
        if not poly2.is_valid:
            poly2 = poly2.buffer(0)

        # Compute intersection and union
        intersection = poly1.intersection(poly2).area
        union = poly1.union(poly2).area

        if union <= 0:
            return 0.0

        return intersection / union
    except Exception:
        return 0.0


def compute_height_overlap(corners1: np.ndarray, corners2: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute height overlap between two boxes.

    Args:
        corners1: 8x3 array of first box corners
        corners2: 8x3 array of second box corners

    Returns:
        Tuple of (height_intersection, height1, height2)
    """
    # Get z ranges from corners
    z1_min, z1_max = corners1[:, 2].min(), corners1[:, 2].max()
    z2_min, z2_max = corners2[:, 2].min(), corners2[:, 2].max()

    h1 = z1_max - z1_min
    h2 = z2_max - z2_min

    # Height intersection
    z_overlap_min = max(z1_min, z2_min)
    z_overlap_max = min(z1_max, z2_max)
    height_intersection = max(0, z_overlap_max - z_overlap_min)

    return height_intersection, h1, h2


def box_to_corners(box: Dict) -> np.ndarray:
    """
    Convert axis-aligned box dict to 8 corner points.

    Args:
        box: Dict with min_x, max_x, min_y, max_y, min_z, max_z

    Returns:
        8x3 array of corner points
    """
    x0, x1 = box['min_x'], box['max_x']
    y0, y1 = box['min_y'], box['max_y']
    z0, z1 = box['min_z'], box['max_z']

    # Corner ordering matching KITTI: bottom 4, then top 4
    corners = np.array([
        [x0, y0, z0],
        [x1, y0, z0],
        [x1, y1, z0],
        [x0, y1, z0],
        [x0, y0, z1],
        [x1, y0, z1],
        [x1, y1, z1],
        [x0, y1, z1],
    ])
    return corners


def compute_3d_iou_oriented(box1: Dict, box2: Dict) -> float:
    """
    Compute oriented 3D IoU using BEV polygon intersection.

    Works with rotated boxes using their corner points.
    Falls back to axis-aligned IoU if shapely is not available.

    Args:
        box1: Cuboid dict with 'corners' (8x3) or min/max bounds
        box2: Cuboid dict with 'corners' (8x3) or min/max bounds

    Returns:
        3D IoU value between 0 and 1
    """
    # Get corners for both boxes
    if 'corners' in box1 and box1['corners'] is not None:
        corners1 = np.asarray(box1['corners'])
    else:
        corners1 = box_to_corners(box1)

    if 'corners' in box2 and box2['corners'] is not None:
        corners2 = np.asarray(box2['corners'])
    else:
        corners2 = box_to_corners(box2)

    # Compute BEV IoU
    bev_iou = compute_bev_iou_shapely(corners1, corners2)
    if bev_iou <= 0:
        return 0.0

    # Compute height overlap
    height_inter, h1, h2 = compute_height_overlap(corners1, corners2)
    if height_inter <= 0:
        return 0.0

    # Get BEV areas
    bev1 = corners_to_bev_polygon(corners1)
    bev2 = corners_to_bev_polygon(corners2)

    try:
        poly1 = Polygon(bev1)
        poly2 = Polygon(bev2)

        if not poly1.is_valid:
            poly1 = poly1.buffer(0)
        if not poly2.is_valid:
            poly2 = poly2.buffer(0)

        bev_inter = poly1.intersection(poly2).area
        bev_area1 = poly1.area
        bev_area2 = poly2.area
    except Exception:
        return compute_3d_iou_axis_aligned(box1, box2)

    # 3D intersection volume
    vol_inter = bev_inter * height_inter

    # 3D volumes
    vol1 = bev_area1 * h1
    vol2 = bev_area2 * h2

    # 3D union
    vol_union = vol1 + vol2 - vol_inter

    if vol_union <= 0:
        return 0.0

    return vol_inter / vol_union


def compute_3d_iou(box1: Dict, box2: Dict, use_oriented: bool = True) -> float:
    """
    Compute 3D IoU between two cuboids.

    Main entry point for 3D IoU calculation. Uses oriented IoU if corners
    are available and shapely is installed, otherwise falls back to
    axis-aligned calculation.

    Args:
        box1: First cuboid dict
        box2: Second cuboid dict
        use_oriented: If True, use oriented IoU when possible

    Returns:
        3D IoU value between 0 and 1
    """
    has_corners = (
        ('corners' in box1 and box1['corners'] is not None) or
        ('corners' in box2 and box2['corners'] is not None)
    )

    if use_oriented:
        return compute_3d_iou_oriented(box1, box2)
    else:
        return compute_3d_iou_axis_aligned(box1, box2)


@dataclass
class MatchResult:
    """
    Result of matching detected objects to ground truth.

    Provides computed properties for common evaluation metrics.
    """
    matches: List[Tuple[int, int, float]]  # (gt_idx, det_idx, distance)
    unmatched_gt: List[int]                # GT indices with no match
    unmatched_det: List[int]               # Detection indices with no match

    @property
    def n_matches(self) -> int:
        """Number of matched pairs (True Positives)."""
        return len(self.matches)

    @property
    def n_false_positives(self) -> int:
        """Number of false positives (detections without GT match)."""
        return len(self.unmatched_det)

    @property
    def n_false_negatives(self) -> int:
        """Number of false negatives (GT without detection match)."""
        return len(self.unmatched_gt)

    @property
    def precision(self) -> float:
        """Precision = TP / (TP + FP)"""
        total = self.n_matches + self.n_false_positives
        if total == 0:
            return 0.0
        return self.n_matches / total

    @property
    def recall(self) -> float:
        """Recall = TP / (TP + FN)"""
        total = self.n_matches + self.n_false_negatives
        if total == 0:
            return 0.0
        return self.n_matches / total

    @property
    def f1_score(self) -> float:
        """F1 Score = 2 * (Precision * Recall) / (Precision + Recall)"""
        p, r = self.precision, self.recall
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)

    def get_metrics_dict(self) -> Dict:
        """Get all metrics as a dictionary."""
        return {
            'true_positives': self.n_matches,
            'false_positives': self.n_false_positives,
            'false_negatives': self.n_false_negatives,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score
        }


class CuboidMatcher:
    """
    Matches detected cuboids to ground truth based on spatial proximity.

    Supports category-aware matching where only same-category objects
    can be matched together.
    """

    def __init__(self, max_distance: float = 5.0, match_by_category: bool = True):
        """
        Initialize CuboidMatcher.

        Args:
            max_distance: Maximum center distance for a valid match (meters)
            match_by_category: If True, only match objects of the same category
        """
        self.max_distance = max_distance
        self.match_by_category = match_by_category

    @staticmethod
    def get_cuboid_center(cuboid: Dict) -> np.ndarray:
        """
        Extract center point from cuboid dictionary.

        Args:
            cuboid: Cuboid dict with either 'corners' array or min/max bounds

        Returns:
            Center point as (3,) numpy array
        """
        if 'corners' in cuboid and cuboid['corners'] is not None:
            return cuboid['corners'].mean(axis=0)

        return np.array([
            (cuboid['min_x'] + cuboid['max_x']) / 2,
            (cuboid['min_y'] + cuboid['max_y']) / 2,
            (cuboid['min_z'] + cuboid['max_z']) / 2
        ])

    def match(
        self,
        gt_cuboids: List[Dict],
        detected_cuboids: List[Dict]
    ) -> MatchResult:
        """
        Match detected cuboids to ground truth based on center distance.

        Uses greedy matching: each detection is matched to its nearest GT
        that hasn't been matched yet (if within max_distance and same category
        when match_by_category is True).

        Args:
            gt_cuboids: List of ground truth cuboid dicts
            detected_cuboids: List of detected cuboid dicts

        Returns:
            MatchResult with matches, unmatched_gt, and unmatched_det
        """
        matches = []
        matched_gt_indices: Set[int] = set()
        matched_det_indices: Set[int] = set()

        # Match each detected cuboid to nearest GT
        for det_idx, det in enumerate(detected_cuboids):
            det_center = self.get_cuboid_center(det)
            det_category = det.get('category', 'Unknown')

            best_match_idx = None
            best_dist = self.max_distance

            for gt_idx, gt in enumerate(gt_cuboids):
                # Skip already matched GT
                if gt_idx in matched_gt_indices:
                    continue

                gt_category = gt.get('category', 'Unknown')

                # Category check if enabled
                if self.match_by_category and det_category != gt_category:
                    continue

                gt_center = self.get_cuboid_center(gt)
                dist = np.linalg.norm(det_center - gt_center)

                if dist < best_dist:
                    best_dist = dist
                    best_match_idx = gt_idx

            if best_match_idx is not None:
                matches.append((best_match_idx, det_idx, best_dist))
                matched_gt_indices.add(best_match_idx)
                matched_det_indices.add(det_idx)

        # Collect unmatched indices
        unmatched_gt = [i for i in range(len(gt_cuboids)) if i not in matched_gt_indices]
        unmatched_det = [i for i in range(len(detected_cuboids)) if i not in matched_det_indices]

        return MatchResult(
            matches=matches,
            unmatched_gt=unmatched_gt,
            unmatched_det=unmatched_det
        )

    def compute_per_category_metrics(
        self,
        gt_cuboids: List[Dict],
        detected_cuboids: List[Dict],
        match_result: MatchResult
    ) -> Dict[str, Dict]:
        """
        Compute evaluation metrics broken down by category.

        Args:
            gt_cuboids: List of ground truth cuboid dicts
            detected_cuboids: List of detected cuboid dicts
            match_result: MatchResult from match() method

        Returns:
            Dict mapping category name to metrics dict with TP, FP, FN, Precision, Recall
        """
        category_stats: Dict[str, Dict] = {}

        # Count true positives per category
        for gt_idx, det_idx, dist in match_result.matches:
            cat = gt_cuboids[gt_idx].get('category', 'Unknown')
            if cat not in category_stats:
                category_stats[cat] = {'TP': 0, 'FP': 0, 'FN': 0}
            category_stats[cat]['TP'] += 1

        # Count false negatives (unmatched GT)
        for gt_idx in match_result.unmatched_gt:
            cat = gt_cuboids[gt_idx].get('category', 'Unknown')
            if cat not in category_stats:
                category_stats[cat] = {'TP': 0, 'FP': 0, 'FN': 0}
            category_stats[cat]['FN'] += 1

        # Count false positives (unmatched detections)
        for det_idx in match_result.unmatched_det:
            cat = detected_cuboids[det_idx].get('category', 'Unknown')
            if cat not in category_stats:
                category_stats[cat] = {'TP': 0, 'FP': 0, 'FN': 0}
            category_stats[cat]['FP'] += 1

        # Compute per-category precision and recall
        for cat, stats in category_stats.items():
            tp, fp, fn = stats['TP'], stats['FP'], stats['FN']
            stats['Precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            stats['Recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        return category_stats


def run_pipeline_on_sample(
    sample_index: int,
    algorithm: str,
    params_dict: Dict
) -> Optional[Dict]:
    """
    Run the full detection pipeline on a single KITTI sample.

    Args:
        sample_index: Index of the sample to process
        algorithm: Clustering algorithm to use ('hdbscan', 'dbscan', 'optics', 'birch', 'agglomerative')
        params_dict: Parameters dict containing 'pipeline' and algorithm-specific params

    Returns dict with detected cuboids, ground truth, and metrics.
    """
    # Extract pipeline parameters
    pipeline_params = params_dict['pipeline']
    clustering_params = params_dict.get(algorithm, {})

    # Load KITTI sample
    dataset_loader = KITTIDatasetLoader(dataroot='dataset/kitti', split='training')
    dataset_loader.load_dataset()
    sample_data = dataset_loader.load_kitti_data(sample_index)

    if sample_data is None:
        print(f"Sample {sample_index}: Failed to load sample data")
        return None

    # Check if pose estimation is enabled - if so, also use depth estimation to get more points
    use_pose_estimation = pipeline_params.get('use_pose_estimation', False)
    use_depth_estimation = pipeline_params.get('use_depth_estimation', use_pose_estimation)
    
    # If depth estimation is enabled, estimate depth and reconstruct points
    original_points = sample_data['point_cloud'].copy()
    if use_depth_estimation:
        try:
            # Load image
            image_path = sample_data.get('image_path')
            if image_path:
                img = cv2.imread(image_path)
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Initialize depth estimator
                    depth_estimator = DepthEstimator(
                        use_marigold=pipeline_params.get('use_marigold', True),
                        use_full_precision=pipeline_params.get('use_full_precision', False),
                        use_tiny_vae=pipeline_params.get('use_tiny_vae', False),
                        camera_intrinsic=sample_data['camera_intrinsic'],
                        camera_to_lidar_transform=sample_data['camera_to_lidar_transform']
                    )
                    
                    # Get depth estimation parameters
                    dc_params = pipeline_params.get('marigold_dc', {})
                    use_sparse_depth = pipeline_params.get('use_sparse_depth_prior', True)
                    
                    # Try to use sparse depth prior if available
                    if use_sparse_depth and len(original_points) > 0:
                        h, w = img_rgb.shape[:2]
                        sparse_depth = depth_estimator.create_sparse_depth_map(
                            point_cloud=original_points,
                            image_shape=(h, w)
                        )
                        
                        if sparse_depth is not None:
                            n_sparse_points = np.sum(sparse_depth > 0)
                            coverage = 100 * n_sparse_points / (h * w)
                            
                            if coverage >= 0.1:  # Use sparse depth if coverage is sufficient
                                depth_map = depth_estimator.complete_depth(
                                    image=img_rgb,
                                    sparse_depth=sparse_depth,
                                    num_inference_steps=dc_params.get('num_inference_steps', 50),
                                    ensemble_size=dc_params.get('ensemble_size', 1),
                                    processing_resolution=dc_params.get('processing_resolution', 768),
                                    seed=dc_params.get('seed', 2024)
                                )
                            else:
                                depth_map = depth_estimator.get_depth_map_marigold(img_rgb)
                        else:
                            depth_map = depth_estimator.get_depth_map_marigold(img_rgb)
                    else:
                        depth_map = depth_estimator.get_depth_map_marigold(img_rgb)
                    
                    # Reconstruct points from depth
                    reconstructed_points = depth_estimator.reconstruct_points_from_depth(
                        depth_map=depth_map,
                        stride=pipeline_params.get('depth_stride', 2),
                        depth_threshold_min=pipeline_params.get('depth_threshold_min', 0.5),
                        depth_threshold_max=pipeline_params.get('depth_threshold_max', 80.0)
                    )
                    
                    # Add reconstructed points to original point cloud
                    if len(reconstructed_points) > 0:
                        original_points = np.vstack([original_points, reconstructed_points])
                        print(f"Sample {sample_index}: Added {len(reconstructed_points):,} reconstructed points from depth estimation")
        except Exception as e:
            print(f"Sample {sample_index}: Depth estimation failed: {str(e)}, using LiDAR points only")
    
    # Process point cloud (with or without reconstructed points)
    point_cloud = PointCloud(original_points)
    point_cloud.remove_ground_plane_ransac(
        distance_threshold=pipeline_params['distance_threshold'],
        ransac_n=pipeline_params['ransac_n'],
        num_iterations=pipeline_params['num_iterations'],
        filter_forward_only=pipeline_params['filter_forward_only']
    )

    ground_truth_boxes = sample_data.get('ground_truth_boxes', [])
    has_2d_bboxes = any(box.get('bbox_2d') is not None for box in ground_truth_boxes)

    if not has_2d_bboxes:
        print(f"Sample {sample_index}: No 2D bboxes available")
        return None

    if len(ground_truth_boxes) == 0:
        print(f"Sample {sample_index}: No ground truth boxes")
        return None

    # Create frustum manager
    fm = FrustumManager(
        sample_data['camera_intrinsic'],
        sample_data['camera_to_lidar_transform']
    )
    frustums = fm.create_frustums_from_bboxes(
        ground_truth_boxes,
        depth=pipeline_params.get('frustum_depth', 100)
    )

    if not frustums:
        print(f"Sample {sample_index}: No frustums created from {len(ground_truth_boxes)} GT boxes")
        return None

    # Get points
    points = point_cloud.point_cloud_plane_removed
    if len(points) == 0:
        print(f"Sample {sample_index}: No points after ground plane removal")
        return None

    # Run frustum-based clustering
    cuboids, per_frustum_results = fm.cluster_in_frustums(
        points, frustums,
        min_cluster_size=clustering_params.get('min_cluster_size', 5),
        min_samples=clustering_params.get('min_samples', 5),
        algorithm=algorithm,
        validate_overlap=pipeline_params['validate_overlap'],
        overlap_threshold=pipeline_params['overlap_threshold'],
        use_templates=pipeline_params['use_templates'],
        clustering_params={algorithm: clustering_params},
        ground_plane_model=point_cloud.ground_plane_model,
        use_pose_estimation=pipeline_params.get('use_pose_estimation', False),
        pose_estimation_method=pipeline_params.get('pose_estimation_method', 'l_shape'),
        template_dims=pipeline_params.get('template_dims', None)
    )

    print(f"Sample {sample_index}: {len(cuboids)} cuboids from {len(frustums)} frustums, {len(ground_truth_boxes)} GT")
    return {
        'sample_index': sample_index,
        'detected_cuboids': cuboids,
        'ground_truth_boxes': ground_truth_boxes,
        'n_frustums': len(frustums),
        'per_frustum_results': per_frustum_results
    }
