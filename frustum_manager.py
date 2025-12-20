"""
Frustum Manager Module

Manages frustum creation from 2D bounding boxes and frustum-based clustering operations.
Provides clean separation of concerns for frustum-related functionality.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import numpy as np

from pointcloud_projection import Projection, filter_points_in_frustum
from clustering_manager import ClusteringManager


def compute_bbox_iou(bbox1: Dict[str, float], bbox2: Dict[str, float]) -> float:
    """
    Compute Intersection over Union (IoU) between two 2D bounding boxes.

    Args:
        bbox1, bbox2: Dicts with 'left', 'top', 'right', 'bottom' keys

    Returns:
        IoU value in range [0, 1]
    """
    # Compute intersection
    x1 = max(bbox1['left'], bbox2['left'])
    y1 = max(bbox1['top'], bbox2['top'])
    x2 = min(bbox1['right'], bbox2['right'])
    y2 = min(bbox1['bottom'], bbox2['bottom'])

    if x2 <= x1 or y2 <= y1:
        return 0.0  # No intersection

    intersection = (x2 - x1) * (y2 - y1)
    area1 = (bbox1['right'] - bbox1['left']) * (bbox1['bottom'] - bbox1['top'])
    area2 = (bbox2['right'] - bbox2['left']) * (bbox2['bottom'] - bbox2['top'])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


@dataclass
class Frustum:
    """Data class representing a single frustum projected from a 2D bounding box."""
    idx: int
    camera_origin: np.ndarray  # (3,) apex point in LiDAR coordinates
    base_corners: np.ndarray   # (4, 3) base corners [TL, TR, BR, BL] in LiDAR coords
    category: str
    bbox_2d: Dict[str, float]  # {'left', 'top', 'right', 'bottom'}

    def to_dict(self) -> Dict:
        """Convert to dictionary for session state storage."""
        return {
            'idx': self.idx,
            'camera_origin': self.camera_origin,
            'base_corners': self.base_corners,
            'category': self.category,
            'bbox_2d': self.bbox_2d
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Frustum':
        """Create Frustum from dictionary."""
        return cls(
            idx=data['idx'],
            camera_origin=np.array(data['camera_origin']),
            base_corners=np.array(data['base_corners']),
            category=data['category'],
            bbox_2d=data['bbox_2d']
        )


@dataclass
class FrustumClusterResult:
    """Result of clustering within a single frustum."""
    frustum_idx: int
    category: str
    points: np.ndarray         # Nx3 points in this frustum
    labels: np.ndarray         # Cluster labels (-1 for noise)
    n_points: int
    n_clusters: int
    status: str                # 'success', 'too_few_points', 'error: ...'

    def to_dict(self) -> Dict:
        """Convert to dictionary for session state storage."""
        return {
            'frustum_idx': self.frustum_idx,
            'category': self.category,
            'points': self.points,
            'labels': self.labels,
            'n_points': self.n_points,
            'n_clusters': self.n_clusters,
            'status': self.status
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'FrustumClusterResult':
        """Create FrustumClusterResult from dictionary."""
        return cls(
            frustum_idx=data['frustum_idx'],
            category=data['category'],
            points=np.array(data['points']),
            labels=np.array(data['labels']),
            n_points=data['n_points'],
            n_clusters=data['n_clusters'],
            status=data['status']
        )


class FrustumManager:
    """
    Manages frustum creation and frustum-based clustering operations.

    This class decouples frustum logic from UI code, making it reusable
    and testable independently.
    """

    def __init__(self, camera_intrinsic: np.ndarray, camera_to_lidar_transform: np.ndarray):
        """
        Initialize FrustumManager with camera calibration.

        Args:
            camera_intrinsic: 3x3 camera intrinsic matrix (K)
            camera_to_lidar_transform: 4x4 transformation from camera to LiDAR coordinates
        """
        self.camera_intrinsic = camera_intrinsic
        self.camera_to_lidar_transform = camera_to_lidar_transform
        self._projection: Optional[Projection] = None

    @property
    def projection(self) -> Projection:
        """Lazy-load Projection instance."""
        if self._projection is None:
            dummy_pc = np.zeros((1, 3))
            self._projection = Projection(
                camera_intrinsic=self.camera_intrinsic,
                camera_extrinsic=np.eye(4),
                camera_to_lidar_transform=self.camera_to_lidar_transform,
                point_cloud=dummy_pc
            )
        return self._projection

    def create_frustums_from_bboxes(self, bboxes: List[Dict], depth: float = 100.0) -> List[Frustum]:
        """
        Create Frustum objects from 2D bounding boxes.

        Args:
            bboxes: List of dicts with 'bbox_2d' key containing
                    {'left', 'top', 'right', 'bottom'} and optional 'category'
            depth: Frustum depth in meters (how far to project)

        Returns:
            List of Frustum objects
        """
        frustums = []

        for i, bbox in enumerate(bboxes):
            bbox_2d = bbox.get('bbox_2d')
            if bbox_2d is None:
                continue

            category = bbox.get('category', 'Unknown')

            # Project 2D bbox corners to 3D frustum
            camera_origin, base_corners = self.projection.project_bbox_corners_to_3d(
                bbox_2d, depth=depth
            )

            frustums.append(Frustum(
                idx=i,
                camera_origin=camera_origin,
                base_corners=base_corners,
                category=category,
                bbox_2d=bbox_2d
            ))

        return frustums

    def find_best_cuboid(
        self,
        cuboids: List[Dict],
        points: np.ndarray,
        labels: np.ndarray,
        original_bbox_2d: Dict[str, float],
        overlap_threshold: float = 0.3
    ) -> Optional[Dict]:
        """
        Find the best cuboid by projecting each to 2D and selecting the one with highest IoU.

        Projects all axis-aligned cuboids to 2D and returns the one with highest IoU overlap
        with the original 2D bounding box. Note: 'need_review' is NOT set here - it should
        be computed after template cuboid generation in cluster_in_frustums().

        Args:
            cuboids: List of cuboid dicts from clustering
            points: Nx3 array of points in the frustum
            labels: N array of cluster labels
            original_bbox_2d: Dict with 'left', 'top', 'right', 'bottom' (the original 2D detection)
            overlap_threshold: IoU threshold (unused here, kept for API compatibility)

        Returns:
            Best cuboid dict with added 'selected_points', 'selected_labels', 'iou',
            and 'projected_bbox_2d' keys, or None if no valid cuboids
        """
        if not cuboids:
            return None

        # Iterate from nearest to farthest
        best_cuboid = {'iou': 0}
        for cuboid in cuboids:
            
            # Project cuboid to 2D
            projected = self.projection.cuboid_to_2d(cuboid)
            if projected is None:
                print(f'cuboid is behind camera')
                continue  # Cuboid is behind camera

            projected_bbox = projected['bbox_2d']
            
            # Compute IoU between projected bbox and original detection bbox
            iou = compute_bbox_iou(projected_bbox, original_bbox_2d)
            if iou > best_cuboid['iou']:
                best_cuboid = cuboid
                best_cuboid['iou'] = iou
                best_cuboid['projected_bbox_2d'] = projected_bbox
                best_cuboid['selected_points'] = points[labels == best_cuboid['label']]
                best_cuboid['selected_labels'] = labels[labels == best_cuboid['label']]

        print(f"best cuboid (axis-aligned): iou={best_cuboid['iou']}, source bbox idx: {best_cuboid['source_bbox_idx']}")
        return best_cuboid

    def cluster_in_frustums(
        self,
        point_cloud: np.ndarray,
        frustums: List[Frustum],
        min_cluster_size: int = 15,
        min_samples: int = 5,
        algorithm: str = "hdbscan",
        validate_overlap: bool = False,
        overlap_threshold: float = 0.7,
        use_templates: bool = True,
        clustering_params: Optional[Dict[str, Dict]] = None,
        ground_plane_model: Optional[np.ndarray] = None
    ) -> Tuple[List[Dict], List[FrustumClusterResult]]:
        """
        Run clustering on points inside each frustum and generate cuboids.

        Args:
            point_cloud: Nx3 array of 3D points (typically ground-removed)
            frustums: List of Frustum objects from create_frustums_from_bboxes()
            min_cluster_size: Minimum points for a valid cluster
            min_samples: Minimum samples for density estimation (HDBSCAN/DBSCAN)
            algorithm: Clustering algorithm ('hdbscan', 'birch', 'agglomerative', 'optics' or 'dbscan')
            validate_overlap: If True, validate cuboids by projecting back to 2D
                             and checking overlap with original 2D bbox
            overlap_threshold: Minimum IoU required to accept a cuboid (0.0-1.0)
            use_templates: If True, use class-specific cuboid templates instead of
                          axis-aligned bounding boxes from cluster points
            clustering_params: Optional dictionary of algorithm-specific parameters from UI.
                              Structure: {'hdbscan': {...}, 'dbscan': {...}, ...}
                              These override the default ClusteringManager params.
            ground_plane_model: Optional [a, b, c, d] plane equation from RANSAC.
                               Used to compute ground z for template cuboids.

        Returns:
            Tuple of:
                - cuboids: List of cuboid dicts with category info
                - results: List of FrustumClusterResult objects
        """
        all_cuboids = []
        results = []

        for frustum in frustums:
            # Filter points inside this frustum
            points_in_frustum, mask = filter_points_in_frustum(
                point_cloud,
                frustum.camera_origin,
                frustum.base_corners
            )

            n_points = len(points_in_frustum)

            # Check minimum points requirement
            if n_points < min_cluster_size:
                print(f"n_points: {n_points} is less than min_cluster_size: {min_cluster_size}")
                results.append(FrustumClusterResult(
                    frustum_idx=frustum.idx,
                    category=frustum.category,
                    points=points_in_frustum,
                    labels=np.array([]),
                    n_points=n_points,
                    n_clusters=0,
                    status='too_few_points'
                ))
                continue

            # Run clustering
            try:
                # Create ClusteringManager with provided params
                cluster_manager = ClusteringManager(points_in_frustum, params=clustering_params)

                # Run clustering using the unified method with stored params
                # Override min_cluster_size and min_samples from function args
                override_params = {
                    'min_cluster_size': min_cluster_size,
                    'min_samples': min_samples
                }
                labels = cluster_manager.run_clustering(algorithm, **override_params)
                print(f"labels: {np.unique(labels)}")
                n_clusters = len(np.unique(labels))
                # First generate axis-aligned cuboids for selection/validation
                cuboids = cluster_manager.generate_cuboids_from_clusters(labels)

                # Add category info to each cuboid
                for cuboid in cuboids:
                    cuboid['category'] = frustum.category
                    cuboid['source_bbox_idx'] = frustum.idx
                # Find best cuboid via overlap validation, then create template cuboid
                if validate_overlap and cuboids:
                    selected = self.find_best_cuboid(
                        cuboids,
                        points_in_frustum,
                        labels,
                        frustum.bbox_2d,
                        overlap_threshold
                    )
                    if selected and use_templates:
                        # Create template cuboid for the selected cluster
                        cluster_label = selected.get('label', -1)
                        if cluster_label >= 0:
                            cluster_points = points_in_frustum[labels == cluster_label]

                            # Compute ground_z at cluster centroid using plane model
                            ground_z = None
                            if ground_plane_model is not None:
                                a, b, c, d = ground_plane_model
                                if abs(c) > 1e-6:
                                    center_x = np.mean(cluster_points[:, 0])
                                    center_y = np.mean(cluster_points[:, 1])
                                    ground_z = -(a * center_x + b * center_y + d) / c

                            template_cuboid = cluster_manager.generate_cuboid_from_template(
                                cluster_points, frustum.category, cluster_label, ground_z=ground_z
                            )
                            if template_cuboid:
                                template_cuboid['category'] = frustum.category
                                template_cuboid['source_bbox_idx'] = frustum.idx
                                template_cuboid['selected_points'] = selected.get('selected_points')
                                template_cuboid['selected_labels'] = selected.get('selected_labels')

                                # Project template cuboid back to 2D and recalculate IoU
                                template_projected = self.projection.cuboid_to_2d(template_cuboid)
                                if template_projected is not None:
                                    template_bbox_2d = template_projected['bbox_2d']
                                    template_iou = compute_bbox_iou(template_bbox_2d, frustum.bbox_2d)
                                    template_cuboid['projected_bbox_2d'] = template_bbox_2d
                                    template_cuboid['iou'] = template_iou
                                    template_cuboid['need_review'] = template_iou < overlap_threshold
                                    print(f"template cuboid: iou={template_iou:.3f}, need_review={template_cuboid['need_review']}")
                                else:
                                    # Template cuboid is behind camera, use axis-aligned values
                                    template_cuboid['projected_bbox_2d'] = selected.get('projected_bbox_2d')
                                    template_cuboid['iou'] = selected.get('iou')
                                    template_cuboid['need_review'] = True

                                cuboid = template_cuboid
                            else:
                                cuboid = selected
                        else:
                            cuboid = selected
                    elif selected:
                        # Not using templates - set need_review based on axis-aligned IoU
                        selected['need_review'] = selected.get('iou', 0) < overlap_threshold
                        cuboid = selected
                    else:
                        cuboid = None  # No cuboid met the overlap threshold

                if cuboid:
                    all_cuboids.append(cuboid)

                results.append(FrustumClusterResult(
                    frustum_idx=frustum.idx,
                    category=frustum.category,
                    points=points_in_frustum,
                    labels=labels,
                    n_points=n_points,
                    n_clusters=n_clusters,
                    status=f"need_review: {cuboid.get('need_review') if cuboid else 'no_cuboid'}"
                ))
            except Exception as e:
                results.append(FrustumClusterResult(
                    frustum_idx=frustum.idx,
                    category=frustum.category,
                    points=points_in_frustum,
                    labels=np.array([]),
                    n_points=n_points,
                    n_clusters=0,
                    status=f'error: {str(e)}'
                ))

        return all_cuboids, results

    @staticmethod
    def combine_cluster_results(
        results: List[FrustumClusterResult]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Combine per-frustum cluster results for visualization with distinct colors.

        Each cluster gets a unique global label to ensure different colors
        across all frustums.

        Args:
            results: List of FrustumClusterResult objects

        Returns:
            Tuple of (combined_points, combined_labels) for visualization
            Returns (None, None) if no valid clusters
        """
        all_points = []
        all_labels = []
        current_label_offset = 0

        for result in results:
            if result.status != 'success' or result.n_clusters == 0:
                continue

            points = result.points
            labels = result.labels

            if len(points) == 0 or len(labels) == 0:
                continue

            # Offset labels to make them unique across frustums
            # -1 (noise) stays as -1, other labels get offset
            adjusted_labels = np.where(
                labels >= 0,
                labels + current_label_offset,
                labels
            )

            all_points.append(points)
            all_labels.append(adjusted_labels)

            # Update offset for next frustum
            max_label = labels.max() if len(labels) > 0 else -1
            if max_label >= 0:
                current_label_offset += max_label + 1

        if len(all_points) == 0:
            return None, None

        combined_points = np.vstack(all_points)
        combined_labels = np.concatenate(all_labels)

        return combined_points, combined_labels

    @staticmethod
    def results_to_bbox_summary(results: List[FrustumClusterResult]) -> List[Dict]:
        """
        Convert FrustumClusterResult list to summary dicts for display.

        Args:
            results: List of FrustumClusterResult objects

        Returns:
            List of summary dicts with keys: bbox_idx, category, points, clusters, status
        """
        return [
            {
                'bbox_idx': r.frustum_idx,
                'category': r.category,
                'points': r.n_points,
                'clusters': r.n_clusters,
                'status': r.status
            }
            for r in results
        ]
