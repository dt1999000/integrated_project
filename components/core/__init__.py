"""
Core modules for projection, frustum management, clustering, and evaluation.
"""

# Import constants first (no dependencies)
from .constants import KITTI_CUBOID_TEMPLATES

# Then import other modules
# Note: ClusteringManager is NOT imported here to avoid circular dependencies
# Pages should import it directly: from components.core.clustering_manager import ClusteringManager
from .pointcloud_projection import Projection, PointCloud
from .evaluation import compute_3d_iou
from .pose_estimation import estimate_pose_pca, estimate_pose_l_shape, cuboid_from_pose
from .sam_integration import SAMIntegration, assign_points_to_masks

__all__ = [
    'Projection',
    'PointCloud',
    'filter_points_in_frustum',
    'compute_frustum_planes',
    # 'ClusteringManager',  # Not imported to avoid circular dependencies - import directly
    'KITTI_CUBOID_TEMPLATES',
    'compute_3d_iou',
    'estimate_pose_pca',
    'estimate_pose_l_shape',
    'cuboid_from_pose',
    'SAMIntegration',
    'assign_points_to_masks',
]

