"""
2D to 3D Projection and Point Cloud Visualization
This module provides classes for projecting 2D pixels to 3D rays and visualizing point clouds.
"""

import numpy as np
import cv2
from typing import Any, List, Tuple, Optional, Dict
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import open3d as o3d

# Import the new clustering manager
from clustering_manager import ClusteringManager


# =============================================================================
# Frustum Filtering Utility Functions
# =============================================================================

def compute_frustum_planes(camera_origin: np.ndarray,
                           base_corners: np.ndarray) -> List[np.ndarray]:
    """
    Compute the 5 plane equations for a frustum pyramid.

    A frustum is bounded by 4 side planes (connecting apex to base edges)
    and 1 base plane (the far end at specified depth).

    Args:
        camera_origin: (3,) apex of frustum (camera center in LiDAR coords)
        base_corners: (4, 3) base corners [TL, TR, BR, BL] in LiDAR coords

    Returns:
        List of 5 plane equations as [a, b, c, d] where ax + by + cz + d = 0
    """
    planes = []

    # 4 side planes: each from apex + 2 adjacent base corners
    # Order: TL-TR, TR-BR, BR-BL, BL-TL
    for i in range(4):
        p1 = camera_origin
        p2 = base_corners[i]
        p3 = base_corners[(i + 1) % 4]

        # Compute plane normal using cross product
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm_length = np.linalg.norm(normal)
        if norm_length > 1e-10:
            normal = normal / norm_length
        else:
            # Degenerate case: skip this plane
            continue

        # Plane equation: normal · (point - p1) = 0
        # ax + by + cz + d = 0 where d = -normal · p1
        d = -np.dot(normal, p1)
        planes.append(np.array([normal[0], normal[1], normal[2], d]))

    # Base plane: defined by 3 base corners
    v1 = base_corners[1] - base_corners[0]
    v2 = base_corners[2] - base_corners[0]
    normal = np.cross(v1, v2)
    norm_length = np.linalg.norm(normal)
    if norm_length > 1e-10:
        normal = normal / norm_length

        # Ensure normal points toward camera (so points between camera and base are inside)
        to_camera = camera_origin - base_corners[0]
        if np.dot(normal, to_camera) < 0:
            normal = -normal

        d = -np.dot(normal, base_corners[0])
        planes.append(np.array([normal[0], normal[1], normal[2], d]))

    return planes


def filter_points_in_frustum(points: np.ndarray,
                             camera_origin: np.ndarray,
                             base_corners: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter points to keep only those inside a frustum pyramid.

    Uses half-space method: a point is inside if it's on the correct side
    of all 5 bounding planes.

    Args:
        points: Nx3 array of 3D points
        camera_origin: (3,) frustum apex (camera center)
        base_corners: (4, 3) frustum base corners [TL, TR, BR, BL]

    Returns:
        filtered_points: Mx3 array of points inside frustum
        mask: N boolean array indicating which points are inside
    """
    if len(points) == 0:
        return np.array([]).reshape(0, 3), np.array([], dtype=bool)

    planes = compute_frustum_planes(camera_origin, base_corners)

    if len(planes) == 0:
        # No valid planes, return empty
        return np.array([]).reshape(0, 3), np.zeros(len(points), dtype=bool)

    # For each plane, compute signed distance
    # Point is inside if on correct side of ALL planes
    n_points = len(points)
    inside = np.ones(n_points, dtype=bool)

    # Determine correct sign by testing centroid of frustum
    centroid = (camera_origin + base_corners.mean(axis=0)) / 2

    for plane in planes:
        normal = plane[:3]
        d = plane[3]

        # Signed distance: normal · point + d
        distances = points @ normal + d
        centroid_dist = np.dot(centroid, normal) + d

        # Points should be on same side as centroid
        if centroid_dist >= 0:
            inside &= (distances >= 0)
        else:
            inside &= (distances <= 0)

    return points[inside], inside


def filter_points_in_multiple_frustums(points: np.ndarray,
                                       frustums: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter points to keep those inside ANY of the provided frustums.

    Args:
        points: Nx3 array of 3D points
        frustums: List of (camera_origin, base_corners) tuples

    Returns:
        filtered_points: Mx3 array of points inside any frustum
        mask: N boolean array indicating which points are inside
    """
    if len(points) == 0 or len(frustums) == 0:
        return np.array([]).reshape(0, 3), np.zeros(len(points), dtype=bool)

    combined_mask = np.zeros(len(points), dtype=bool)

    for camera_origin, base_corners in frustums:
        _, mask = filter_points_in_frustum(points, camera_origin, base_corners)
        combined_mask |= mask

    return points[combined_mask], combined_mask


class Projection:
    """
    Bidirectional projection class for 2D↔3D transformations using camera intrinsics,
    extrinsics, and LiDAR transformations.

    Supports:
    - 2D to 3D: pixel_to_ray, find_closest_point_on_ray, project_pixels_to_3d
    - 3D to 2D: point_to_pixel, cuboid_to_2d, points_to_pixels
    """
    
    def __init__(self, camera_intrinsic: np.ndarray, camera_extrinsic: np.ndarray,
                 camera_to_lidar_transform: np.ndarray, point_cloud: np.ndarray,
                 image: Optional[np.ndarray] = None):
        """
        Initialize the 2D to 3D projection class.
        
        Args:
            camera_intrinsic: 3x3 camera intrinsic matrix (K)
            camera_extrinsic: 4x4 camera extrinsic matrix (world to camera)
            camera_to_lidar_transform: 4x4 transformation matrix from camera to lidar coordinates
            point_cloud: Nx3 or Nx4 array of point cloud points (x, y, z) or (x, y, z, intensity)
            image: Optional 2D image corresponding to the point cloud
        """
        self.camera_intrinsic = camera_intrinsic.astype(np.float32)
        self.camera_extrinsic = camera_extrinsic.astype(np.float32)
        self.camera_to_lidar_transform = camera_to_lidar_transform.astype(np.float32)
        self.point_cloud = point_cloud[:, :3] if point_cloud.shape[1] > 3 else point_cloud
        self.image = image
        
        # Compute inverse transformations
        self.lidar_to_camera_transform = np.linalg.inv(self.camera_to_lidar_transform)
        self.camera_to_world_transform = np.linalg.inv(self.camera_extrinsic)
        
        # Compute coordinate system origins and axes
        # Lidar coordinate system (origin at lidar position, typically at origin)
        self.lidar_origin = np.array([0, 0, 0])
        self.lidar_axes = np.eye(3)  # X, Y, Z axes
        
        # Camera coordinate system in lidar coordinates
        camera_center_cam = np.array([0, 0, 0, 1])
        self.camera_origin = (self.camera_to_lidar_transform @ camera_center_cam)[:3]
        
        # Camera axes in lidar coordinates (X-right, Y-down, Z-forward in camera coords)
        camera_axes_cam = np.eye(3)
        camera_axes_lidar = (self.camera_to_lidar_transform[:3, :3] @ camera_axes_cam.T).T
        self.camera_axes = camera_axes_lidar
        
    def pixel_to_ray(self, pixel_coords: np.ndarray) -> np.ndarray:
        """
        Convert 2D pixel coordinates to 3D rays in lidar coordinate system.
        
        Args:
            pixel_coords: Nx2 array of pixel coordinates (u, v)
            
        Returns:
            Dictionary containing:
                - 'origins': Nx3 array of ray origins in lidar coordinates
                - 'directions': Nx3 array of normalized ray directions in lidar coordinates
        """
        if pixel_coords.ndim == 1:
            pixel_coords = pixel_coords.reshape(1, -1)
        
        num_pixels = pixel_coords.shape[0]
        rays = {
            'origins': np.zeros((num_pixels, 3)),
            'directions': np.zeros((num_pixels, 3))
        }
        
        # Camera center in camera coordinates (origin)
        camera_center_cam = np.array([0, 0, 0, 1])
        
        # Transform camera center to lidar coordinates
        camera_center_lidar = (self.camera_to_lidar_transform @ camera_center_cam)[:3]
        
        for i, (u, v) in enumerate(pixel_coords):
            # Convert pixel to normalized camera coordinates
            pixel_homogeneous = np.array([u, v, 1.0])
            
            # Back-project to camera coordinate system (3D point at z=1)
            K_inv = np.linalg.inv(self.camera_intrinsic)
            point_cam = K_inv @ pixel_homogeneous
            
            # Normalize to get direction vector in camera coordinates
            direction_cam = point_cam / np.linalg.norm(point_cam)
            direction_cam_homogeneous = np.append(direction_cam, 0)
            
            # Transform direction to lidar coordinates
            direction_lidar = (self.camera_to_lidar_transform[:3, :3] @ direction_cam)
            direction_lidar = direction_lidar / np.linalg.norm(direction_lidar)
            
            # Ray origin is camera center in lidar coordinates
            rays['origins'][i] = camera_center_lidar
            rays['directions'][i] = direction_lidar
        
        return rays
    
    def find_closest_point_on_ray(self, rays: Dict[str, np.ndarray], 
                                  max_distance: float = 100.0,
                                  distance_threshold: float = 0.05) -> np.ndarray:
        """
        Find the closest point in the point cloud to each ray and project it onto the ray.
        
        Args:
            rays: Dictionary with 'origins' (Nx3) and 'directions' (Nx3) arrays
            max_distance: Maximum distance along ray to search for points
            distance_threshold: Maximum perpendicular distance from ray to consider a point
            
        Returns:
            Nx3 array of projected points on rays (closest to point cloud points)
        """
        origins = rays['origins']
        directions = rays['directions']
        num_rays = origins.shape[0]
        projected_points = np.zeros((num_rays, 3))
        
        for i in range(num_rays):
            origin = origins[i]
            direction = directions[i]
            
            # Find points within max_distance from ray origin
            distances_from_origin = np.linalg.norm(self.point_cloud - origin, axis=1)
            nearby_mask = distances_from_origin < max_distance
            nearby_points = self.point_cloud[nearby_mask]
            
            if len(nearby_points) == 0:
                # No nearby points, extend ray to max_distance
                projected_points[i] = origin + direction * max_distance
                continue
            
            # For each nearby point, find closest point on ray
            # Point on ray: origin + t * direction
            # Distance from point to ray: ||(point - origin) - ((point - origin) · direction) * direction||
            
            vectors_to_points = nearby_points - origin
            t_values = np.dot(vectors_to_points, direction)
            t_values = np.clip(t_values, 0, max_distance)  # Only consider forward direction
            
            points_on_ray = origin + t_values[:, np.newaxis] * direction
            distances_to_ray = np.linalg.norm(nearby_points - points_on_ray, axis=1)
            
            # Find point with minimum distance to ray
            min_idx = np.argmin(distances_to_ray)
            
            if distances_to_ray[min_idx] < distance_threshold:
                # Use the point on ray closest to the nearest point cloud point
                projected_points[i] = points_on_ray[min_idx]
            else:
                # No point close enough, extend ray to max_distance
                projected_points[i] = origin + direction * max_distance
        
        return projected_points
    

    def project_pixels_to_3d(self, pixel_coords: np.ndarray, 
                            max_distance: float = 100.0,
                            distance_threshold: float = 0.5) -> Dict[str, np.ndarray]:
        """
        Complete pipeline: pixels -> rays -> 3D points.
        
        Args:
            pixel_coords: Nx2 array of pixel coordinates (u, v)
            max_distance: Maximum distance along ray to search for points
            distance_threshold: Maximum perpendicular distance from ray to consider a point
            
        Returns:
            Dictionary containing:
                - 'rays': Dictionary with 'origins' and 'directions'
                - 'projected_points': Nx3 array of 3D points on rays
        """
        rays = self.pixel_to_ray(pixel_coords)
        projected_points = self.find_closest_point_on_ray(rays, max_distance, distance_threshold)
        return {
            'rays': rays,
            'projected_points': projected_points
        }
    
    def get_coordinate_systems(self) -> Dict[str, np.ndarray]:
        """
        Get coordinate system origins and axes for visualization.

        Returns:
            Dictionary with 'lidar_origin', 'lidar_axes', 'camera_origin', 'camera_axes'
        """
        return {
            'lidar_origin': self.lidar_origin,
            'lidar_axes': self.lidar_axes,
            'camera_origin': self.camera_origin,
            'camera_axes': self.camera_axes
        }

    def project_bbox_corners_to_3d(self, bbox_2d: Dict,
                                    depth: float = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project 2D bounding box corners to 3D using ray casting.
        Creates a frustum/pyramid by projecting the 4 corners of a 2D bbox to 3D.

        Args:
            bbox_2d: Dict with 'left', 'top', 'right', 'bottom' pixel coords
            depth: Distance to extend rays (meters)

        Returns:
            camera_origin: np.ndarray (3,) - Camera center in LiDAR coords (apex of frustum)
            base_corners: np.ndarray (4, 3) - Projected corner positions in LiDAR coords (base of frustum)
        """
        # Get camera origin in LiDAR coordinates
        camera_center_cam = np.array([0, 0, 0, 1])
        camera_origin = (self.camera_to_lidar_transform @ camera_center_cam)[:3]

        # Extract 4 corners from 2D bbox (in pixel coordinates)
        corners_2d = np.array([
            [bbox_2d['left'], bbox_2d['top']],      # Top-left
            [bbox_2d['right'], bbox_2d['top']],     # Top-right
            [bbox_2d['right'], bbox_2d['bottom']],  # Bottom-right
            [bbox_2d['left'], bbox_2d['bottom']]    # Bottom-left
        ])

        # Inverse camera intrinsic for back-projection
        K_inv = np.linalg.inv(self.camera_intrinsic)

        base_corners = []
        for u, v in corners_2d:
            # Back-project pixel to camera coordinates (normalized at z=1)
            pixel_homogeneous = np.array([u, v, 1.0])
            point_cam = K_inv @ pixel_homogeneous

            # Normalize to get direction vector in camera coordinates
            direction_cam = point_cam / np.linalg.norm(point_cam)

            # Transform direction to LiDAR coordinates
            direction_lidar = self.camera_to_lidar_transform[:3, :3] @ direction_cam
            direction_lidar = direction_lidar / np.linalg.norm(direction_lidar)

            # Extend ray to specified depth
            corner_3d = camera_origin + direction_lidar * depth
            base_corners.append(corner_3d)

        return camera_origin, np.array(base_corners)

    def get_frustums_from_bboxes(self, bboxes: List[Dict],
                                 depth: float = 30.0) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate frustum data for multiple 2D bounding boxes.

        Args:
            bboxes: List of bbox dicts with 'bbox_2d' key containing
                    {'left', 'top', 'right', 'bottom'} pixel coordinates
            depth: Frustum depth in meters

        Returns:
            List of (camera_origin, base_corners) tuples where:
                - camera_origin: (3,) camera center in LiDAR coords
                - base_corners: (4, 3) frustum base corners in LiDAR coords
        """
        frustums = []
        for bbox in bboxes:
            bbox_2d = bbox.get('bbox_2d')
            if bbox_2d is None:
                continue
            camera_origin, base_corners = self.project_bbox_corners_to_3d(bbox_2d, depth)
            frustums.append((camera_origin, base_corners))
        return frustums

    # =========================================================================
    # 3D to 2D Projection Methods
    # =========================================================================

    def point_to_pixel(self, points_3d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project 3D points (in LiDAR coordinates) to 2D pixel coordinates.

        Args:
            points_3d: Nx3 array of 3D points in LiDAR coordinates

        Returns:
            pixels: Nx2 array of pixel coordinates (u, v)
            valid_mask: N boolean array indicating which points project in front of camera
        """
        if points_3d.ndim == 1:
            points_3d = points_3d.reshape(1, -1)

        n_points = len(points_3d)

        # Convert to homogeneous coordinates
        points_homo = np.hstack([points_3d, np.ones((n_points, 1))])

        # Transform from LiDAR to camera coordinates
        points_cam = (self.lidar_to_camera_transform @ points_homo.T).T[:, :3]

        # Filter points behind camera (z <= 0 in camera coords)
        valid_mask = points_cam[:, 2] > 0

        # Initialize output
        pixels = np.zeros((n_points, 2))

        # Project valid points to image plane: pixel = K @ [x/z, y/z, 1]
        if np.any(valid_mask):
            z = points_cam[valid_mask, 2:3]
            normalized = points_cam[valid_mask, :2] / z
            pixels_homo = np.hstack([normalized, np.ones((np.sum(valid_mask), 1))])
            pixels[valid_mask] = (self.camera_intrinsic @ pixels_homo.T).T[:, :2]

        return pixels, valid_mask

    def cuboid_to_2d(self, cuboid: Dict) -> Optional[Dict]:
        """
        Project a 3D cuboid to 2D bounding box and projected corners.

        Args:
            cuboid: Dict with either 'corners' (8x3) or min/max bounds
                    (min_x, max_x, min_y, max_y, min_z, max_z)

        Returns:
            Dict with:
                - 'bbox_2d': {'left', 'top', 'right', 'bottom'} in pixels
                - 'corners_2d': 8x2 array of projected corner pixels
                - 'valid_mask': 8 boolean array for which corners are visible
                - 'visible': boolean indicating if cuboid is at least partially visible
            Returns None if cuboid is entirely behind camera
        """
        # Get 8 corners of cuboid
        if 'corners' in cuboid and cuboid['corners'] is not None:
            corners_3d = np.array(cuboid['corners'])
        else:
            # Build corners from min/max bounds
            min_x, max_x = cuboid['min_x'], cuboid['max_x']
            min_y, max_y = cuboid['min_y'], cuboid['max_y']
            min_z, max_z = cuboid['min_z'], cuboid['max_z']
            corners_3d = np.array([
                [min_x, min_y, min_z], [max_x, min_y, min_z],
                [max_x, max_y, min_z], [min_x, max_y, min_z],
                [min_x, min_y, max_z], [max_x, min_y, max_z],
                [max_x, max_y, max_z], [min_x, max_y, max_z]
            ])

        # Project corners to 2D
        corners_2d, valid_mask = self.point_to_pixel(corners_3d)

        # If no corners visible, return None
        if not np.any(valid_mask):
            return None

        # Get 2D bounding box from visible corners
        visible_corners = corners_2d[valid_mask]
        bbox_2d = {
            'left': float(np.min(visible_corners[:, 0])),
            'top': float(np.min(visible_corners[:, 1])),
            'right': float(np.max(visible_corners[:, 0])),
            'bottom': float(np.max(visible_corners[:, 1]))
        }

        return {
            'bbox_2d': bbox_2d,
            'corners_2d': corners_2d,
            'valid_mask': valid_mask,
            'visible': True
        }

    def points_to_pixels(self, points_3d: np.ndarray,
                         image_shape: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project 3D points to 2D and optionally filter by image bounds.

        Args:
            points_3d: Nx3 array of 3D points in LiDAR coordinates
            image_shape: Optional (height, width) to filter points outside image

        Returns:
            pixels: Mx2 array of valid pixel coordinates
            indices: M array of indices into original points array
        """
        pixels, valid_mask = self.point_to_pixel(points_3d)

        if image_shape is not None:
            h, w = image_shape
            in_bounds = (
                (pixels[:, 0] >= 0) & (pixels[:, 0] < w) &
                (pixels[:, 1] >= 0) & (pixels[:, 1] < h)
            )
            valid_mask &= in_bounds

        indices = np.where(valid_mask)[0]
        return pixels[valid_mask], indices

    def project_cuboids_to_2d(self, cuboids: List[Dict]) -> List[Optional[Dict]]:
        """
        Project multiple 3D cuboids to 2D.

        Args:
            cuboids: List of cuboid dicts

        Returns:
            List of 2D projection results (None for cuboids behind camera)
        """
        return [self.cuboid_to_2d(cuboid) for cuboid in cuboids]


# Backward compatibility alias
Projection2DTo3D = Projection


class PointCloud:
    """
    Class for representing a point cloud.
    """
    def __init__(self, point_cloud: np.ndarray, coordinate_systems: Optional[Dict[str, np.ndarray]] = None):
        self.original_point_cloud = point_cloud[:, :3] if point_cloud.shape[1] > 3 else point_cloud
        self.coordinate_systems = coordinate_systems
        self.ground_removed = False
        
    def copy(self):
        """
        Create a copy of the point cloud.
        
        Returns:
            A new PointCloud instance with the same data
        """
        new_point_cloud = PointCloud(self.original_point_cloud.copy(), 
                                    self.coordinate_systems.copy() if self.coordinate_systems else None)
        
        # Copy additional attributes if they exist
        if self.ground_removed:
            new_point_cloud.ground_removed = True
            new_point_cloud.point_cloud_plane_removed = self.point_cloud_plane_removed.copy()
            if hasattr(self, 'ground_plane_model'):
                new_point_cloud.ground_plane_model = self.ground_plane_model.copy()
            if hasattr(self, 'ground_inliers'):
                new_point_cloud.ground_inliers = self.ground_inliers.copy()
        
        return new_point_cloud

    def filter_forward_points(self, points: np.ndarray) -> np.ndarray:
        """
        Filter points to only include those in front of the vehicle (positive x in LiDAR coords).
        This is necessary because KITTI only has forward-facing cameras.

        Args:
            points: Nx3 array of points

        Returns:
            Filtered Nx3 array with only forward-facing points
        """
        forward_mask = points[:, 0] > 0  # Keep only positive x values
        filtered_points = points[forward_mask]

        n_removed = len(points) - len(filtered_points)
        if n_removed > 0:
            print(f"Forward-facing filter:")
            print(f"  Removed {n_removed} points behind/beside vehicle (x <= 0)")
            print(f"  Remaining forward-facing points: {len(filtered_points)}")

        return filtered_points

    def filter_by_frustums(self, frustums: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter point cloud to keep only points inside frustum regions.

        Args:
            frustums: List of (camera_origin, base_corners) tuples where:
                - camera_origin: (3,) frustum apex
                - base_corners: (4, 3) frustum base corners

        Returns:
            filtered_points: Mx3 array of points inside any frustum
            mask: N boolean array indicating which points are inside
        """
        points = self.point_cloud_plane_removed if self.ground_removed else self.original_point_cloud

        if len(frustums) == 0:
            print("Frustum filtering: No frustums provided, returning all points")
            return points, np.ones(len(points), dtype=bool)

        filtered_points, mask = filter_points_in_multiple_frustums(points, frustums)

        return filtered_points, mask

    def remove_ground_plane_ransac(self, distance_threshold: float = 0.3,
                                   ransac_n: int = 3, num_iterations: int = 1000,
                                   remove_ego_car: bool = True, filter_forward_only: bool = True) -> np.ndarray:
        """
        Remove ground plane from point cloud using RANSAC.

        Args:
            distance_threshold: Maximum distance from plane to be considered inlier
            ransac_n: Number of points to sample for plane fitting
            num_iterations: Number of RANSAC iterations
            remove_ego_car: Whether to remove points near the ego vehicle
            filter_forward_only: Whether to keep only forward-facing points (x > 0)

        Returns:
            Filtered point cloud with ground plane removed
        """

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.original_point_cloud)

        # Segment plane using RANSAC
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations
        )

        # Extract non-ground points (outliers)
        outlier_cloud = pcd.select_by_index(inliers, invert=True)

        # Convert back to numpy array
        filtered_points = np.asarray(outlier_cloud.points)

        n_ground_points = len(inliers)
        n_remaining_points = len(filtered_points)

        if remove_ego_car:
            # Calculate distance from LiDAR origin (0, 0, 0) for each point
            distances = np.linalg.norm(filtered_points[:, :3], axis=1)
            ego_mask = distances > 2.5
            filtered_points = filtered_points[ego_mask]

        # Filter to only forward-facing points (positive x)
        if filter_forward_only:
            filtered_points = self.filter_forward_points(filtered_points)

        self.ground_removed = True
        self.point_cloud_plane_removed = filtered_points
        self.ground_plane_model = plane_model
        self.ground_inliers = self.original_point_cloud[inliers]

    def get_ground_z(self, x: float = 0.0, y: float = 0.0) -> Optional[float]:
        """
        Compute the ground z value at a given (x, y) location using the ground plane model.

        The ground plane equation is: ax + by + cz + d = 0
        Solving for z: z = -(ax + by + d) / c

        Args:
            x: X coordinate (forward direction)
            y: Y coordinate (lateral direction)

        Returns:
            Ground z value at (x, y), or None if ground plane not computed
        """
        if not self.ground_removed or self.ground_plane_model is None:
            return None

        a, b, c, d = self.ground_plane_model
        if abs(c) < 1e-6:
            # Plane is nearly vertical, can't solve for z
            return None

        z = -(a * x + b * y + d) / c
        return float(z)

    def add_projected_points(self, projected_points: np.ndarray):
        """
        Add projected points to the point cloud.
        """
        self.point_cloud_plane_removed = np.concatenate((self.point_cloud_plane_removed, projected_points), axis=0)
        
    def cluster_with_dbscan(self, eps: float = 0.5, min_samples: int = 10,
                           metric: str = 'euclidean', algorithm: str = 'auto',
                           leaf_size: int = 30) -> List[np.ndarray]:
        """
        Cluster point cloud using DBSCAN algorithm.
        Uses ground-removed point cloud if available.
        
        Args:
            eps: The maximum distance between two samples for one to be considered
                as in the neighborhood of the other. This is the most important DBSCAN
                parameter to choose appropriately for your data set and distance function.
            min_samples: The number of samples (or total weight) in a neighborhood
                for a point to be considered as a core point. This includes the point itself.
            metric: The metric to use when calculating distance between instances.
                Options: 'euclidean', 'manhattan', 'chebyshev', 'minkowski', etc.
            algorithm: The algorithm to use for finding nearest neighbors.
                Options: 'auto', 'ball_tree', 'kd_tree', 'brute'
            leaf_size: Leaf size passed to BallTree or KDTree. This can affect the
                speed of the construction and query, as well as the memory required.
        
        Returns:
            List of numpy arrays, where each array contains the indices of points
            belonging to that cluster. Points with label -1 (noise) are excluded.
            Note: Indices correspond to the filtered (ground-removed) point cloud.
        """
        
        # Perform DBSCAN clustering on filtered point cloud
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric,
                       algorithm=algorithm, leaf_size=leaf_size)
        labels = dbscan.fit_predict(self.point_cloud_plane_removed)
        
        # Organize clusters
        unique_labels = np.unique(labels)
        clusters = []
        
        for label in unique_labels:
            if label == -1:
                # Skip noise points (label -1)
                continue
            cluster_indices = np.where(labels == label)[0]
            clusters.append(cluster_indices)
        
        n_noise = np.sum(labels == -1)
        n_clusters = len(clusters)
        
        print(f"DBSCAN clustering completed:")
        print(f"  Number of clusters found: {n_clusters}")
        print(f"  Number of noise points: {n_noise}")
        print(f"  Total points: {len(self.point_cloud_plane_removed)}")
        print(f"  Parameters: eps={eps}, min_samples={min_samples}")
        
        self.clusters = clusters
    
    def cluster_with_dbscan_adaptive(self, k: int = 5, min_samples: int = 10,
                                    percentile: float = 50.0) -> List[np.ndarray]:
        """
        Cluster point cloud using DBSCAN with adaptive eps parameter.
        The eps is automatically determined using k-nearest neighbors distance.
        
        Args:
            k: Number of nearest neighbors to consider for adaptive eps calculation
            min_samples: The number of samples in a neighborhood for a point to be
                considered as a core point
            percentile: Percentile of k-NN distances to use as eps (default: 50th percentile)
        
        Returns:
            List of numpy arrays, where each array contains the indices of points
            belonging to that cluster. Points with label -1 (noise) are excluded.
        """
        
            
        # Calculate k-nearest neighbors distances
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(self.point_cloud_plane_removed)  # k+1 because point itself is included
        distances = nbrs.kneighbors(self.point_cloud_plane_removed)
        
        # Get k-th nearest neighbor distances (skip the first one which is the point itself)
        k_distances = distances[:, k]
        
        # Use percentile of k-distances as eps
        eps = np.percentile(k_distances, percentile)
        
        print(f"Adaptive DBSCAN: Calculated eps={eps:.3f} from {percentile}th percentile of {k}-NN distances")
        
        # Perform DBSCAN with adaptive eps
        self.cluster_with_dbscan(eps=eps, min_samples=min_samples)



    def cluster_with_segmentation_masks(self, mask_points: Dict[int, np.ndarray],
                                       min_points_per_cluster: int = 10) -> List[np.ndarray]:
        """
        Create clusters based on segmentation mask projections.
        
        Args:
            mask_points: Dictionary mapping mask_id to Nx3 array of projected 3D points
            min_points_per_cluster: Minimum number of points required for a valid cluster
            
        Returns:
            List of numpy arrays, where each array contains the indices of points
            belonging to that cluster in the ground-removed point cloud.
        """
        if not self.ground_removed:
            raise ValueError("Ground plane removal required before clustering. Call remove_ground_plane_ransac() first.")
        
        clusters = []
        
        # For each mask projection, find corresponding points in the ground-removed point cloud
        for mask_id, points in mask_points.items():
            if len(points) < min_points_per_cluster:
                print(f"Skipping mask {mask_id}: only {len(points)} points (minimum: {min_points_per_cluster})")
                continue
            
            # Find points in the ground-removed point cloud that are close to the projected points
            # This is a simplified approach - you might want to use a more sophisticated method
            cluster_indices = []
            
            # For each projected point, find the closest point in the ground-removed point cloud
            for point in points:
                # Calculate distances to all points in the ground-removed point cloud
                distances = np.linalg.norm(self.point_cloud_plane_removed - point, axis=1)
                
                # Find the closest point
                closest_idx = np.argmin(distances)
                
                # If the closest point is within a reasonable distance, add it to the cluster
                if distances[closest_idx] < 0.5:  # 0.5m threshold
                    cluster_indices.append(closest_idx)
            
            # Remove duplicates
            cluster_indices = list(set(cluster_indices))
            
            if len(cluster_indices) >= min_points_per_cluster:
                clusters.append(np.array(cluster_indices))
                print(f"Created cluster for mask {mask_id}: {len(cluster_indices)} points")
            else:
                print(f"Skipping mask {mask_id}: only {len(cluster_indices)} unique points found (minimum: {min_points_per_cluster})")
        
        print(f"Created {len(clusters)} clusters from {len(mask_points)} segmentation masks")
        self.clusters = clusters
        return clusters
    
    def add_segmentation_projected_points(self, mask_points: Dict[int, np.ndarray]):
        """
        Add projected points from segmentation masks to the point cloud.
        
        Args:
            mask_points: Dictionary mapping mask_id to Nx3 array of projected 3D points
        """
        all_points = []
        for mask_id, points in mask_points.items():
            if len(points) > 0:
                all_points.append(points)
        
        if all_points:
            all_points = np.vstack(all_points)
            self.add_projected_points(all_points)
            print(f"Added {len(all_points)} points from {len(mask_points)} segmentation masks")
        else:
            print("No points to add from segmentation masks")

class PointCloudVisualizer:
    """
    Class for visualizing point clouds, clusters, and rays.
    """
    
    def __init__(self, point_cloud: PointCloud):
        """
        Initialize the point cloud visualizer.
        
        Args:
            point_cloud: Nx3 array of point cloud points (x, y, z)
            coordinate_systems: Optional coordinate system information
            remove_ground: Whether to automatically remove ground plane using RANSAC
        """
        
        if point_cloud.ground_removed:
            self.original_point_cloud = point_cloud.point_cloud_plane_removed
        else:
            self.original_point_cloud = point_cloud.original_point_cloud
    
    def visualize_point_cloud(self, points: Optional[np.ndarray] = None,
                                       rays: Optional[Dict[str, np.ndarray]] = None,
                                       clusters: Optional[List[np.ndarray]] = None,
                                       title: str = "Point Cloud with Projected Points"):
        """
        Visualize point cloud with projected 3D points from rays using Open3D. It has the option to visualize the ground plane, projected points, rays, and clusters along with the point cloud.
        
        Args:
            points: Nx3 array of projected 3D points
            rays: Optional dictionary with 'origins' (Nx3) and 'directions' (Nx3)
            clusters: Optional list of point indices arrays for each cluster
            title: Window title
            draw_coordinate_systems: Whether to draw coordinate system axes
            axis_length: Length of coordinate system axes
        """
        geometries = []
    
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.original_point_cloud)
        pcd.paint_uniform_color([0, 0, 1])
        geometries.append(pcd)
        
        # Add clusters if provided
        if clusters is not None:
            cluster_colors = np.zeros((len(clusters), 3))
            cluster_colors[:, 0] = 0.9  # High red channel for all clusters
            cluster_colors[:, 1] = np.linspace(0, 0.3, len(clusters))  # Vary green from 0 to 0.3
            cluster_colors[:, 2] = np.linspace(0, 0.2, len(clusters))  # Vary blue from 0 to 0.2
            print("Cluster colors:")
            print(cluster_colors)
            for i, cluster_indices in enumerate(clusters):
                if len(cluster_indices) > 0:
                    cluster_pcd = o3d.geometry.PointCloud()
                    cluster_pcd.points = o3d.utility.Vector3dVector(
                        self.original_point_cloud[cluster_indices]
                    )
                    cluster_pcd.paint_uniform_color(cluster_colors[i])
                    geometries.append(cluster_pcd)
        
        # Add projected points
        if points is not None and len(points) > 0:
            projected_pcd = o3d.geometry.PointCloud()
            projected_pcd.points = o3d.utility.Vector3dVector(points)
            projected_pcd.paint_uniform_color([1, 0, 0])  # Red
            geometries.append(projected_pcd)
            
        
        # Add rays if provided
        if rays is not None:
            origins = rays['origins']
            directions = rays['directions']
            for i in range(len(origins)):
                origin = origins[i]
                if points is not None and i < len(points):
                    projected = points[i]
                else:
                    # Extend ray if no projected point
                    projected = origin + directions[i] * 20.0
                
                # Create line segment for ray
                line_points = np.array([origin, projected])
                line = o3d.geometry.LineSet()
                line.points = o3d.utility.Vector3dVector(line_points)
                line.lines = o3d.utility.Vector2iVector([[0, 1]])
                line.colors = o3d.utility.Vector3dVector([[1, 0.5, 0.2]])  # Orange
                geometries.append(line)
        
        
        # Visualize
        o3d.visualization.draw_geometries(geometries, window_name=title)


def visualize_image_with_pixels(image: np.ndarray, pixel_coords: np.ndarray,
                                save_path: Optional[str] = None, show: bool = True):
    """
    Visualize image with marked pixels that are being projected.
    
    Args:
        image: 2D image (can be path or numpy array)
        pixel_coords: Nx2 array of pixel coordinates (u, v)
        save_path: Optional path to save the visualization
        show: Whether to display the plot
    """
    # Load image if it's a path
    if isinstance(image, str):
        img = cv2.imread(image)
        if img is None:
            raise ValueError(f"Could not load image: {image}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = image.copy()
        if len(img.shape) == 3 and img.shape[2] == 3:
            # Already RGB
            pass
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(img)
    
    # Mark pixels
    if pixel_coords.ndim == 1:
        pixel_coords = pixel_coords.reshape(1, -1)
    
    for i, (u, v) in enumerate(pixel_coords):
        # Draw circle at pixel location
        circle = plt.Circle((u, v), 4, color='red', fill=True, alpha=0.5)
        ax.add_patch(circle)
        
    ax.set_title(f"Image with {len(pixel_coords)} Projected Pixels", 
                fontsize=14, weight='bold')
    ax.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Image visualization saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()