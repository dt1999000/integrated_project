"""
2D to 3D Projection, Sparse Depth, and Point Cloud Visualization.
"""

import numpy as np
import cv2
from typing import Any, List, Tuple, Optional, Dict
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import open3d as o3d

# Import the new clustering manager
from .clustering_manager import ClusteringManager

import os


DEBUG_PROJECTION = os.getenv("PROJECTION_DEBUG", "0").strip() in {"1", "true", "yes", "on"}


def _debug_projection_log(label: str, details: str) -> None:
    if DEBUG_PROJECTION:
        print(f"[Projection][{label}] {details}")


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
        _debug_projection_log(
            "init",
            f"n_points={len(self.point_cloud)}, "
            f"camera_to_lidar={self.camera_to_lidar_transform}, "
            f"lidar_to_camera[0,:]={self.lidar_to_camera_transform}",
        )
    
    def filter_forward_points(self, points_3d_lidar: np.ndarray) -> np.ndarray:
        """
        Filter points to only keep those in front of the camera (z_cam > 0).
 
        Args:
            points_3d_lidar: Nx3 (or Nx>=3) array in LiDAR coordinates.
 
        Returns:
            Filtered array with only forward-facing points.
        """
        if points_3d_lidar.ndim != 2 or points_3d_lidar.shape[1] < 3:
            raise ValueError(
                f"Expected points with shape (N, 3) or (N, >=3), got {points_3d_lidar.shape}"
            )
        ones = np.ones((points_3d_lidar.shape[0], 1), dtype=points_3d_lidar.dtype)
        pts_h = np.hstack([points_3d_lidar[:, :3], ones])
        pts_cam = (self.lidar_to_camera_transform @ pts_h.T).T[:, :3]
        z_cam = pts_cam[:, 2]
        mask = z_cam > 0.0
        filtered = points_3d_lidar[mask]
        if DEBUG_PROJECTION:
            for i in range(min(5, len(filtered))):
                _debug_projection_log(
                    "sample_filter_forward",
                    f"x_in,y_in,z_in={filtered[i,0]:.2f}, {filtered[i,1]:.2f}, {filtered[i,2]:.2f}, x_cam,y_cam,z_cam={pts_cam[i,0]:.2f}, {pts_cam[i,1]:.2f}, {pts_cam[i,2]:.2f}"
                ) 
        n_removed = len(points_3d_lidar) - len(filtered)
        if n_removed > 0:
            _debug_projection_log(
                "filter_forward_points",
                f"removed={n_removed}, remaining={len(filtered)}, "
                f"z_cam_range=({z_cam.min():.3f}, {z_cam.max():.3f})",
            )
        
        return filtered
    
    def compute_sparse_depth_map(self, image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Create sparse depth map by back-projecting this instance's LiDAR points
        onto the image, using positive camera z as depth.
 
        Args:
            image_shape: (height, width) of the image
 
        Returns:
            sparse_depth: HxW numpy array with depth values at projected pixel
                          locations, zeros elsewhere.
        """
        h, w = image_shape
        pixels, valid_mask = self.point_to_pixel(self.point_cloud)
        in_bounds = (
            (pixels[:, 0] >= 0)
            & (pixels[:, 0] < w)
            & (pixels[:, 1] >= 0)
            & (pixels[:, 1] < h)
        )
        valid_mask &= in_bounds
 
        sparse_depth = np.zeros((h, w), dtype=np.float32)
        if np.any(valid_mask):
            valid_pixels = pixels[valid_mask].astype(int)
            valid_points = self.point_cloud[valid_mask]
 
            ones = np.ones((valid_points.shape[0], 1), dtype=valid_points.dtype)
            pts_h = np.hstack([valid_points[:, :3], ones])
            pts_cam = (self.lidar_to_camera_transform @ pts_h.T).T[:, :3]
            depths = pts_cam[:, 2]
 
            positive_depth_mask = depths > 0
            valid_pixels = valid_pixels[positive_depth_mask]
            depths = depths[positive_depth_mask]
            if DEBUG_PROJECTION:
                for i in range(min(5, len(valid_points))):
                    _debug_projection_log(
                        "sample",
                        f"x_in,y_in,z_in={valid_points[i,0]:.2f}, {valid_points[i,1]:.2f}, {valid_points[i,2]:.2f}, x_cam,y_cam,z_cam={pts_cam[i,0]:.2f}, {pts_cam[i,1]:.2f}, {pts_cam[i,2]:.2f}"
                    ) 
 
            for (u, v), depth in zip(valid_pixels, depths):
                if sparse_depth[v, u] == 0 or depth < sparse_depth[v, u]:
                    sparse_depth[v, u] = depth
 
        n_points = np.sum(sparse_depth > 0)
        if n_points > 0:
            non_zero = sparse_depth[sparse_depth > 0]
            _debug_projection_log(
                "compute_sparse_depth_map",
                f"valid_depths={n_points}/{len(self.point_cloud)}, "
                f"depth_range=({non_zero.min():.2f}, {non_zero.max():.2f}), "
                f"coverage={100 * n_points / (h * w):.2f}%"
            )
 
        return sparse_depth
        
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
        z_cam = points_cam[:, 2]
        valid_mask = z_cam > 0

        # Initialize output
        pixels = np.zeros((n_points, 2))

        # Project valid points to image plane: pixel = K @ [x/z, y/z, 1]
        if np.any(valid_mask):
            z = points_cam[valid_mask, 2:3]
            normalized = points_cam[valid_mask, :2] / z
            pixels_homo = np.hstack([normalized, np.ones((np.sum(valid_mask), 1))])
            pixels[valid_mask] = (self.camera_intrinsic @ pixels_homo.T).T[:, :2]

        if DEBUG_PROJECTION and n_points > 0:
            n_valid = int(valid_mask.sum())
            z_min = float(z_cam[valid_mask].min()) if n_valid > 0 else 0.0
            z_max = float(z_cam[valid_mask].max()) if n_valid > 0 else 0.0
            _debug_projection_log(
                "point_to_pixel",
                f"n_points={n_points}, n_valid={n_valid}, "
                f"z_cam_range=({z_min:.3f}, {z_max:.3f})"
            )

        return pixels, valid_mask

    def cuboid_to_2d(self, cuboid: Dict) -> Optional[Dict]:
        """
        Project a 3D cuboid to 2D bounding box and projected corners.

        Args:
            cuboid: Dict with one of:
                    - 'corners' (8x3) - 8 corner points
                    - min/max bounds: 'min_x', 'max_x', 'min_y', 'max_y', 'min_z', 'max_z'
                    - KITTI format: 'center' (3,), 'yaw' (float), 'length', 'width', 'height'
                      (if 'format' == 'kitti' or all KITTI keys are present)

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
        elif cuboid.get('format') == 'kitti' or ('center' in cuboid and 'yaw' in cuboid and 
                                                   'length' in cuboid and 'width' in cuboid and 'height' in cuboid):
            # KITTI format: center, yaw, length, width, height
            center = np.asarray(cuboid['center']).flatten()
            yaw = cuboid['yaw']
            length = cuboid['length']
            width = cuboid['width']
            height = cuboid['height']
            
            # Half-dimensions
            l_half = length / 2.0
            w_half = width / 2.0
            h_half = height / 2.0
            
            # Create 8 corners in local coordinate system (centered at origin, axis-aligned)
            corners_local = np.array([
                [-l_half, -w_half, -h_half],  # 0: bottom front-left
                [ l_half, -w_half, -h_half],  # 1: bottom front-right
                [ l_half,  w_half, -h_half],  # 2: bottom back-right
                [-l_half,  w_half, -h_half],  # 3: bottom back-left
                [-l_half, -w_half,  h_half],  # 4: top front-left
                [ l_half, -w_half,  h_half],  # 5: top front-right
                [ l_half,  w_half,  h_half],  # 6: top back-right
                [-l_half,  w_half,  h_half],  # 7: top back-left
            ])
            
            # Rotation matrix around Z-axis (yaw)
            cos_yaw = np.cos(yaw)
            sin_yaw = np.sin(yaw)
            R_z = np.array([
                [cos_yaw, -sin_yaw, 0],
                [sin_yaw,  cos_yaw, 0],
                [0,        0,       1]
            ])
            
            # Rotate corners and translate to center
            corners_rotated = (R_z @ corners_local.T).T
            corners_3d = corners_rotated + center
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

    def backproject_sparse_depth_map_with_colors(
        self,
        sparse_depth_map: np.ndarray,
        image: np.ndarray,
        depth_threshold_min: float = 0.1,
        depth_threshold_max: float = 100.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Backproject sparse depth map to 3D LiDAR point cloud with colors from the image.

        Args:
            sparse_depth_map: HxW numpy array with depth values at projected pixel locations, zeros elsewhere
            image: HxWx3 numpy array (RGB format) corresponding to the depth map
            depth_threshold_min: Minimum valid depth value (meters)
            depth_threshold_max: Maximum valid depth value (meters)

        Returns:
            Tuple of (points_lidar, colors) where:
                - points_lidar: Nx3 array of 3D points in LiDAR coordinates
                - colors: Nx3 array of RGB colors (0-255) corresponding to each point
        """
        h, w = sparse_depth_map.shape

        if len(image.shape) == 3 and image.shape[2] == 3:
            if image.dtype != np.uint8:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
        else:
            raise ValueError(f"Image must be HxWx3 RGB format, got shape {image.shape}")

        valid_mask = (sparse_depth_map > depth_threshold_min) & (sparse_depth_map <= depth_threshold_max)

        if not np.any(valid_mask):
            return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3)

        v, u = np.where(valid_mask)
        depths = sparse_depth_map[valid_mask]

        colors = image[v, u]

        K_inv = np.linalg.inv(self.camera_intrinsic)
        pixels_homogeneous = np.stack([u, v, np.ones_like(u)], axis=0)
        points_normalized = K_inv @ pixels_homogeneous
        points_camera = points_normalized * depths
        points_camera = points_camera.T

        points_camera_homo = np.hstack([points_camera, np.ones((len(points_camera), 1))])
        points_lidar = (self.camera_to_lidar_transform @ points_camera_homo.T).T[:, :3]

        print(f"Backprojected {len(points_lidar)} colored points from sparse depth map")
        print(f"  X range: [{points_lidar[:, 0].min():.2f}, {points_lidar[:, 0].max():.2f}]")
        print(f"  Y range: [{points_lidar[:, 1].min():.2f}, {points_lidar[:, 1].max():.2f}]")
        print(f"  Z range: [{points_lidar[:, 2].min():.2f}, {points_lidar[:, 2].max():.2f}]")

        return points_lidar, colors

    def project_masked_depth_to_3d(
        self,
        depth_map: np.ndarray,
        mask: Optional[np.ndarray] = None,
        depth_threshold_min: float = 0.1,
        depth_threshold_max: float = 100.0,
        stride: int = 1
    ) -> np.ndarray:
        """
        Project a depth map (optionally masked) to 3D LiDAR coordinates using stored camera parameters.

        Args:
            depth_map: Metric depth map (H, W) in meters
            mask: Optional binary mask (H, W). Only pixels where mask>0 are projected.
            depth_threshold_min: Minimum valid depth value (meters)
            depth_threshold_max: Maximum valid depth value (meters)
            stride: Sampling stride for point cloud (1 = all pixels, 2 = every other pixel, etc.)

        Returns:
            points_lidar: Nx3 array of 3D points in LiDAR coordinates
        """
        depth_map = np.asarray(depth_map)
        original_shape = depth_map.shape

        if depth_map.ndim == 1:
            raise ValueError(f"Depth map is 1D with shape {depth_map.shape}, expected 2D (H, W)")
        elif depth_map.ndim == 2:
            pass
        elif depth_map.ndim == 3:
            if depth_map.shape[0] == 1:
                depth_map = depth_map[0, :, :]
            elif depth_map.shape[2] == 1:
                depth_map = depth_map[:, :, 0]
            else:
                depth_map = depth_map[0, :, :]
        elif depth_map.ndim == 4:
            depth_map = depth_map.squeeze()
            if depth_map.ndim != 2:
                depth_map = depth_map.reshape(-1, depth_map.shape[-1])[0].reshape(depth_map.shape[-2:])
        else:
            raise ValueError(f"Unexpected depth map shape: {depth_map.shape}, expected 2D (H, W)")

        if depth_map.ndim != 2:
            raise ValueError(f"After processing, depth map still has {depth_map.ndim} dimensions with shape {depth_map.shape} (original shape: {original_shape})")

        height, width = depth_map.shape

        u_coords, v_coords = np.meshgrid(
            np.arange(0, width, stride),
            np.arange(0, height, stride)
        )

        u = u_coords.flatten()
        v = v_coords.flatten()
        depths = depth_map[v, u]

        valid_mask = (depths >= depth_threshold_min) & (depths <= depth_threshold_max)

        if mask is not None:
            if mask.shape != depth_map.shape:
                raise ValueError(f"Mask shape {mask.shape} does not match depth map shape {depth_map.shape}")
            mask_values = mask[v, u]
            valid_mask = valid_mask & (mask_values > 0)
            print(f"Applied mask: {np.sum(valid_mask):,} pixels remain after mask filtering")

        u_valid = u[valid_mask]
        v_valid = v[valid_mask]
        depths_valid = depths[valid_mask]

        print("Reconstructing masked point cloud from depth:")
        print(f"  Total pixels: {len(u)}")
        print(f"  Valid depth pixels: {len(depths_valid)} ({100 * len(depths_valid) / len(u):.1f}%)")

        K_inv = np.linalg.inv(self.camera_intrinsic)
        pixels_homogeneous = np.stack([u_valid, v_valid, np.ones_like(u_valid)], axis=0)
        points_normalized = K_inv @ pixels_homogeneous
        points_camera = points_normalized * depths_valid
        points_camera = points_camera.T

        points_camera_homo = np.hstack([points_camera, np.ones((len(points_camera), 1))])
        points_lidar = (self.camera_to_lidar_transform @ points_camera_homo.T).T[:, :3]

        if len(points_lidar) > 0:
            print(f"  Reconstructed {len(points_lidar)} masked points in LiDAR coordinates")
            print(f"  X range: [{points_lidar[:, 0].min():.2f}, {points_lidar[:, 0].max():.2f}]")
            print(f"  Y range: [{points_lidar[:, 1].min():.2f}, {points_lidar[:, 1].max():.2f}]")
            print(f"  Z range: [{points_lidar[:, 2].min():.2f}, {points_lidar[:, 2].max():.2f}]")
        else:
            print("  No valid points reconstructed from masked depth")

        return points_lidar


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
                                   remove_ego_car: bool = False, filter_forward_only: bool = False,
                                   ground_z_range: Optional[Tuple[float, float]] = (-3.0, 3.0),
                                   ground_z_percentile: float = 0.35,
                                   ground_fit_max_xy_distance: Optional[float] = 15.0,
                                   camera_to_lidar_transform: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Remove ground plane from point cloud using RANSAC.

        RANSAC is run on a subset of points that are likely ground to avoid
        fitting walls/ceiling in indoor scenes (where they can have more points).
        The plane normal is required to be roughly horizontal and pointing up.

        Args:
            distance_threshold: Maximum distance from plane to be considered inlier
            ransac_n: Number of points to sample for plane fitting
            num_iterations: Number of RANSAC iterations
            remove_ego_car: Whether to remove points near the ego vehicle
            filter_forward_only: Whether to keep only forward-facing points in
                                 front of the camera. If camera_to_lidar_transform
                                 is provided, this uses the camera z-axis in LiDAR
                                 coordinates; otherwise it falls back to x > 0.
            ground_z_range: (z_min, z_max) - hard clip on z for candidates. Typical: (-3, 3) m.
            ground_z_percentile: Use only points with z in [0, this percentile] of the cloud
                                (e.g. 0.35 = lowest 35%%). Favors floor over walls in indoor.
            ground_fit_max_xy_distance: If set, only points within this horizontal distance
                                       from origin are used to fit the plane (avoids far walls).

        Returns:
            Filtered point cloud with ground plane removed
        """
        pts = self.original_point_cloud
        z_min_range, z_max_range = ground_z_range
        # 1) Restrict to z band
        ground_candidate_mask = (pts[:, 2] >= z_min_range) & (pts[:, 2] <= z_max_range)
        # 2) Further restrict to lowest percentile of z (floor, not walls/ceiling)
        band = pts[(pts[:, 2] >= z_min_range) & (pts[:, 2] <= z_max_range)]
        z_lo_pct, z_hi_pct = np.percentile(band[:, 2], [0, ground_z_percentile * 100])
        low_z_mask = (pts[:, 2] >= z_lo_pct) & (pts[:, 2] <= z_hi_pct)
        ground_candidate_mask = ground_candidate_mask & low_z_mask
        # 3) Optional: only points near the sensor (avoids far walls in same z band)
        if ground_fit_max_xy_distance is not None and ground_fit_max_xy_distance > 0:
            xy_dist = np.linalg.norm(pts[:, :2], axis=1)
            ground_candidate_mask = ground_candidate_mask & (xy_dist <= ground_fit_max_xy_distance)
        if np.sum(ground_candidate_mask) < ransac_n * 10:
            # Fallback: relax to lower half of z range, no xy cap
            z_lo, z_hi = np.percentile(pts[:, 2], [0, 50])
            ground_candidate_mask = (pts[:, 2] >= z_lo) & (pts[:, 2] <= z_hi)
        ground_candidates = pts[ground_candidate_mask]
        # Ensure we have enough candidates for Open3D's RANSAC
        if len(ground_candidates) < ransac_n:
            # Not enough points to fit a plane; keep original cloud unchanged
            self.point_cloud_plane_removed = pts.copy()
            self.ground_plane_model = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64)
            self.ground_plane_inliers = np.zeros(len(pts), dtype=bool)
            self.ground_plane_outliers = np.ones(len(pts), dtype=bool)
            return self.point_cloud_plane_removed

        # Create Open3D point cloud from ground-candidate points only
        pcd_fit = o3d.geometry.PointCloud()
        pcd_fit.points = o3d.utility.Vector3dVector(ground_candidates)

        plane_model, inliers_fit = pcd_fit.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations
        )
        a, b, c, d = plane_model
        n_norm = np.sqrt(a * a + b * b + c * c) + 1e-9
        # Require roughly horizontal (|n_z| dominant)
        if abs(c) / n_norm < 0.7:
            z_median = float(np.median(ground_candidates[:, 2]))
            plane_model = np.array([0.0, 0.0, 1.0, -z_median], dtype=np.float64)
            a, b, c, d = plane_model
            n_norm = 1.0
        else:
            # Prefer normal pointing up (floor = ground below); flip if ceiling
            if c < 0:
                plane_model = np.array([-a, -b, -c, -d], dtype=np.float64)
                a, b, c, d = plane_model

        # Apply plane to full point cloud: compute distances and mark inliers
        n_pts = len(pts)
        dists = np.abs(pts[:, 0] * a + pts[:, 1] * b + pts[:, 2] * c + d) / n_norm
        inliers_full = np.where(dists <= distance_threshold)[0]
        outlier_mask = np.ones(n_pts, dtype=bool)
        outlier_mask[inliers_full] = False
        filtered_points = pts[outlier_mask]

        n_ground_points = len(inliers_full)
        n_remaining_points = len(filtered_points)

        if remove_ego_car:
            # Calculate distance from LiDAR origin (0, 0, 0) for each point
            distances = np.linalg.norm(filtered_points[:, :3], axis=1)
            ego_mask = distances > 2.5
            filtered_points = filtered_points[ego_mask]

        # Filter to only forward-facing points (z_cam > 0 if transform available,
        # otherwise fallback to x > 0 in LiDAR frame).
        if filter_forward_only:
            if camera_to_lidar_transform is not None:
                filtered_points = Projection.filter_forward_points(
                    Projection(
                        camera_intrinsic=np.eye(3, dtype=filtered_points.dtype),
                        camera_extrinsic=np.eye(4, dtype=filtered_points.dtype),
                        camera_to_lidar_transform=camera_to_lidar_transform,
                        point_cloud=filtered_points,
                    ),
                    filtered_points,
                )
            else:
                forward_mask = filtered_points[:, 0] > 0
                filtered_points = filtered_points[forward_mask]

        self.ground_removed = True
        self.point_cloud_plane_removed = filtered_points
        self.ground_plane_model = plane_model
        self.ground_inliers = self.original_point_cloud[inliers_full]

    def get_ground_z(self, x: float = 0.0, y: float = 0.0,
                     z_range: Optional[Tuple[float, float]] = (-10.0, 10.0)) -> Optional[float]:
        """
        Compute the ground z value at a given (x, y) location using the ground plane model.

        The ground plane equation is: ax + by + cz + d = 0
        Solving for z: z = -(ax + by + d) / c

        Args:
            x: X coordinate (forward direction)
            y: Y coordinate (lateral direction)
            z_range: (z_min, z_max) - if computed z is outside this range, return None.
                      Prevents invalid values (e.g. -80m) from bad plane fits.

        Returns:
            Ground z value at (x, y), or None if ground plane not computed or z out of range
        """
        if not self.ground_removed or self.ground_plane_model is None:
            return None

        a, b, c, d = self.ground_plane_model
        if abs(c) < 1e-6:
            # Plane is nearly vertical, can't solve for z
            return None

        z = -(a * x + b * y + d) / c
        z = float(z)
        if z_range is not None:
            z_lo, z_hi = z_range
            if z < z_lo or z > z_hi:
                return None
        return z

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

    @staticmethod
    def from_rgbd_sunrgbd(
        rgb_image: np.ndarray,
        depth_image: np.ndarray,
        camera_intrinsic: np.ndarray,
        depth_scale: float = 10000.0,
        depth_trunc: float = 10.0,
        stride: int = 1,
        keep_fraction: float = 0.8,
    ) -> "PointCloud":
        """
        Create a PointCloud from an RGBD image pair following the SunRGBD dataset conventions.

        SunRGBD stores depth as 16-bit PNG. The ``depth_bfx/`` directory uses a
        scale of 10000 (depth_meters = raw / 10000). Raw sensor directories may
        use 1000 (millimetres). Adjust ``depth_scale`` accordingly.

        The returned point cloud lives in the **camera** coordinate system
        (X-right, Y-down, Z-forward) because SunRGBD has no separate LiDAR
        frame.  Colours are carried on the ``PointCloud`` via the
        ``.colors`` attribute (Nx3, 0-255 uint8).

        Args:
            rgb_image:        H×W×3 uint8 RGB image.
            depth_image:      H×W uint16 / float depth map (raw sensor values).
            camera_intrinsic: 3×3 intrinsic matrix ``[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]``.
            depth_scale:      Divisor that converts raw depth to **metres**
                              (10000 for ``depth_bfx``, 1000 for raw mm).
            depth_trunc:      Maximum depth in metres; farther points are discarded.
            stride:           Pixel sampling stride (1 = every pixel).

        Returns:
            A ``PointCloud`` instance whose ``.original_point_cloud`` is Nx3 in
            camera coords and ``.colors`` is the corresponding Nx3 RGB array.
        """
        print(
            "[SUNRGBD_RGBD_PC] from_rgbd_sunrgbd called "
            f"rgb_shape={getattr(rgb_image, 'shape', None)} "
            f"depth_shape={getattr(depth_image, 'shape', None)} "
            f"depth_dtype={getattr(depth_image, 'dtype', None)} "
            f"depth_scale={depth_scale} depth_trunc={depth_trunc} stride={stride} "
            f"keep_fraction={keep_fraction}"
        )

        # Convert depth to metres using float32 to reduce memory footprint.
        depth_m = depth_image.astype(np.float32) / float(depth_scale)

        h, w = depth_m.shape
        u_coords, v_coords = np.meshgrid(
            np.arange(0, w, stride),
            np.arange(0, h, stride),
        )
        u = u_coords.ravel()
        v = v_coords.ravel()
        z = depth_m[v, u]

        valid = (z > 0.0) & (z <= depth_trunc)
        u, v, z = u[valid], v[valid], z[valid]

        # Optional random downsampling to limit point count for memory-intensive
        # operations such as DBSCAN in batch mode.
        n_valid = z.size
        if n_valid > 0:
            if keep_fraction <= 0.0:
                keep_fraction = 0.0
            if keep_fraction < 1.0:
                target_n = int(float(n_valid) * float(keep_fraction))
                if target_n <= 0:
                    target_n = 1
                if target_n < n_valid:
                    rng = np.random.default_rng()
                    keep_idx = rng.choice(n_valid, size=target_n, replace=False)
                    u = u[keep_idx]
                    v = v[keep_idx]
                    z = z[keep_idx]

        fx = camera_intrinsic[0, 0]
        fy = camera_intrinsic[1, 1]
        cx = camera_intrinsic[0, 2]
        cy = camera_intrinsic[1, 2]

        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        points = np.stack([x, y, z], axis=-1).astype(np.float32)
        colors = rgb_image[v, u]  # Nx3 uint8

        pc = PointCloud(points)
        pc.colors = colors

        print(
            "[SUNRGBD_RGBD_PC] from_rgbd_sunrgbd finished "
            f"valid_points={len(points)}"
        )

        return pc


def load_sunrgbd_intrinsics(intrinsics_path: str) -> np.ndarray:
    """
    Load a 3×3 camera intrinsic matrix from a SunRGBD ``intrinsics.txt`` file.

    The file contains 3 lines with 3 space-separated floats each::

        fx  0  cx
         0 fy  cy
         0  0   1

    Args:
        intrinsics_path: Path to the ``intrinsics.txt`` file.

    Returns:
        3×3 numpy float64 intrinsic matrix.
    """
    K = np.loadtxt(intrinsics_path, dtype=np.float64).reshape(3, 3)
    return K


def create_o3d_pointcloud_from_rgbd_sunrgbd(
    rgb_image: np.ndarray,
    depth_image: np.ndarray,
    camera_intrinsic: np.ndarray,
    depth_scale: float = 10000.0,
    depth_trunc: float = 10.0,
) -> o3d.geometry.PointCloud:
    """
    Create an Open3D PointCloud directly from SunRGBD RGBD data using the
    Open3D RGBD pipeline for maximum efficiency.

    Args:
        rgb_image:        H×W×3 uint8 RGB image.
        depth_image:      H×W uint16 / float raw depth values.
        camera_intrinsic: 3×3 numpy intrinsic matrix.
        depth_scale:      Divisor converting raw depth to metres.
        depth_trunc:      Maximum depth in metres.

    Returns:
        ``open3d.geometry.PointCloud`` with colours.
    """
    h, w = depth_image.shape[:2]

    o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=w,
        height=h,
        fx=float(camera_intrinsic[0, 0]),
        fy=float(camera_intrinsic[1, 1]),
        cx=float(camera_intrinsic[0, 2]),
        cy=float(camera_intrinsic[1, 2]),
    )

    rgb_o3d = o3d.geometry.Image(np.ascontiguousarray(rgb_image))
    depth_o3d = o3d.geometry.Image(np.ascontiguousarray(depth_image.astype(np.float32)))

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_o3d,
        depth_o3d,
        depth_scale=depth_scale,
        depth_trunc=depth_trunc,
        convert_rgb_to_intensity=False,
    )

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, o3d_intrinsic)

    return pcd
