"""
Pose Estimation Module

Provides different methods for estimating object pose (position and orientation)
from 3D point clouds. Supports PCA-based and L-shape fitting approaches.
"""

import numpy as np
from typing import Dict, Optional


def estimate_pose_pca(points: np.ndarray) -> Dict:
    """
    Estimates x, y, z, and yaw using Principal Component Analysis.
    
    PCA finds the direction of maximum variance. In a dense, well-reconstructed 
    point cloud, the first principal component usually aligns with the length of the object.
    
    Args:
        points: np.ndarray (N, 3) - 3D points in LiDAR coordinates
    
    Returns:
        Dictionary containing:
            - 'center': np.ndarray (3,) - Centroid [x, y, z]
            - 'yaw': float - Rotation angle around Z-axis (radians)
            - 'method': str - 'pca'
    
    Pros: 
        - Mathematically elegant
        - Extremely fast
        - Handles 360° point distributions perfectly
    
    Cons:
        - If the object is a perfect square, the yaw becomes unstable
        - Fails if the densification is uneven (e.g., more points on one side than the other)
    """
    if len(points) == 0:
        raise ValueError("Cannot estimate pose from empty point cloud")
    
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected points shape (N, 3), got {points.shape}")
    
    # 1. Calculate Centroid (x_c, y_c, z_c)
    centroid = np.mean(points, axis=0)
    
    # 2. Center the points
    centered_points = points - centroid
    
    # 3. Covariance Matrix (using only X-Y for yaw calculation, ignoring Z)
    points_2d = centered_points[:, :2]
    
    # Handle edge case where all points are at the same location
    if np.allclose(points_2d, 0):
        # All points are at the same (x, y), use default yaw
        return {
            'center': centroid,
            'yaw': 0.0,
            'method': 'pca'
        }
    
    cov = np.cov(points_2d.T)
    
    # 4. Eigen-decomposition
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    
    # 5. The primary eigenvector corresponds to the orientation
    # Use the eigenvector with the largest eigenvalue
    primary_idx = np.argmax(eigenvalues)
    primary_axis = eigenvectors[:, primary_idx]
    
    # Ensure consistent direction (point towards positive x)
    if primary_axis[0] < 0:
        primary_axis = -primary_axis
    
    # Calculate yaw angle (rotation around Z-axis)
    yaw = np.arctan2(primary_axis[1], primary_axis[0])
    
    return {
        'center': centroid,
        'yaw': yaw,
        'method': 'pca'
    }


def estimate_pose_l_shape(points: np.ndarray, d_theta: float = 0.01, 
                          ground_plane_model: Optional[np.ndarray] = None) -> Dict:
    """
    Finds the best yaw by minimizing the bounding box area (L-shape fitting).
    
    This is the industry standard for LiDAR because we usually only see two sides 
    of an object. It searches for the angle that best aligns the points with a 
    bounding box.
    
    Args:
        points: np.ndarray (N, 3) - 3D points in LiDAR coordinates
        d_theta: float - Angular step size for search (radians). Default 0.01 (~0.57 degrees)
        ground_plane_model: Optional [a, b, c, d] plane equation from RANSAC.
                          Used to compute ground z for height calculation.
    
    Returns:
        Dictionary containing:
            - 'center': np.ndarray (3,) - Centroid [x, y, z]
            - 'yaw': float - Rotation angle around Z-axis (radians)
            - 'method': str - 'l_shape'
            - 'length': float - Estimated length along primary axis
            - 'width': float - Estimated width along secondary axis
            - 'height': float - Estimated height (if ground_plane_model provided)
    
    Pros:
        - Highly robust to "partial views" (where you only see the corner of a car/box)
        - Works well with sparse or uneven point distributions
    
    Cons:
        - Computationally more expensive than PCA due to the search loop
        - Requires a ground-plane assumption for accurate height
    """
    if len(points) == 0:
        raise ValueError("Cannot estimate pose from empty point cloud")
    
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected points shape (N, 3), got {points.shape}")
    
    # Use only X-Y coordinates for 2D rotation search
    points_2d = points[:, :2]
    
    # Handle edge case where all points are at the same location
    if np.allclose(points_2d, points_2d[0]):
        centroid = np.mean(points, axis=0)
        return {
            'center': centroid,
            'yaw': 0.0,
            'method': 'l_shape',
            'length': 0.0,
            'width': 0.0,
            'height': np.ptp(points[:, 2]) if len(points) > 0 else 0.0
        }
    
    best_yaw = 0.0
    min_area = float('inf')
    best_length = 0.0
    best_width = 0.0
    
    # Search through angles (0 to 90 degrees is sufficient for a box due to symmetry)
    for theta in np.arange(0, np.pi/2, d_theta):
        # Rotation matrix
        R = np.array([
            [np.cos(theta), -np.sin(theta)], 
            [np.sin(theta),  np.cos(theta)]
        ])
        
        # Rotate points
        rotated_points = points_2d @ R.T
        
        # Calculate axis-aligned bounding box area in rotated space
        maxs = np.max(rotated_points, axis=0)
        mins = np.min(rotated_points, axis=0)
        dims = maxs - mins
        area = np.prod(dims)
        
        if area < min_area:
            min_area = area
            best_yaw = theta
            # Store dimensions (length is typically the larger dimension)
            if dims[0] > dims[1]:
                best_length = dims[0]
                best_width = dims[1]
            else:
                best_length = dims[1]
                best_width = dims[0]
    
    # Center calculation (use full 3D)
    center_3d = np.mean(points, axis=0)
    
    # Calculate height
    if ground_plane_model is not None:
        a, b, c, d = ground_plane_model
        if abs(c) > 1e-6:
            # Compute ground z at center
            ground_z = -(a * center_3d[0] + b * center_3d[1] + d) / c
            # Height is difference between max z and ground z
            height = np.max(points[:, 2]) - ground_z
        else:
            height = np.ptp(points[:, 2])
    else:
        # Use min z as ground level
        height = np.max(points[:, 2]) - np.min(points[:, 2])
    
    return {
        'center': center_3d,
        'yaw': best_yaw,
        'method': 'l_shape',
        'length': best_length,
        'width': best_width,
        'height': height
    }


def cuboid_from_pose(pose_result: Dict, category: str, 
                     template_dims: Optional[Dict[str, float]] = None,
                     ground_z: Optional[float] = None) -> Dict:
    """
    Create a KITTI-format cuboid dictionary from pose estimation result.
    
    Args:
        pose_result: Dictionary from estimate_pose_pca or estimate_pose_l_shape
        category: Object category (e.g., 'Car', 'Pedestrian')
        template_dims: Optional dict with 'length', 'width', 'height' from templates.
                      If provided, uses these instead of estimated dimensions.
        ground_z: Optional ground z value at cuboid center. If provided, uses this
                  for base_z calculation.
    
    Returns:
        Dictionary with KITTI format cuboid:
            - 'center': np.ndarray (3,) - Center position
            - 'yaw': float - Rotation angle
            - 'length': float - Length dimension
            - 'width': float - Width dimension
            - 'height': float - Height dimension
            - 'category': str - Object category
            - 'corners': np.ndarray (8, 3) - 8 corner points
            - 'min_x', 'max_x', etc. - Bounding box bounds
            - 'format': str - 'kitti' to indicate format
    """
    center = pose_result['center']
    yaw = pose_result['yaw']
    
    # Use template dimensions if available, otherwise use estimated dimensions
    if template_dims is not None:
        length = template_dims.get('length', pose_result.get('length', 4.0))
        width = template_dims.get('width', pose_result.get('width', 1.8))
        height = template_dims.get('height', pose_result.get('height', 1.5))
    else:
        length = pose_result.get('length', 4.0)
        width = pose_result.get('width', 1.8)
        height = pose_result.get('height', 1.5)
    
    # Calculate base z (ground level)
    if ground_z is not None:
        base_z = ground_z
    else:
        # Use center z minus half height as approximation
        base_z = center[2] - height / 2.0
    
    # Create corners using the same logic as cuboid_kitti_format
    l_half = length / 2.0
    w_half = width / 2.0
    h_half = height / 2.0
    
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
    
    # Adjust z to use base_z instead of center[2]
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
    
    # Calculate bounding box
    min_x = float(np.min(corners[:, 0]))
    max_x = float(np.max(corners[:, 0]))
    min_y = float(np.min(corners[:, 1]))
    max_y = float(np.max(corners[:, 1]))
    min_z = float(np.min(corners[:, 2]))
    max_z = float(np.max(corners[:, 2]))
    
    return {
        'center': center,
        'yaw': yaw,
        'length': length,
        'width': width,
        'height': height,
        'category': category,
        'corners': corners,
        'min_x': min_x,
        'max_x': max_x,
        'min_y': min_y,
        'max_y': max_y,
        'min_z': min_z,
        'max_z': max_z,
        'format': 'kitti',
        'method': pose_result.get('method', 'unknown')
    }

