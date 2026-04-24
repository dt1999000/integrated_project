"""
Pose Estimation Module

Provides different methods for estimating object pose (position and orientation)
from 3D point clouds. Supports PCA-based and L-shape fitting approaches.
"""

import numpy as np
from typing import Dict, Optional, Tuple

# Import templates from constants (avoid circular import by checking if already imported)
try:
    from .constants import KITTI_CUBOID_TEMPLATES
except ImportError:
    # Fallback if constants not available
    KITTI_CUBOID_TEMPLATES = {
        'Car': {'length': 3.64, 'width': 1.86, 'height': 1.58},
        'Pedestrian': {'length': 0.88, 'width': 0.90, 'height': 1.77},
        'Cyclist': {'length': 1.68, 'width': 0.75, 'height': 1.76},
        'Van': {'length': 4.76, 'width': 2.22, 'height': 2.27},
        'Truck': {'length': 9.82, 'width': 2.99, 'height': 3.38},
        'Tram': {'length': 15.59, 'width': 3.66, 'height': 3.73},
        'Misc': {'length': 2.56, 'width': 1.91, 'height': 1.68},
        'Person_sitting': {'length': 0.72, 'width': 0.80, 'height': 1.29},
        'Unknown': {'length': 2.0, 'width': 1.5, 'height': 1.5},
    }
def estimate_pose_pca(points: np.ndarray, category: Optional[str] = None) -> Dict:
    """
    Estimates x, y, z, and yaw using Principal Component Analysis.
    
    PCA finds the direction of maximum variance. In a dense, well-reconstructed 
    point cloud, the first principal component usually aligns with the length of the object.
    
    Args:
        points: np.ndarray (N, 3) - 3D points in LiDAR coordinates
        category: Optional object category (e.g., 'Car', 'Pedestrian') - not used in PCA but kept for API consistency
    
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


def estimate_pose_l_shape(points: np.ndarray,
                          category: Optional[str] = None,
                          d_theta: float = 0.01,
                          ground_plane_model: Optional[np.ndarray] = None,
                          dimensions: Optional[Tuple[float, float, float]] = None
                          ) -> Dict:
    """
    Finds the best yaw by minimizing the bounding box area (L-shape fitting).
    
    This is the industry standard for LiDAR because we usually only see two sides 
    of an object. It searches for the angle that best aligns the points with a 
    bounding box.
    
    Args:
        points: np.ndarray (N, 3) - 3D points in LiDAR coordinates
        category: Optional object category (e.g., 'Car', 'Pedestrian'). Used for metadata.
        d_theta: float - Angular step size for search (radians). Default 0.01 (~0.57 degrees)
        ground_plane_model: Optional [a, b, c, d] plane equation from RANSAC.
                            Used to compute ground z for height calculation.
        dimensions: Optional (length, width, height) prior. If provided, used when the
                    raw L-shape estimate is unreliable (e.g. too small). Pass from caller.
    
    Returns:
        Dictionary containing:
            - 'center': np.ndarray (3,) - Centroid [x, y, z]
            - 'yaw': float - Rotation angle around Z-axis (radians)
            - 'method': str - 'l_shape'
            - 'length': float - Estimated length along primary axis (or from template)
            - 'width': float - Estimated width along secondary axis (or from template)
            - 'height': float - Estimated height (or from template)
    
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
    # Actually, we need 0 to 180 degrees because length/width are different
    for theta in np.arange(0, np.pi, d_theta):
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
            best_yaw = theta+np.pi/2
            # Store dimensions (length is typically the larger dimension)
            if dims[0] > dims[1]:
                best_length = dims[0]
                best_width = dims[1]
            else:
                best_length = dims[1]
                best_width = dims[0]
    print(f"Best yaw: {best_yaw}")
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
    
    # Use prior dimensions if provided and estimated dimensions seem unreliable.
    # Caller must pass dimensions when category-based priors are needed.
    if dimensions is not None:
        prior_length, prior_width, prior_height = dimensions
        # Use prior if estimated dimensions are very small (likely unreliable)
        if best_length < 0.5 or best_width < 0.3:
            best_length = float(prior_length)
            best_width = float(prior_width)
        # Use prior height if estimated height is very small
        if height < 0.5:
            height = float(prior_height)
    
    return {
        'center': center_3d,
        'yaw': best_yaw,
        'method': 'l_shape',
        'length': best_length,
        'width': best_width,
        'height': height,
        'category': category,
    }


def cuboid_from_pose(pose_result: Dict,
                     category: Optional[str] = None,
                     template_dims: Optional[Dict[str, float]] = None,
                     dimensions: Optional[Tuple[float, float, float]] = None,
                     ground_z: Optional[float] = None) -> Dict:
    """
    Create a KITTI-format cuboid dictionary from pose estimation result.
    
    Args:
        pose_result: Dictionary from estimate_pose_pca or estimate_pose_l_shape.
                     May already contain 'length', 'width', and 'height'.
        category: Optional object category (e.g., 'Car', 'Pedestrian'). Used for
                  metadata and, if needed, to derive fallback dimensions.
        template_dims: Optional dict with 'length', 'width', 'height' from
                       category templates. Used when pose_result does not already
                       contain dimensions (e.g., PCA-based pose).
        dimensions: Optional (length, width, height) tuple. Used when pose_result
                    does not contain dimensions and template_dims is not provided.
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
    
    # Prefer dimensions from pose_result (L-shape returns them); if not present,
    # fall back to provided template dimensions or category-based priors.
    length = pose_result.get('length')
    width = pose_result.get('width')
    height = pose_result.get('height')
    
    if length is None or width is None or height is None:
        if template_dims is not None:
            length = float(template_dims.get('length', 4.0))
            width = float(template_dims.get('width', 1.8))
            height = float(template_dims.get('height', 1.6))
        elif dimensions is not None:
            length, width, height = float(dimensions[0]), float(dimensions[1]), float(dimensions[2])
        else:
            # Fallback to KITTI template for category (no LLM)
            t = KITTI_CUBOID_TEMPLATES.get(category, KITTI_CUBOID_TEMPLATES['Unknown'])
            length = float(t['length'])
            width = float(t['width'])
            height = float(t['height'])
    else:
        length = float(length)
        width = float(width)
        height = float(height)
    
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
        'category': category or pose_result.get('category', None),
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


def fit_cuboid_to_points_outdoor(points: np.ndarray,
                                 dimensions: Tuple[float, float, float],
                                 step_center_search: float,
                                 max_step_center: int = 10,
                                 d_theta: float = 0.05,
                                 normals: Optional[np.ndarray] = None,
                                 score_weights: Tuple[float, float, float] = (1.0, 0.5, 2.0),
                                 ground_z: Optional[float] = None) -> Dict:
    """
    Outdoor / automotive-style cuboid fitting with fixed template dimensions (e.g. from LLM).

    Uses center line-search and yaw search over known length, width, height.

    Search is performed along a ray starting at the mean point and going in the
    direction of the mean (towards / away from the origin), and over yaw
    angles.     If ground_z is provided, fitting uses bird's eye view (BEV): points are
    projected to the xy-plane and the cuboid bottom is fixed at ground_z
    (center z = ground_z + height/2). Only the two rectangle edges closest to
    the LiDAR origin are used for the distance term. The best hypothesis
    minimizes:

    - Squared distance from 2D points to the two closest edges of the
      rectangle (BEV), plus outlier fraction.
    - Geometric consistency: local surface normals should align with the
      vector from the cuboid center to each point.
    - Outlier penalty: fraction of points that fall outside the cuboid.
    
    Args:
        points: np.ndarray (N, 3) - 3D points in LiDAR coordinates
        dimensions: Tuple (length, width, height) in meters
        step_center_search: Step size for center search along the ray
        max_step_center: Maximum number of steps for center search
        d_theta: Angular step size for yaw search (radians)
        normals: Optional (N, 3) array of surface normals for geometric consistency
        score_weights: Tuple (w_dist, w_geo, w_out) - weights for distance, 
                      geometric consistency, and outlier penalty terms
        ground_z: If provided, cuboid bottom is fixed at this z (center z = ground_z + height/2).
    
    Returns:
        Dictionary with 'center', 'yaw', 'length', 'width', 'height', 'score', 'method'
    """
    if points.size == 0:
        raise ValueError("Cannot fit cuboid to empty point cloud")

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected points shape (N, 3), got {points.shape}")

    if normals is not None and normals.shape != points.shape:
        raise ValueError("normals must have shape (N, 3) if provided")

    length, width, height = map(float, dimensions)
    w_dist, w_geo, w_out = map(float, score_weights)
    half_h = height / 2.0

    # Mean of the points
    mu = np.mean(points, axis=0)

    use_bev = ground_z is not None
    if use_bev:
        # Bird's eye view: work in 2D (xy only); center z fixed at ground_z + half_h
        points_xy = points[:, :2]
        mu_xy = np.mean(points_xy, axis=0)
        ray_dir_xy = mu_xy.copy()
        norm_xy = np.linalg.norm(ray_dir_xy)
        if norm_xy > 1e-6:
            ray_dir_xy /= norm_xy
        else:
            ray_dir_xy = np.array([1.0, 0.0], dtype=float)
    else:
        # Full 3D: ray from origin towards mean
        ray_dir = mu.copy()
        norm_ray = np.linalg.norm(ray_dir)
        if norm_ray > 1e-6:
            ray_dir /= norm_ray
        else:
            ray_dir = np.array([0.0, 0.0, 1.0], dtype=float)

    def _score_bev(center_xy: np.ndarray, yaw: float) -> float:
        """Score for bird's eye view: 2D rectangle, two edges closest to LiDAR origin."""
        half_l = length / 2.0
        half_w = width / 2.0
        cos_y = np.cos(yaw)
        sin_y = np.sin(yaw)
        u_xy = np.array([cos_y, sin_y])
        v_xy = np.array([-sin_y, cos_y])
        rel_xy = points_xy - center_xy[None, :]
        x_c = rel_xy @ u_xy
        y_c = rel_xy @ v_xy
        outside = (np.abs(x_c) > half_l) | (np.abs(y_c) > half_w)
        outlier_frac = float(np.mean(outside))
        # Face centers (2D) for the four edges
        center_front = center_xy + u_xy * half_l
        center_back = center_xy - u_xy * half_l
        center_right = center_xy + v_xy * half_w
        center_left = center_xy - v_xy * half_w
        lateral_centers = np.stack([center_front, center_back, center_right, center_left], axis=0)
        dists_to_origin = np.linalg.norm(lateral_centers, axis=1)
        two_closest_idx = np.argsort(dists_to_origin)[:2]
        sq_dists = []
        for idx in two_closest_idx:
            if idx == 0:
                n_vec, p0 = u_xy, center_front
            elif idx == 1:
                n_vec, p0 = -u_xy, center_back
            elif idx == 2:
                n_vec, p0 = v_xy, center_right
            else:
                n_vec, p0 = -v_xy, center_left
            diff = points_xy - p0[None, :]
            d_plane = np.abs(diff @ n_vec)
            sq_dists.append(d_plane ** 2)
        sq_dists = np.stack(sq_dists, axis=1)
        mean_min_sq_dist = float(np.mean(np.min(sq_dists, axis=1)))
        return w_dist * mean_min_sq_dist + w_out * outlier_frac

    def _score_for_hypothesis(center: np.ndarray, yaw: float, 
                              dims: Tuple[float, float, float],
                              weights: Tuple[float, float, float]) -> float:
        # Extract dimensions
        l, w_dim, h = dims
        half_l = l / 2.0
        half_w = w_dim / 2.0
        half_h_inner = h / 2.0
        
        # Rotation matrix around Z
        cos_y = np.cos(yaw)
        sin_y = np.sin(yaw)
        R_z = np.array([
            [cos_y, -sin_y, 0.0],
            [sin_y,  cos_y, 0.0],
            [0.0,    0.0,   1.0],
        ])

        # Local cuboid axes in world frame
        u = R_z @ np.array([1.0, 0.0, 0.0])  # length axis
        v = R_z @ np.array([0.0, 1.0, 0.0])  # width axis
        w = np.array([0.0, 0.0, 1.0])        # height axis (z-up)

        # Transform points relative to center
        rel = points - center[None, :]

        # Coordinates in cuboid local frame
        x_c = rel @ u
        y_c = rel @ v
        z_c = rel @ w

        # Outlier mask: outside if any coordinate exceeds half-extent
        outside = (
            (np.abs(x_c) > half_l) |
            (np.abs(y_c) > half_w) |
            (np.abs(z_c) > half_h_inner)
        )
        outlier_frac = float(np.mean(outside))

        # Plane centers for the four lateral faces in world frame (no top/bottom for scoring)
        center_front = center + u * half_l
        center_back = center - u * half_l
        center_right = center + v * half_w
        center_left = center - v * half_w
        
        # Distances of lateral face centers to origin (only lateral faces used)
        lateral_centers = np.stack([
            center_front, center_back,
            center_right, center_left,
        ], axis=0)
        dists_to_origin = np.linalg.norm(lateral_centers, axis=1)
        # Two lateral faces whose centers are closest to origin
        two_closest_idx = np.argsort(dists_to_origin)[:2]

        face_normals = []
        for idx in two_closest_idx:
            if idx == 0:
                face_normals.append((u, center_front))
            elif idx == 1:
                face_normals.append((-u, center_back))
            elif idx == 2:
                face_normals.append((v, center_right))
            elif idx == 3:
                face_normals.append((-v, center_left))

        # Distance to visible lateral faces only (no top, no z-direction)
        sq_dists = []
        for n_vec, p0 in face_normals:
            diff = points - p0[None, :]
            d_plane = np.abs(diff @ n_vec)
            sq_dists.append(d_plane ** 2)
        sq_dists = np.stack(sq_dists, axis=1)  # (N, 2)
        min_sq_dist = np.min(sq_dists, axis=1)
        mean_min_sq_dist = float(np.mean(min_sq_dist))

        # Geometric consistency term
        geo_term = 0.0
        if normals is not None:
            vec_center_to_point = rel
            norms_v = np.linalg.norm(vec_center_to_point, axis=1) + 1e-8
            dir_center_to_point = vec_center_to_point / norms_v[:, None]
            cos_angles = np.sum(normals * dir_center_to_point, axis=1)
            geo_term = float(np.mean(1.0 - np.abs(cos_angles)))

        w_d, w_g, w_o = weights
        score = (
            w_d * mean_min_sq_dist +
            w_g * geo_term +
            w_o * outlier_frac
        )
        return score
    
    best_score = float('inf')
    best_center = None
    best_yaw = 0.0

    if use_bev:
        # Center and yaw search in 2D; z fixed at ground_z + half_h
        for n in range(max_step_center + 1):
            center_xy = mu_xy + step_center_search * n * ray_dir_xy
            center = np.array([center_xy[0], center_xy[1], ground_z + half_h], dtype=float)
            for yaw in np.arange(0.0, np.pi, d_theta):
                score = _score_bev(center_xy, yaw)
                if score < best_score:
                    best_score = score
                    best_center = center.copy()
                    best_yaw = float(yaw)
    else:
        # Center and yaw search in 3D
        for n in range(max_step_center + 1):
            center = mu + step_center_search * n * ray_dir
            for yaw in np.arange(0.0, np.pi, d_theta):
                score = _score_for_hypothesis(center, yaw, (length, width, height), (w_dist, w_geo, w_out))
                if score < best_score:
                    best_score = score
                    best_center = center.copy()
                    best_yaw = float(yaw)

    return {
        'center': best_center,
        'yaw': best_yaw,
        'length': length,
        'width': width,
        'height': height,
        'score': best_score,
        'method': 'cuboid_fit_outdoor',
    }


def fit_cuboid_to_points_indoor(
    points: np.ndarray,
    d_theta: float = 0.05,
    margin: float = 0.05,
    min_extent: float = 0.05,
    inlier_quantile: float = 0.96,
    coverage_weight: float = 4.0,
) -> Dict:
    """
    Indoor-oriented cuboid: deformable axis-aligned box in yaw, minimizing volume so that
    (with margin) all cluster points lie inside — equivalent to maximizing point density N/V.

    For each yaw, points are expressed in a horizontal frame (length along u(yaw), width along v).
    Extents are the minimal L, W, H containing the projected coordinates plus margin.

    Args:
        points: (N, 3) LiDAR points for one cluster.
        d_theta: Yaw sampling step (radians), 0..pi (rectangle symmetry).
        margin: Extra half-padding (meters) on each axis in the local frame.
        min_extent: Minimum L, W, H (meters) to avoid degenerate boxes.
        inlier_quantile: Central quantile used to reject sparse outliers before
                         building the fitted box (higher keeps more points).
        coverage_weight: Penalty weight for excluding too many points. Higher
                         values keep more points at the cost of looser boxes.

    Returns:
        Same keys as outdoor fit: center, yaw, length, width, height, score (volume), method.
    """
    if points.size == 0:
        raise ValueError("Cannot fit cuboid to empty point cloud")

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected points shape (N, 3), got {points.shape}")

    margin = float(margin)
    min_extent = float(min_extent)
    d_theta = float(d_theta)
    inlier_quantile = float(inlier_quantile)
    coverage_weight = float(coverage_weight)
    inlier_quantile = float(np.clip(inlier_quantile, 0.80, 1.0))
    coverage_weight = max(0.0, coverage_weight)

    mu = np.mean(points, axis=0)
    rel = points - mu[None, :]

    best_score = float("inf")
    best_yaw = 0.0
    best_center = mu.copy()
    best_l = best_w = best_h = min_extent

    thetas = np.arange(0.0, np.pi, d_theta, dtype=float)
    if thetas.size == 0:
        thetas = np.array([0.0], dtype=float)

    for yaw in thetas:
        cos_y = float(np.cos(yaw))
        sin_y = float(np.sin(yaw))
        u = np.array([cos_y, sin_y, 0.0], dtype=float)
        v = np.array([-sin_y, cos_y, 0.0], dtype=float)
        w_axis = np.array([0.0, 0.0, 1.0], dtype=float)
        qx = rel @ u
        qy = rel @ v
        qz = rel @ w_axis

        tail_q = 0.5 * (1.0 - inlier_quantile)
        low_q = tail_q
        high_q = 1.0 - tail_q
        xmin, xmax = float(np.quantile(qx, low_q)), float(np.quantile(qx, high_q))
        ymin, ymax = float(np.quantile(qy, low_q)), float(np.quantile(qy, high_q))
        zmin, zmax = float(np.quantile(qz, low_q)), float(np.quantile(qz, high_q))
        length = max(min_extent, xmax - xmin + 2.0 * margin)
        width = max(min_extent, ymax - ymin + 2.0 * margin)
        height = max(min_extent, zmax - zmin + 2.0 * margin)
        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)
        cz = 0.5 * (zmin + zmax)
        c_world = mu + u * cx + v * cy + w_axis * cz

        half_l = 0.5 * length
        half_w = 0.5 * width
        half_h = 0.5 * height
        inlier_mask = (
            (np.abs(qx - cx) <= half_l) &
            (np.abs(qy - cy) <= half_w) &
            (np.abs(qz - cz) <= half_h)
        )
        inlier_count = int(np.count_nonzero(inlier_mask))
        coverage = inlier_count / float(points.shape[0])
        volume = length * width * height
        density = inlier_count / max(volume, 1e-6)
        score = (1.0 / max(density, 1e-6)) + coverage_weight * (1.0 - coverage)

        if score < best_score:
            best_score = score
            best_yaw = float(yaw)
            best_center = c_world
            best_l, best_w, best_h = length, width, height

    return {
        "center": best_center,
        "yaw": best_yaw,
        "length": float(best_l),
        "width": float(best_w),
        "height": float(best_h),
        "score": float(best_score),
        "method": "cuboid_fit_indoor_dense",
    }


# Backward-compatible name: outdoor / fixed-dimension pipeline.
fit_cuboid_to_points = fit_cuboid_to_points_outdoor

