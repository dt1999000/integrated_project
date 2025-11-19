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


class Projection2DTo3D:
    """
    Class for 2D to 3D projection using camera intrinsics, extrinsics, and lidar transformations.
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
                                  distance_threshold: float = 0.5) -> np.ndarray:
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

class PointCloud:
    """
    Class for representing a point cloud.
    """
    def __init__(self, point_cloud: np.ndarray, coordinate_systems: Optional[Dict[str, np.ndarray]] = None):
        self.original_point_cloud = point_cloud[:, :3] if point_cloud.shape[1] > 3 else point_cloud
        self.coordinate_systems = coordinate_systems
        self.ground_removed = False

    def remove_ground_plane_ransac(self, distance_threshold: float = 0.3,
                                   ransac_n: int = 3, num_iterations: int = 1000) -> np.ndarray:
        """
        Remove ground plane from point cloud using RANSAC.
        
        Args:
            distance_threshold: Maximum distance from plane to be considered inlier
            ransac_n: Number of points to sample for plane fitting
            num_iterations: Number of RANSAC iterations
            
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
        
        print(f"RANSAC ground plane removal:")
        print(f"  Ground plane equation: {plane_model[:3]} * x + {plane_model[3]} = 0")
        print(f"  Ground points removed: {n_ground_points}")
        print(f"  Remaining points: {n_remaining_points}")
        print(f"  Removal ratio: {n_ground_points / len(self.original_point_cloud) * 100:.2f}%")
        
        self.ground_removed = True
        self.point_cloud_plane_removed = filtered_points
        self.ground_plane_model = plane_model
        self.ground_inliers = self.original_point_cloud[inliers]
        
        
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
            cluster_colors = plt.cm.tab20(np.linspace(0, 1, len(clusters)))[:, :3]
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
        circle = plt.Circle((u, v), 10, color='red', fill=False, linewidth=2.5)
        ax.add_patch(circle)
        # Draw crosshair
        ax.plot([u-15, u+15], [v, v], 'r-', linewidth=2)
        ax.plot([u, u], [v-15, v+15], 'r-', linewidth=2)
        # Add label
        ax.text(u+20, v, f'P{i+1}', color='red', fontsize=12, 
                weight='bold', bbox=dict(boxstyle='round,pad=0.3', 
                                        facecolor='yellow', alpha=0.7))
    
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