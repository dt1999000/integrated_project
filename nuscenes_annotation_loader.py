"""
nuScenes Annotation Loader and Visualizer
This module provides functionality to load and visualize nuScenes LiDAR annotations.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points, box_in_image, transform_matrix
from pyquaternion import Quaternion
import open3d as o3d


class NuScenesAnnotationLoader:
    """
    Class to load and process nuScenes annotations for LiDAR point clouds.
    """
    
    def __init__(self, nusc: NuScenes):
        """
        Initialize with a NuScenes instance.
        
        Args:
            nusc: NuScenes instance
        """
        self.nusc = nusc
    
    def get_lidar_annotations(self, sample_token: str) -> List[Dict]:
        """
        Get all annotations for a sample, transformed to LiDAR coordinate frame.
        
        Args:
            sample_token: Token of the sample
            
        Returns:
            List of dictionaries, each containing:
            - 'token': Annotation token
            - 'category_name': Category name (e.g., 'car', 'pedestrian')
            - 'category_token': Category token
            - 'bbox_3d': 3D bounding box in LiDAR coordinates
            - 'bbox_2d': 2D bounding box in image coordinates (if visible)
            - 'visibility': Visibility level (if available)
        """
        sample = self.nusc.get('sample', sample_token)
        
        # Get all annotation tokens for this sample
        ann_tokens = sample['anns']
        
        annotations = []
        
        for ann_token in ann_tokens:
            ann = self.nusc.get('sample_annotation', ann_token)
            
            # Get category information from instance record
            instance_token = ann['instance_token']
            instance = self.nusc.get('instance', instance_token)
            category_token = instance['category_token']
            category = self.nusc.get('category', category_token)
            category_name = category['name']
            
            # Create 3D bounding box in global coordinates
            bbox_3d_global = Box(
                center=ann['translation'],
                size=ann['size'],
                orientation=Quaternion(ann['rotation'])
            )
            
            # Transform to LiDAR coordinates
            # First, get the sample data token for the LiDAR sensor
            lidar_sd_token = sample['data']['LIDAR_TOP']
            lidar_sd = self.nusc.get('sample_data', lidar_sd_token)
            
            # Get the calibrated sensor token for LiDAR
            lidar_cs_token = lidar_sd['calibrated_sensor_token']
            lidar_cs = self.nusc.get('calibrated_sensor', lidar_cs_token)
            
            # Get the ego pose token for LiDAR
            lidar_ep_token = lidar_sd['ego_pose_token']
            lidar_ep = self.nusc.get('ego_pose', lidar_ep_token)
            
            # Create transformation matrices
            # Global to ego vehicle
            global_to_ego = transform_matrix(
                lidar_ep['translation'],
                Quaternion(lidar_ep['rotation']),
                inverse=False
            )
            
            # Ego vehicle to sensor (LiDAR)
            ego_to_sensor = transform_matrix(
                lidar_cs['translation'],
                Quaternion(lidar_cs['rotation']),
                inverse=False
            )
            
            # Global to sensor (LiDAR)
            global_to_sensor = np.dot(ego_to_sensor, global_to_ego)
            
            # Transform the 3D bounding box to LiDAR coordinates
            bbox_3d_lidar = bbox_3d_global.transform(global_to_sensor)
            
            # Check if the annotation is visible in the front camera
            # Get the sample data token for the front camera
            cam_front_sd_token = sample['data']['CAM_FRONT']
            cam_front_sd = self.nusc.get('sample_data', cam_front_sd_token)
            
            # Get camera intrinsic and extrinsic
            cam_cs_token = cam_front_sd['calibrated_sensor_token']
            cam_cs = self.nusc.get('calibrated_sensor', cam_cs_token)
            
            cam_intrinsic = np.array(cam_cs['camera_intrinsic'])
            
            # Get ego pose for camera
            cam_ep_token = cam_front_sd['ego_pose_token']
            cam_ep = self.nusc.get('ego_pose', cam_ep_token)
            
            # Create transformation matrices for camera
            global_to_ego_cam = transform_matrix(
                cam_ep['translation'],
                Quaternion(cam_ep['rotation']),
                inverse=False
            )
            
            ego_to_sensor_cam = transform_matrix(
                cam_cs['translation'],
                Quaternion(cam_cs['rotation']),
                inverse=False
            )
            
            global_to_sensor_cam = np.dot(ego_to_sensor_cam, global_to_ego_cam)
            
            # Transform the 3D bounding box to camera coordinates
            bbox_3d_cam = bbox_3d_global.transform(global_to_sensor_cam)
            
            # Check if the box is in the image
            _, bbox_2d, _ = view_points(
                bbox_3d_cam.corners(),
                cam_intrinsic,
                normalize=True
            )
            
            # Check if the box is visible in the image
            in_image = box_in_image(bbox_2d, cam_front_sd['width'], cam_front_sd['height'])
            
            # Get visibility if available
            visibility = ann.get('visibility', None)
            if visibility:
                visibility = self.nusc.get('visibility', visibility)['token']
            
            annotation = {
                'token': ann_token,
                'category_name': category_name,
                'category_token': category_token,
                'bbox_3d': bbox_3d_lidar,
                'bbox_2d': bbox_2d if in_image else None,
                'visibility': visibility
            }
            
            annotations.append(annotation)
        
        return annotations
    
    def get_annotation_points(self, annotations: List[Dict], 
                            num_points_per_box: int = 100) -> Dict[str, np.ndarray]:
        """
        Get sample points from 3D bounding boxes.
        
        Args:
            annotations: List of annotation dictionaries
            num_points_per_box: Number of points to sample per bounding box
            
        Returns:
            Dictionary mapping annotation token to Nx3 array of points
        """
        annotation_points = {}
        
        for ann in annotations:
            bbox_3d = ann['bbox_3d']
            
            # Sample points inside the 3D bounding box
            # Get the corners of the box
            corners = bbox_3d.corners().T  # Shape: (3, 8)
            
            # Get the min and max coordinates
            min_coords = np.min(corners, axis=1)
            max_coords = np.max(corners, axis=1)
            
            # Sample points uniformly inside the box
            points = np.random.uniform(
                low=min_coords,
                high=max_coords,
                size=(num_points_per_box, 3)
            )
            
            annotation_points[ann['token']] = points
        
        return annotation_points


class AnnotationVisualizer:
    """
    Class to visualize nuScenes annotations with point clouds.
    """
    
    def __init__(self, point_cloud: np.ndarray):
        """
        Initialize with a point cloud.
        
        Args:
            point_cloud: Nx3 array of 3D points
        """
        self.point_cloud = point_cloud
    
    def visualize_with_annotations(self, annotations: List[Dict],
                                 annotation_points: Optional[Dict[str, np.ndarray]] = None,
                                 title: str = "Point Cloud with Annotations"):
        """
        Visualize point cloud with annotations.
        
        Args:
            annotations: List of annotation dictionaries
            annotation_points: Optional dictionary of annotation points
            title: Window title
        """
        geometries = []
        
        # Add point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.point_cloud)
        pcd.paint_uniform_color([0.5, 0.5, 0.5])  # Gray
        geometries.append(pcd)
        
        # Add annotation bounding boxes
        for ann in annotations:
            bbox_3d = ann['bbox_3d']
            
            # Create a bounding box visualization
            lines = [
                [0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3],
                [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]
            ]
            
            colors = [
                [1, 0, 0] for _ in lines
            ]  # Red for all lines
            
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(bbox_3d.corners().T)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector(colors)
            geometries.append(line_set)
        
        # Add annotation points if provided
        if annotation_points:
            for ann_token, points in annotation_points.items():
                if len(points) > 0:
                    ann_pcd = o3d.geometry.PointCloud()
                    ann_pcd.points = o3d.utility.Vector3dVector(points)
                    ann_pcd.paint_uniform_color([1, 0, 0])  # Red
                    geometries.append(ann_pcd)
        
        # Visualize
        o3d.visualization.draw_geometries(geometries, window_name=title)
    
    def visualize_annotation_comparison(self, 
                                      annotation_points: Dict[str, np.ndarray],
                                      detected_clusters: List[np.ndarray],
                                      title: str = "Annotation vs Detection Comparison"):
        """
        Visualize comparison between annotations and detected clusters.
        
        Args:
            annotation_points: Dictionary of annotation points
            detected_clusters: List of detected clusters
            title: Window title
        """
        geometries = []
        
        # Add point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.point_cloud)
        pcd.paint_uniform_color([0.5, 0.5, 0.5])  # Gray
        geometries.append(pcd)
        
        # Add annotation points in red
        for ann_token, points in annotation_points.items():
            if len(points) > 0:
                ann_pcd = o3d.geometry.PointCloud()
                ann_pcd.points = o3d.utility.Vector3dVector(points)
                ann_pcd.paint_uniform_color([1, 0, 0])  # Red
                geometries.append(ann_pcd)
        
        # Add detected clusters in different colors
        colors = plt.cm.tab20(np.linspace(0, 1, len(detected_clusters)))[:, :3]
        for i, cluster_indices in enumerate(detected_clusters):
            if len(cluster_indices) > 0:
                cluster_pcd = o3d.geometry.PointCloud()
                cluster_pcd.points = o3d.utility.Vector3dVector(
                    self.point_cloud[cluster_indices]
                )
                cluster_pcd.paint_uniform_color(colors[i])
                geometries.append(cluster_pcd)
        
        # Visualize
        o3d.visualization.draw_geometries(geometries, window_name=title)
