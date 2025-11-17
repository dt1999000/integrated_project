"""
nuScenes Dataset Loader with YOLOWorld Object Detection
This script loads the nuScenes dataset and performs object detection using YOLOWorld model.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from ultralytics import YOLO
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points, box_in_image
from pyquaternion import Quaternion
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
from nuscenes.utils.data_classes import LidarPointCloud


class NuScenesDatasetLoader:
    """Class to load and process nuScenes dataset."""
    
    def __init__(self, dataroot: str = "v1.0-mini", version: str = "v1.0-mini", verbose: bool = True):
        """
        Initialize nuScenes dataset loader.
        
        Args:
            dataroot: Root directory of the dataset
            version: Version of the dataset (e.g., 'v1.0-mini')
            verbose: Whether to print verbose information
        """
        self.dataroot = dataroot
        self.version = version
        self.verbose = verbose
        self.nusc = None
        
    def load_dataset(self):
        """Load the nuScenes dataset."""
        if not os.path.exists(self.dataroot):
            raise FileNotFoundError(f"Dataset directory not found: {self.dataroot}")
        
        self.nusc = NuScenes(version=self.version, dataroot=self.dataroot, verbose=self.verbose)
        print(f"nuScenes dataset loaded: {self.version}")
        print(f"Number of scenes: {len(self.nusc.scene)}")
        print(f"Number of samples: {len(self.nusc.sample)}")
    
    def get_camera_samples(self, camera_channel: str = "CAM_FRONT") -> List[Dict]:
        """
        Get all camera samples for a specific camera channel.
        
        Args:
            camera_channel: Camera channel name (e.g., 'CAM_FRONT', 'CAM_BACK', etc.)
            
        Returns:
            List of sample dictionaries with image paths and metadata
        """
        if self.nusc is None:
            raise RuntimeError("Dataset must be loaded first. Call load_dataset() first.")
        
        samples = []
        for sample in self.nusc.sample:
            sample_token = sample['token']
            sample_data_token = sample['data'][camera_channel]
            sample_data = self.nusc.get('sample_data', sample_data_token)
            
            image_path = os.path.join(self.nusc.dataroot, sample_data['filename'])
            
            samples.append({
                'sample_token': sample_token,
                'sample_data_token': sample_data_token,
                'image_path': image_path,
                'timestamp': sample['timestamp'],
                'scene_token': sample['scene_token'],
                'camera_channel': camera_channel
            })
        
        return samples
    
    def get_all_camera_samples(self) -> List[Dict]:
        """
        Get all camera samples from all camera channels.
        
        Returns:
            List of sample dictionaries from all cameras
        """
        camera_channels = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                          'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
        all_samples = []
        
        for channel in camera_channels:
            samples = self.get_camera_samples(channel)
            all_samples.extend(samples)
        
        return all_samples

    
    def load_nuscenes_data(self, sample_token: str, camera_channel: str = "CAM_FRONT"):
        """
        Load camera and lidar data from nuScenes for a given sample.
        
        Args:
            nusc: NuScenes object
            sample_token: Token of the sample
            camera_channel: Camera channel name
            
        Returns:
            Dictionary with camera data, lidar data, and transformation matrices
        """

        sample = self.nusc.get('sample', sample_token)
        
        # Get camera data
        sample_data_token = sample['data'][camera_channel]
        sample_data = self.nusc.get('sample_data', sample_data_token)
        calibrated_sensor = self.nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
        
        # Get lidar data
        lidar_token = sample['data']['LIDAR_TOP']
        lidar_data = self.nusc.get('sample_data', lidar_token)
        
        # Load point cloud
        pc_path = os.path.join(self.nusc.dataroot, lidar_data['filename'])
        pc = LidarPointCloud.from_file(pc_path)
        point_cloud = pc.points[:3, :].T  # Nx3 array
        
        # Camera intrinsic
        K = np.array(calibrated_sensor['camera_intrinsic'])
        
        cs_record = self.nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
        
        
        # Camera extrinsic: ego to camera
        cam_rotation_quat = Quaternion(cs_record['rotation'])
        T_ego_to_cam = np.eye(4)
        T_ego_to_cam[:3, :3] = cam_rotation_quat.rotation_matrix
        T_ego_to_cam[:3, 3] = np.array(cs_record['translation'])
        
        # Ego to lidar
        lidar_cs = self.nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
        lidar_rotation_quat = Quaternion(lidar_cs['rotation'])
        T_ego_to_lidar = np.eye(4)
        T_ego_to_lidar[:3, :3] = lidar_rotation_quat.rotation_matrix
        T_ego_to_lidar[:3, 3] = np.array(lidar_cs['translation'])
        
        # Camera to lidar
        T_cam_to_lidar = T_ego_to_lidar @ np.linalg.inv(T_ego_to_cam)
        
        return {
                "sample_token": sample_token,
                "image_path": os.path.join(self.nusc.dataroot, sample_data['filename']),
                "point_cloud": point_cloud,                 # Nx3 numpy array
                "camera_intrinsic": K,                      # 3x3
                "camera_extrinsic": T_ego_to_cam,           # 4x4
                "camera_to_lidar_transform": T_cam_to_lidar,              # 4x4
                "nusc": self.nusc
            }


