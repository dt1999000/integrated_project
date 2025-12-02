"""
KITTI Dataset Loader - Simplified to match NuScenes pattern
This script loads the KITTI dataset for use with the 3D detection pipeline.
"""

import os
import numpy as np
from typing import List, Dict, Tuple, Optional


class KITTIDatasetLoader:
    """Class to load and process KITTI dataset - matches NuScenes interface."""

    def __init__(self, dataroot: str = "dataset/kitti", split: str = "training", verbose: bool = True):
        """
        Initialize KITTI dataset loader.

        Args:
            dataroot: Root directory of the dataset
            split: Dataset split ('training' or 'testing')
            verbose: Whether to print verbose information
        """
        self.dataroot = dataroot
        self.split = split
        self.verbose = verbose

        # Set up paths
        self.split_dir = os.path.join(dataroot, split)
        self.image_dir = os.path.join(self.split_dir, "image_2")
        self.velodyne_dir = os.path.join(self.split_dir, "velodyne")
        self.calib_dir = os.path.join(self.split_dir, "calib")
        self.label_dir = os.path.join(self.split_dir, "label_2")

        self.num_samples = 0
        self.samples = []

    def load_dataset(self):
        """Load and validate the KITTI dataset structure."""
        # Validate required directories exist
        required_dirs = [self.image_dir, self.velodyne_dir, self.calib_dir]
        if self.split == "training":
            required_dirs.append(self.label_dir)

        for dir_path in required_dirs:
            if not os.path.exists(dir_path):
                raise FileNotFoundError(f"Required directory not found: {dir_path}")

        # Count available samples and create sample list (like NuScenes)
        image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.png')])
        self.num_samples = len(image_files)

        # Create simple sample list with indices
        self.samples = [{'index': i} for i in range(self.num_samples)]

        if self.verbose:
            print(f"KITTI dataset loaded: {self.split} split")
            print(f"Number of samples: {self.num_samples}")
            print(f"Dataset root: {self.dataroot}")

    def load_kitti_data(self, sample_index: int) -> Dict:
        """
        Load camera, LiDAR, and ground truth data for a sample.
        Returns data in the same format as NuScenes loader.

        Args:
            sample_index: Index of the sample to load (0 to num_samples-1)

        Returns:
            Dictionary matching NuScenes format:
            {
                "sample_index": int,
                "image_path": str,
                "point_cloud": np.ndarray (Nx3),
                "camera_intrinsic": np.ndarray (3x3),
                "camera_extrinsic": np.ndarray (4x4),
                "camera_to_lidar_transform": np.ndarray (4x4),
                "ground_truth_boxes": List[Dict]  # KITTI-specific
            }
        """
        if sample_index < 0 or sample_index >= self.num_samples:
            raise ValueError(f"Sample index {sample_index} out of range [0, {self.num_samples-1}]")

        # Load components
        image_path = os.path.join(self.image_dir, f"{sample_index:06d}.png")
        point_cloud = self._load_point_cloud(sample_index)
        calib = self._load_calibration(sample_index)
        camera_intrinsic, camera_to_lidar_transform = self._compute_transforms(calib)

        # Load ground truth labels and convert to cuboid format
        ground_truth_boxes = []
        if self.split == "training":
            ground_truth_boxes = self._load_ground_truth_cuboids(sample_index, calib)

        return {
            "sample_index": sample_index,
            "image_path": image_path,
            "point_cloud": point_cloud,
            "camera_intrinsic": camera_intrinsic,
            "camera_extrinsic": np.eye(4),
            "camera_to_lidar_transform": camera_to_lidar_transform,
            "ground_truth_boxes": ground_truth_boxes
        }

    def _load_point_cloud(self, idx: int) -> np.ndarray:
        """Load point cloud from binary file."""
        pc_path = os.path.join(self.velodyne_dir, f"{idx:06d}.bin")
        point_cloud = np.fromfile(pc_path, dtype=np.float32)
        point_cloud = point_cloud.reshape(-1, 4)
        return point_cloud[:, :3]  # Return Nx3 (x, y, z)

    def _load_calibration(self, idx: int) -> Dict:
        """Load and parse calibration file."""
        calib_path = os.path.join(self.calib_dir, f"{idx:06d}.txt")
        calib = {}

        with open(calib_path, 'r') as f:
            for line in f.readlines():
                if ':' not in line:
                    continue
                key, value = line.split(':', 1)
                key = key.strip()
                values = np.array([float(x) for x in value.split()])

                if key == 'P2':
                    calib['P2'] = values.reshape(3, 4)
                elif key == 'R0_rect':
                    calib['R0_rect'] = values.reshape(3, 3)
                elif key == 'Tr_velo_to_cam':
                    calib['Tr_velo_to_cam'] = values.reshape(3, 4)

        return calib

    def _compute_transforms(self, calib: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Compute camera intrinsic and camera-to-lidar transformation matrices."""
        # Camera intrinsic from P2
        P2 = calib['P2']
        camera_intrinsic = P2[:, :3]

        # Camera-to-lidar transform (inverse of lidar-to-camera)
        Tr_velo_to_cam = calib['Tr_velo_to_cam']
        R0_rect = calib['R0_rect']

        # Convert to 4x4 matrices
        Tr_velo_to_cam_4x4 = np.vstack([Tr_velo_to_cam, [0, 0, 0, 1]])
        R0_rect_4x4 = np.eye(4)
        R0_rect_4x4[:3, :3] = R0_rect

        # Full transform: LiDAR → Camera
        lidar_to_cam = R0_rect_4x4 @ Tr_velo_to_cam_4x4

        # Invert to get Camera → LiDAR
        camera_to_lidar_transform = np.linalg.inv(lidar_to_cam)

        return camera_intrinsic, camera_to_lidar_transform

    def _load_ground_truth_cuboids(self, idx: int, calib: Dict) -> List[Dict]:
        """
        Load ground truth labels and convert to cuboid format for visualization.
        Returns cuboids in LiDAR coordinates compatible with element.py.
        """
        label_path = os.path.join(self.label_dir, f"{idx:06d}.txt")
        if not os.path.exists(label_path):
            return []

        # Get transformation matrix
        Tr_velo_to_cam = calib['Tr_velo_to_cam']
        R0_rect = calib['R0_rect']

        Tr_velo_to_cam_4x4 = np.vstack([Tr_velo_to_cam, [0, 0, 0, 1]])
        R0_rect_4x4 = np.eye(4)
        R0_rect_4x4[:3, :3] = R0_rect

        lidar_to_cam = R0_rect_4x4 @ Tr_velo_to_cam_4x4
        cam_to_lidar = np.linalg.inv(lidar_to_cam)

        cuboids = []
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) < 15:
                    continue

                obj_type = parts[0]
                if obj_type == 'DontCare':
                    continue

                # Parse KITTI label: Type Truncated Occluded Alpha bbox2d(4) dim(3) loc(3) Rotation_y
                h, w, l = float(parts[8]), float(parts[9]), float(parts[10])  # height, width, length
                x, y, z = float(parts[11]), float(parts[12]), float(parts[13])  # location in camera coords
                rotation_y = float(parts[14])

                # Compute 8 corners in camera coordinates
                # KITTI: bottom center of 3D box is at (x, y, z), box extends up by h
                corners_cam = np.array([
                    [-l/2, -l/2,  l/2,  l/2, -l/2, -l/2,  l/2,  l/2],  # x: length
                    [   0,    0,    0,    0,   -h,   -h,   -h,   -h],  # y: height (bottom at 0)
                    [-w/2,  w/2,  w/2, -w/2, -w/2,  w/2,  w/2, -w/2]   # z: width
                ])

                # Apply rotation around Y-axis
                R_y = np.array([
                    [ np.cos(rotation_y), 0, np.sin(rotation_y)],
                    [                  0, 1,                  0],
                    [-np.sin(rotation_y), 0, np.cos(rotation_y)]
                ])
                corners_cam = R_y @ corners_cam

                # Translate to object location
                corners_cam[0, :] += x
                corners_cam[1, :] += y
                corners_cam[2, :] += z

                # Transform to LiDAR coordinates
                corners_cam_homogeneous = np.vstack([corners_cam, np.ones((1, 8))])
                corners_lidar_homogeneous = cam_to_lidar @ corners_cam_homogeneous
                corners_lidar = corners_lidar_homogeneous[:3, :].T  # 8x3

                # Create cuboid dict compatible with element.py
                cuboid = {
                    'category': obj_type,
                    'corners': corners_lidar,
                    'min_x': float(corners_lidar[:, 0].min()),
                    'max_x': float(corners_lidar[:, 0].max()),
                    'min_y': float(corners_lidar[:, 1].min()),
                    'max_y': float(corners_lidar[:, 1].max()),
                    'min_z': float(corners_lidar[:, 2].min()),
                    'max_z': float(corners_lidar[:, 2].max())
                }
                cuboids.append(cuboid)

        return cuboids
