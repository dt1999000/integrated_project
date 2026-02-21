"""
Dataset loading utilities for multi-format support (KITTI, nuScenes, sim/LinkedDataHandler).
Contains helper functions for loading samples from different dataset formats.
"""
import os
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
import cv2

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from .kitti_dataset_loader import KITTIDatasetLoader
from .nuscenes_dataset_loader import NuScenesDatasetLoader
from .dataset_loader import LinkedDataHandler


def detect_dataset_type(dataset_path: str) -> Optional[str]:
    """
    Detect dataset type based on folder structure and JSON files.
    
    Detection priority:
    1. LinkedDataHandler/sim: Check for dataset.json in root
    2. KITTI: Check for training/ or testing/ with image_2/, velodyne/, calib/
    3. nuScenes: Check for samples/, sweeps/, v1.0-*/ folders
    
    Args:
        dataset_path: Root directory of the dataset
        
    Returns:
        Dataset type: 'kitti', 'nuscenes', 'sim', or None if cannot determine
    """
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        return None
    
    # Check for LinkedDataHandler/sim format (dataset.json in root)
    dataset_json = dataset_path / "dataset.json"
    if dataset_json.exists():
        return "sim"
    
    # Check for KITTI structure
    training_dir = dataset_path / "training"
    testing_dir = dataset_path / "testing"
    
    if training_dir.exists() or testing_dir.exists():
        split_dir = training_dir if training_dir.exists() else testing_dir
        has_image_2 = (split_dir / "image_2").exists()
        has_velodyne = (split_dir / "velodyne").exists()
        has_calib = (split_dir / "calib").exists()
        
        if has_image_2 and has_velodyne and has_calib:
            return "kitti"
    
    # Check for nuScenes structure
    has_samples = (dataset_path / "samples").exists()
    has_sweeps = (dataset_path / "sweeps").exists()
    has_v1 = any(d.name.startswith("v1.0-") for d in dataset_path.iterdir() if d.is_dir())
    
    if has_samples and (has_sweeps or has_v1):
        return "nuscenes"
    
    return None


def load_dataset_sample(
    dataset_path: str,
    sample_index: int = 0,
    dataset_type: Optional[str] = None,
    filter_forward_only: bool = True
) -> Tuple[Optional[Dict], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load sample from dataset (KITTI, nuScenes, or sim format).
    
    Note: This function does NOT remove ground plane. Ground plane removal
    should be done in the detection pipeline (Step 1).
    
    Args:
        dataset_path: Root directory of dataset
        sample_index: Index or token of sample to load (int for KITTI, str for nuScenes/sim)
        dataset_type: 'kitti', 'nuscenes', 'sim', or None (auto-detect)
        filter_forward_only: Whether to keep only forward-facing points (x > 0) - for KITTI
        
    Returns:
        Tuple of (sample_meta_data dict, image array, point_cloud array)
        Returns (None, None, None) on error
    """
    # Auto-detect dataset type if not provided
    if dataset_type is None:
        dataset_type = detect_dataset_type(dataset_path)
        if dataset_type is None:
            print(f"Error: Cannot determine dataset type for {dataset_path}")
            return None, None, None
    
    # Route to appropriate loader
    if dataset_type == "kitti":
        return _load_kitti_sample(dataset_path, sample_index, filter_forward_only)
    elif dataset_type == "nuscenes":
        return _load_nuscenes_sample(dataset_path, sample_index)
    elif dataset_type == "sim":
        return _load_sim_sample(dataset_path, sample_index)
    else:
        print(f"Error: Unsupported dataset type: {dataset_type}")
        return None, None, None


def _load_kitti_sample(
    dataset_path: str,
    sample_index: int,
    filter_forward_only: bool
) -> Tuple[Optional[Dict], Optional[np.ndarray], Optional[np.ndarray]]:
    """Load KITTI sample."""
    try:
        # Determine split (prefer training, fallback to testing)
        training_dir = Path(dataset_path) / "training"
        testing_dir = Path(dataset_path) / "testing"
        
        if training_dir.exists():
            split = "training"
        elif testing_dir.exists():
            split = "testing"
        else:
            print(f"Error: Neither training/ nor testing/ found in {dataset_path}")
            return None, None, None
        
        # Load KITTI data
        dataset_loader = KITTIDatasetLoader(dataroot=str(dataset_path), split=split)
        dataset_loader.load_dataset()
        
        # Load synchronized camera, LiDAR, and ground truth data
        sample_data = dataset_loader.load_kitti_data(sample_index)
        
        if sample_data is None:
            print(f"Error: Failed to load KITTI sample {sample_index}")
            return None, None, None
        
        # Load image
        image_path = sample_data['image_path']
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image from {image_path}")
            return None, None, None
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get point cloud (raw, no ground removal)
        point_cloud = sample_data['point_cloud'].copy()
        
        # Filter forward-facing points if requested
        if filter_forward_only:
            point_cloud = point_cloud[point_cloud[:, 0] > 0]
        
        # Create normalized sample_meta_data
        sample_meta_data = {
            'image_path': image_path,
            'point_cloud_path': None,  # KITTI uses binary files
            'camera_intrinsic': sample_data['camera_intrinsic'],
            'camera_extrinsic': sample_data.get('camera_extrinsic', np.eye(4)),
            'camera_to_lidar_transform': sample_data['camera_to_lidar_transform'],
            'ground_truth_boxes': sample_data.get('ground_truth_boxes', []),
            'sample_index': sample_index,
            'dataset_type': 'kitti'
        }
        
        return sample_meta_data, image_rgb, point_cloud
        
    except Exception as e:
        print(f"Error loading KITTI sample: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def _load_nuscenes_sample(
    dataset_path: str,
    sample_token: str
) -> Tuple[Optional[Dict], Optional[np.ndarray], Optional[np.ndarray]]:
    """Load nuScenes sample."""
    try:
        # Determine version from directory structure
        version = None
        for d in Path(dataset_path).iterdir():
            if d.is_dir() and d.name.startswith("v1.0-"):
                version = d.name
                break
        
        if version is None:
            print(f"Error: Could not determine nuScenes version in {dataset_path}")
            return None, None, None
        
        # Load nuScenes data
        dataset_loader = NuScenesDatasetLoader(dataroot=str(dataset_path), version=version)
        dataset_loader.load_dataset()
        
        # Load sample data (assuming sample_token is provided)
        sample_data = dataset_loader.load_nuscenes_data(sample_token, camera_channel="CAM_FRONT")
        
        if sample_data is None:
            print(f"Error: Failed to load nuScenes sample {sample_token}")
            return None, None, None
        
        # Load image
        image_path = sample_data.get('image_path')
        if image_path:
            image = cv2.imread(image_path)
            if image is not None:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = None
        else:
            image_rgb = None
        
        # Get point cloud
        point_cloud = sample_data.get('point_cloud')
        if point_cloud is None:
            print(f"Error: No point cloud data in nuScenes sample")
            return None, None, None
        
        # Create normalized sample_meta_data
        sample_meta_data = {
            'image_path': image_path,
            'point_cloud_path': None,
            'camera_intrinsic': sample_data.get('camera_intrinsic'),
            'camera_extrinsic': sample_data.get('camera_extrinsic', np.eye(4)),
            'camera_to_lidar_transform': sample_data.get('camera_to_lidar_transform'),
            'ground_truth_boxes': sample_data.get('ground_truth_boxes', []),
            'sample_index': sample_token,
            'dataset_type': 'nuscenes'
        }
        
        return sample_meta_data, image_rgb, point_cloud
        
    except Exception as e:
        print(f"Error loading nuScenes sample: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def _load_sim_sample(
    dataset_path: str,
    link_token: str
) -> Tuple[Optional[Dict], Optional[np.ndarray], Optional[np.ndarray]]:
    """Load sim/LinkedDataHandler sample."""
    try:
        # Load LinkedDataHandler
        handler = LinkedDataHandler(root_dir=str(dataset_path), load_dataset=True)
        
        # Find the link and its subset
        link = None
        found_subset_name = None
        for subset_name in handler.list_subsets():
            subset = handler.subsets[subset_name]
            for l in subset['links']:
                if l['token'] == link_token:
                    link = l
                    found_subset_name = subset_name
                    break
            if link:
                break
        
        if link is None:
            print(f"Error: Link token {link_token} not found")
            return None, None, None
        
        if found_subset_name is None:
            print(f"Error: Could not determine subset for link token {link_token}")
            return None, None, None
        
        # Get image and point cloud paths from link
        rgb_sample = link['samples'].get('rgb')
        lidar_sample = link['samples'].get('lidar')
        
        if not rgb_sample or not lidar_sample:
            print(f"Error: Missing rgb or lidar sample in link")
            return None, None, None
        
        # Helper function to normalize filename (remove leading / and handle absolute paths)
        def normalize_filename(filename):
            """Normalize filename to relative path"""
            if not filename:
                return filename
            # Remove leading slashes
            filename = filename.lstrip('/').lstrip('\\')
            # If it's an absolute path (starts with drive letter), extract relative part
            if len(filename) > 1 and filename[1] == ':':
                # Windows absolute path like C:\rgb\file.jpg
                # Extract everything after the first backslash after the drive
                parts = filename.split('\\', 2)
                if len(parts) > 2:
                    filename = parts[2]
                else:
                    # Just drive and filename, take filename
                    filename = parts[-1]
            return filename
        
        # Load image
        image_path = None
        image_rgb = None
        if 'filename' in rgb_sample:
            filename = normalize_filename(rgb_sample['filename'])
            # Construct path: dataset_path / subset_name / samples / filename
            image_path = Path(dataset_path) / found_subset_name / "samples" / filename
            if image_path.exists():
                image = cv2.imread(str(image_path))
                if image is not None:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load point cloud
        point_cloud = None
        point_cloud_path = None
        if 'filename' in lidar_sample:
            filename = normalize_filename(lidar_sample['filename'])
            # Construct path: dataset_path / subset_name / samples / filename
            point_cloud_path = Path(dataset_path) / found_subset_name / "samples" / filename
            if point_cloud_path.exists():
                import open3d as o3d
                pcd = o3d.io.read_point_cloud(str(point_cloud_path))
                points = np.asarray(pcd.points)
                if len(points) > 0:
                    point_cloud = points
        
        if image_rgb is None or point_cloud is None:
            print(f"Error: Could not load image or point cloud")
            return None, None, None
        
        # Get calibration data
        calibration = link.get('calibration', {})
        camera_intrinsic = calibration.get('camera_intrinsic')
        camera_to_lidar_transform = calibration.get('camera_to_lidar_transform')
        
        # Create normalized sample_meta_data
        sample_meta_data = {
            'image_path': image_path,
            'point_cloud_path': point_cloud_path,
            'camera_intrinsic': camera_intrinsic,
            'camera_extrinsic': np.eye(4),  # Default if not available
            'camera_to_lidar_transform': camera_to_lidar_transform,
            'ground_truth_boxes': link.get('samples', {}).get('lidar', {}).get('annotations', []),
            'sample_index': link_token,
            'dataset_type': 'sim'
        }
        
        return sample_meta_data, image_rgb, point_cloud
        
    except Exception as e:
        print(f"Error loading sim sample: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

