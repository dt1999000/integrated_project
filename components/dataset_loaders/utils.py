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
import open3d as o3d
from scipy.spatial.transform import Rotation as R

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from .kitti_dataset_loader import KITTIDatasetLoader
from .nuscenes_dataset_loader import NuScenesDatasetLoader
from .sunrgbd_dataset_loader import SUNRGBDDatasetLoader
from .dataset_loader import LinkedDataHandler
from components.core.pointcloud_projection import load_sunrgbd_calibration


def _looks_like_sunrgbd_trainval(root: Path) -> bool:
    """
    True if ``root`` matches the SUNRGBD trainval release layout:
    calib/, depth/, image/ (depth holds .mat point clouds; calib holds per-frame intrinsics .txt).
    """
    if not root.is_dir():
        return False
    return (
        (root / "calib").is_dir()
        and (root / "depth").is_dir()
        and (root / "image").is_dir()
    )


def detect_dataset_type(dataset_path: str) -> Optional[str]:
    """
    Detect dataset type based on folder structure and JSON files.
    
    Detection priority:
    1. LinkedDataHandler/sim: Check for dataset.json in root
    2. KITTI: Check for training/ or testing/ with image_2/, velodyne/, calib/
    3. nuScenes: Check for samples/, sweeps/, v1.0-*/ folders
    4. SUNRGBD: ``sunrgbd_trainval/`` with calib/, depth/, image/ (or that layout at dataset root)
    5. SUNRGBD (legacy): sensor folders + nested scenes with intrinsics.txt + image/ + depth_bfx/
    
    Args:
        dataset_path: Root directory of the dataset
        
    Returns:
        Dataset type: 'kitti', 'nuscenes', 'sim', 'sunrgbd', or None if cannot determine
    """
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        return None
    
    # If a file is passed (e.g. rosbag file), don't try to walk it as a directory.
    # This avoids NotADirectoryError when the user points directly to a bag file.
    if dataset_path.is_file():
        # Detect rosbag-like files (ROS1 .bag, ROS2 .db3, MCAP .mcap).
        # Compressed variants like *.mcap.zstd are not treated as rosbag here
        # unless explicitly supported by the rosbag reader.
        suffixes = "".join(dataset_path.suffixes).lower()
        if any(ext in suffixes for ext in [".bag", ".db3", ".mcap"]):
            return "rosbag"
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
    
    # Check for nuScenes structure (only if dataset_path is a directory)
    has_samples = (dataset_path / "samples").exists()
    has_sweeps = (dataset_path / "sweeps").exists()
    has_v1 = any(d.name.startswith("v1.0-") for d in dataset_path.iterdir() if d.is_dir())
    
    if has_samples and (has_sweeps or has_v1):
        return "nuscenes"

    # SUNRGBD — current release layout (matches SUNRGBDDatasetLoader)
    trainval_root = dataset_path / "sunrgbd_trainval"
    if _looks_like_sunrgbd_trainval(trainval_root):
        return "sunrgbd"
    if _looks_like_sunrgbd_trainval(dataset_path):
        return "sunrgbd"

    # SUNRGBD — legacy toolbox export (per-scene intrinsics.txt + depth_bfx PNG)
    has_sensor_dirs = any((dataset_path / name).exists() for name in ["kv1", "kv2", "realsense", "xtion"])
    has_intrinsics = any(dataset_path.glob("**/intrinsics.txt"))
    has_rgb_scene = any(dataset_path.glob("**/image/*.jpg")) or any(dataset_path.glob("**/image/*.png"))
    has_depth_scene = any(dataset_path.glob("**/depth_bfx/*.png"))
    if has_sensor_dirs and has_intrinsics and has_rgb_scene and has_depth_scene:
        return "sunrgbd"

    return None


def load_dataset_sample(
    dataset_path: str,
    sample_index: int = 0,
    dataset_type: Optional[str] = None,
    filter_forward_only: bool = True,
    use_saved_media_paths: bool = False,
    saved_image_path: Optional[str] = None,
    saved_point_cloud_path: Optional[str] = None,
    sunrgbd_keep_fraction: Optional[float] = None,
) -> Tuple[Optional[Dict], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load sample from dataset (KITTI, nuScenes, sim, or SUNRGBD format).
    
    Note: This function does NOT remove ground plane. Ground plane removal
    should be done in the detection pipeline (Step 1).
    
    Args:
        dataset_path: Root directory of dataset
        sample_index: Index or token of sample to load (int for KITTI, str for nuScenes/sim)
        dataset_type: 'kitti', 'nuscenes', 'sim', 'sunrgbd', or None (auto-detect)
        filter_forward_only: Whether to keep only forward-facing points (x > 0) - for KITTI
        use_saved_media_paths: For SUNRGBD, prefer pre-saved image/point cloud paths
        saved_image_path: Optional pre-saved image path for SUNRGBD batch mode
        saved_point_cloud_path: Optional pre-saved PCD path for SUNRGBD batch mode
        
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
    elif dataset_type == "sunrgbd":
        return _load_sunrgbd_sample(
            dataset_path,
            sample_index,
            use_saved_media_paths=use_saved_media_paths,
            saved_image_path=saved_image_path,
            saved_point_cloud_path=saved_point_cloud_path,
            keep_fraction=sunrgbd_keep_fraction,
        )
    elif dataset_type == "rosbag":
        # For rosbag we expect dataset_path to point to an extracted folder
        # produced by components.dataset_loaders.rosbag_extractor.extract_bag_to_folder.
        return _load_rosbag_sample(dataset_path, sample_index)
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
            'dataset_type': 'kitti',
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
            'dataset_type': 'nuscenes',
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
        
        # Get calibration data from rgb and lidar samples
        rgb_calib = rgb_sample.get('calibration', {})
        lidar_calib = lidar_sample.get('calibration', {})
        
        # Extract camera intrinsic (handle typo in LinkedDataHandler: "camera_intrinisc")
        camera_intrinsic = None
        if 'camera_intrinisc' in rgb_calib:  # Typo version
            camera_intrinsic = np.array(rgb_calib['camera_intrinisc'])
        elif 'camera_intrinsic' in rgb_calib:  # Correct version
            camera_intrinsic = np.array(rgb_calib['camera_intrinsic'])
        
        # Compute camera_to_lidar_transform from rotation and translation
        camera_to_lidar_transform = None
        if rgb_calib and lidar_calib:
            try:
                # Camera rotation and translation (quaternion [x,y,z,w] and translation)
                q_cam = rgb_calib.get('rotation', [0, 0, 0, 1])  # [x, y, z, w]
                t_cam = np.array(rgb_calib.get('translation', [0, 0, 0]))
                
                # LiDAR rotation and translation
                q_lidar = lidar_calib.get('rotation', [0, 0, 0, 1])  # [x, y, z, w]
                t_lidar = np.array(lidar_calib.get('translation', [0, 0, 0]))
                
                # Convert quaternions to rotation matrices
                R_cam2world = R.from_quat(q_cam).as_matrix()  # Camera to world
                R_lidar2world = R.from_quat(q_lidar).as_matrix()  # LiDAR to world
                
                # Build transformation matrices (4x4)
                T_cam2world = np.eye(4)
                T_cam2world[:3, :3] = R_cam2world
                T_cam2world[:3, 3] = t_cam
                
                T_lidar2world = np.eye(4)
                T_lidar2world[:3, :3] = R_lidar2world
                T_lidar2world[:3, 3] = t_lidar
                
                # Camera to LiDAR: T_cam2lidar = T_lidar2world^-1 @ T_cam2world
                T_world2lidar = np.linalg.inv(T_lidar2world)
                camera_to_lidar_transform = T_world2lidar @ T_cam2world
                
            except Exception as e:
                print(f"Warning: Could not compute camera_to_lidar_transform: {e}")
                camera_to_lidar_transform = None
        
        # Validate that we have required calibration data
        if camera_intrinsic is None:
            print(f"Warning: Camera intrinsic not found in calibration data")
        if camera_to_lidar_transform is None:
            print(f"Warning: Camera to LiDAR transform could not be computed")
        
        # Build normalized GT boxes for downstream evaluation/visualization.
        # Sim annotations are native 3D boxes (translation + size), so we derive:
        # - axis-aligned min/max cuboid
        # - projected 2D bbox from camera projection
        lidar_annotations = link.get('samples', {}).get('lidar', {}).get('annotations', [])
        normalized_ground_truth_boxes = []
        image_height, image_width = image_rgb.shape[:2]
        for annotation in lidar_annotations:
            translation = annotation.get('translation')
            size = annotation.get('size')
            if translation is None or size is None or len(translation) != 3 or len(size) != 3:
                continue

            center = np.asarray(translation, dtype=np.float64)
            size_np = np.asarray(size, dtype=np.float64)
            half_size = size_np / 2.0
            bbox_min = center - half_size
            bbox_max = center + half_size

            projected_points = handler.getAnnotationInCameraFrame(annotation, link, camera="rgb")
            bbox_2d = None
            if projected_points:
                pts = np.asarray(projected_points, dtype=np.int32)
                left = int(np.clip(np.min(pts[:, 0]), 0, image_width - 1))
                right = int(np.clip(np.max(pts[:, 0]), 0, image_width - 1))
                top = int(np.clip(np.min(pts[:, 1]), 0, image_height - 1))
                bottom = int(np.clip(np.max(pts[:, 1]), 0, image_height - 1))
                if right > left and bottom > top:
                    bbox_2d = {
                        "left": left,
                        "top": top,
                        "right": right,
                        "bottom": bottom,
                    }

            normalized_ground_truth_boxes.append({
                "token": annotation.get("token"),
                "category": annotation.get("class", "Person"),
                "track_id": annotation.get("track_id", -1),
                "num_points": annotation.get("num_points", 0),
                "translation": translation,
                "size": size,
                "min_x": float(bbox_min[0]),
                "min_y": float(bbox_min[1]),
                "min_z": float(bbox_min[2]),
                "max_x": float(bbox_max[0]),
                "max_y": float(bbox_max[1]),
                "max_z": float(bbox_max[2]),
                "bbox_2d": bbox_2d,
            })

        # Create normalized sample_meta_data
        sample_meta_data = {
            'image_path': str(image_path) if image_path else None,
            'point_cloud_path': str(point_cloud_path) if point_cloud_path else None,
            'camera_intrinsic': camera_intrinsic,
            'camera_extrinsic': np.eye(4),
            'camera_to_lidar_transform': camera_to_lidar_transform,
            'ground_truth_boxes': normalized_ground_truth_boxes,
            'sample_index': link_token,
            'dataset_type': 'sim',
            'subset_name': found_subset_name,
        }
        
        return sample_meta_data, image_rgb, point_cloud
        
    except Exception as e:
        print(f"Error loading sim sample: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def _load_sunrgbd_sample(
    dataset_path: str,
    sample_index: int,
    use_saved_media_paths: bool = False,
    saved_image_path: Optional[str] = None,
    saved_point_cloud_path: Optional[str] = None,
    keep_fraction: Optional[float] = None,
) -> Tuple[Optional[Dict], Optional[np.ndarray], Optional[np.ndarray]]:
    """Load SUNRGBD sample; optionally bypass RGB-D reconstruction with saved media."""
    loader = SUNRGBDDatasetLoader(dataroot=str(dataset_path))
    loader.load_dataset()
    sample_idx = int(sample_index)
    if sample_idx < 0 or sample_idx >= len(loader.samples):
        print(f"Error: SUNRGBD sample_index out of range: {sample_idx} (n={len(loader.samples)})")
        return None, None, None

    sample = loader.samples[sample_idx]

    if use_saved_media_paths:
        image_file = Path(saved_image_path).expanduser() if saved_image_path else Path(sample["image_path"])
        pcd_file = Path(saved_point_cloud_path).expanduser() if saved_point_cloud_path else None

        image_bgr = cv2.imread(str(image_file))
        if image_bgr is None:
            print(f"Error: Could not load image from {image_file}")
            return None, None, None
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        if pcd_file is None or not pcd_file.exists():
            print(f"Error: Saved SUNRGBD point cloud file not found for sample {sample_idx}: {pcd_file}")
            return None, None, None
        pcd = o3d.io.read_point_cloud(str(pcd_file))
        point_cloud = np.asarray(pcd.points, dtype=np.float32)
        if point_cloud.size == 0:
            print(f"Error: Saved SUNRGBD point cloud is empty for sample {sample_idx}: {pcd_file}")
            return None, None, None

        camera_intrinsic, camera_rtilt = load_sunrgbd_calibration(sample["intrinsics_path"])
        camera_to_lidar_transform = SUNRGBDDatasetLoader._sunrgbd_camera_to_depth_transform(camera_rtilt)
        gt_boxes_cam = SUNRGBDDatasetLoader._load_ground_truth_boxes(
            sample.get("annotation_path"),
            image_rgb.shape,
        )
        gt_boxes = SUNRGBDDatasetLoader._transform_gt_boxes_to_pipeline_lidar(
            gt_boxes_cam,
            camera_to_lidar_transform,
        )
        sample_meta_data = {
            "image_path": str(image_file),
            "point_cloud_path": str(pcd_file),
            "camera_intrinsic": camera_intrinsic,
            "camera_rtilt": camera_rtilt,
            "camera_extrinsic": np.eye(4, dtype=np.float64),
            "camera_to_lidar_transform": camera_to_lidar_transform,
            "camera_frame": "camera_optical",
            "lidar_frame": "sunrgbd_upright_depth",
            "ground_truth_boxes": gt_boxes,
            "sample_index": sample_idx,
            "dataset_type": "sunrgbd",
            "scene_id": sample.get("scene_id"),
            "depth_scale": 10000.0,
            "image_shape": image_rgb.shape[:2],
        }
        return sample_meta_data, image_rgb, point_cloud

    if keep_fraction is None:
        keep_fraction = 0.8

    sample_data = loader.load_sunrgbd_data(
        sample_index=sample_idx,
        keep_fraction=float(keep_fraction),
    )
    image_path = sample_data["image_path"]
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return None, None, None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    point_cloud = sample_data["point_cloud"]
    if point_cloud is None or len(point_cloud) == 0:
        print(f"Error: Reconstructed SUNRGBD point cloud is empty for sample {sample_idx}")
        return None, None, None

    sample_meta_data = {
        "image_path": sample_data["image_path"],
        "point_cloud_path": sample_data["depth_path"],
        "camera_intrinsic": sample_data["camera_intrinsic"],
        "camera_rtilt": sample_data.get("camera_rtilt"),
        "camera_extrinsic": sample_data.get("camera_extrinsic", np.eye(4)),
        "camera_to_lidar_transform": sample_data["camera_to_lidar_transform"],
        "camera_frame": sample_data.get("camera_frame", "camera_optical"),
        "lidar_frame": sample_data.get("lidar_frame", "sunrgbd_upright_depth"),
        "ground_truth_boxes": sample_data.get("ground_truth_boxes", []),
        "sample_index": sample_idx,
        "dataset_type": "sunrgbd",
        "scene_id": sample_data.get("scene_id"),
        "depth_scale": sample_data.get("depth_scale", 10000.0),
        "image_shape": image_rgb.shape[:2],
        "sunrgbd_keep_fraction": float(sample_data.get("keep_fraction", keep_fraction)),
        "sunrgbd_stride": int(sample_data.get("stride", 1)),
    }

    return sample_meta_data, image_rgb, point_cloud


def _load_rosbag_sample(
    dataset_path: str,
    sample_index: int,
) -> Tuple[Optional[Dict], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load a sample that was extracted from a ROS bag into a KITTI-like folder.
    
    Expected layout (produced by rosbag_extractor.extract_bag_to_folder):
        dataset_path/
          image_2/    000000.png, 000001.png, ...
          velodyne/   000000.pcd, 000001.pcd, ... or 000000.bin (legacy)
          calib.npz   camera_intrinsic, camera_to_lidar, camera_frame, lidar_frame
    """
    try:
        root = Path(dataset_path)
        image_dir = root / "image_2"
        velodyne_dir = root / "velodyne"
        calib_path = root / "calib.npz"
        
        img_path = image_dir / f"{int(sample_index):06d}.png"
        pcd_path = velodyne_dir / f"{int(sample_index):06d}.pcd"
        bin_path = velodyne_dir / f"{int(sample_index):06d}.bin"
        
        if not img_path.exists() or (not pcd_path.exists() and not bin_path.exists()):
            print(f"Error: ROS bag extracted sample {sample_index} not found in {dataset_path}")
            return None, None, None

        # Load image (BGR → RGB)
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            print(f"Error: Could not read image {img_path}")
            return None, None, None
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # Load point cloud (prefer PCD, fall back to legacy BIN)
        point_cloud = None
        pc_path_str: Optional[str] = None
        
        if pcd_path.exists():
            try:
                import open3d as o3d
                pcd = o3d.io.read_point_cloud(str(pcd_path))
                pts = np.asarray(pcd.points)
                if pts.size == 0:
                    print(f"Error: PCD file {pcd_path} contains no points")
                    return None, None, None
                point_cloud = pts.astype(np.float32)
                pc_path_str = str(pcd_path)
            except Exception as e:
                print(f"Error: Failed to read PCD file {pcd_path}: {e}")
                return None, None, None
        elif bin_path.exists():
            pc_raw = np.fromfile(str(bin_path), dtype=np.float32)
            if pc_raw.size % 3 != 0:
                print(f"Error: Unexpected point cloud size in {bin_path}")
                return None, None, None
            point_cloud = pc_raw.reshape(-1, 3)
            pc_path_str = str(bin_path)
        else:
            print(f"Error: No point cloud file found for sample {sample_index} in {velodyne_dir}")
            return None, None, None

        # Load calibration if present
        if calib_path.exists():
            calib = np.load(str(calib_path), allow_pickle=True)
            camera_intrinsic = calib.get("camera_intrinsic", np.eye(3, dtype=np.float64))
            camera_to_lidar = calib.get("camera_to_lidar", np.eye(4, dtype=np.float64))
        else:
            camera_intrinsic = np.eye(3, dtype=np.float64)
            camera_to_lidar = np.eye(4, dtype=np.float64)

        sample_meta_data = {
            "image_path": str(img_path),
            "point_cloud_path": pc_path_str,
            "camera_intrinsic": camera_intrinsic,
            "camera_extrinsic": np.eye(4, dtype=np.float64),
            "camera_to_lidar_transform": camera_to_lidar,
            "ground_truth_boxes": [],
            "sample_index": int(sample_index),
            "dataset_type": "rosbag",
        }

        return sample_meta_data, image_rgb, point_cloud

    except Exception as e:
        print(f"Error loading ROS bag extracted sample: {e}")
        import traceback

        traceback.print_exc()
        return None, None, None