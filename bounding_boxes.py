"""
BoundingBoxes class for extracting bounding boxes from different data sources.
Currently supports nuScenes dataset, designed to be extensible for other formats.
"""

import numpy as np
from typing import List, Dict, Optional
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points, box_in_image
from pyquaternion import Quaternion


class BoundingBox:
    """Represents a single bounding box with its metadata."""
    
    def __init__(self, bbox_2d: List[float], category: str, 
                 bbox_3d: Optional[Box] = None, token: Optional[str] = None,
                 visibility: Optional[str] = None):
        self.bbox_2d = bbox_2d  # [min_x, min_y, max_x, max_y]
        self.category = category
        self.bbox_3d = bbox_3d
        self.token = token
        self.visibility = visibility
    
    def get_corners(self) -> np.ndarray:
        """
        Get 4 corner pixels of the 2D bounding box.
        
        Returns:
            4x2 array of pixel coordinates: [top-left, top-right, bottom-left, bottom-right]
        """
        min_x, min_y, max_x, max_y = self.bbox_2d
        return np.array([
            [min_x, min_y],  # Top-left
            [max_x, min_y],  # Top-right
            [min_x, max_y],  # Bottom-left
            [max_x, max_y]   # Bottom-right
        ])


class BoundingBoxes:
    
    def __init__(self, nusc: Optional[NuScenes] = None, data_format: str = "nuscenes"):
        """
        Initialize BoundingBoxes extractor.
        
        Args:
            nusc: NuScenes object (required for nuScenes format)
            data_format: Data format to use ("nuscenes" for now)
        """
        self.data_format = data_format.lower()
        self.nusc = nusc
        
        if self.data_format == "nuscenes" and nusc is None:
            raise ValueError("NuScenes object is required for 'nuscenes' data format")
    
    def get_boxes_for_sample(self, sample_token: str, camera_channel: str = "CAM_FRONT",
                            max_boxes: Optional[int] = None) -> List[BoundingBox]:
        """
        Get bounding boxes for a specific sample.
        
        Args:
            sample_token: Token of the sample
            camera_channel: Camera channel name (e.g., 'CAM_FRONT')
            max_boxes: Maximum number of bounding boxes to return (None for all)
            
        Returns:
            List of BoundingBox objects
        """
        if self.data_format == "nuscenes":
            return self._get_boxes_from_nuscenes(sample_token, camera_channel, max_boxes)
        else:
            raise NotImplementedError(f"Data format '{self.data_format}' is not yet implemented")
    
    def _get_boxes_from_nuscenes(self, sample_token: str, camera_channel: str,
                                 max_boxes: Optional[int] = None) -> List[BoundingBox]:
        """
        Extract bounding boxes from nuScenes dataset.
        
        Args:
            sample_token: Token of the sample
            camera_channel: Camera channel name
            max_boxes: Maximum number of boxes to return
            
        Returns:
            List of BoundingBox objects
        """
        sample = self.nusc.get('sample', sample_token)
        sample_data_token = sample['data'][camera_channel]
        sample_data = self.nusc.get('sample_data', sample_data_token)
        
        calibrated_sensor_token = sample_data['calibrated_sensor_token']
        cs_record = self.nusc.get('calibrated_sensor', calibrated_sensor_token)
        pose_record = self.nusc.get('ego_pose', sample_data['ego_pose_token'])
        
        bounding_boxes = []
        annotation_tokens = sample['anns']
        
        if max_boxes is not None:
            annotation_tokens = annotation_tokens[:max_boxes]
        
        for ann_token in annotation_tokens:
            ann_record = self.nusc.get('sample_annotation', ann_token)
            instance_token = ann_record['instance_token']
            instance = self.nusc.get('instance', instance_token)
            category_token = instance['category_token']
            category = self.nusc.get('category', category_token)
            
            # Create 3D box in global coordinate system
            rotation_quat = Quaternion(ann_record['rotation'])
            box_3d = Box(ann_record['translation'], ann_record['size'], rotation_quat)
            
            # Transform box to ego vehicle coordinate system
            box_3d.translate(-np.array(pose_record['translation']))
            pose_quat = Quaternion(pose_record['rotation'])
            box_3d.rotate(pose_quat.inverse)
            
            # Transform box to camera coordinate system
            box_3d.translate(-np.array(cs_record['translation']))
            cs_quat = Quaternion(cs_record['rotation'])
            box_3d.rotate(cs_quat.inverse)
            
            # Project 3D box to 2D image coordinates
            corners_3d = box_3d.corners()
            view = np.array(cs_record['camera_intrinsic'])
            corners_2d = view_points(corners_3d, view, normalize=True)[:2, :]
            
            # Filter out points behind the camera
            valid = corners_2d[0, :] > 0
            if not np.any(valid):
                continue
            
            corners_2d = corners_2d[:, valid]
            
            min_x = max(0, int(np.min(corners_2d[0, :])))
            max_x = min(sample_data['width'], int(np.max(corners_2d[0, :])))
            min_y = max(0, int(np.min(corners_2d[1, :])))
            max_y = min(sample_data['height'], int(np.max(corners_2d[1, :])))
            
            # Check if box is visible in image
            if min_x < max_x and min_y < max_y and max_x > 0 and max_y > 0:
                if box_in_image(box_3d, view, (sample_data['width'], sample_data['height'])):
                    bbox_2d = [min_x, min_y, max_x, max_y]
                    
                    bounding_box = BoundingBox(
                        bbox_2d=bbox_2d,
                        category=category['name'],
                        bbox_3d=box_3d,
                        token=ann_token,
                        visibility=ann_record['visibility_token']
                    )
                    bounding_boxes.append(bounding_box)
        
        return bounding_boxes
    
    def get_all_bb_pixels(self, sample_token: str, camera_channel: str = "CAM_FRONT",
                             max_boxes: Optional[int] = None, sample_ratio: float = 0.1) -> np.ndarray:
        """
        Get pixels inside bounding boxes (randomly sampled for better performance).
        
        Args:
            sample_token: Token of the sample
            camera_channel: Camera channel name
            max_boxes: Maximum number of boxes to process
            sample_ratio: Ratio of pixels to sample from each bounding box (default: 0.1 = 10%)
            
        Returns:
            Nx2 array of pixel coordinates (sampled pixels from all boxes)
        """
        bounding_boxes = self.get_boxes_for_sample(sample_token, camera_channel, max_boxes)
        
        if len(bounding_boxes) == 0:
            return np.array([])
        
        all_pixels = []
        for bbox in bounding_boxes:
            min_x, min_y, max_x, max_y = bbox.bbox_2d
            # Generate all pixels inside the bounding box
            box_pixels = []
            for x in range(int(min_x), int(max_x) + 1):
                for y in range(int(min_y), int(max_y) + 1):
                    box_pixels.append([x, y])
            
            box_pixels = np.array(box_pixels)
            
            # Randomly sample specified ratio of pixels
            n_total = len(box_pixels)
            n_sample = max(1, int(n_total * sample_ratio))
            
            if n_sample < n_total:
                sample_indices = np.random.choice(n_total, n_sample, replace=False)
                sampled_pixels = box_pixels[sample_indices]
            else:
                sampled_pixels = box_pixels
            
            all_pixels.extend(sampled_pixels)
        
        return np.array(all_pixels)
        