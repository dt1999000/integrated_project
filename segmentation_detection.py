"""
Segmentation-based Object Detection for 3D Point Cloud Projection
This module provides classes for using SAM2 or DeepLabv3 to get segmentation masks
and project them onto 3D LiDAR scenes.
"""

import numpy as np
import cv2
import torch
from typing import List, Dict, Tuple, Optional, Union
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics.models.sam import SAM2DynamicInteractivePredictor
from bounding_boxes import BoundingBoxes


class SegmentationDetector:
    """
    Base class for segmentation-based object detection.
    """
    
    def __init__(self, model_type: str = "sam2"):
        """
        Initialize the segmentation detector.
        
        Args:
            model_type: Type of model to use ("sam2" or "deeplabv3")
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        overrides=dict(conf=0.01, task="segment", mode="predict", imgsz=1024, model="sam2_t.pt", save=False)
        self.predictor = SAM2DynamicInteractivePredictor(overrides = overrides, max_obj_num=10)
    
    def downsample_segmentation_mask(self, mask: np.ndarray, scale_factor: float = 0.5) -> np.ndarray:
        """
        Downsample a segmentation mask for better performance.
        
        Args:
            mask: Input segmentation mask (H, W) with integer labels
            scale_factor: Scale factor for downsampling (0.5 = half size)
            
        Returns:
            Downsampled mask with preserved labels
        """
        if scale_factor >= 1.0:
            return mask  # No downsampling needed
            
        # Get mask dimensions
        h, w = mask.shape
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        
        # Create empty downsampled mask
        downsampled_mask = np.zeros((new_h, new_w), dtype=mask.dtype)
        
        # Get unique labels (excluding background 0)
        unique_labels = np.unique(mask)
        unique_labels = unique_labels[unique_labels > 0]
        
        # Process each label separately to preserve them
        for label in unique_labels:
            # Create binary mask for this label
            binary_mask = (mask == label).astype(np.uint8)
            
            # Downsample binary mask
            downsampled_binary = cv2.resize(
                binary_mask, 
                (new_w, new_h), 
                interpolation=cv2.INTER_NEAREST
            )
            
            # Add to result with correct label
            downsampled_mask[downsampled_binary > 0] = label
            
        return downsampled_mask
    
    def upsample_segmentation_mask(self, mask: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """
        Upsample a segmentation mask back to original size.
        
        Args:
            mask: Downsampled segmentation mask
            target_shape: Target shape (height, width)
            
        Returns:
            Upsampled mask with preserved labels
        """
        if mask.shape == target_shape:
            return mask  # No upsampling needed
            
        h, w = target_shape
        
        # Create empty upsampled mask
        upsampled_mask = np.zeros((h, w), dtype=mask.dtype)
        
        # Get unique labels (excluding background 0)
        unique_labels = np.unique(mask)
        unique_labels = unique_labels[unique_labels > 0]
        
        # Process each label separately to preserve them
        for label in unique_labels:
            # Create binary mask for this label
            binary_mask = (mask == label).astype(np.uint8)
            
            # Upsample binary mask
            upsampled_binary = cv2.resize(
                binary_mask, 
                (w, h), 
                interpolation=cv2.INTER_NEAREST
            )
            
            # Add to result with correct label
            upsampled_mask[upsampled_binary > 0] = label
            
        return upsampled_mask
    
    def get_segmentation_mask(self, image: np.ndarray, bboxes: BoundingBoxes, use_downsampling: bool = True) -> np.ndarray:
        """
        Get segmentation mask using SAM2.
        
        Args:
            image: Input RGB image
            bboxes: Bounding boxes to use as prompts
            use_downsampling: Whether to use downsampling for better performance
            
        Returns:
            Segmentation mask with integer labels
        """
        # Get max_obj_num from predictor to limit boxes
        max_obj_num = self.predictor._max_obj_num
        
        # Limit boxes to max_obj_num and create sequential integer obj_ids
        boxes_2d = bboxes.boxes_2d[:max_obj_num] if len(bboxes.boxes_2d) > max_obj_num else bboxes.boxes_2d
        obj_ids = list(range(len(boxes_2d)))  # Sequential integers: 0, 1, 2, ...
        
        if len(boxes_2d) == 0:
            return np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        # Optionally downsample the image for faster processing
        original_shape = image.shape[:2]
        if use_downsampling:
            # Downsample image to half size for faster processing
            scale_factor = 0.5
            h, w = original_shape
            new_h, new_w = int(h * scale_factor), int(w * scale_factor)
            
            # Resize image
            resized_image = cv2.resize(image, (new_w, new_h))
            
            # Scale bounding boxes
            scaled_boxes = []
            for box in boxes_2d:
                scaled_box = [
                    box[0] * scale_factor,
                    box[1] * scale_factor,
                    box[2] * scale_factor,
                    box[3] * scale_factor
                ]
                scaled_boxes.append(scaled_box)
            
            # Run prediction on downsampled image
            results = self.predictor(
                source=resized_image,
                bboxes=scaled_boxes,
                obj_ids=obj_ids,
                update_memory=True
            )
        else:
            # Run prediction on original image
            results = self.predictor(
                source=image,
                bboxes=boxes_2d,
                obj_ids=obj_ids,
                update_memory=True
            )
        
        # Extract masks from results and combine into single mask array
        if results and len(results) > 0:
            # Create empty mask with appropriate dimensions
            if use_downsampling:
                # Create mask with downsampled dimensions
                h, w = int(original_shape[0] * 0.5), int(original_shape[1] * 0.5)
                mask = np.zeros((h, w), dtype=np.uint8)
            else:
                # Create mask with original dimensions
                mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            
            # Process each result
            for i, result in enumerate(results):
                if result.masks is not None and len(result.masks) > 0:
                    # Get the first mask from the result
                    # result.masks[0] is a Masks object, need to access .data attribute
                    mask_obj = result.masks[0]
                    
                    # Get the actual mask data (tensor/array) from the Masks object
                    # Handle different Masks object structures
                    if hasattr(mask_obj, 'data'):
                        seg_mask = mask_obj.data
                    elif hasattr(mask_obj, 'masks'):
                        seg_mask = mask_obj.masks
                    else:
                        # Try to get the mask data directly
                        seg_mask = mask_obj
                    
                    # Convert to numpy if it's a tensor
                    if torch.is_tensor(seg_mask):
                        seg_mask = seg_mask.cpu().numpy()
                    
                    # Reshape if needed (masks might be 2D or need reshaping)
                    # Masks are typically stored as (1, H, W) or (H, W)
                    if seg_mask.ndim > 2:
                        seg_mask = seg_mask.squeeze()
                    
                    # Debug: print mask info
                    print(f"Mask {i}: shape={seg_mask.shape}, dtype={seg_mask.dtype}, min={seg_mask.min()}, max={seg_mask.max()}")
                    
                    # Ensure mask matches current dimensions
                    if seg_mask.shape != mask.shape:
                        seg_mask = cv2.resize(seg_mask.astype(np.float32), 
                                             (mask.shape[1], mask.shape[0]), 
                                             interpolation=cv2.INTER_NEAREST).astype(np.float32)
                    
                    # Add to mask with unique label (i+1 to avoid 0 which is background)
                    # Threshold the mask (masks are typically 0-1 range)
                    mask[seg_mask > 0.5] = i + 1
            
            # Upsample mask if needed
            if use_downsampling:
                mask = self.upsample_segmentation_mask(mask, original_shape)
            
            return mask
        else:
            # Return empty mask if no results
            print("segmentation returns empty mask")
            return np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
 
    def get_mask_pixels(self, mask: np.ndarray, 
                       target_classes: Optional[List[int]] = None,
                       min_area: int = 100) -> np.ndarray:
        """
        Extract pixel coordinates from segmentation mask.
        
        Args:
            mask: Segmentation mask (H, W) with integer labels
            target_classes: List of class IDs to extract (None for all)
            min_area: Minimum area in pixels for a region to be included
            
        Returns:
            Nx2 array of pixel coordinates
        """
        if target_classes is None:
            # Use all non-zero pixels
            binary_mask = mask > 0
        else:
            # Create binary mask for target classes
            binary_mask = np.zeros_like(mask, dtype=bool)
            for class_id in target_classes:
                binary_mask |= (mask == class_id)
        
        # Find connected components to filter by area
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_mask.astype(np.uint8), connectivity=8
        )
        
        # Filter components by minimum area
        valid_mask = np.zeros_like(binary_mask)
        for i in range(1, num_labels):  # Skip background (0)
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                valid_mask |= (labels == i)
        
        # Get pixel coordinates
        y_coords, x_coords = np.where(valid_mask)
        pixels = np.column_stack((x_coords, y_coords))
        
        return pixels


class SegmentationToPointCloud:
    """
    Class to project segmentation masks onto 3D LiDAR point clouds.
    """
    
    def __init__(self, projection_2d_to_3d):
        """
        Initialize with a Projection2DTo3D instance.
        
        Args:
            projection_2d_to_3d: Instance of Projection2DTo3D class
        """
        self.projection = projection_2d_to_3d
    
    def project_mask_to_pointcloud(self, mask: np.ndarray, 
                                 mask_id: int = 1,
                                 max_distance: float = 100.0,
                                 distance_threshold: float = 0.5) -> np.ndarray:
        """
        Project segmentation mask to 3D point cloud.
        
        Args:
            mask: Segmentation mask (H, W) with integer labels
            mask_id: ID of the mask to project
            max_distance: Maximum ray extension distance
            distance_threshold: Maximum perpendicular distance to consider a point on the ray
            
        Returns:
            Nx3 array of projected 3D points
        """
        # Get pixels belonging to the specified mask
        y_coords, x_coords = np.where(mask == mask_id)
        pixels = np.column_stack((x_coords, y_coords))
        
        if len(pixels) == 0:
            return np.array([])
        
        # Project pixels to 3D rays
        rays = self.projection.pixel_to_ray(pixels)
        
        # Find closest points on rays
        projected_points = self.projection.find_closest_point_on_ray(
            rays, max_distance=max_distance, distance_threshold=distance_threshold
        )
        
        return rays, projected_points
    
    def project_all_masks(self, mask: np.ndarray,
                         max_distance: float = 100.0,
                         distance_threshold: float = 0.5) -> Dict[int, np.ndarray]:
        """
        Project all masks in the segmentation to 3D point clouds.
        
        Args:
            mask: Segmentation mask (H, W) with integer labels
            max_distance: Maximum ray extension distance
            distance_threshold: Maximum perpendicular distance to consider a point on the ray
            
        Returns:
            Dictionary mapping mask_id to Nx3 array of projected 3D points
        """
        unique_mask_ids = np.unique(mask)
        # Skip background (usually 0)
        mask_ids = [id for id in unique_mask_ids if id != 0]
        
        results = {}
        for mask_id in mask_ids:
            rays, projected_points = self.project_mask_to_pointcloud(
                mask, mask_id, max_distance, distance_threshold
            )
            if len(projected_points) > 0:
                results[mask_id] = {
                    'rays': rays,
                    'projected_points': projected_points
                }
        
        return results


