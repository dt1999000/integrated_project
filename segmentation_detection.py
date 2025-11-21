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

try:
    # Try to import SAM2
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_AVAILABLE = True
except ImportError:
    print("SAM2 not available. Install with: pip install git+https://github.com/facebookresearch/segment-anything-2.git")
    SAM2_AVAILABLE = False

try:
    # Try to import DeepLabv3
    from torchvision import models
    from torchvision.transforms import functional as F
    DEEPLABV3_AVAILABLE = True
except ImportError:
    print("DeepLabv3 not available. Install with: pip install torchvision")
    DEEPLABV3_AVAILABLE = False


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
        self.model_type = model_type
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if model_type == "sam2" and SAM2_AVAILABLE:
            self._init_sam2()
        elif model_type == "deeplabv3" and DEEPLABV3_AVAILABLE:
            self._init_deeplabv3()
        else:
            raise ValueError(f"Model type {model_type} not available or not installed")
    
    def _init_sam2(self):
        """Initialize SAM2 model."""
        # Using the default SAM2 model configuration
        sam2_cfg = "sam2_hiera_l.yaml"
        sam2_checkpoint = "sam2_hiera_large.pt"  # You need to download this
        
        try:
            self.model = build_sam2(sam2_cfg, sam2_checkpoint, device=self.device)
            self.predictor = SAM2ImagePredictor(self.model)
            print("SAM2 model initialized successfully")
        except Exception as e:
            print(f"Failed to initialize SAM2: {e}")
            raise
    
    def _init_deeplabv3(self):
        """Initialize DeepLabv3 model."""
        try:
            # Load pre-trained DeepLabv3 with ResNet-101 backbone
            self.model = models.segmentation.deeplabv3_resnet101(
                pretrained=True, progress=True
            ).to(self.device)
            self.model.eval()
            print("DeepLabv3 model initialized successfully")
        except Exception as e:
            print(f"Failed to initialize DeepLabv3: {e}")
            raise
    
    def get_segmentation_mask(self, image: np.ndarray, 
                             prompts: Optional[Dict] = None) -> np.ndarray:
        """
        Get segmentation mask for the given image.
        
        Args:
            image: Input image as numpy array (H, W, 3)
            prompts: Optional prompts for SAM2 (points, boxes, etc.)
            
        Returns:
            Segmentation mask as numpy array (H, W) with integer labels
        """
        if self.model_type == "sam2":
            return self._get_sam2_mask(image, prompts)
        elif self.model_type == "deeplabv3":
            return self._get_deeplabv3_mask(image)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _get_sam2_mask(self, image: np.ndarray, 
                      prompts: Optional[Dict] = None) -> np.ndarray:
        """Get segmentation mask using SAM2."""
        if prompts is None:
            # If no prompts provided, use automatic mask generation
            # This is a simplified version - you might want to implement
            # more sophisticated automatic mask generation
            raise NotImplementedError("Automatic mask generation for SAM2 not implemented")
        
        # Set image for SAM2
        self.predictor.set_image(image)
        
        # Get masks from prompts
        masks, scores, logits = self.predictor.predict(
            point_coords=prompts.get("points", None),
            point_labels=prompts.get("point_labels", None),
            box=prompts.get("box", None),
            multimask_output=True
        )
        
        # Return the mask with highest score
        best_mask_idx = np.argmax(scores)
        return masks[best_mask_idx].astype(np.uint8)
    
    def _get_deeplabv3_mask(self, image: np.ndarray) -> np.ndarray:
        """Get segmentation mask using DeepLabv3."""
        # Preprocess image
        input_tensor = F.to_tensor(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)["out"]
        
        # Get the most likely class for each pixel
        masks = output.argmax(1).squeeze().cpu().numpy()
        
        return masks.astype(np.uint8)
    
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
        
        return projected_points
    
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
            projected_points = self.project_mask_to_pointcloud(
                mask, mask_id, max_distance, distance_threshold
            )
            if len(projected_points) > 0:
                results[mask_id] = projected_points
        
        return results


