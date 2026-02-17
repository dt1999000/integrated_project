"""
SAM Integration Module for 3D Object Detection Pipeline
This module provides unified class for integrating SAM2 and SAM3 models.
Supports bounding box-based segmentation (SAM2 & SAM3) and text-based semantic segmentation (SAM3 only).
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import torch
import cv2
import os

try:
    from ultralytics import ASSETS, SAM, YOLO, FastSAM
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    print("Ultralytics not available. Install with: pip install ultralytics")
    ULTRALYTICS_AVAILABLE = False

try:
    from ultralytics.models.sam import SAM2DynamicInteractivePredictor
    SAM2_AVAILABLE = True
except ImportError:
    print("SAM2 not available. Install with: pip install ultralytics")
    SAM2_AVAILABLE = False

try:
    from ultralytics.models.sam import SAM3SemanticPredictor
    SAM3_AVAILABLE = True
except ImportError:
    print("SAM3 not available. Install with: pip install ultralytics")
    SAM3_AVAILABLE = False


class SAMIntegration:
    """
    Unified class for SAM model management and segmentation operations.
    Supports SAM2 (bounding box segmentation) and SAM3 (bounding box + text-based semantic segmentation).
    """
    
    def __init__(self, model_type: str = "sam2_t"):
        """
        Initialize SAM integration manager.
        
        Args:
            model_type: Type of SAM model to use
                      Options: "sam2_t", "sam2_b", "sam2_l", "sam3", "sam_b", "mobile_sam"
        """
        self.model_type = model_type
        self.model = None
        self.predictor = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.current_image = None
        
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("Ultralytics not available. Install with: pip install ultralytics")
        
        self._load_model()
    
    def _load_model(self):
        """Load the specified SAM model from ./models directory."""
        # Base directory for models - get absolute path to ensure it works correctly
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        models_dir = os.path.join(project_root, "models")
        models_dir = os.path.abspath(models_dir)  # Convert to absolute path
        
        if self.model_type == "sam3":
            if not SAM3_AVAILABLE:
                raise ImportError("SAM3 not available. Install with: pip install ultralytics")
            
            model_path = os.path.join(models_dir, "sam3.pt")
            model_path = os.path.abspath(model_path)  # Ensure absolute path
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"SAM3 model not found at {model_path}")
            
            overrides = dict(
                conf=0.25,
                task="segment",
                mode="predict",
                model=model_path,
                half=True,  # Use FP16 for faster inference
                save=True,
            )
            self.predictor = SAM3SemanticPredictor(overrides=overrides)
            print(f"Loaded SAM3 model from {model_path}")
            
        elif self.model_type.startswith("sam2"):
            if not ULTRALYTICS_AVAILABLE:
                raise ImportError("Ultralytics not available. Install with: pip install ultralytics")
            
            if self.model_type == "sam2_t":
                model_file = "sam2_t.pt"
            elif self.model_type == "sam2_b":
                model_file = "sam2_b.pt"
            elif self.model_type == "sam2_l":
                model_file = "sam2_l.pt"
            else:
                raise ValueError(f"Unknown SAM2 model type: {self.model_type}")
            
            model_path = os.path.join(models_dir, model_file)
            model_path = os.path.abspath(model_path)  # Ensure absolute path
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"SAM2 model '{model_file}' not found at {model_path}")
            
            # Use SAM class directly (like the example) instead of SAM2DynamicInteractivePredictor
            self.model = SAM(model_path)
            self.model.info()
            print(f"Loaded SAM2 model from {model_path}")
            
        elif self.model_type == "sam_b":
            model_path = os.path.join(models_dir, "sam_b.pt")
            model_path = os.path.abspath(model_path)  # Ensure absolute path
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"SAM-b model not found at {model_path}")
            self.model = SAM(model_path)
            self.model.info()
            print(f"Loaded SAM-b model from {model_path}")
            
        elif self.model_type == "mobile_sam":
            model_path = os.path.join(models_dir, "FastSAM-s.pt")
            model_path = os.path.abspath(model_path)  # Ensure absolute path
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"MobileSAM model not found at {model_path}")
            self.model = FastSAM(model_path)
            self.model.info()
            print(f"Loaded MobileSAM model from {model_path}")
            
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def get_mask_from_bbox(self, image: np.ndarray, bbox: List[float]) -> np.ndarray:
        """
        Get segmentation mask from a single bounding box.
        Works for both SAM2 and SAM3.
        
        Args:
            image: Input image as numpy array (H, W, 3)
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Segmentation mask as numpy array (H, W) with binary mask (0 or 1)
        """
        if self.model_type == "sam3":
            # For SAM3, set image first if not already set
            if self.current_image is None or not np.array_equal(self.current_image, image):
                self.predictor.set_image(image)
                self.current_image = image.copy()
            
            # Convert bbox to format expected by SAM3
            # SAM3 expects bboxes in format [[x1, y1, x2, y2]]
            bboxes = [bbox]
            results = self.predictor(bboxes=bboxes)
            
            # Extract mask from results
            if results and len(results) > 0:
                result = results[0]
                if result.masks is not None and len(result.masks) > 0:
                    mask = result.masks[0].cpu().numpy()
                    # Validate mask dimensions
                    if mask.size == 0 or mask.ndim < 2:
                        h, w = image.shape[:2]
                        return np.zeros((h, w), dtype=np.uint8)
                    
                    # Resize to image size if needed
                    h, w = image.shape[:2]
                    if h <= 0 or w <= 0:
                        return np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
                    
                    if mask.shape[-2:] != (h, w):
                        # Only resize if mask has valid dimensions
                        if mask.shape[-2] > 0 and mask.shape[-1] > 0:
                            mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
                        else:
                            return np.zeros((h, w), dtype=np.uint8)
                    # Convert to binary mask
                    mask = (mask > 0.5).astype(np.uint8)
                    return mask
            return np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            
        elif self.model_type.startswith("sam2"):
            # For SAM2, use the SAM model directly with bboxes parameter (like the example)
            # bbox format: [x1, y1, x2, y2]
            if isinstance(bbox, np.ndarray):
                bbox = bbox.tolist()
            
            # Ensure bbox is a list with 4 elements
            if not isinstance(bbox, list) or len(bbox) != 4:
                raise ValueError(f"Bbox must be a list with 4 elements [x1, y1, x2, y2], got {bbox}")
            
            # Run inference with bboxes parameter directly (mimicking the example)
            # The model accepts bboxes as a list: bboxes=[x1, y1, x2, y2]
            results = self.model(image, bboxes=bbox)
            
            # Extract mask from results
            if results and len(results) > 0:
                result = results[0] if isinstance(results, list) else results
                if hasattr(result, 'masks') and result.masks is not None:
                    # result.masks is a Masks object, not a list
                    # Use the .data attribute to get the tensor, then convert to numpy
                    mask_obj = result.masks
                    
                    # Get the mask data from the Masks object
                    # Masks object has .data attribute which contains the tensor
                    if hasattr(mask_obj, 'data'):
                        mask_tensor = mask_obj.data
                        # Convert tensor to numpy array
                        if hasattr(mask_tensor, 'cpu'):
                            mask = mask_tensor.cpu().numpy()
                        elif hasattr(mask_tensor, 'numpy'):
                            mask = mask_tensor.numpy()
                        else:
                            mask = np.array(mask_tensor)
                    else:
                        # Fallback: use the numpy() method on the Masks object
                        mask = mask_obj.numpy() if hasattr(mask_obj, 'numpy') else mask_obj.cpu().numpy()
                    
                    # Handle different mask formats - get first mask if batch dimension exists
                    if mask.ndim > 2:
                        mask = mask[0] if mask.shape[0] == 1 else mask.squeeze()
                    
                    # Ensure mask is 2D
                    if mask.ndim != 2:
                        mask = mask.squeeze()
                    
                    # Validate mask dimensions
                    if mask.ndim != 2 or mask.size == 0:
                        # Invalid mask, return empty mask
                        h, w = image.shape[:2]
                        return np.zeros((h, w), dtype=np.uint8)
                    
                    # Resize to image size if needed
                    h, w = image.shape[:2]
                    if h <= 0 or w <= 0:
                        # Invalid image dimensions
                        return np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
                    
                    if mask.shape != (h, w):
                        # Only resize if mask has valid dimensions
                        if mask.shape[0] > 0 and mask.shape[1] > 0:
                            mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
                        else:
                            # Invalid mask dimensions, return empty mask
                            return np.zeros((h, w), dtype=np.uint8)
                    
                    # Convert to binary mask
                    mask = (mask > 0.5).astype(np.uint8)
                    return mask
            return np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            
        else:
            # For other SAM models
            if self.model is None:
                raise RuntimeError("Model not loaded")
            
            results = self.model(image)
            # Extract mask from first result
            if results and len(results) > 0:
                result = results[0]
                if result.masks is not None and len(result.masks) > 0:
                    mask = result.masks[0].cpu().numpy()
                    # Validate mask dimensions
                    if mask.size == 0 or mask.ndim < 2:
                        h, w = image.shape[:2]
                        return np.zeros((h, w), dtype=np.uint8)
                    
                    h, w = image.shape[:2]
                    if h <= 0 or w <= 0:
                        return np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
                    
                    if mask.shape[-2:] != (h, w):
                        # Only resize if mask has valid dimensions
                        if mask.shape[-2] > 0 and mask.shape[-1] > 0:
                            mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
                        else:
                            return np.zeros((h, w), dtype=np.uint8)
                    mask = (mask > 0.5).astype(np.uint8)
                    return mask
            return np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    
    def segment_by_classes(self, image: np.ndarray, class_names: List[str]) -> Dict:
        """
        Segment all objects of specified classes using text prompts (SAM3 only).
        
        Args:
            image: Input image as numpy array (H, W, 3)
            class_names: List of class names to segment (e.g., ["person", "bus", "car"])
            
        Returns:
            Dictionary with keys:
                - "masks": List of segmentation masks (one per instance)
                - "labels": List of class labels for each mask
                - "instances": List of instance IDs
                
        Raises:
            RuntimeError: If model is not SAM3
        """
        if self.model_type != "sam3":
            raise RuntimeError("segment_by_classes is only available for SAM3 model")
        
        if not SAM3_AVAILABLE:
            raise ImportError("SAM3 not available")
        
        # Set image if not already set
        if self.current_image is None or not np.array_equal(self.current_image, image):
            self.predictor.set_image(image)
            self.current_image = image.copy()
        
        # Query with text prompts
        results = self.predictor(text=class_names)
        
        # Extract masks and labels
        masks = []
        labels = []
        instance_ids = []
        
        if results and len(results) > 0:
            for i, result in enumerate(results):
                if result.masks is not None and len(result.masks) > 0:
                    for j, mask in enumerate(result.masks):
                        mask_np = mask.cpu().numpy()
                        # Validate mask dimensions
                        if mask_np.size == 0 or mask_np.ndim < 2:
                            h, w = image.shape[:2]
                            mask_np = np.zeros((h, w), dtype=np.uint8)
                        else:
                            # Resize to image size if needed
                            h, w = image.shape[:2]
                            if h <= 0 or w <= 0:
                                mask_np = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
                            elif mask_np.shape[-2:] != (h, w):
                                # Only resize if mask has valid dimensions
                                if mask_np.shape[-2] > 0 and mask_np.shape[-1] > 0:
                                    mask_np = cv2.resize(mask_np.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
                                else:
                                    mask_np = np.zeros((h, w), dtype=np.uint8)
                            # Convert to binary mask
                            mask_np = (mask_np > 0.5).astype(np.uint8)
                        masks.append(mask_np)
                        labels.append(class_names[i] if i < len(class_names) else "unknown")
                        instance_ids.append(len(masks) - 1)
        
        return {
            "masks": masks,
            "labels": labels,
            "instances": instance_ids
        }
    
    def match_instances_to_bboxes(self, masks: List[np.ndarray], bboxes: List[List[float]], 
                                  iou_threshold: float = 0.5) -> Dict[int, int]:
        """
        Match segmented instances to bounding boxes based on IoU (Intersection over Union).
        
        Args:
            masks: List of segmentation masks (H, W) as numpy arrays
            bboxes: List of bounding boxes [x1, y1, x2, y2]
            iou_threshold: Minimum IoU threshold for matching (default: 0.5)
            
        Returns:
            Dictionary mapping mask index to bbox index: {mask_idx: bbox_idx}
            If a mask doesn't match any bbox, it won't be in the dictionary
        """
        matches = {}
        
        for mask_idx, mask in enumerate(masks):
            # Get bounding box of the mask
            mask_bbox = self._get_bbox_from_mask(mask)
            
            best_iou = 0.0
            best_bbox_idx = -1
            
            for bbox_idx, bbox in enumerate(bboxes):
                iou = self._calculate_iou(mask_bbox, bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_bbox_idx = bbox_idx
            
            if best_iou >= iou_threshold:
                matches[mask_idx] = best_bbox_idx
        
        return matches
    
    def draw_bbox_around_mask(self, mask: np.ndarray) -> List[float]:
        """
        Draw bounding box around a segmented instance.
        
        Args:
            mask: Segmentation mask as numpy array (H, W) with binary values (0 or 1)
            
        Returns:
            Bounding box as [x1, y1, x2, y2]
        """
        return self._get_bbox_from_mask(mask)
    
    def _get_bbox_from_mask(self, mask: np.ndarray) -> List[float]:
        """
        Get bounding box coordinates from a binary mask.
        
        Args:
            mask: Binary mask as numpy array (H, W)
            
        Returns:
            Bounding box as [x1, y1, x2, y2]
        """
        # Find all non-zero pixels
        coords = np.column_stack(np.where(mask > 0))
        
        if len(coords) == 0:
            return [0, 0, 0, 0]
        
        # Get min and max coordinates
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        return [int(x_min), int(y_min), int(x_max), int(y_max)]
    
    def segment_everything(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Segment everything from an image without specific prompts (SAM2 only).
        Uses automatic mask generation to find all objects in the image.
        
        Args:
            image: Input image as numpy array (H, W, 3)
            
        Returns:
            List of binary masks (H, W) for each detected object
            
        Raises:
            RuntimeError: If model is not SAM2
        """
        if not self.model_type.startswith("sam2"):
            raise RuntimeError("segment_everything is only available for SAM2 model")
        
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Run inference without prompts - SAM2 will automatically segment everything
        results = self.model(image)
        
        masks = []
        if results and len(results) > 0:
            result = results[0] if isinstance(results, list) else results
            if hasattr(result, 'masks') and result.masks is not None:
                mask_obj = result.masks
                
                # Get all masks from the result
                if hasattr(mask_obj, 'data'):
                    mask_tensor = mask_obj.data
                    if hasattr(mask_tensor, 'cpu'):
                        all_masks = mask_tensor.cpu().numpy()
                    elif hasattr(mask_tensor, 'numpy'):
                        all_masks = mask_tensor.numpy()
                    else:
                        all_masks = np.array(mask_tensor)
                else:
                    all_masks = mask_obj.numpy() if hasattr(mask_obj, 'numpy') else mask_obj.cpu().numpy()
                
                # Handle different mask formats
                if all_masks.ndim == 2:
                    # Single mask
                    masks.append(all_masks)
                elif all_masks.ndim == 3:
                    # Multiple masks (N, H, W)
                    for i in range(all_masks.shape[0]):
                        mask = all_masks[i]
                        # Validate mask dimensions
                        if mask.size == 0 or mask.ndim < 2:
                            h, w = image.shape[:2]
                            mask = np.zeros((h, w), dtype=np.uint8)
                        else:
                            h, w = image.shape[:2]
                            if h <= 0 or w <= 0:
                                mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
                            elif mask.shape != (h, w):
                                # Only resize if mask has valid dimensions
                                if mask.shape[0] > 0 and mask.shape[1] > 0:
                                    mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
                                else:
                                    mask = np.zeros((h, w), dtype=np.uint8)
                            mask = (mask > 0.5).astype(np.uint8)
                        masks.append(mask)
                elif all_masks.ndim == 4:
                    # Batch dimension (B, N, H, W) - take first batch
                    for i in range(all_masks.shape[1]):
                        mask = all_masks[0, i]
                        # Validate mask dimensions
                        if mask.size == 0 or mask.ndim < 2:
                            h, w = image.shape[:2]
                            mask = np.zeros((h, w), dtype=np.uint8)
                        else:
                            h, w = image.shape[:2]
                            if h <= 0 or w <= 0:
                                mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
                            elif mask.shape != (h, w):
                                # Only resize if mask has valid dimensions
                                if mask.shape[0] > 0 and mask.shape[1] > 0:
                                    mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
                                else:
                                    mask = np.zeros((h, w), dtype=np.uint8)
                            mask = (mask > 0.5).astype(np.uint8)
                        masks.append(mask)
        
        print(f"Segmented {len(masks)} objects from image using SAM2 automatic segmentation")
        return masks
    
    def get_object_boundaries(self, masks: List[np.ndarray]) -> np.ndarray:
        """
        Get combined object boundaries from masks as a binary mask.
        Uses morphological operations to extract boundaries efficiently.
        
        Args:
            masks: List of binary masks (H, W)
            
        Returns:
            Binary mask (H, W) with 1 at object boundaries, 0 elsewhere
        """
        if len(masks) == 0:
            return np.zeros((1, 1), dtype=np.uint8)
        
        # Get shape from first mask
        h, w = masks[0].shape
        
        # Combine all masks
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        for mask in masks:
            if mask is not None:
                combined_mask = np.logical_or(combined_mask, mask > 0).astype(np.uint8)
        
        # Use morphological operations to get boundaries (erosion then difference)
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(combined_mask, kernel, iterations=1)
        boundaries = combined_mask - eroded
        
        return boundaries.astype(np.uint8)
    
    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            bbox1: First bounding box [x1, y1, x2, y2]
            bbox2: Second bounding box [x1, y1, x2, y2]
            
        Returns:
            IoU value between 0 and 1
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union


def assign_points_to_masks(
    points_3d: np.ndarray,
    masks: List[np.ndarray],
    projection,
    image_shape: Tuple[int, int],
) -> np.ndarray:
    """
    Assign each 3D point to a mask by projecting to 2D and checking mask membership.

    Args:
        points_3d: Nx3 array of 3D points in LiDAR coordinates.
        masks: List of binary masks (H, W).
        projection: Projection object with a point_to_pixel method.
        image_shape: (height, width) of the image.

    Returns:
        N array of mask indices (-1 for points not in any mask).
    """
    n_points = len(points_3d)
    if n_points == 0 or len(masks) == 0:
        return np.full(n_points, -1, dtype=int)

    pixels, valid_mask = projection.point_to_pixel(points_3d)
    h, w = image_shape

    in_bounds = (
        (pixels[:, 0] >= 0)
        & (pixels[:, 0] < w)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < h)
    )
    valid_mask = valid_mask & in_bounds

    mask_assignments = np.full(n_points, -1, dtype=int)

    for mask_idx, mask in enumerate(masks):
        if mask is None:
            continue

        valid_pixels = pixels[valid_mask].astype(int)
        valid_indices = np.where(valid_mask)[0]

        for i, (u, v) in enumerate(valid_pixels):
            if 0 <= v < h and 0 <= u < w:
                if mask[v, u] > 0:
                    point_idx = valid_indices[i]
                    if mask_assignments[point_idx] == -1:
                        mask_assignments[point_idx] = mask_idx

    return mask_assignments
