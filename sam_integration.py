"""
SAM Integration Module for 3D Object Detection Pipeline
This module provides classes for integrating SAM models with the existing pipeline.
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional, Union
import torch
from pathlib import Path

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


class SAMModelManager:
    """
    Class to manage different SAM models for segmentation.
    """
    
    def __init__(self, model_type: str = "sam2_t"):
        """
        Initialize SAM model manager.
        
        Args:
            model_type: Type of SAM model to use
                      Options: "sam2_t", "sam2_b", "sam2_l", "sam_b", "mobile_sam"
        """
        self.model_type = model_type
        self.model = None
        self.predictor = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("Ultralytics not available. Install with: pip install ultralytics")
        
        self._load_model()
    
    def _load_model(self):
        """Load the specified SAM model."""
        try:
            if self.model_type.startswith("sam2"):
                if self.model_type == "sam2_t":
                    model_file = "sam2_t.pt"
                elif self.model_type == "sam2_b":
                    model_file = "sam2_b.pt"
                elif self.model_type == "sam2_l":
                    model_file = "sam2_l.pt"
                else:
                    raise ValueError(f"Unknown SAM2 model type: {self.model_type}")
                
                # Create SAM2DynamicInteractivePredictor
                overrides = dict(conf=0.01, task="segment", mode="predict", 
                                imgsz=1024, model=model_file, save=False)
                self.predictor = SAM2DynamicInteractivePredictor(overrides=overrides, max_obj_num=10)
                print(f"Loaded SAM2 model: {model_file}")
                
            elif self.model_type == "sam_b":
                self.model = SAM("sam_b.pt")
                self.model.info()
                print("Loaded SAM-b model")
                
            elif self.model_type == "mobile_sam":
                self.model = FastSAM("FastSAM-s.pt")
                self.model.info()
                print("Loaded MobileSAM model")
                
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
                
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def predict_from_bboxes(self, image: np.ndarray, bboxes: List[List[int]], 
                          obj_ids: Optional[List[int]] = None,
                          update_memory: bool = True) -> Dict:
        """
        Predict segmentation masks from bounding boxes.
        
        Args:
            image: Input image as numpy array (H, W, 3)
            bboxes: List of bounding boxes [x1, y1, x2, y2]
            obj_ids: Optional list of object IDs for each bbox
            update_memory: Whether to update memory with new objects
            
        Returns:
            Dictionary with segmentation results
        """
        if self.predictor is None:
            raise RuntimeError("Model not loaded")
        
        if obj_ids is None:
            obj_ids = [0] * len(bboxes)
        
        if self.model_type.startswith("sam2"):
            # Use SAM2DynamicInteractivePredictor
            results = self.predictor(
                source=image,
                bboxes=bboxes,
                obj_ids=obj_ids,
                update_memory=update_memory
            )
            return results
        else:
            # For other SAM models, use standard predict method
            results = self.model(image)
            return results
    
    def predict_from_points(self, image: np.ndarray, points: List[List[int]], 
                         labels: Optional[List[int]] = None,
                         obj_ids: Optional[List[int]] = None,
                         update_memory: bool = True) -> Dict:
        """
        Predict segmentation masks from point prompts.
        
        Args:
            image: Input image as numpy array (H, W, 3)
            points: List of point coordinates [x, y]
            labels: Optional list of point labels (1 for positive, 0 for negative)
            obj_ids: Optional list of object IDs for each point
            update_memory: Whether to update memory with new objects
            
        Returns:
            Dictionary with segmentation results
        """
        if self.predictor is None:
            raise RuntimeError("Model not loaded")
        
        if labels is None:
            labels = [1] * len(points)
        
        if obj_ids is None:
            obj_ids = [0] * len(points)
        
        if self.model_type.startswith("sam2"):
            # Use SAM2DynamicInteractivePredictor
            results = self.predictor(
                source=image,
                points=points,
                labels=labels,
                obj_ids=obj_ids,
                update_memory=update_memory
            )
            return results
        else:
            # For other SAM models, use standard predict method
            results = self.model(image)
            return results
    
    def predict(self, image: np.ndarray) -> Dict:
        """
        Predict segmentation masks from image without prompts.
        
        Args:
            image: Input image as numpy array (H, W, 3)
            
        Returns:
            Dictionary with segmentation results
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        results = self.model(image)
        return results
    
    def get_segmentation_masks(self, results: Dict) -> np.ndarray:
        """
        Extract segmentation masks from prediction results.
        
        Args:
            results: Dictionary with prediction results
            
        Returns:
            Segmentation mask as numpy array (H, W) with integer labels
        """
        if self.model_type.startswith("sam2"):
            # Extract masks from SAM2 results
            if results and len(results) > 0:
                # Create empty mask
                mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
                
                # Process each result
                for i, result in enumerate(results):
                    if result.masks is not None and len(result.masks) > 0:
                        # Get the first mask
                        seg_mask = result.masks[0].cpu().numpy()
                        # Add to mask with unique label
                        mask[seg_mask > 0.5] = i + 1
                
                return mask
            else:
                return np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        else:
            # For other SAM models, extract masks differently
            if results and hasattr(results[0], 'masks'):
                # Create empty mask
                mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
                
                # Process each result
                for i, result in enumerate(results):
                    if result.masks is not None and len(result.masks) > 0:
                        # Get the first mask
                        seg_mask = result.masks[0].cpu().numpy()
                        # Add to mask with unique label
                        mask[seg_mask > 0.5] = i + 1
                
                return mask
            else:
                return np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)


class BoundingBoxToSAM:
    """
    Class to convert bounding boxes to SAM prompts.
    """
    
    def __init__(self, sam_manager: SAMModelManager):
        """
        Initialize with a SAM model manager.
        
        Args:
            sam_manager: SAMModelManager instance
        """
        self.sam_manager = sam_manager
    
    def get_sam_prompts_from_bboxes(self, image: np.ndarray, 
                                   bboxes: List[List[float]]) -> Dict:
        """
        Convert bounding boxes to SAM prompts.
        
        Args:
            image: Input image as numpy array (H, W, 3)
            bboxes: List of bounding boxes [x1, y1, x2, y2]
            
        Returns:
            Dictionary with prompts for SAM
        """
        # Get center points of bounding boxes
        center_points = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            center_points.append([center_x, center_y])
        
        # Create prompts
        prompts = {
            "points": np.array(center_points),
            "labels": np.array([1] * len(center_points)),  # All positive points
            "bboxes": bboxes
        }
        
        return prompts
    
    def segment_from_bboxes(self, image: np.ndarray, 
                         bboxes: List[List[float]]) -> np.ndarray:
        """
        Segment objects using bounding boxes as prompts.
        
        Args:
            image: Input image as numpy array (H, W, 3)
            bboxes: List of bounding boxes [x1, y1, x2, y2]
            
        Returns:
            Segmentation mask as numpy array (H, W) with integer labels
        """
        # Convert bounding boxes to SAM prompts
        prompts = self.get_sam_prompts_from_bboxes(image, bboxes)
        
        # Predict with SAM
        results = self.sam_manager.predict_from_bboxes(image, bboxes)
        
        # Extract masks
        mask = self.sam_manager.get_segmentation_masks(results)
        
        return mask
