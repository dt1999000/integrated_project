"""
SAM Integration Module for 3D Object Detection Pipeline
This module provides unified class for integrating SAM2 and SAM3 models.
Supports bounding box-based segmentation (SAM2 & SAM3) and text-based semantic segmentation (SAM3 only).
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Set, Any
import torch
import cv2
import os
from components.utils.mask_utils import get_bbox_from_mask

try:
    from ultralytics import YOLO, SAM
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    print("Ultralytics not available. Install with: pip install ultralytics")
    ULTRALYTICS_AVAILABLE = False

try:
    from ultralytics.models.sam import SAM2DynamicInteractivePredictor
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False

try:
    from ultralytics.models.sam import SAM3SemanticPredictor
    SAM3_AVAILABLE = True
except ImportError:
    print("SAM3 not available. Install with: pip install ultralytics")
    SAM3_AVAILABLE = False

try:
    from transformers import pipeline as _hf_zero_shot_od_pipeline
    from PIL import Image as _PIL_Image

    TRANSFORMERS_GROUNDING_AVAILABLE = True
except ImportError:
    TRANSFORMERS_GROUNDING_AVAILABLE = False
    _hf_zero_shot_od_pipeline = None  # type: ignore
    _PIL_Image = None  # type: ignore

def get_available_models() -> Dict[str, List[str]]:
    """
    Discover available model files in the project's models directory.
    Returns base names for SAM models and filenames (with extension) for YOLO models.
    """
    models_dir = get_models_dir()

    discovered = {"sam2": [], "sam3": [], "yolo": []}
    if not os.path.isdir(models_dir):
        return discovered

    files = sorted(os.listdir(models_dir))
    for fname in files:
        fpath = os.path.join(models_dir, fname)
        if not os.path.isfile(fpath):
            continue

        base, ext = os.path.splitext(fname)
        if ext.lower() not in {".pt", ".pth"}:
            continue

        b = base.lower()
        if b.startswith("sam2_") or b == "sam2":
            discovered["sam2"].append(base)
        elif b.startswith("sam3"):
            discovered["sam3"].append(base)
        elif b.startswith("yolo") or "world" in b:
            discovered["yolo"].append(fname)

    return discovered


def get_models_dir() -> str:
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.abspath(os.path.join(project_root, "models"))


def calculate_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes
    in [x_min, y_min, x_max, y_max] format.
    """
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_w = max(0.0, inter_x_max - inter_x_min)
    inter_h = max(0.0, inter_y_max - inter_y_min)
    inter_area = inter_w * inter_h

    area1 = max(0.0, x1_max - x1_min) * max(0.0, y1_max - y1_min)
    area2 = max(0.0, x2_max - x2_min) * max(0.0, y2_max - y2_min)

    union = area1 + area2 - inter_area
    if union <= 0.0:
        return 0.0

    return inter_area / union


def _binary_mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """
    IoU between two binary masks (accepts any numeric dtype).
    If shapes differ, b is resized to a's spatial shape (nearest).
    """
    if a is None or b is None:
        return 0.0
    if a.size == 0 or b.size == 0:
        return 0.0
    aa = (a > 0).astype(np.uint8)
    bb = (b > 0).astype(np.uint8)
    if aa.ndim != 2:
        aa = np.squeeze(aa)
    if bb.ndim != 2:
        bb = np.squeeze(bb)
    if aa.ndim != 2 or bb.ndim != 2:
        return 0.0
    if aa.shape != bb.shape:
        h, w = aa.shape[:2]
        if h <= 0 or w <= 0:
            return 0.0
        bb = cv2.resize(bb, (w, h), interpolation=cv2.INTER_NEAREST)
    inter = int(np.sum((aa > 0) & (bb > 0)))
    union = int(np.sum((aa > 0) | (bb > 0)))
    return float(inter / union) if union > 0 else 0.0


class SAMIntegration:
    """
    Unified class for SAM model management and segmentation operations.
    Supports SAM2 (bounding box segmentation) and SAM3 (bounding box + text-based semantic segmentation).
    """

    def __init__(self, model_type: str = "sam2_t", use_gpu: bool = True):
        """
        Initialize SAM integration manager.

        Args:
            model_type: Type of SAM model to use
                      Options: "sam2_t", "sam2_b", "sam2_l", "sam3"
            use_gpu: Boolean flag to enable GPU usage (if available)
        """
        self.model_type = model_type
        self.model = None
        self.predictor = None
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        self.current_image = None
        self._yolo_integration = None  # Lazy initialization for SAM2 pipeline
        self._grounding_dino_integration = None
        self._grounding_dino_model_id_cached: Optional[str] = None
        self._sam3_prev_tracked: List[Dict[str, Union[str, np.ndarray, float]]] = []
        self._sam3_prev_mask_tracks: List[Dict[str, Any]] = []
        self._sam3_next_track_id: int = 0
        # SAM2 memory-bank slot management: maps track_id -> predictor slot index (0..max-1)
        self._sam2_max_obj_num: int = 50
        self._sam2_slot_map: Dict[int, int] = {}
        self._sam2_free_slots: List[int] = list(range(50))

        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("Ultralytics not available. Install with: pip install ultralytics")

        self._load_model()

    def _empty_mask_for_image(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        return np.zeros((h, w), dtype=np.uint8)

    def _ensure_sam3_image(self, image: np.ndarray) -> None:
        if self.current_image is None or not np.array_equal(self.current_image, image):
            self.predictor.set_image(image)
            self.current_image = image.copy()

    def _masks_obj_to_numpy(self, mask_obj: Any) -> np.ndarray:
        """
        Convert Ultralytics Masks wrapper / tensor-like object to a numpy array.
        Returns an empty array if conversion fails.
        """
        if mask_obj is None:
            return np.array([])

        mask_tensor = getattr(mask_obj, "data", None)
        src = mask_tensor if mask_tensor is not None else mask_obj
        cpu_fn = getattr(src, "cpu", None)
        numpy_fn = getattr(src, "numpy", None)

        if callable(cpu_fn):
            return cpu_fn().numpy()
        if callable(numpy_fn):
            return numpy_fn()
        return np.array(src)

    def _normalize_mask_to_image(
        self,
        mask: Any,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Normalize any mask-like input into a (H, W) binary np.uint8 mask.
        Accepts 2D/3D/4D arrays and picks the first mask when batched.
        """
        h, w = image.shape[:2]
        if h <= 0 or w <= 0:
            return np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        if mask is None:
            return np.zeros((h, w), dtype=np.uint8)

        m = np.array(mask)
        if m.size == 0:
            return np.zeros((h, w), dtype=np.uint8)

        if m.ndim == 2:
            m2 = m
        elif m.ndim == 3:
            m2 = m[0]
        elif m.ndim == 4:
            m2 = m[0, 0]
        else:
            return np.zeros((h, w), dtype=np.uint8)

        if m2.ndim != 2 or m2.size == 0:
            m2 = np.squeeze(m2)
        if m2.ndim != 2 or m2.size == 0:
            return np.zeros((h, w), dtype=np.uint8)

        if m2.shape[-2:] != (h, w):
            if m2.shape[-2] > 0 and m2.shape[-1] > 0:
                m2 = cv2.resize(m2.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
            else:
                return np.zeros((h, w), dtype=np.uint8)

        return (m2 > 0.5).astype(np.uint8)

    def _load_model(self):
        """Load the specified SAM model from ./models directory."""
        models_dir = get_models_dir()

        if self.model_type.startswith("sam3"):
            if not SAM3_AVAILABLE:
                raise ImportError("SAM3 not available. Install with: pip install ultralytics")

            model_file = f"{self.model_type}.pt"
            model_path = os.path.abspath(os.path.join(models_dir, model_file))
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"SAM3 model not found at {model_path}")

            overrides = dict(
                conf=0.25,
                task="segment",
                mode="predict",
                model=model_path,
                half=self.use_gpu,
                save=True,
                device=0 if self.use_gpu else "cpu",
            )
            self.predictor = SAM3SemanticPredictor(overrides=overrides)
            print(f"Loaded SAM3 model from {model_path} on {self.device}")

        elif self.model_type.startswith("sam2"):
            if not ULTRALYTICS_AVAILABLE:
                raise ImportError("Ultralytics not available. Install with: pip install ultralytics")

            model_file = f"{self.model_type}.pt"
            model_path = os.path.abspath(os.path.join(models_dir, model_file))
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"SAM2 model '{model_file}' not found at {model_path}")

            # Use the standard SAM class for per-frame bbox-prompted segmentation.
            # SAM2DynamicInteractivePredictor is an interactive video predictor and
            # does not support the model(image, bboxes=...) inference API.
            self.model = SAM(model_path)
            print(f"Loaded SAM2 model from {model_path} on {self.device}")

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def get_mask_from_bbox(
        self,
        image: np.ndarray,
        bbox: List[float]
    ) -> np.ndarray:
        """
        Get segmentation mask from a single bounding box on the current frame.

        Args:
            image: Current frame as numpy array (H, W, 3).
            bbox: Bounding box [x1, y1, x2, y2] in pixel coordinates of ``image``.

        Returns:
            Segmentation mask as numpy array (H, W) with binary values (0 or 1).
        """
        if self.model_type == "sam3":
            self._ensure_sam3_image(image)
            results = self.predictor(bboxes=[bbox])
            if results and len(results) > 0:
                result = results[0]
                masks_obj = getattr(result, "masks", None)
                masks_np = self._masks_obj_to_numpy(masks_obj)
                if masks_np.size > 0:
                    return self._normalize_mask_to_image(masks_np, image)
            return self._empty_mask_for_image(image)

        elif self.model_type.startswith("sam2"):
            if self.model is None:
                raise RuntimeError("Model not loaded")

            if isinstance(bbox, np.ndarray):
                bbox = bbox.tolist()

            if not isinstance(bbox, list) or len(bbox) != 4:
                raise ValueError(f"Bbox must be a list with 4 elements [x1, y1, x2, y2], got {bbox}")

            # Clamp and normalize bbox to image coordinates, then pass as a single-box batch.
            h, w = image.shape[:2]
            x1, y1, x2, y2 = [float(v) for v in bbox]
            x1 = max(0.0, min(x1, float(w - 1)))
            x2 = max(0.0, min(x2, float(w - 1)))
            y1 = max(0.0, min(y1, float(h - 1)))
            y2 = max(0.0, min(y2, float(h - 1)))
            if x2 <= x1 or y2 <= y1:
                return self._empty_mask_for_image(image)
            bbox_prompt = [x1, y1, x2, y2]

            # SAM (Ultralytics) bbox inference: bboxes must be a list of boxes,
            # so a single box is passed as [[x1, y1, x2, y2]].
            results = self.model(
                image,
                bboxes=[bbox_prompt],
                device=0 if self.use_gpu else "cpu",
            )
            #print(f'results: {results}', flush=True)
            if results and len(results) > 0:
                result = results[0] if isinstance(results, list) else results
                masks_obj = getattr(result, "masks", None)
                masks_np = self._masks_obj_to_numpy(masks_obj)
                if masks_np.size > 0:
                    return self._normalize_mask_to_image(masks_np, image)
            return self._empty_mask_for_image(image)

        else:
            if self.model is None:
                raise RuntimeError("Model not loaded")

            results = self.model(image, device=0 if self.use_gpu else "cpu")
            if results and len(results) > 0:
                result = results[0]
                masks_obj = getattr(result, "masks", None)
                masks_np = self._masks_obj_to_numpy(masks_obj)
                if masks_np.size > 0:
                    return self._normalize_mask_to_image(masks_np, image)
            return self._empty_mask_for_image(image)

    # ------------------------------------------------------------------
    # SAM2 memory-bank tracking API
    # ------------------------------------------------------------------

    def _sam2_alloc_slot(self, track_id: int) -> Optional[int]:
        """Return the existing slot for ``track_id``, or allocate a new one."""
        if track_id in self._sam2_slot_map:
            return self._sam2_slot_map[track_id]
        if not self._sam2_free_slots:
            print(
                f"[sam2] No free predictor slots for track_id={track_id}; "
                f"max_obj_num={self._sam2_max_obj_num} exceeded."
            )
            return None
        slot = self._sam2_free_slots.pop(0)
        self._sam2_slot_map[track_id] = slot
        return slot

    def sam2_register_tracks(
        self,
        image: np.ndarray,
        track_bboxes: Dict[int, List[float]],
    ) -> None:
        """Register or refresh tracks in the SAM2 memory bank.

        For each entry in ``track_bboxes``, allocates a predictor slot (reusing
        any existing slot for that track) and calls the predictor with
        ``update_memory=True`` using the bbox as prompt on ``image``.  After
        this call, ``sam2_propagate_all_tracks`` can locate those objects on
        subsequent frames without any prompt.

        Args:
            image: Frame on which the bboxes are defined (H, W, 3).
            track_bboxes: Mapping of track_id -> [x1, y1, x2, y2].
        """
        if not self.model_type.startswith("sam2") or not track_bboxes:
            return
        h, w = image.shape[:2]

        # Collect all valid (slot, bbox) pairs first, then register in a single
        # predictor call so image features are extracted only once for all tracks.
        batch_slots: List[int] = []
        batch_bboxes: List[List[float]] = []
        for track_id, bbox in track_bboxes.items():
            slot = self._sam2_alloc_slot(track_id)
            if slot is None:
                continue
            x1, y1, x2, y2 = [float(v) for v in bbox]
            x1 = max(0.0, min(x1, float(w - 1)))
            x2 = max(0.0, min(x2, float(w - 1)))
            y1 = max(0.0, min(y1, float(h - 1)))
            y2 = max(0.0, min(y2, float(h - 1)))
            if x2 <= x1 or y2 <= y1:
                continue
            batch_slots.append(slot)
            batch_bboxes.append([x1, y1, x2, y2])

        if not batch_slots:
            return

        self.model(
            image,
            bboxes=batch_bboxes,
            obj_ids=batch_slots,
            update_memory=True,
        )

    def sam2_propagate_all_tracks(self, image: np.ndarray) -> Dict[int, np.ndarray]:
        """Propagate all registered tracks into ``image`` using the SAM2 memory bank.

        Calls the predictor with no prompt so SAM2 uses its memory from prior
        ``sam2_register_tracks`` calls to locate each object in the new frame.

        Args:
            image: Current frame (H, W, 3).

        Returns:
            Dict mapping track_id -> binary segmentation mask (H, W) for every
            track whose propagated mask is non-empty.
        """
        if not self.model_type.startswith("sam2"):
            return {}
        if not self._sam2_slot_map or len(self.model.obj_idx_set) == 0:
            return {}

        results = self.model(image)

        slot_to_track = {slot: tid for tid, slot in self._sam2_slot_map.items()}
        # obj_idx_set contains the active slots; their order matches the returned masks.
        active_slots = list(self.model.obj_idx_set)

        obj_masks: Dict[int, np.ndarray] = {}
        if not results:
            return obj_masks
        result = results[0] if isinstance(results, list) else results
        masks_obj = getattr(result, "masks", None)
        if masks_obj is None:
            return obj_masks
        masks_np = self._masks_obj_to_numpy(masks_obj)
        if masks_np.size == 0:
            return obj_masks
        if masks_np.ndim == 2:
            masks_np = masks_np[None]

        for i, slot in enumerate(active_slots):
            if i >= masks_np.shape[0]:
                break
            track_id = slot_to_track.get(slot)
            if track_id is None:
                continue
            mask = self._normalize_mask_to_image(masks_np[i], image)
            if np.sum(mask > 0) > 0:
                obj_masks[track_id] = mask

        return obj_masks

    def sam2_release_tracks(self, track_ids: List[int]) -> None:
        """Free SAM2 predictor slots for tracks that are no longer active.

        Released slots are returned to the free pool and become available for
        future ``sam2_register_tracks`` calls.

        Args:
            track_ids: Track IDs to release.
        """
        for track_id in track_ids:
            slot = self._sam2_slot_map.pop(track_id, None)
            if slot is not None:
                self.model.obj_idx_set.discard(slot)
                self._sam2_free_slots.append(slot)

    def sam2_trim_memory_bank(self, max_frames: int) -> None:
        """Keep only most recent SAM2 memory entries to bound VRAM."""
        if not self.model_type.startswith("sam2"):
            return
        if max_frames <= 0:
            return
        bank = getattr(self.model, "memory_bank", None)
        if not bank or len(bank) <= int(max_frames):
            return
        self.model.memory_bank = bank[-int(max_frames):]

    def sam2_reset_state(self) -> None:
        """Clear SAM2 slot assignments and memory bank state."""
        if not self.model_type.startswith("sam2"):
            return
        self._sam2_slot_map.clear()
        self._sam2_free_slots = list(range(self._sam2_max_obj_num))
        memory_bank = getattr(self.model, "memory_bank", None)
        if memory_bank is not None:
            memory_bank.clear()
        obj_idx_set = getattr(self.model, "obj_idx_set", None)
        if obj_idx_set is not None:
            obj_idx_set.clear()

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
                - "scores": List of confidence scores for each mask (if available)

        Raises:
            RuntimeError: If model is not SAM3
        """
        if self.model_type != "sam3":
            raise RuntimeError("segment_by_classes is only available for SAM3 model")

        if not SAM3_AVAILABLE:
            raise ImportError("SAM3 not available")

        self._ensure_sam3_image(image)

        # Query with text prompts
        results = self.predictor(text=class_names)

        # Extract masks, labels, and scores
        masks = []
        labels = []
        instance_ids = []
        scores = []

        if results and len(results) > 0:
            for i, result in enumerate(results):
                if result.masks is not None and len(result.masks) > 0:
                    result_scores = None
                    scores_obj = getattr(result, "scores", None)
                    score_obj = getattr(result, "score", None)
                    if scores_obj is not None:
                        cpu_fn = getattr(scores_obj, "cpu", None)
                        result_scores = cpu_fn().numpy() if callable(cpu_fn) else scores_obj
                    elif score_obj is not None:
                        cpu_fn = getattr(score_obj, "cpu", None)
                        result_scores = cpu_fn().numpy() if callable(cpu_fn) else score_obj

                    all_masks = self._masks_obj_to_numpy(result.masks)

                    mask_list: List[np.ndarray] = []
                    if all_masks.ndim == 2:
                        mask_list = [all_masks]
                    elif all_masks.ndim == 3:
                        mask_list = [all_masks[k] for k in range(all_masks.shape[0])]
                    elif all_masks.ndim == 4:
                        mask_list = [all_masks[0, k] for k in range(all_masks.shape[1])]

                    for j, mask_np in enumerate(mask_list):
                        mask_np = self._normalize_mask_to_image(mask_np, image)
                        masks.append(mask_np)
                        labels.append(class_names[i] if i < len(class_names) else "unknown")
                        instance_ids.append(len(masks) - 1)

                        if result_scores is not None:
                            if isinstance(result_scores, np.ndarray):
                                if result_scores.ndim == 0:
                                    score = float(result_scores)
                                elif len(result_scores) > j:
                                    score = float(result_scores[j])
                                else:
                                    score = float(result_scores[0]) if len(result_scores) > 0 else 1.0
                            else:
                                score = float(result_scores)
                        else:
                            score = None
                        scores.append(score)

        return {
            "masks": masks,
            "labels": labels,
            "instances": instance_ids,
            "scores": scores
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
            mask_bbox = self._get_bbox_from_mask(mask)

            best_iou = 0.0
            best_bbox_idx = -1

            for bbox_idx, bbox in enumerate(bboxes):
                iou = calculate_iou(mask_bbox, bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_bbox_idx = bbox_idx

            if best_iou >= iou_threshold:
                matches[mask_idx] = best_bbox_idx

        return matches

    def _get_bbox_from_mask(self, mask: np.ndarray) -> List[float]:
        return get_bbox_from_mask(mask)

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

        results = self.model(image, device=0 if self.use_gpu else "cpu")

        masks = []
        if results and len(results) > 0:
            result = results[0] if isinstance(results, list) else results
            mask_obj = getattr(result, "masks", None)
            if mask_obj is not None:
                mask_tensor = getattr(mask_obj, "data", None)
                if mask_tensor is not None:
                    cpu_fn = getattr(mask_tensor, "cpu", None)
                    numpy_fn = getattr(mask_tensor, "numpy", None)
                    if callable(cpu_fn):
                        all_masks = cpu_fn().numpy()
                    elif callable(numpy_fn):
                        all_masks = numpy_fn()
                    else:
                        all_masks = np.array(mask_tensor)
                else:
                    numpy_fn = getattr(mask_obj, "numpy", None)
                    cpu_fn = getattr(mask_obj, "cpu", None)
                    if callable(numpy_fn):
                        all_masks = numpy_fn()
                    elif callable(cpu_fn):
                        all_masks = cpu_fn().numpy()
                    else:
                        all_masks = np.array(mask_obj)

                if all_masks.ndim == 2:
                    masks.append(all_masks)
                elif all_masks.ndim == 3:
                    for i in range(all_masks.shape[0]):
                        mask = all_masks[i]
                        if mask.size == 0 or mask.ndim < 2:
                            h, w = image.shape[:2]
                            mask = np.zeros((h, w), dtype=np.uint8)
                        else:
                            h, w = image.shape[:2]
                            if h <= 0 or w <= 0:
                                mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
                            elif mask.shape != (h, w):
                                if mask.shape[0] > 0 and mask.shape[1] > 0:
                                    mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
                                else:
                                    mask = np.zeros((h, w), dtype=np.uint8)
                            mask = (mask > 0.5).astype(np.uint8)
                        masks.append(mask)
                elif all_masks.ndim == 4:
                    for i in range(all_masks.shape[1]):
                        mask = all_masks[0, i]
                        if mask.size == 0 or mask.ndim < 2:
                            h, w = image.shape[:2]
                            mask = np.zeros((h, w), dtype=np.uint8)
                        else:
                            h, w = image.shape[:2]
                            if h <= 0 or w <= 0:
                                mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
                            elif mask.shape != (h, w):
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

        h, w = masks[0].shape

        combined_mask = np.zeros((h, w), dtype=np.uint8)
        for mask in masks:
            if mask is not None:
                combined_mask = np.logical_or(combined_mask, mask > 0).astype(np.uint8)

        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(combined_mask, kernel, iterations=1)
        boundaries = combined_mask - eroded

        return boundaries.astype(np.uint8)

    def segment_by_class_names(self, image: np.ndarray, class_names: List[str],
                               yolo_model_path: Optional[str] = None,
                               conf_threshold: float = 0.25,
                               open_vocab_detector: str = "yolo",
                               grounding_dino_model_id: Optional[str] = None) -> Dict:
        """
        Unified pipeline for segmentation by class names.
        For SAM3: Directly segments using class names.
        For SAM2: Uses an open-vocabulary detector (YOLO-World or Grounding DINO) for boxes,
        then segments with SAM2.

        Args:
            image: Input image as numpy array (H, W, 3) in RGB format
            class_names: List of class names to segment (e.g., ["car", "person", "bicycle"])
            yolo_model_path: Optional path to YOLO-World model (SAM2 + open_vocab_detector="yolo")
            conf_threshold: Confidence threshold for open-vocab detections (default: 0.25)
            open_vocab_detector: ``"yolo"`` (YOLO-World) or ``"grounding_dino"`` (HF Grounding DINO)
            grounding_dino_model_id: Hugging Face model id for Grounding DINO (default:
                ``IDEA-Research/grounding-dino-base``)

        Returns:
            Dictionary with:
                - 'masks': List of segmentation masks (H, W) as numpy arrays
                - 'bboxes': List of bounding boxes [x1, y1, x2, y2] for each mask
                - 'class_names': List of class names for each mask
                - 'confidences': List of confidence scores for each mask
        """
        if self.model_type == "sam3":
            return self._segment_by_class_names_sam3(image, class_names)
        elif self.model_type.startswith("sam2"):
            return self._segment_by_class_names_sam2(
                image, class_names, yolo_model_path, conf_threshold,
                open_vocab_detector=open_vocab_detector,
                grounding_dino_model_id=grounding_dino_model_id,
            )
        else:
            raise RuntimeError(f"segment_by_class_names not supported for model type: {self.model_type}")

    def _segment_by_class_names_sam3(self, image: np.ndarray, class_names: List[str]) -> Dict:
        """
        SAM3 pipeline: Directly segment using class names, then draw boxes around masks.

        Args:
            image: Input image as numpy array (H, W, 3)
            class_names: List of class names to segment

        Returns:
            Dictionary with masks, bboxes, class_names, and confidences
        """
        segment_results = self.segment_by_classes(image, class_names)

        masks = segment_results['masks']
        labels = segment_results['labels']
        scores = segment_results.get('scores', [])

        bboxes = []
        for mask in masks:
            bbox = get_bbox_from_mask(mask)
            bboxes.append(bbox)

        confidences = []
        for i, mask in enumerate(masks):
            if i < len(scores) and scores[i] is not None:
                confidence = float(scores[i])
            else:
                # Fallback: Use mask area as a proxy for confidence (normalized)
                mask_area = np.sum(mask > 0)
                h, w = image.shape[:2]
                normalized_area = mask_area / (h * w)
                confidence = min(1.0, normalized_area * 10.0)

            confidences.append(confidence)

        masks_per_class: Dict[str, int] = {}
        for lbl in labels:
            masks_per_class[str(lbl)] = masks_per_class.get(str(lbl), 0) + 1

        return {
            'masks': masks,
            'bboxes': bboxes,
            'class_names': labels,
            'confidences': confidences,
            'debug': {
                'sam_model': 'sam3',
                'n_class_prompts': len(class_names),
                'n_masks': len(masks),
                'masks_per_class': masks_per_class,
            },
        }

    def _segment_by_class_names_sam2(self, image: np.ndarray, class_names: List[str],
                                     yolo_model_path: Optional[str] = None,
                                     conf_threshold: float = 0.25,
                                     open_vocab_detector: str = "yolo",
                                     grounding_dino_model_id: Optional[str] = None) -> Dict:
        """
        SAM2 pipeline: Use an open-vocabulary detector for boxes, then segment with SAM2.

        Args:
            image: Input image as numpy array (H, W, 3)
            class_names: List of class names to detect and segment
            yolo_model_path: Optional path to YOLO-World model (when using YOLO)
            conf_threshold: Confidence threshold for detector
            open_vocab_detector: ``"yolo"`` or ``"grounding_dino"``
            grounding_dino_model_id: Hugging Face model id for Grounding DINO

        Returns:
            Dictionary with masks, bboxes, class_names, and confidences
        """
        backend = (open_vocab_detector or "yolo").strip().lower()
        if backend == "grounding_dino":
            model_id = grounding_dino_model_id or "IDEA-Research/grounding-dino-base"
            if (
                self._grounding_dino_integration is None
                or self._grounding_dino_model_id_cached != model_id
            ):
                self._grounding_dino_integration = GroundingDINOIntegration(
                    model_id=model_id, use_gpu=self.use_gpu
                )
                self._grounding_dino_model_id_cached = model_id
            detector = self._grounding_dino_integration
            print(f"detector: {detector}, pipeline: {detector._pipe}", flush=True)
            ov_detections = detector.detect_with_classes(
                image, class_names, conf_threshold=conf_threshold
            )
        else:
            if self._yolo_integration is None:
                if yolo_model_path is None:
                    yolo_model_path = "yolov8s-world.pt"
                self._yolo_integration = YOLOIntegration(yolo_model_path, use_gpu=self.use_gpu)
            detector = self._yolo_integration
            ov_detections = detector.detect_with_classes(
                image, class_names, conf_threshold=conf_threshold
            )

        print(f'ov_detections: {ov_detections}', flush=True)
        if len(ov_detections) == 0:
            print('no detections', flush=True)
            return {
                'masks': [],
                'bboxes': [],
                'class_names': [],
                'confidences': [],
                'debug': {
                    'sam_model': 'sam2',
                    'open_vocab_detector': backend,
                    'n_class_prompts': len(class_names),
                    'yolo_conf_threshold': float(conf_threshold),
                    'n_yolo_detections': 0,
                    'n_sam_masks': 0,
                    'yolo_class_hist': {},
                },
            }

        masks = []
        bboxes = []
        class_names_list = []
        confidences = []

        yolo_class_hist: Dict[str, int] = {}
        for detection in ov_detections:
            bbox = detection['bbox']
            class_name = detection['class_name']
            confidence = detection['confidence']
            yolo_class_hist[str(class_name)] = yolo_class_hist.get(str(class_name), 0) + 1
            print(f'starting get_mask_from_bbox for bbox: {bbox}')
            mask = self.get_mask_from_bbox(image, bbox)

            if np.sum(mask > 0) > 0:
                masks.append(mask)
                bboxes.append(bbox)
                class_names_list.append(class_name)
                confidences.append(confidence)

        return {
            'masks': masks,
            'bboxes': bboxes,
            'class_names': class_names_list,
            'confidences': confidences,
            'debug': {
                'sam_model': 'sam2',
                'open_vocab_detector': backend,
                'grounding_dino_model_id': (
                    self._grounding_dino_model_id_cached if backend == "grounding_dino" else None
                ),
                'n_class_prompts': len(class_names),
                'yolo_conf_threshold': float(conf_threshold),
                'n_yolo_detections': len(ov_detections),
                'n_sam_masks': len(masks),
                'yolo_class_hist': yolo_class_hist,
            },
        }


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

class YOLOIntegration:
    """
    YOLO Integration Module for 3D Object Detection Pipeline
    This module provides unified class for integrating YOLO models.
    Supports YOLO-World models for open-vocabulary object detection.
    """
    def __init__(self, model_path: str = "yolov8x-worldv2.pt", use_gpu: bool = True):
        """
        Initialize YOLO integration manager.

        Args:
            model_path: Path to YOLO-World model file (default: "yolov8s-world.pt")
            use_gpu: Boolean flag to enable GPU usage (if available)
        """
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("Ultralytics not available. Install with: pip install ultralytics")

        if not os.path.isabs(model_path):
            model_path = os.path.abspath(os.path.join(get_models_dir(), model_path))

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YOLO model not found at {model_path}")

        self.model_path = model_path
        self.model = None
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = "cuda:0" if self.use_gpu else "cpu"
        self._load_model()

    def _load_model(self):
        """Load the YOLO model."""
        self.model = YOLO(self.model_path)
        print(f"Loaded YOLO model from {self.model_path} on {self.device}")

    def detect_with_classes(self, image: np.ndarray, class_names: List[str],
                           conf_threshold: float = 0.25) -> List[Dict]:
        """
        Detect objects in image using YOLO-World with specified class names.

        Args:
            image: Input image as numpy array (H, W, 3) in RGB format
            class_names: List of class names to detect (e.g., ["car", "person", "bicycle"])
            conf_threshold: Confidence threshold for detections (default: 0.25)

        Returns:
            List of detection dictionaries, each with:
                - 'bbox': [x1, y1, x2, y2] bounding box coordinates
                - 'class_name': Detected class name
                - 'confidence': Detection confidence score
                - 'class_id': Class index in class_names list
        """
        print(f'starting yolo detect with class_names: {class_names}')
        if self.model is None:
            print('model is None')
            raise RuntimeError("Model not loaded")

        self.model.set_classes(class_names)
        
        results = self.model(image, conf=conf_threshold, verbose=False, device=self.device)

        detections = []
        if results and len(results) > 0:
            result = results[0]

            if result.boxes is not None:
                boxes = result.boxes
                num_detections = len(boxes)

                for i in range(num_detections):
                    bbox = boxes.xyxy[i].cpu().numpy().tolist()

                    confidence = float(boxes.conf[i].cpu().numpy())

                    class_id = int(boxes.cls[i].cpu().numpy())
                    if class_id < len(class_names):
                        class_name = class_names[class_id]
                    else:
                        class_name = "unknown"

                    detections.append({
                        'bbox': [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
                        'class_name': class_name,
                        'confidence': confidence,
                        'class_id': class_id
                    })
        print(f'yolo detections: {detections}')
        return detections


def _resolve_open_vocab_class_id(label: str, class_names: List[str]) -> Tuple[int, str]:
    """
    Map a detector label string to an index in ``class_names`` and the canonical name.
    """
    lab = str(label).strip()
    for i, n in enumerate(class_names):
        cn = str(n).strip()
        if cn == lab or cn.lower() == lab.lower():
            return i, cn
    return -1, lab


class GroundingDINOIntegration:
    """
    Open-vocabulary detection via Hugging Face ``zero-shot-object-detection`` (Grounding DINO).
    """

    def __init__(self, model_id: str = "IDEA-Research/grounding-dino-base", use_gpu: bool = True):
        if not TRANSFORMERS_GROUNDING_AVAILABLE:
            raise ImportError(
                "Grounding DINO requires transformers and Pillow. "
                "Install with: pip install transformers pillow"
            )

        self.model_id = model_id
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self._pipe = None
        self._load_pipeline()

    def _load_pipeline(self) -> None:
        device = 0 if self.use_gpu else -1
        self._pipe = _hf_zero_shot_od_pipeline(
            "zero-shot-object-detection",
            model=self.model_id,
            device=device,
        )
        dev_name = "cuda" if self.use_gpu else "cpu"
        print(f"Loaded Grounding DINO pipeline {self.model_id} on {dev_name}", flush=True)

    def detect_with_classes(self, image: np.ndarray, class_names: List[str],
                            conf_threshold: float = 0.25) -> List[Dict]:
        """
        Run Grounding DINO on ``image`` with ``class_names`` as candidate labels.

        Returns the same structure as :meth:`YOLOIntegration.detect_with_classes`.
        """
        print(f"starting groundingdino detect", flush=True)
        if self._pipe is None:
            print("pipeline is None", flush=True)
            raise RuntimeError("Grounding DINO pipeline not loaded")
        print("pipeline is not None", flush=True)

        arr = np.ascontiguousarray(image)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        pil_image = _PIL_Image.fromarray(arr)

        raw = self._pipe(
            pil_image,
            candidate_labels=list(class_names),
            threshold=conf_threshold,
        )

        detections: List[Dict] = []
        print(f"raw: {raw}", flush=True)
        if not raw:
            return detections
        print("raw is not empty", flush=True)
        for item in raw:
            score = float(item["score"])
            label = item["label"]
            box = item["box"]
            if isinstance(box, dict):
                xmin = float(box["xmin"])
                ymin = float(box["ymin"])
                xmax = float(box["xmax"])
                ymax = float(box["ymax"])
            else:
                xmin = float(box.xmin)
                ymin = float(box.ymin)
                xmax = float(box.xmax)
                ymax = float(box.ymax)

            class_id, class_name = _resolve_open_vocab_class_id(str(label), class_names)
            detections.append({
                'bbox': [int(round(xmin)), int(round(ymin)), int(round(xmax)), int(round(ymax))],
                'class_name': class_name,
                'confidence': score,
                'class_id': class_id,
            })
        print(f"groundingdino detections: {detections}", flush=True)
        return detections
