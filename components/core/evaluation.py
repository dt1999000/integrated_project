"""
Evaluation Module

Provides utilities for evaluating object detection results by matching
detected cuboids to ground truth annotations.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set, Callable
import math
import numpy as np

from shapely.geometry import Polygon

from ..dataset_loaders.kitti_dataset_loader import KITTIDatasetLoader
import cv2

KITTI_DIFFICULTY_RULES: Dict[str, Dict[str, float]] = {
    "easy": {"min_height_px": 40.0, "max_occlusion": 0, "max_truncation": 0.15},
    "moderate": {"min_height_px": 25.0, "max_occlusion": 1, "max_truncation": 0.30},
    "hard": {"min_height_px": 25.0, "max_occlusion": 2, "max_truncation": 0.50},
}


def _normalize_eval_category_key(category: object) -> str:
    """
    Canonical class label for category-aware matching and per-class bookkeeping.

    YOLO-World etc. often emit ``person`` while KITTI GT uses ``Pedestrian``.
    """
    raw = category
    if raw is None:
        label = ""
    else:
        label = str(raw).strip().lower()
    if label in {"", "unknown"}:
        return "unknown"
    if label in {"person", "pedestrian"}:
        return "pedestrian"
    return label


# =============================================================================
# 3D IoU Calculation Functions
# =============================================================================

def compute_3d_iou_axis_aligned(box1: Dict, box2: Dict) -> float:
    """
    Compute 3D Intersection over Union between two axis-aligned cuboids.

    Args:
        box1: First cuboid with min_x, max_x, min_y, max_y, min_z, max_z
        box2: Second cuboid with min_x, max_x, min_y, max_y, min_z, max_z

    Returns:
        IoU value between 0 and 1
    """
    # Get intersection box
    x_min = max(box1['min_x'], box2['min_x'])
    x_max = min(box1['max_x'], box2['max_x'])
    y_min = max(box1['min_y'], box2['min_y'])
    y_max = min(box1['max_y'], box2['max_y'])
    z_min = max(box1['min_z'], box2['min_z'])
    z_max = min(box1['max_z'], box2['max_z'])

    # Check if there is an intersection
    if x_max <= x_min or y_max <= y_min or z_max <= z_min:
        return 0.0

    # Intersection volume
    intersection = (x_max - x_min) * (y_max - y_min) * (z_max - z_min)

    # Individual volumes
    vol1 = (box1['max_x'] - box1['min_x']) * (box1['max_y'] - box1['min_y']) * (box1['max_z'] - box1['min_z'])
    vol2 = (box2['max_x'] - box2['min_x']) * (box2['max_y'] - box2['min_y']) * (box2['max_z'] - box2['min_z'])

    # Union volume
    union = vol1 + vol2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def corners_to_bev_polygon(corners: np.ndarray) -> np.ndarray:
    """
    Extract bird's eye view (BEV) polygon from 8 corner points.

    Args:
        corners: 8x3 array of corner points

    Returns:
        4x2 array of BEV polygon vertices (x, y)
    """
    # Take bottom 4 corners (assuming corners 0-3 are bottom face)
    # KITTI corner ordering: 0-3 bottom, 4-7 top
    bev_corners = corners[:4, :2]  # x, y only
    return bev_corners


def compute_bev_iou_shapely(corners1: np.ndarray, corners2: np.ndarray) -> float:
    """
    Compute BEV IoU using shapely polygon intersection.

    Args:
        corners1: 8x3 array of first box corners
        corners2: 8x3 array of second box corners

    Returns:
        BEV IoU value
    """

    try:
        # Get BEV polygons
        bev1 = corners_to_bev_polygon(corners1)
        bev2 = corners_to_bev_polygon(corners2)

        # Create shapely polygons
        poly1 = Polygon(bev1)
        poly2 = Polygon(bev2)

        if not poly1.is_valid:
            poly1 = poly1.buffer(0)
        if not poly2.is_valid:
            poly2 = poly2.buffer(0)

        # Compute intersection and union
        intersection = poly1.intersection(poly2).area
        union = poly1.union(poly2).area

        if union <= 0:
            return 0.0

        return intersection / union
    except Exception:
        return 0.0


def compute_height_overlap(corners1: np.ndarray, corners2: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute height overlap between two boxes.

    Args:
        corners1: 8x3 array of first box corners
        corners2: 8x3 array of second box corners

    Returns:
        Tuple of (height_intersection, height1, height2)
    """
    # Get z ranges from corners
    z1_min, z1_max = corners1[:, 2].min(), corners1[:, 2].max()
    z2_min, z2_max = corners2[:, 2].min(), corners2[:, 2].max()

    h1 = z1_max - z1_min
    h2 = z2_max - z2_min

    # Height intersection
    z_overlap_min = max(z1_min, z2_min)
    z_overlap_max = min(z1_max, z2_max)
    height_intersection = max(0, z_overlap_max - z_overlap_min)

    return height_intersection, h1, h2


def box_to_corners(box: Dict) -> np.ndarray:
    """
    Convert axis-aligned box dict to 8 corner points.

    Args:
        box: Dict with min_x, max_x, min_y, max_y, min_z, max_z

    Returns:
        8x3 array of corner points
    """
    x0, x1 = box['min_x'], box['max_x']
    y0, y1 = box['min_y'], box['max_y']
    z0, z1 = box['min_z'], box['max_z']

    # Corner ordering matching KITTI: bottom 4, then top 4
    corners = np.array([
        [x0, y0, z0],
        [x1, y0, z0],
        [x1, y1, z0],
        [x0, y1, z0],
        [x0, y0, z1],
        [x1, y0, z1],
        [x1, y1, z1],
        [x0, y1, z1],
    ])
    return corners


def compute_3d_iou_oriented(box1: Dict, box2: Dict) -> float:
    """
    Compute oriented 3D IoU using BEV polygon intersection.

    Works with rotated boxes using their corner points.
    Falls back to axis-aligned IoU if shapely is not available.

    Args:
        box1: Cuboid dict with 'corners' (8x3) or min/max bounds
        box2: Cuboid dict with 'corners' (8x3) or min/max bounds

    Returns:
        3D IoU value between 0 and 1
    """
    # Get corners for both boxes
    if 'corners' in box1 and box1['corners'] is not None:
        corners1 = np.asarray(box1['corners'])
    else:
        corners1 = box_to_corners(box1)

    if 'corners' in box2 and box2['corners'] is not None:
        corners2 = np.asarray(box2['corners'])
    else:
        corners2 = box_to_corners(box2)

    # Compute BEV IoU
    bev_iou = compute_bev_iou_shapely(corners1, corners2)
    if bev_iou <= 0:
        return 0.0

    # Compute height overlap
    height_inter, h1, h2 = compute_height_overlap(corners1, corners2)
    if height_inter <= 0:
        return 0.0

    # Get BEV areas
    bev1 = corners_to_bev_polygon(corners1)
    bev2 = corners_to_bev_polygon(corners2)

    try:
        poly1 = Polygon(bev1)
        poly2 = Polygon(bev2)

        if not poly1.is_valid:
            poly1 = poly1.buffer(0)
        if not poly2.is_valid:
            poly2 = poly2.buffer(0)

        bev_inter = poly1.intersection(poly2).area
        bev_area1 = poly1.area
        bev_area2 = poly2.area
    except Exception:
        return compute_3d_iou_axis_aligned(box1, box2)

    # 3D intersection volume
    vol_inter = bev_inter * height_inter

    # 3D volumes
    vol1 = bev_area1 * h1
    vol2 = bev_area2 * h2

    # 3D union
    vol_union = vol1 + vol2 - vol_inter

    if vol_union <= 0:
        return 0.0

    return vol_inter / vol_union


def compute_3d_iou(box1: Dict, box2: Dict, use_oriented: bool = True) -> float:
    """
    Compute 3D IoU between two cuboids.

    Main entry point for 3D IoU calculation. Uses oriented IoU if corners
    are available and shapely is installed, otherwise falls back to
    axis-aligned calculation.

    Args:
        box1: First cuboid dict
        box2: Second cuboid dict
        use_oriented: If True, use oriented IoU when possible

    Returns:
        3D IoU value between 0 and 1
    """
    has_corners = (
        ('corners' in box1 and box1['corners'] is not None) or
        ('corners' in box2 and box2['corners'] is not None)
    )

    if use_oriented:
        return compute_3d_iou_oriented(box1, box2)
    else:
        return compute_3d_iou_axis_aligned(box1, box2)


@dataclass
class MatchResult:
    """
    Result of matching detected objects to ground truth.

    Provides computed properties for common evaluation metrics.
    """
    matches: List[Tuple[int, int, float]]  # (gt_idx, det_idx, distance)
    unmatched_gt: List[int]                # GT indices with no match
    unmatched_det: List[int]               # Detection indices with no match

    @property
    def n_matches(self) -> int:
        """Number of matched pairs (True Positives)."""
        return len(self.matches)

    @property
    def n_false_positives(self) -> int:
        """Number of false positives (detections without GT match)."""
        return len(self.unmatched_det)

    @property
    def n_false_negatives(self) -> int:
        """Number of false negatives (GT without detection match)."""
        return len(self.unmatched_gt)

    @property
    def precision(self) -> float:
        """Precision = TP / (TP + FP)"""
        total = self.n_matches + self.n_false_positives
        if total == 0:
            return 0.0
        return self.n_matches / total

    @property
    def recall(self) -> float:
        """Recall = TP / (TP + FN)"""
        total = self.n_matches + self.n_false_negatives
        if total == 0:
            return 0.0
        return self.n_matches / total

    @property
    def f1_score(self) -> float:
        """F1 Score = 2 * (Precision * Recall) / (Precision + Recall)"""
        p, r = self.precision, self.recall
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)

    def get_metrics_dict(self) -> Dict:
        """Get all metrics as a dictionary."""
        return {
            'true_positives': self.n_matches,
            'false_positives': self.n_false_positives,
            'false_negatives': self.n_false_negatives,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score
        }


class CuboidMatcher:
    """
    Matches detected cuboids to ground truth based on spatial proximity.

    Supports category-aware matching where only same-category objects
    can be matched together.
    """

    def __init__(self, max_distance: float = 5.0, match_by_category: bool = True):
        """
        Initialize CuboidMatcher.

        Args:
            max_distance: Maximum center distance for a valid match (meters)
            match_by_category: If True, only match objects of the same category
        """
        self.max_distance = max_distance
        self.match_by_category = match_by_category

    @staticmethod
    def get_cuboid_center(cuboid: Dict) -> np.ndarray:
        """
        Extract center point from cuboid dictionary.

        Args:
            cuboid: Cuboid dict with either 'corners' array or min/max bounds

        Returns:
            Center point as (3,) numpy array
        """
        if 'corners' in cuboid and cuboid['corners'] is not None:
            return cuboid['corners'].mean(axis=0)

        return np.array([
            (cuboid['min_x'] + cuboid['max_x']) / 2,
            (cuboid['min_y'] + cuboid['max_y']) / 2,
            (cuboid['min_z'] + cuboid['max_z']) / 2
        ])

    def match(
        self,
        gt_cuboids: List[Dict],
        detected_cuboids: List[Dict]
    ) -> MatchResult:
        """
        Match detected cuboids to ground truth based on center distance.

        Uses greedy matching: each detection is matched to its nearest GT
        that hasn't been matched yet (if within max_distance and same category
        when match_by_category is True).

        Args:
            gt_cuboids: List of ground truth cuboid dicts
            detected_cuboids: List of detected cuboid dicts

        Returns:
            MatchResult with matches, unmatched_gt, and unmatched_det
        """
        matches = []
        matched_gt_indices: Set[int] = set()
        matched_det_indices: Set[int] = set()

        # Match each detected cuboid to nearest GT
        for det_idx, det in enumerate(detected_cuboids):
            det_center = self.get_cuboid_center(det)
            det_category = det.get("category", det.get("class", "Unknown"))

            best_match_idx = None
            best_dist = self.max_distance

            for gt_idx, gt in enumerate(gt_cuboids):
                # Skip already matched GT
                if gt_idx in matched_gt_indices:
                    continue

                gt_category = gt.get("category", gt.get("class", "Unknown"))

                # Category check if enabled
                if self.match_by_category and _normalize_eval_category_key(
                    det_category
                ) != _normalize_eval_category_key(gt_category):
                    continue

                gt_center = self.get_cuboid_center(gt)
                dist = np.linalg.norm(det_center - gt_center)

                if dist < best_dist:
                    best_dist = dist
                    best_match_idx = gt_idx

            if best_match_idx is not None:
                matches.append((best_match_idx, det_idx, best_dist))
                matched_gt_indices.add(best_match_idx)
                matched_det_indices.add(det_idx)

        # Collect unmatched indices
        unmatched_gt = [i for i in range(len(gt_cuboids)) if i not in matched_gt_indices]
        unmatched_det = [i for i in range(len(detected_cuboids)) if i not in matched_det_indices]

        return MatchResult(
            matches=matches,
            unmatched_gt=unmatched_gt,
            unmatched_det=unmatched_det
        )

    def compute_per_category_metrics(
        self,
        gt_cuboids: List[Dict],
        detected_cuboids: List[Dict],
        match_result: MatchResult
    ) -> Dict[str, Dict]:
        """
        Compute evaluation metrics broken down by category.

        Args:
            gt_cuboids: List of ground truth cuboid dicts
            detected_cuboids: List of detected cuboid dicts
            match_result: MatchResult from match() method

        Returns:
            Dict mapping category name to metrics dict with TP, FP, FN, Precision, Recall
        """
        category_stats: Dict[str, Dict] = {}

        # Count true positives per category
        for gt_idx, det_idx, dist in match_result.matches:
            cat = _normalize_eval_category_key(gt_cuboids[gt_idx].get("category", gt_cuboids[gt_idx].get("class", "Unknown")))
            if cat not in category_stats:
                category_stats[cat] = {'TP': 0, 'FP': 0, 'FN': 0}
            category_stats[cat]['TP'] += 1

        # Count false negatives (unmatched GT)
        for gt_idx in match_result.unmatched_gt:
            cat = _normalize_eval_category_key(gt_cuboids[gt_idx].get("category", gt_cuboids[gt_idx].get("class", "Unknown")))
            if cat not in category_stats:
                category_stats[cat] = {'TP': 0, 'FP': 0, 'FN': 0}
            category_stats[cat]['FN'] += 1

        # Count false positives (unmatched detections)
        for det_idx in match_result.unmatched_det:
            cat = _normalize_eval_category_key(detected_cuboids[det_idx].get("category", detected_cuboids[det_idx].get("class", "Unknown")))
            if cat not in category_stats:
                category_stats[cat] = {'TP': 0, 'FP': 0, 'FN': 0}
            category_stats[cat]['FP'] += 1

        # Compute per-category precision and recall
        for cat, stats in category_stats.items():
            tp, fp, fn = stats['TP'], stats['FP'], stats['FN']
            stats['Precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            stats['Recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        return category_stats


# =============================================================================
# Batch Evaluation: AP_50 / AP_25
# =============================================================================

def _normalize_gt_cuboids(cuboids: List[Dict]) -> List[Dict]:
    """
    Normalize a list of GT cuboid dicts to the min/max corner format
    required by the IoU functions.

    Handles both the already-normalized min/max format and the
    translation+size format used by nuScenes-style annotations.
    """
    normalized = []
    for gt in cuboids:
        if all(
            k in gt and gt[k] is not None
            for k in ("min_x", "min_y", "min_z", "max_x", "max_y", "max_z")
        ):
            norm_gt = dict(gt)
            norm_gt["category"] = norm_gt.get("category", norm_gt.get("class", "Unknown"))
            normalized.append(norm_gt)
            continue

        translation = gt.get("translation")
        size = gt.get("size")
        if translation is None or size is None or len(translation) != 3 or len(size) != 3:
            continue

        center = np.asarray(translation, dtype=np.float64)
        half = np.asarray(size, dtype=np.float64) / 2.0
        norm_gt = dict(gt)
        norm_gt["min_x"] = float(center[0] - half[0])
        norm_gt["min_y"] = float(center[1] - half[1])
        norm_gt["min_z"] = float(center[2] - half[2])
        norm_gt["max_x"] = float(center[0] + half[0])
        norm_gt["max_y"] = float(center[1] + half[1])
        norm_gt["max_z"] = float(center[2] + half[2])
        norm_gt["category"] = norm_gt.get("category", norm_gt.get("class", "Unknown"))
        normalized.append(norm_gt)
    return normalized


def greedy_iou_match(
    gt_cuboids: List[Dict],
    detected_cuboids: List[Dict],
    iou_threshold: float,
    match_by_category: bool,
) -> Tuple[List[Tuple[int, int]], Set[int], Set[int]]:
    """
    Greedy maximum-IoU matching: pairs sorted by IoU descending, each GT/det used once.

    Returns:
        tp_pairs: list of (gt_idx, det_idx) for matched pairs (IoU >= threshold)
        unmatched_gt: GT indices with no matched detection
        unmatched_det: detection indices with no matched GT
    """
    n_gt = len(gt_cuboids)
    n_det = len(detected_cuboids)
    if n_gt == 0 or n_det == 0:
        return (
            [],
            set(range(n_gt)),
            set(range(n_det)),
        )

    iou_matrix = np.zeros((n_gt, n_det), dtype=np.float64)
    for gi, gt in enumerate(gt_cuboids):
        gt_cat = gt.get("category", gt.get("class", "Unknown"))
        for di, det in enumerate(detected_cuboids):
            if match_by_category and _normalize_eval_category_key(gt_cat) != _normalize_eval_category_key(
                det.get("category", det.get("class", "Unknown"))
            ):
                continue
            iou_matrix[gi, di] = compute_3d_iou(gt, det)

    pairs = [
        (iou_matrix[gi, di], gi, di)
        for gi in range(n_gt)
        for di in range(n_det)
        if iou_matrix[gi, di] >= iou_threshold
    ]
    pairs.sort(reverse=True)

    matched_gt: Set[int] = set()
    matched_det: Set[int] = set()
    tp_pairs: List[Tuple[int, int]] = []
    for _, gi, di in pairs:
        if gi not in matched_gt and di not in matched_det:
            matched_gt.add(gi)
            matched_det.add(di)
            tp_pairs.append((gi, di))

    unmatched_gt = set(range(n_gt)) - matched_gt
    unmatched_det = set(range(n_det)) - matched_det
    return tp_pairs, unmatched_gt, unmatched_det


def compute_frame_metrics_at_iou(
    gt_cuboids: List[Dict],
    detected_cuboids: List[Dict],
    iou_threshold: float = 0.5,
    match_by_category: bool = False,
) -> Dict:
    """
    Compute per-frame TP, FP, FN counts at a given 3D IoU threshold.

    Uses greedy matching: all (GT, detection) pairs whose IoU meets the
    threshold are sorted by IoU descending and greedily assigned so that
    each GT and each detection is used at most once.

    Args:
        gt_cuboids: Ground truth cuboid dicts (already normalized to min/max).
        detected_cuboids: Detected cuboid dicts.
        iou_threshold: IoU value a pair must meet to count as a TP.
        match_by_category: When True, a detection may only match a GT of the
            same category.

    Returns:
        Dict with keys TP, FP, FN, precision, recall, f1, n_gt, n_det,
        and per_class (dict mapping category -> {TP, FP, FN}).
    """
    n_gt = len(gt_cuboids)
    n_det = len(detected_cuboids)

    per_class_tp: Dict[str, int] = {}
    per_class_fp: Dict[str, int] = {}
    per_class_fn: Dict[str, int] = {}

    if n_gt == 0 and n_det == 0:
        return {
            "TP": 0, "FP": 0, "FN": 0,
            "precision": 1.0, "recall": 1.0, "f1": 1.0,
            "n_gt": 0, "n_det": 0, "per_class": {},
        }

    if n_gt == 0:
        for det in detected_cuboids:
            cat = _normalize_eval_category_key(det.get("category", det.get("class", "Unknown")))
            per_class_fp[cat] = per_class_fp.get(cat, 0) + 1
        per_class = {c: {"TP": 0, "FP": per_class_fp[c], "FN": 0} for c in per_class_fp}
        return {
            "TP": 0, "FP": n_det, "FN": 0,
            "precision": 0.0, "recall": 1.0, "f1": 0.0,
            "n_gt": 0, "n_det": n_det, "per_class": per_class,
        }

    if n_det == 0:
        for gt in gt_cuboids:
            cat = _normalize_eval_category_key(gt.get("category", gt.get("class", "Unknown")))
            per_class_fn[cat] = per_class_fn.get(cat, 0) + 1
        per_class = {c: {"TP": 0, "FP": 0, "FN": per_class_fn[c]} for c in per_class_fn}
        return {
            "TP": 0, "FP": 0, "FN": n_gt,
            "precision": 1.0, "recall": 0.0, "f1": 0.0,
            "n_gt": n_gt, "n_det": 0, "per_class": per_class,
        }

    tp_pairs, unmatched_gt, unmatched_det = greedy_iou_match(
        gt_cuboids, detected_cuboids, iou_threshold, match_by_category
    )

    for gi, di in tp_pairs:
        cat = _normalize_eval_category_key(gt_cuboids[gi].get("category", gt_cuboids[gi].get("class", "Unknown")))
        per_class_tp[cat] = per_class_tp.get(cat, 0) + 1

    for di in unmatched_det:
        cat = _normalize_eval_category_key(detected_cuboids[di].get("category", detected_cuboids[di].get("class", "Unknown")))
        per_class_fp[cat] = per_class_fp.get(cat, 0) + 1

    for gi in unmatched_gt:
        cat = _normalize_eval_category_key(gt_cuboids[gi].get("category", gt_cuboids[gi].get("class", "Unknown")))
        per_class_fn[cat] = per_class_fn.get(cat, 0) + 1

    tp = len(tp_pairs)
    fp = n_det - tp
    fn = n_gt - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    all_cats = set(list(per_class_tp) + list(per_class_fp) + list(per_class_fn))
    per_class = {
        cat: {
            "TP": per_class_tp.get(cat, 0),
            "FP": per_class_fp.get(cat, 0),
            "FN": per_class_fn.get(cat, 0),
        }
        for cat in all_cats
    }

    return {
        "TP": tp, "FP": fp, "FN": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "n_gt": n_gt, "n_det": n_det,
        "per_class": per_class,
    }


def _detection_confidence_for_pr(det: Dict) -> float:
    """
    Sort key for PR-curve AP: higher means process this detection earlier
    (higher "confidence" first).
    """
    c = det.get("confidence")
    if c is None:
        return 0.0
    cf = float(c)
    if not math.isfinite(cf):
        return 0.0
    return float(np.clip(cf, 0.0, 1.0))


def average_precision_under_pr_curve(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """
    Detection AP computed as step-wise area under the monotonic precision
    envelope (not trapezoidal interpolation).
    """
    if recalls.size == 0:
        return 0.0
    rec = np.concatenate(([0.0], recalls.astype(np.float64), [1.0]))
    prec = np.concatenate(([0.0], precisions.astype(np.float64), [0.0]))
    for i in range(len(prec) - 2, -1, -1):
        prec[i] = max(prec[i], prec[i + 1])
    idx = np.where(rec[1:] != rec[:-1])[0] + 1
    return float(np.sum((rec[idx] - rec[idx - 1]) * prec[idx]))


def ap_r11(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """
    AP on the classic 11-point interpolation grid (0.0..1.0 recall).
    """
    recalls = np.asarray(recalls, dtype=np.float64)
    precisions = np.asarray(precisions, dtype=np.float64)
    recall_levels = np.linspace(0.0, 1.0, 11)
    ap = 0.0
    for r in recall_levels:
        valid = precisions[recalls >= r]
        p_interp = valid.max() if valid.size > 0 else 0.0
        ap += float(p_interp) / 11.0
    return float(ap)


def _kitti_box_height_px(cuboid: Dict) -> float:
    if cuboid.get("bbox_height_px") is not None:
        return max(0.0, float(cuboid.get("bbox_height_px")))
    bbox_2d = cuboid.get("bbox_2d")
    if not isinstance(bbox_2d, dict):
        return 0.0
    top = bbox_2d.get("top")
    bottom = bbox_2d.get("bottom")
    if top is None or bottom is None:
        return 0.0
    return max(0.0, float(bottom) - float(top))


def _kitti_occlusion_level(cuboid: Dict) -> int:
    occ_raw = cuboid.get("occlusion", cuboid.get("occluded", 3))
    if isinstance(occ_raw, bool):
        return 1 if occ_raw else 0
    if isinstance(occ_raw, (int, np.integer)):
        return int(occ_raw)
    if isinstance(occ_raw, (float, np.floating)):
        return int(occ_raw)
    if isinstance(occ_raw, str):
        value = occ_raw.strip().lower()
        occ_map = {
            "0": 0, "fully_visible": 0, "fully visible": 0,
            "1": 1, "partly_occluded": 1, "partly occluded": 1, "partly": 1,
            "2": 2, "largely_occluded": 2, "difficult_to_see": 2, "difficult to see": 2,
            "3": 3, "unknown": 3,
        }
        if value in occ_map:
            return occ_map[value]
    return 3


def _kitti_truncation(cuboid: Dict) -> float:
    trunc = cuboid.get("truncation", cuboid.get("truncated", 1.0))
    if trunc is None:
        return 1.0
    try:
        return float(trunc)
    except (TypeError, ValueError):
        return 1.0


def matches_kitti_difficulty(cuboid: Dict, difficulty: str) -> bool:
    """
    Check if a GT cuboid satisfies KITTI Easy/Moderate/Hard rules.
    """
    rules = KITTI_DIFFICULTY_RULES.get(str(difficulty).lower())
    if rules is None:
        return True
    h = _kitti_box_height_px(cuboid)
    occ = _kitti_occlusion_level(cuboid)
    trunc = _kitti_truncation(cuboid)
    return (
        h >= rules["min_height_px"]
        and occ <= int(rules["max_occlusion"])
        and trunc <= rules["max_truncation"]
    )


def filter_kitti_samples_by_difficulty(
    batch_results: List[Dict],
    difficulty: str,
) -> List[Dict]:
    """
    Return a copy of ``batch_results`` where each sample keeps only GT cuboids
    that satisfy the selected KITTI difficulty.
    """
    filtered: List[Dict] = []
    for sample in batch_results:
        sample_copy = dict(sample)
        gt_raw = sample_copy.get("ground_truth_cuboids")
        if gt_raw is not None:
            sample_copy["ground_truth_cuboids"] = [
                gt for gt in (gt_raw or []) if matches_kitti_difficulty(gt, difficulty)
            ]
        filtered.append(sample_copy)
    return filtered


def _closest_gt_index_for_detection(
    gt_cuboids: List[Dict],
    detection: Dict,
    *,
    match_by_category: bool,
) -> int:
    """
    Return index of closest GT by 3D center distance, or -1 when no candidate exists.
    """
    det_center = CuboidMatcher.get_cuboid_center(detection)
    det_category = _normalize_eval_category_key(
        detection.get("category", detection.get("class", "Unknown"))
    )
    best_idx = -1
    best_dist = float("inf")
    for gi, gt in enumerate(gt_cuboids):
        gt_category = _normalize_eval_category_key(gt.get("category", gt.get("class", "Unknown")))
        if match_by_category and gt_category != det_category:
            continue
        gt_center = CuboidMatcher.get_cuboid_center(gt)
        dist = float(np.linalg.norm(det_center - gt_center))
        if dist < best_dist:
            best_dist = dist
            best_idx = gi
    return best_idx


def compute_kitti_difficulty_ap(
    batch_results: List[Dict],
    iou_threshold: float = 0.5,
    match_by_category: bool = True,
) -> Dict:
    """
    KITTI AP split by difficulty using nearest-GT pairing first.

    Flow:
    1) For each detection, find its closest GT (optionally class-constrained).
    2) Assign detection to that GT's difficulty bucket (Easy/Moderate/Hard).
    3) Per (difficulty, class), sort detections by confidence and compute PR:
       - TP: IoU>=threshold with paired GT, and that GT not claimed yet.
       - FP: otherwise.
    4) Report AP under PR curve and AP_R11.
    """
    difficulties = ("easy", "moderate", "hard")
    samples: Dict[int, Dict[str, List[Dict]]] = {}
    n_gt: Dict[str, Dict[str, int]] = {d: {} for d in difficulties}
    det_entries: Dict[str, Dict[str, List[Tuple[int, Dict, int, float]]]] = {
        d: {} for d in difficulties
    }

    for fi, sample in enumerate(batch_results):
        gt_list = _normalize_gt_cuboids(sample.get("ground_truth_cuboids", []) or [])
        det_list = list(sample.get("detected_cuboids", []) or [])
        samples[fi] = {"gt": gt_list, "det": det_list}

        gt_difficulties: List[List[str]] = []
        for gt in gt_list:
            gt_cat = _normalize_eval_category_key(gt.get("category", gt.get("class", "Unknown")))
            gt_diff_list: List[str] = []
            for d in difficulties:
                if matches_kitti_difficulty(gt, d):
                    n_gt[d][gt_cat] = n_gt[d].get(gt_cat, 0) + 1
                    gt_diff_list.append(d)
            gt_difficulties.append(gt_diff_list)

        for det in det_list:
            gi = _closest_gt_index_for_detection(
                gt_list, det, match_by_category=match_by_category
            )
            if gi < 0:
                continue
            d_list = gt_difficulties[gi]
            if not d_list:
                continue
            gt_cat = _normalize_eval_category_key(
                gt_list[gi].get("category", gt_list[gi].get("class", "Unknown"))
            )
            score = _detection_confidence_for_pr(det)
            for d in d_list:
                det_entries[d].setdefault(gt_cat, []).append(
                    (fi, det, gi, score)
                )

    per_difficulty_per_class: Dict[str, Dict[str, Dict]] = {d: {} for d in difficulties}
    for d in difficulties:
        for cat, entries in det_entries[d].items():
            entries.sort(key=lambda x: (-x[3], x[0]))
            npos = int(n_gt[d].get(cat, 0))
            matched_gt: Set[Tuple[int, int]] = set()
            precisions: List[float] = []
            recalls: List[float] = []
            cum_tp = 0
            cum_fp = 0

            for fi, det, gi, _score in entries:
                gt = samples[fi]["gt"][gi]
                gt_key = (fi, gi)
                valid_pair = True
                gt_cat_norm = _normalize_eval_category_key(gt.get("category", gt.get("class", "Unknown")))
                det_cat_norm = _normalize_eval_category_key(det.get("category", det.get("class", "Unknown")))
                if match_by_category and gt_cat_norm != det_cat_norm:
                    valid_pair = False
                iou = compute_3d_iou(gt, det) if valid_pair else 0.0
                if gt_key not in matched_gt and iou >= iou_threshold:
                    matched_gt.add(gt_key)
                    cum_tp += 1
                else:
                    cum_fp += 1
                denom = cum_tp + cum_fp
                precisions.append(cum_tp / denom if denom > 0 else 0.0)
                recalls.append(cum_tp / npos if npos > 0 else 0.0)

            rec_np = np.asarray(recalls, dtype=np.float64)
            prec_np = np.asarray(precisions, dtype=np.float64)
            per_difficulty_per_class[d][cat] = {
                "n_gt": npos,
                "ap_pr": average_precision_under_pr_curve(rec_np, prec_np),
                "ap_r11": ap_r11(rec_np, prec_np),
                "precisions": [float(v) for v in precisions],
                "recalls": [float(v) for v in recalls],
            }

        for cat, npos in n_gt[d].items():
            if cat not in per_difficulty_per_class[d]:
                per_difficulty_per_class[d][cat] = {
                    "n_gt": int(npos),
                    "ap_pr": 0.0,
                    "ap_r11": 0.0,
                    "precisions": [],
                    "recalls": [],
                }

    return {
        "iou_threshold": float(iou_threshold),
        "per_difficulty_per_class": per_difficulty_per_class,
    }


def _ap_pr_single_bucket(
    samples: List[Tuple[int, List[Dict], List[Dict]]],
    iou_threshold: float,
    *,
    category_filter: Optional[str],
) -> Tuple[float, float, List[float], List[float]]:
    """
    One AP from a PR curve. If ``category_filter`` is None, pool all classes:
    any GT may match any detection (IoU only). Otherwise only that category's
    GT and detections participate; matching is restricted to same category.
    """
    if category_filter is None:
        n_gt_total = sum(len(gt) for _, gt, _ in samples)
        det_entries: List[Tuple[int, Dict, float]] = []
        for fi, _gt, det_list in samples:
            for det in det_list:
                det_entries.append((fi, det, _detection_confidence_for_pr(det)))
    else:
        c = category_filter
        n_gt_total = 0
        det_entries = []
        cn = _normalize_eval_category_key(c)
        for fi, gt_list, det_list in samples:
            for gi, gt in enumerate(gt_list):
                if _normalize_eval_category_key(gt.get("category", gt.get("class", "Unknown"))) == cn:
                    n_gt_total += 1
            for det in det_list:
                if _normalize_eval_category_key(det.get("category", det.get("class", "Unknown"))) == cn:
                    det_entries.append((fi, det, _detection_confidence_for_pr(det)))

    det_entries.sort(key=lambda x: (-x[2], x[0]))

    matched_gt: Set[Tuple[int, int]] = set()
    precisions: List[float] = []
    recalls: List[float] = []
    cum_tp = 0
    cum_fp = 0
    category_norm = (
        None if category_filter is None else _normalize_eval_category_key(category_filter)
    )

    for fi, det, _sc in det_entries:
        gt_list = samples[fi][1]
        best_gi = -1
        best_iou = 0.0
        for gi, gt in enumerate(gt_list):
            if (fi, gi) in matched_gt:
                continue
            if category_norm is not None:
                if _normalize_eval_category_key(gt.get("category", gt.get("class", "Unknown"))) != category_norm:
                    continue
                if _normalize_eval_category_key(det.get("category", det.get("class", "Unknown"))) != category_norm:
                    continue
            iou = compute_3d_iou(gt, det)
            if iou > best_iou:
                best_iou = iou
                best_gi = gi

        if best_iou >= iou_threshold and best_gi >= 0:
            matched_gt.add((fi, best_gi))
            cum_tp += 1
        else:
            cum_fp += 1

        denom = cum_tp + cum_fp
        precisions.append(cum_tp / denom if denom > 0 else 0.0)
        recalls.append(cum_tp / n_gt_total if n_gt_total > 0 else 0.0)

    ap = average_precision_under_pr_curve(
        np.asarray(recalls, dtype=np.float64),
        np.asarray(precisions, dtype=np.float64),
    )
    ap11 = ap_r11(
        np.asarray(recalls, dtype=np.float64),
        np.asarray(precisions, dtype=np.float64),
    )
    return ap, ap11, precisions, recalls


def compute_pr_map_at_iou(
    batch_results: List[Dict],
    iou_threshold: float,
    match_by_category: bool,
) -> Dict:
    """
    Mean AP (mAP) at a fixed 3D IoU threshold as the mean of per-class AP,
    where each AP is the area under the precision–recall curve.

    For each class, detections of that class are sorted by
    ``_detection_confidence_for_pr`` (global across frames); each GT box is
    matched at most once per class.

    Args:
        batch_results: Same as ``compute_batch_ap`` (evaluable samples with GT).
        iou_threshold: IoU threshold for a TP (e.g. 0.5 or 0.25).
        match_by_category: If True, mAP is the unweighted mean of per-class AP
            over classes with at least one GT. If False, one pooled AP treats
            all categories together (``map_pr`` equals ``micro_ap_pr``).

    Returns:
        Dict with ``map_pr``, ``micro_ap_pr`` (pooled PR-AUC), ``per_class_pr_ap``,
        ``per_class_pr_curves`` (raw per-class PR points), ``n_gt_per_class``,
        optional micro PR point lists, and ``iou_threshold``.
    """
    samples: List[Tuple[int, List[Dict], List[Dict]]] = []
    for i, sample in enumerate(batch_results):
        gt = _normalize_gt_cuboids(sample.get("ground_truth_cuboids", []))
        det = list(sample.get("detected_cuboids", []))
        samples.append((i, gt, det))

    n_gt_per_class: Dict[str, int] = {}
    for _fi, gt_list, _ in samples:
        for gt in gt_list:
            lab = _normalize_eval_category_key(gt.get("category", gt.get("class", "Unknown")))
            n_gt_per_class[lab] = n_gt_per_class.get(lab, 0) + 1

    micro_ap, micro_ap_r11, micro_prec, micro_rec = _ap_pr_single_bucket(
        samples, iou_threshold, category_filter=None
    )

    per_class_ap: Dict[str, float] = {}
    per_class_ap_r11: Dict[str, float] = {}
    per_class_pr_curves: Dict[str, Dict[str, List[float]]] = {}
    if match_by_category:
        for c in sorted(n_gt_per_class.keys()):
            nk = n_gt_per_class[c]
            if nk == 0:
                continue
            ap_c, ap11_c, prec_c, rec_c = _ap_pr_single_bucket(
                samples, iou_threshold, category_filter=c
            )
            per_class_ap[c] = ap_c
            per_class_ap_r11[c] = ap11_c
            per_class_pr_curves[c] = {
                "precisions": [float(v) for v in prec_c],
                "recalls": [float(v) for v in rec_c],
                "n_gt": int(nk),
            }
        classes_with_gt = [c for c, nk in n_gt_per_class.items() if nk > 0]
        map_pr = (
            float(np.mean([per_class_ap[c] for c in classes_with_gt]))
            if classes_with_gt
            else 0.0
        )
        map_pr_r11 = (
            float(np.mean([per_class_ap_r11[c] for c in classes_with_gt]))
            if classes_with_gt
            else 0.0
        )
    else:
        map_pr = micro_ap
        map_pr_r11 = micro_ap_r11

    return {
        "map_pr": map_pr,
        "map_pr_r11": map_pr_r11,
        "micro_ap_pr": micro_ap,
        "micro_ap_r11": micro_ap_r11,
        "per_class_pr_ap": per_class_ap,
        "per_class_pr_ap_r11": per_class_ap_r11,
        "per_class_pr_curves": per_class_pr_curves,
        "n_gt_per_class": dict(n_gt_per_class),
        "pr_recalls_micro": micro_rec,
        "pr_precisions_micro": micro_prec,
        "iou_threshold": iou_threshold,
    }


def compute_batch_ap(
    batch_results: List[Dict],
    iou_threshold: float = 0.5,
    match_by_category: bool = False,
) -> Dict:
    """
    Batch-level detection metrics at a fixed 3D IoU threshold.

    Returns **micro** precision/recall/F1 over all TPs/FPs/FNs pooled across
    frames, **macro_f1**, and **PR-curve mAP** (``map_pr``): mean of per-class
    AP as area under the precision–recall curve when ``match_by_category`` is
    True; otherwise a single pooled AP (see ``compute_pr_map_at_iou``).
    Detection ordering uses the exported ``confidence`` score.

    Args:
        batch_results: List of sample dicts from
            ``batch_export_results['samples']``.  Each dict must contain
            ``detected_cuboids`` and optionally ``ground_truth_cuboids``.
        iou_threshold: IoU threshold used to decide TP vs FP/FN.
        match_by_category: Whether to restrict GT↔detection matching to
            pairs with the same category label.

    Returns:
        Dict with macro_f1, precision, recall, f1 (micro), total_tp/fp/fn,
        per_frame_metrics (list), per_class (dict with precision, recall, f1),
        iou_threshold, and PR fields from ``compute_pr_map_at_iou``.
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0
    per_frame_metrics: List[Dict] = []

    class_tp: Dict[str, int] = {}
    class_fp: Dict[str, int] = {}
    class_fn: Dict[str, int] = {}

    for i, sample in enumerate(batch_results):
        gt = _normalize_gt_cuboids(sample.get("ground_truth_cuboids", []))
        det = sample.get("detected_cuboids", [])

        frame_m = compute_frame_metrics_at_iou(gt, det, iou_threshold, match_by_category)
        frame_m["frame_index"] = i
        frame_m["sample_index"] = sample.get("metadata", {}).get("sample_index", str(i))
        per_frame_metrics.append(frame_m)

        total_tp += frame_m["TP"]
        total_fp += frame_m["FP"]
        total_fn += frame_m["FN"]

        for cat, stats in frame_m["per_class"].items():
            class_tp[cat] = class_tp.get(cat, 0) + stats["TP"]
            class_fp[cat] = class_fp.get(cat, 0) + stats["FP"]
            class_fn[cat] = class_fn.get(cat, 0) + stats["FN"]

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    all_cats = set(list(class_tp) + list(class_fp) + list(class_fn))
    per_class: Dict[str, Dict] = {}
    f1_per_class: List[float] = []
    for cat in all_cats:
        tp_c = class_tp.get(cat, 0)
        fp_c = class_fp.get(cat, 0)
        fn_c = class_fn.get(cat, 0)
        p_c = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0.0
        r_c = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0.0
        f1_c = 2 * p_c * r_c / (p_c + r_c) if (p_c + r_c) > 0 else 0.0
        per_class[cat] = {
            "TP": tp_c, "FP": fp_c, "FN": fn_c,
            "precision": p_c, "recall": r_c, "f1": f1_c,
        }
        f1_per_class.append(f1_c)

    macro_f1 = float(np.mean(f1_per_class)) if f1_per_class else f1

    pr_stats = compute_pr_map_at_iou(batch_results, iou_threshold, match_by_category)

    return {
        "macro_f1": macro_f1,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "per_frame_metrics": per_frame_metrics,
        "per_class": per_class,
        "iou_threshold": iou_threshold,
        "map_pr": pr_stats["map_pr"],
        "map_pr_r11": pr_stats["map_pr_r11"],
        "micro_ap_pr": pr_stats["micro_ap_pr"],
        "micro_ap_r11": pr_stats["micro_ap_r11"],
        "per_class_pr_ap": pr_stats["per_class_pr_ap"],
        "per_class_pr_ap_r11": pr_stats["per_class_pr_ap_r11"],
        "per_class_pr_curves": pr_stats["per_class_pr_curves"],
        "n_gt_per_class_pr": pr_stats["n_gt_per_class"],
        "pr_recalls_micro": pr_stats["pr_recalls_micro"],
        "pr_precisions_micro": pr_stats["pr_precisions_micro"],
    }


def compute_omni3d_class_agnostic_map(
    batch_results: List[Dict],
    iou_thresholds: Optional[List[float]] = None,
    progress_callback: Optional[Callable[[int, int, float], None]] = None,
) -> Dict:
    """
    omni3d-style class-agnostic 3D mAP.

    All GT and detections are evaluated as a single pooled "object" class
    (i.e., category labels are ignored), AP is computed at each IoU threshold,
    and the final score is the mean AP over thresholds.

    Default threshold grid:
        0.05, 0.10, 0.15, ..., 0.50
    """
    if iou_thresholds is None:
        iou_thresholds = [round(0.05 * i, 2) for i in range(1, 11)]

    thresholds: List[float] = []
    ap_per_threshold: Dict[str, float] = {}
    total_thresholds = len(iou_thresholds)
    for idx, thr in enumerate(iou_thresholds, start=1):
        thr_f = float(thr)
        thresholds.append(thr_f)
        ap_stats = compute_batch_ap(
            batch_results,
            iou_threshold=thr_f,
            match_by_category=False,
        )
        ap_per_threshold[f"{thr_f:.2f}"] = float(ap_stats.get("map_pr", 0.0))
        if progress_callback is not None:
            progress_callback(idx, total_thresholds, thr_f)

    mean_ap = float(np.mean(list(ap_per_threshold.values()))) if ap_per_threshold else 0.0
    return {
        "protocol": "omni3d_class_agnostic",
        "iou_thresholds": thresholds,
        "ap_per_threshold": ap_per_threshold,
        "map": mean_ap,
    }


def _azimuth_deg_xy(x: float, y: float) -> float:
    """atan2(y, x) in degrees, range (-180, 180]."""
    return math.degrees(math.atan2(float(y), float(x)))


def _wrap360(deg: float) -> float:
    return (deg + 360.0) % 360.0


def _azimuth_bin_index(
    x: float,
    y: float,
    n_bins: int,
    *,
    angular_mode: str = "full360",
    angle_offset_deg: float = 0.0,
    angle_span_deg: float = 360.0,
) -> int:
    """
    Map (x, y) to an angular bin index.

    ``angular_mode``:
    - ``full360``: uniform bins on ``(atan2_deg wrapped to [0,360) - offset) % 360``.
    - ``rot_clip``: same rotation then clip to ``[0, angle_span_deg)`` before binning
      (values beyond the span map to the last bin; useful when detections sit in a
      camera-forward arc such as former 90°–270° after a 90° offset → 0°–180°).
    - ``abs180``: bins on ``abs(atan2_deg)`` in ``[0, 180°]`` (symmetric about +x).
    """
    n_bins = max(1, int(n_bins))
    deg = _azimuth_deg_xy(x, y)
    a360 = _wrap360(deg)

    if angular_mode == "abs180":
        span = 180.0
        u = abs(deg)
        if u >= span:
            u = span - 1e-9
        width = span / float(n_bins)
        idx = int(u / width)
    elif angular_mode == "rot_clip":
        span = max(1e-9, float(angle_span_deg))
        u = (a360 - float(angle_offset_deg) + 360.0) % 360.0
        u_clipped = min(max(u, 0.0), span - 1e-9)
        width = span / float(n_bins)
        idx = int(u_clipped / width)
    else:
        # full360
        span = 360.0
        u = (a360 - float(angle_offset_deg) + 360.0) % 360.0
        width = span / float(n_bins)
        idx = int(u / width)

    if idx >= n_bins:
        idx = n_bins - 1
    return idx


def _azimuth_bin_label(
    i: int,
    n_bins: int,
    *,
    angular_mode: str = "full360",
    angle_offset_deg: float = 0.0,
    angle_span_deg: float = 360.0,
) -> str:
    if angular_mode == "abs180":
        span = 180.0
        lo = i * span / float(n_bins)
        hi = (i + 1) * span / float(n_bins)
        return f"|θ| {lo:.0f}°–{hi:.0f}°"

    if angular_mode == "rot_clip":
        span = max(1e-9, float(angle_span_deg))
        lo = i * span / float(n_bins)
        hi = (i + 1) * span / float(n_bins)
        return f"{lo:.0f}°–{hi:.0f}°"

    span = 360.0 if angular_mode == "full360" else max(1e-9, float(angle_span_deg))
    lo = i * span / float(n_bins)
    hi = (i + 1) * span / float(n_bins)
    w_lo = (lo + float(angle_offset_deg)) % 360.0
    w_hi = (hi + float(angle_offset_deg)) % 360.0
    return f"{w_lo:.0f}°–{w_hi:.0f}°"


def compute_batch_azimuth_bin_metrics(
    batch_results: List[Dict],
    iou_threshold: float,
    n_bins: int,
    match_by_category: bool,
    *,
    angular_mode: str = "full360",
    angle_offset_deg: float = 0.0,
    angle_span_deg: float = 360.0,
) -> Dict:
    """
    Per horizontal azimuth bin: TP, FP, FN and precision (per-bin precision = TP/(TP+FP)).

    Objects are assigned using the 3D center of the GT (TP, FN) or detection (FP).
    Default ``full360`` matches the legacy behaviour: ``atan2(y, x)`` wrapped to
    ``[0, 360°)``, 0° along +x, equal-width bins.

    Args:
        batch_results: Evaluable samples (``ground_truth_cuboids`` may be empty lists).
        iou_threshold: Same IoU rule as ``compute_frame_metrics_at_iou``.
        n_bins: Number of equal angular bins (>= 1).
        match_by_category: Same as other batch metrics.
        angular_mode: ``full360`` | ``rot_clip`` | ``abs180`` (see ``_azimuth_bin_index``).
        angle_offset_deg: Subtracted after wrapping to ``[0, 360)`` (for ``full360`` /
            ``rot_clip``). Typical sim “camera-forward” remap: ``90`` with ``rot_clip``
            and ``angle_span_deg=180`` maps the former ``90°–270°`` arc onto ``0°–180°``.
        angle_span_deg: Span of the binned axis after rotation (``rot_clip`` only;
            clipped to this interval before binning).

    Returns:
        Dict with ``bins``, ``n_bins``, ``iou_threshold``, and angular metadata.
    """
    n_bins = max(1, int(n_bins))
    tp_bins = [0] * n_bins
    fp_bins = [0] * n_bins
    fn_bins = [0] * n_bins

    for sample in batch_results:
        gt = _normalize_gt_cuboids(sample.get("ground_truth_cuboids", []))
        det = sample.get("detected_cuboids", [])

        tp_pairs, unmatched_gt, unmatched_det = greedy_iou_match(
            gt, det, iou_threshold, match_by_category
        )

        for gi, _di in tp_pairs:
            c = CuboidMatcher.get_cuboid_center(gt[gi])
            bi = _azimuth_bin_index(
                float(c[0]),
                float(c[1]),
                n_bins,
                angular_mode=angular_mode,
                angle_offset_deg=angle_offset_deg,
                angle_span_deg=angle_span_deg,
            )
            tp_bins[bi] += 1

        for di in unmatched_det:
            c = CuboidMatcher.get_cuboid_center(det[di])
            bi = _azimuth_bin_index(
                float(c[0]),
                float(c[1]),
                n_bins,
                angular_mode=angular_mode,
                angle_offset_deg=angle_offset_deg,
                angle_span_deg=angle_span_deg,
            )
            fp_bins[bi] += 1

        for gi in unmatched_gt:
            c = CuboidMatcher.get_cuboid_center(gt[gi])
            bi = _azimuth_bin_index(
                float(c[0]),
                float(c[1]),
                n_bins,
                angular_mode=angular_mode,
                angle_offset_deg=angle_offset_deg,
                angle_span_deg=angle_span_deg,
            )
            fn_bins[bi] += 1

    bins_out: List[Dict] = []
    for i in range(n_bins):
        tpi, fpi, fni = tp_bins[i], fp_bins[i], fn_bins[i]
        prec = tpi / (tpi + fpi) if (tpi + fpi) > 0 else 0.0
        bins_out.append({
            "bin_index": i,
            "label": _azimuth_bin_label(
                i,
                n_bins,
                angular_mode=angular_mode,
                angle_offset_deg=angle_offset_deg,
                angle_span_deg=angle_span_deg,
            ),
            "TP": tpi,
            "FP": fpi,
            "FN": fni,
            "precision": prec,
        })

    span_used = (
        180.0
        if angular_mode == "abs180"
        else (360.0 if angular_mode == "full360" else max(1e-9, float(angle_span_deg)))
    )

    return {
        "bins": bins_out,
        "n_bins": n_bins,
        "iou_threshold": iou_threshold,
        "angular_mode": angular_mode,
        "angle_offset_deg": float(angle_offset_deg),
        "angle_span_deg": float(angle_span_deg),
        "binned_span_deg": span_used,
    }


def compute_batch_statistics(
    batch_results: List[Dict],
    total_queued: int = 0,
    match_by_category: bool = False,
    omni3d_progress_callback: Optional[Callable[[int, int, float], None]] = None,
) -> Dict:
    """
    Compute comprehensive batch evaluation statistics at IoU 0.5 and 0.25.

    ``ap_50`` / ``ap_25`` hold micro precision/recall/F1, macro-F1 per class,
    per-frame breakdowns (see ``compute_batch_ap``), and **mAP** as mean
    per-class AP under the PR curve (``map_pr``), using exported detection
    ``confidence`` for ranking.

    Args:
        batch_results: List of sample dicts from
            ``batch_export_results['samples']``.
        total_queued: Number of samples originally queued for processing
            (from ``batch_samples``).  Used to compute failure count.
        match_by_category: Forwarded to the underlying batch metric computation.

    Returns:
        Dict with:
            n_total_samples, n_evaluable_samples, n_skipped_no_gt,
            n_failed_pipeline, total_detections, total_ground_truth,
            ap_50 (full result dict), ap_25 (full result dict).
    """
    n_total = len(batch_results)
    results_with_gt = [r for r in batch_results if r.get("ground_truth_cuboids") is not None]
    n_with_gt = len(results_with_gt)
    n_failed = max(0, total_queued - n_total)

    total_detections = sum(len(r.get("detected_cuboids", [])) for r in batch_results)
    total_gt = sum(len(r.get("ground_truth_cuboids", [])) for r in results_with_gt)

    omni3d_class_agnostic = compute_omni3d_class_agnostic_map(
        results_with_gt,
        progress_callback=omni3d_progress_callback,
    )
    ap_50 = compute_batch_ap(results_with_gt, iou_threshold=0.5, match_by_category=match_by_category)
    ap_25 = compute_batch_ap(results_with_gt, iou_threshold=0.25, match_by_category=match_by_category)

    return {
        "n_total_samples": n_total,
        "n_evaluable_samples": n_with_gt,
        "n_skipped_no_gt": n_total - n_with_gt,
        "n_failed_pipeline": n_failed,
        "total_detections": total_detections,
        "total_ground_truth": total_gt,
        "ap_50": ap_50,
        "ap_25": ap_25,
        "omni3d_class_agnostic": omni3d_class_agnostic,
    }
