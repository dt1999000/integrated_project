"""
Evaluation Module

Provides utilities for evaluating object detection results by matching
detected cuboids to ground truth annotations.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
import numpy as np


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
            det_category = det.get('category', 'Unknown')

            best_match_idx = None
            best_dist = self.max_distance

            for gt_idx, gt in enumerate(gt_cuboids):
                # Skip already matched GT
                if gt_idx in matched_gt_indices:
                    continue

                gt_category = gt.get('category', 'Unknown')

                # Category check if enabled
                if self.match_by_category and det_category != gt_category:
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
            cat = gt_cuboids[gt_idx].get('category', 'Unknown')
            if cat not in category_stats:
                category_stats[cat] = {'TP': 0, 'FP': 0, 'FN': 0}
            category_stats[cat]['TP'] += 1

        # Count false negatives (unmatched GT)
        for gt_idx in match_result.unmatched_gt:
            cat = gt_cuboids[gt_idx].get('category', 'Unknown')
            if cat not in category_stats:
                category_stats[cat] = {'TP': 0, 'FP': 0, 'FN': 0}
            category_stats[cat]['FN'] += 1

        # Count false positives (unmatched detections)
        for det_idx in match_result.unmatched_det:
            cat = detected_cuboids[det_idx].get('category', 'Unknown')
            if cat not in category_stats:
                category_stats[cat] = {'TP': 0, 'FP': 0, 'FN': 0}
            category_stats[cat]['FP'] += 1

        # Compute per-category precision and recall
        for cat, stats in category_stats.items():
            tp, fp, fn = stats['TP'], stats['FP'], stats['FN']
            stats['Precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            stats['Recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        return category_stats
