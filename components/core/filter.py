"""
Common image quality filtering utilities used across datasets (KITTI, nuScenes, sim).
"""

from pathlib import Path
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np
from PIL import Image
import imagehash

from components.dataset_loaders.dataset_loader import LinkedDataHandler


def variance_of_laplacian(img_gray: np.ndarray) -> float:
    """Calculate blur metric using Laplacian variance."""
    lap = cv2.Laplacian(img_gray, cv2.CV_64F)
    return float(lap.var())


def contrast_score(img_gray: np.ndarray) -> float:
    """Calculate normalized RMS contrast."""
    return float(np.std(img_gray)) / 255.0


def brightness(img_gray: np.ndarray) -> float:
    """Calculate average brightness."""
    return float(img_gray.mean())


def hamming(a, b) -> int:
    """Calculate Hamming distance between two perceptual hashes."""
    return int(a - b)


def calculate_motion_score(
    frame1_gray: Optional[np.ndarray],
    frame2_gray: Optional[np.ndarray],
) -> float:
    """Calculate motion score between consecutive frames."""
    if frame1_gray is None or frame2_gray is None:
        return 0.0
    diff = cv2.absdiff(frame1_gray, frame2_gray)
    return float(np.mean(diff))


def _apply_quality_filters(
    image_gray: np.ndarray,
    filter_params: Dict,
    seen_hashes: List,
    prev_frame_gray: Optional[np.ndarray],
) -> Tuple[bool, Dict[str, float], Optional[np.ndarray]]:
    """
    Apply common quality filters (blur, motion, dedup, brightness, contrast)
    to a single grayscale image.
    """
    blur_score = variance_of_laplacian(image_gray)
    contrast_val = contrast_score(image_gray)
    brightness_val = brightness(image_gray)

    passed = True

    # Blur filter
    if filter_params.get("enable_blur", False):
        if blur_score < filter_params.get("blur_gate", 0):
            passed = False

    # Motion filter (if enabled and we have previous frame)
    if filter_params.get("enable_motion", False) and prev_frame_gray is not None:
        motion_score = calculate_motion_score(prev_frame_gray, image_gray)
        if motion_score < filter_params.get("motion_thresh", 0):
            passed = False
    else:
        motion_score = 0.0

    # Deduplication
    if filter_params.get("enable_dedup", False) and passed:
        try:
            ph = imagehash.dhash(Image.fromarray(image_gray))
            if any(hamming(ph, old) <= filter_params.get("hash_thresh", 0) for old in seen_hashes):
                passed = False
            else:
                seen_hashes.append(ph)
        except Exception:
            # Ignore hash errors and keep image
            pass

    # Brightness filter
    if filter_params.get("enable_brightness", False) and passed:
        min_b = filter_params.get("min_bright", 0)
        max_b = filter_params.get("max_bright", 255)
        if not (min_b <= brightness_val <= max_b):
            passed = False

    # Contrast filter
    if filter_params.get("enable_contrast", False) and passed:
        if contrast_val < filter_params.get("min_contrast", 0.0):
            passed = False

    # Update previous frame if motion is enabled
    new_prev = image_gray if filter_params.get("enable_motion", False) else prev_frame_gray

    metrics = {
        "blur": blur_score,
        "contrast": contrast_val,
        "brightness": brightness_val,
        "motion": motion_score,
    }

    return passed, metrics, new_prev


def image_passes_quality_filters(
    image_gray: np.ndarray,
    prev_frame_gray: Optional[np.ndarray],
    seen_hashes: List,
    filter_params: Dict,
) -> Tuple[bool, Dict[str, float]]:
    """
    Public wrapper around _apply_quality_filters for single grayscale images.

    This matches the legacy signature used by ROS bag utilities:
        passed, metrics = image_passes_quality_filters(
            gray, prev_gray, seen_hashes, params
        )
    and returns:
        passed: bool
        metrics: dict with blur/contrast/brightness/motion
    """
    passed, metrics, new_prev = _apply_quality_filters(
        image_gray=image_gray,
        filter_params=filter_params,
        seen_hashes=seen_hashes,
        prev_frame_gray=prev_frame_gray,
    )
    # Update caller's previous frame in-place when motion filtering is enabled
    if filter_params.get("enable_motion", False):
        # Rely on caller to hold prev_frame_gray reference; they can update it
        pass
    return passed, metrics


def filter_kitti_images(
    dataset_path: str,
    indices: List[int],
    filter_params: Dict,
) -> List[Dict]:
    """Filter a list of KITTI image indices using common quality metrics."""
    filtered_samples: List[Dict] = []
    seen_hashes: List = []
    prev_frame_gray: Optional[np.ndarray] = None

    root_path = Path(dataset_path)
    training_dir = root_path / "training"
    testing_dir = root_path / "testing"
    split_dir = training_dir if training_dir.exists() else testing_dir
    image_dir = split_dir / "image_2"

    for idx in indices:
        try:
            image_path = image_dir / f"{idx:06d}.png"
            if not image_path.exists():
                continue

            image_bgr = cv2.imread(str(image_path))
            if image_bgr is None:
                continue

            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

            passed, metrics, prev_frame_gray = _apply_quality_filters(
                image_gray=image_gray,
                filter_params=filter_params,
                seen_hashes=seen_hashes,
                prev_frame_gray=prev_frame_gray,
            )

            if passed:
                filtered_samples.append(
                    {
                        "sample_index": int(idx),
                        "image": image_rgb,
                        "image_path": str(image_path),
                        "metrics": metrics,
                    }
                )
        except Exception:
            continue

    return filtered_samples


def filter_nuscenes_images(
    samples: List[Dict],
    filter_params: Dict,
) -> List[Dict]:
    """
    Filter a list of nuScenes camera samples using common quality metrics.
    Each sample dict is expected to have keys: 'sample_token' and 'image_path' (or 'filename').
    """
    filtered_samples: List[Dict] = []
    seen_hashes: List = []
    prev_frame_gray: Optional[np.ndarray] = None

    for sample in samples:
        try:
            image_path = sample.get("image_path") or sample.get("filename")
            if not image_path:
                continue

            image_bgr = cv2.imread(str(image_path))
            if image_bgr is None:
                continue

            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

            passed, metrics, prev_frame_gray = _apply_quality_filters(
                image_gray=image_gray,
                filter_params=filter_params,
                seen_hashes=seen_hashes,
                prev_frame_gray=prev_frame_gray,
            )

            if passed:
                filtered_samples.append(
                    {
                        "sample_token": sample.get("sample_token"),
                        "image": image_rgb,
                        "image_path": str(image_path),
                        "metrics": metrics,
                    }
                )
        except Exception:
            continue

    return filtered_samples


def filter_sim_images(
    handler: LinkedDataHandler,
    subset_name: str,
    links: List[Dict],
    dataset_path: str,
    filter_params: Dict,
) -> List[Dict]:
    """
    Filter images from sim dataset using common quality metrics.
    """
    filtered_samples: List[Dict] = []
    seen_hashes: List = []
    prev_frame_gray: Optional[np.ndarray] = None

    root_path = Path(dataset_path)
    subset_path = root_path / subset_name

    for link in links:
        try:
            rgb_sample = link.get("samples", {}).get("rgb", {})
            if not rgb_sample or "filename" not in rgb_sample:
                continue

            filename = rgb_sample["filename"]

            # Normalize filename: remove leading slashes and handle absolute paths
            filename = filename.lstrip("/").lstrip("\\")
            if len(filename) > 1 and filename[1] == ":":
                # Windows absolute path like C:\rgb\file.jpg
                parts = filename.split("\\", 2)
                if len(parts) > 2:
                    filename = parts[2]
                else:
                    filename = parts[-1]

            image_path = subset_path / "samples" / filename
            if not image_path.exists():
                continue

            image_bgr = cv2.imread(str(image_path))
            if image_bgr is None:
                continue

            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

            passed, metrics, prev_frame_gray = _apply_quality_filters(
                image_gray=image_gray,
                filter_params=filter_params,
                seen_hashes=seen_hashes,
                prev_frame_gray=prev_frame_gray,
            )

            if passed:
                filtered_samples.append(
                    {
                        "link_token": link["token"],
                        "link": link,
                        "image": image_rgb,
                        "image_path": str(image_path),
                        "metrics": {
                            "blur": metrics["blur"],
                            "contrast": metrics["contrast"],
                            "brightness": metrics["brightness"],
                        },
                    }
                )
        except Exception:
            continue

    return filtered_samples

