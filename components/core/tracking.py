from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from components.utils.export_utils import Export


def _clamp_bbox(bbox: List[float], image_shape: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
    """Clamp a 2D bounding box to image bounds."""
    h, w = image_shape[:2]
    x1, y1, x2, y2 = bbox
    x1_i = max(0, min(int(x1), w - 1))
    y1_i = max(0, min(int(y1), h - 1))
    x2_i = max(0, min(int(x2), w - 1))
    y2_i = max(0, min(int(y2), h - 1))
    if x2_i <= x1_i:
        x2_i = min(w - 1, x1_i + 1)
    if y2_i <= y1_i:
        y2_i = min(h - 1, y1_i + 1)
    return x1_i, y1_i, x2_i, y2_i


def _compute_patch_feature(patch: np.ndarray) -> np.ndarray:
    """
    Compute a compact feature vector for an object patch.

    Uses a small grayscale histogram as a lightweight, comparable encoding.
    """
    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist.astype(np.float32)


def _similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two feature vectors."""
    if a.size == 0 or b.size == 0:
        return 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


@dataclass
class TrackedObject:
    track_id: int
    label: str
    feature: np.ndarray
    first_frame: int
    first_bbox: Tuple[int, int, int, int]
    detections: List[Dict[str, Any]] = field(default_factory=list)

    def is_similar(self, candidate_feature: np.ndarray, threshold: float) -> bool:
        """Return True if the candidate patch feature matches this object."""
        score = _similarity(self.feature, candidate_feature)
        return score >= threshold


class ObjectTracker:
    """
    Simple appearance-based tracker over batch samples.

    Each new detection is matched to existing tracks via a cosine similarity
    on a compact image patch feature; if no good match is found a new track
    is created.
    """

    def __init__(self, similarity_threshold: float = 0.8, patch_size: Tuple[int, int] = (64, 64)) -> None:
        self.similarity_threshold = similarity_threshold
        self.patch_size = patch_size
        self._next_track_id = 0
        self._tracks: List[TrackedObject] = []
        self._frames: Dict[int, Dict[str, Any]] = {}

    def _crop_patch(self, image: np.ndarray, bbox: List[float]) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = _clamp_bbox(bbox, image.shape)
        if x2 <= x1 or y2 <= y1:
            return None
        patch = image[y1:y2, x1:x2]
        if patch.size == 0:
            return None
        resized = cv2.resize(patch, self.patch_size, interpolation=cv2.INTER_AREA)
        return resized

    def _match_or_create_track(
        self,
        frame_index: int,
        image: np.ndarray,
        bbox: List[float],
        label: str,
        detection: Dict[str, Any],
    ) -> int:
        patch = self._crop_patch(image, bbox)
        if patch is None:
            track_id = self._next_track_id
            self._next_track_id += 1
            self._tracks.append(
                TrackedObject(
                    track_id=track_id,
                    label=label,
                    feature=np.zeros(32, dtype=np.float32),
                    first_frame=frame_index,
                    first_bbox=_clamp_bbox(bbox, image.shape),
                )
            )
            return track_id

        feature = _compute_patch_feature(patch)

        best_track: Optional[TrackedObject] = None
        best_score = 0.0

        for tr in self._tracks:
            if tr.label != label:
                continue
            score = _similarity(tr.feature, feature)
            if score > best_score:
                best_score = score
                best_track = tr

        if best_track is not None and best_score >= self.similarity_threshold:
            return best_track.track_id

        track_id = self._next_track_id
        self._next_track_id += 1
        new_track = TrackedObject(
            track_id=track_id,
            label=label,
            feature=feature,
            first_frame=frame_index,
            first_bbox=_clamp_bbox(bbox, image.shape),
        )
        self._tracks.append(new_track)
        return track_id

    def update_for_frame(
        self,
        frame_index: int,
        image: np.ndarray,
        detected_cuboids: List[Dict[str, Any]],
        mask_bboxes: List[List[float]],
        class_names: List[str],
        meta: Dict[str, Any],
    ) -> None:
        """
        Update tracker state for a single frame.

        Each detected cuboid is associated with a 2D mask bounding box
        (via its mask_idx) and then matched to existing tracks.
        """
        print(
            f"[tracking] update_for_frame frame_index={frame_index}, "
            f"n_cuboids={len(detected_cuboids)}, "
            f"n_bboxes={len(mask_bboxes)}, n_class_names={len(class_names)}"
        )

        frame_entry = self._frames.get(frame_index)
        if frame_entry is None:
            frame_entry = {
                "frame_index": frame_index,
                "meta": meta,
                "annotations": [],
            }
            self._frames[frame_index] = frame_entry

        for det_idx, det in enumerate(detected_cuboids):
            mask_idx = det.get("mask_idx")
            if mask_idx is None:
                print(f"[tracking] det[{det_idx}] has no mask_idx, skipping")
                continue
            if not (0 <= int(mask_idx) < len(mask_bboxes)):
                print(f"[tracking] det[{det_idx}] mask_idx {mask_idx} out of range, skipping")
                continue

            bbox = mask_bboxes[int(mask_idx)]
            if not (0 <= int(mask_idx) < len(class_names)):
                label = det.get("category", "Unknown")
            else:
                label = class_names[int(mask_idx)]

            track_id = self._match_or_create_track(
                frame_index=frame_index,
                image=image,
                bbox=bbox,
                label=label,
                detection=det,
            )
            print(f"[tracking] det[{det_idx}] assigned track_id={track_id}")

            det["track_id"] = track_id

            annotation = {
                "track_id": track_id,
                "label": label,
                "position": [
                    float(det["center"][0]),
                    float(det["center"][1]),
                    float(det["center"][2]),
                ],
                "rotation": [0.0, 0.0, float(det["yaw"])],
                "scale": [
                    float(det["length"]),
                    float(det["width"]),
                    float(det["height"]),
                ],
                "occluded": False,
            }

            frame_entry["annotations"].append(annotation)

    def build_datumaro_state(self) -> Dict[str, Any]:
        """
        Build a Datumaro/CVAT-style JSON structure summarizing all tracks.

        The structure is aligned with the reference in dataset/tes/annotations/ref.json.
        """
        print(
            f"[tracking] build_datumaro_state with "
            f"{len(self._tracks)} tracks and {len(self._frames)} frames"
        )

        label_names: List[str] = []
        for tr in self._tracks:
            if tr.label not in label_names:
                label_names.append(tr.label)

        label_names_sorted = sorted(label_names)
        label_to_id: Dict[str, int] = {name: idx for idx, name in enumerate(label_names_sorted)}

        categories = {
            "label": {
                "labels": [
                    {"name": name, "parent": "", "attributes": []} for name in label_names_sorted
                ],
                "label_groups": [],
                "attributes": ["occluded"],
            },
            "points": {"items": []},
        }

        items: List[Dict[str, Any]] = []
        sorted_frame_indices = sorted(self._frames.keys())

        for frame_id, frame_index in enumerate(sorted_frame_indices):
            frame_entry = self._frames[frame_index]
            meta = frame_entry["meta"]
            annotations = frame_entry["annotations"]

            item_annotations: List[Dict[str, Any]] = []
            for ann_id, ann in enumerate(annotations):
                label = ann["label"]
                label_id = label_to_id.get(label, 0)
                item_annotations.append(
                    {
                        "id": ann_id,
                        "type": "cuboid_3d",
                        "attributes": {
                            "track_id": ann["track_id"],
                            "keyframe": frame_id == 0,
                            "occluded": ann["occluded"],
                        },
                        "label_id": label_id,
                        "position": ann["position"],
                        "rotation": ann["rotation"],
                        "scale": ann["scale"],
                    }
                )

            # Use sequential frame-based naming to match exported PCD files.
            pcd_name = f"frame_{frame_id:06d}.pcd"

            item = {
                "id": f"frame_{frame_id:06d}",
                "annotations": item_annotations,
                "attr": {"frame": frame_id},
                "point_cloud": {"path": f"lidar/{pcd_name}"},
            }
            items.append(item)

        # Reverse frame order for Datumaro export to match external tools.
        items = Export.reverse_frame_order(items)

        return {
            "info": {},
            "categories": categories,
            "items": items,
        }

