from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

import cv2
import numpy as np

from components.utils.export_utils import Export
from components.utils.mask_utils import get_bbox_from_mask

from components.core.sam_integration import SAMIntegration


def _clamp_bbox(bbox: List[float], image_shape: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
    """Clamp a 2D bounding box to image bounds."""
    h, w = image_shape[:2]
    x1, y1, x2, y2 = bbox
    x1_i = max(0, min(int(x1), w - 1))
    y1_i = max(0, min(int(y1), h - 1))
    x2_i = max(0, min(int(x2), w - 1))
    y2_i = max(0, min(int(y2), h - 1))
    if x1_i != x1 and y1_i != y1 and x2_i != x2 and y2_i != y2:
        print(f'clamped bbox: x1 ={x1}, x2 ={x2}, y1 ={y1}, y2 = {y2} -> x1_i ={x1_i}, x2_i ={x2_i}, y1_i ={y1_i}, y2_i = {y2_i} for h,w = ({(h,w)})')
    if x2_i <= x1_i:
        x2_i = min(w - 1, x1_i + 1)
    if y2_i <= y1_i:
        y2_i = min(h - 1, y1_i + 1)
    return x1_i, y1_i, x2_i, y2_i


def _compute_patch_feature(patch: np.ndarray) -> np.ndarray:
    """
    Compute a compact feature vector for an object patch.

    Uses concatenated HSV histograms and gradient-orientation histograms.
    HSV gives better brightness robustness; gradient bins add shape cues.
    """
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hists: List[np.ndarray] = []
    for ch, bins, value_range in ((0, 12, [0, 180]), (1, 8, [0, 256]), (2, 8, [0, 256])):
        hist = cv2.calcHist([hsv], [ch], None, [bins], value_range)
        hist = cv2.normalize(hist, hist).flatten()
        hists.append(hist.astype(np.float32))

    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag, ang = cv2.cartToPolar(grad_x, grad_y, angleInDegrees=True)
    bins = 8
    bin_edges = np.linspace(0.0, 360.0, bins + 1, dtype=np.float32)
    hog_like = np.zeros(bins, dtype=np.float32)
    for i in range(bins):
        mask = (ang >= bin_edges[i]) & (ang < bin_edges[i + 1])
        hog_like[i] = float(np.sum(mag[mask]))
    hog_like = cv2.normalize(hog_like.reshape(-1, 1), None).flatten().astype(np.float32)
    hists.append(hog_like)
    feature = np.concatenate(hists, axis=0)
    return feature


def _similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two feature vectors."""
    if a.size == 0 or b.size == 0:
        return 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)

def _clamped_bbox_from_mask(mask: np.ndarray, image_shape: Tuple[int, int, int]) -> Optional[Tuple[int, int, int, int]]:
    bbox = get_bbox_from_mask(mask)
    if bbox == [0, 0, 0, 0]:
        return None
    x1, y1, x2, y2 = bbox
    # Convert max coords to right/bottom exclusive bounds for slicing.
    return _clamp_bbox([x1, y1, x2 + 1, y2 + 1], image_shape)


def _bbox_history_to_sam_xyxy(bbox: Tuple[int, int, int, int]) -> List[float]:
    """Convert slice-style (x1,y1,x2,y2) bbox to inclusive xyxy for SAM prompts."""
    x1, y1, x2e, y2e = bbox
    xi2 = max(int(x1), int(x2e) - 1)
    yi2 = max(int(y1), int(y2e) - 1)
    return [float(x1), float(y1), float(xi2), float(yi2)]


def _binary_mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    ab = (a > 0).astype(np.uint8).ravel()
    bb = (b > 0).astype(np.uint8).ravel()
    if ab.size != bb.size or ab.size == 0:
        return 0.0
    inter = int(np.sum((ab > 0) & (bb > 0)))
    union = int(np.sum((ab > 0) | (bb > 0)))
    return float(inter / union) if union > 0 else 0.0


@dataclass
class TrackedObject:
    track_id: int
    label: str
    feature: np.ndarray
    # 2D frame information (image space)
    first_frame: int
    last_frame: int
    # 3D motion information (LiDAR/world space)
    last_center_3d: np.ndarray
    velocity_3d: np.ndarray
    prev_velocity_3d: np.ndarray
    velocity_history: List[np.ndarray] = field(default_factory=list)
    # Per-frame 2D bbox history: frame_index -> (x1, y1, x2, y2)
    bbox_history: Dict[int, Tuple[int, int, int, int]] = field(default_factory=dict)
    occluded_history: List[bool] = field(default_factory=list)
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

    def __init__(
        self,
        similarity_threshold: float = 0.65,
        bag_freq_hz: float = 45.0,
        class_max_speed_mps: Optional[Dict[str, float]] = None,
        sam2_track_iou_threshold: float = 0.2,
    ) -> None:
        # Rosbag/image sampling frequency (Hz). Used to compute Δt between frames.
        if bag_freq_hz <= 0.0:
            self.bag_freq_hz = 45.0
        else:
            self.bag_freq_hz = float(bag_freq_hz)
        # Per-class hard maximum speed priors (m/s).
        default_speed_priors: Dict[str, float] = {
            "car": 40.0,
            "truck": 35.0,
            "bus": 35.0,
            "bicycle": 15.0,
            "bike": 15.0,
            "motorcycle": 25.0,
            "person": 5.0,
            "pedestrian": 5.0,
            "default": 50.0,
        }
        if class_max_speed_mps is None:
            self.class_max_speed_mps = default_speed_priors
        else:
            merged: Dict[str, float] = default_speed_priors.copy()
            for k, v in class_max_speed_mps.items():
                merged[str(k).lower()] = float(v)
            self.class_max_speed_mps = merged
        self.similarity_threshold = similarity_threshold
        self.sam2_track_iou_threshold = float(sam2_track_iou_threshold)
        # Maximum allowed 2D center distance between consecutive frames to still
        # consider detections for the same track.
        self.max_center_distance: float = 100.0
        # Minimum 3D step length (in meters) for a motion segment to be
        # considered when evaluating direction changes between keyframes.
        self.direction_change_min_step: float = 0.25
        # Cosine of the maximum allowed angle between consecutive 3D motion
        # segments before forcing a new keyframe. For example, cos(60°) ≈ 0.5,
        # so anything with an angle larger than 60° will trigger a split.
        self.direction_change_cos_threshold: float = float(np.cos(np.deg2rad(60.0)))
        self._next_track_id = 0
        self._tracks: List[TrackedObject] = []
        self._frames: Dict[int, Dict[str, Any]] = {}
        # frame_index -> per-scene tracking snapshot. This is the canonical 2D
        # export source so scene-wise "appears + bbox" is explicit.
        self._frame_tracking_2d: Dict[int, Dict[str, Any]] = {}

    def _masked_patch_for_feature(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        bbox: Tuple[int, int, int, int],
        reference_size: Optional[Tuple[int, int]] = None,
    ) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            return None
        patch = image[y1:y2, x1:x2]
        patch_mask = mask[y1:y2, x1:x2]
        if patch.size == 0 or patch_mask.size == 0:
            return None
        mask_bin = (patch_mask > 0).astype(np.uint8)
        if not np.any(mask_bin):
            return None
        masked_patch = patch * mask_bin[:, :, None]
        if reference_size is None:
            target_w = max(8, int(x2 - x1))
            target_h = max(8, int(y2 - y1))
        else:
            target_w = max(8, int(reference_size[0]))
            target_h = max(8, int(reference_size[1]))
        resized = cv2.resize(masked_patch, (target_w, target_h), interpolation=cv2.INTER_AREA)
        return resized

    def _update_velocity(
        self,
        tr: TrackedObject,
        curr_center_3d: np.ndarray,
        curr_frame: int,
    ) -> None:
        if tr.last_frame == curr_frame:
            return
        dt_frames = curr_frame - tr.last_frame
        if dt_frames <= 0:
            dt_frames = 1
        dt = float(dt_frames) / self.bag_freq_hz
        disp = curr_center_3d - tr.last_center_3d
        v_new = disp / dt
        tr.prev_velocity_3d = tr.velocity_3d
        tr.velocity_3d = v_new
        tr.velocity_history.append(v_new)
        if len(tr.velocity_history) > 5:
            tr.velocity_history.pop(0)

    def _propose_position(
        self,
        tr: TrackedObject,
        dt_future: float,
    ) -> np.ndarray:
        if len(tr.velocity_history) == 0:
            return tr.last_center_3d
        diffs: List[float] = []
        for i in range(1, len(tr.velocity_history)):
            d = tr.velocity_history[i] - tr.velocity_history[i - 1]
            diffs.append(float(np.linalg.norm(d)))
        if len(diffs) == 0:
            weight = 0.5
        else:
            mean_diff = float(np.mean(diffs))
            weight = 1.0 / (1.0 + mean_diff)
        return tr.last_center_3d + weight * tr.velocity_3d * dt_future



    def track_on_image(
        self,
        frame_index: int,
        image: np.ndarray,
        masks: List[np.ndarray],
        class_names: List[str],
        meta: Dict[str, Any],
    ) -> Dict[int, int]:
        """
        Track objects on image/masks only and return mask_idx -> track_id.
        """
        print(
            f"[tracking] track_on_image frame_index={frame_index}, "
            f"n_masks={len(masks)}, n_class_names={len(class_names)}"
        )

        frame_entry = self._frames.get(frame_index)
        if frame_entry is None:
            frame_entry = {
                "frame_index": frame_index,
                "meta": meta,
                "annotations": [],
            }
            self._frames[frame_index] = frame_entry

        # Build per-mask appearance features and 2D centers.
        mask_bboxes: List[Optional[Tuple[int, int, int, int]]] = []
        mask_centers: List[Tuple[float, float]] = []
        for mask in masks:
            bbox = _clamped_bbox_from_mask(mask, image.shape)
            mask_bboxes.append(bbox)
            if bbox is None:
                mask_centers.append((-1.0, -1.0))
                continue
            x1, y1, x2, y2 = bbox
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            mask_centers.append((cx, cy))

        # Build similarity candidates (sim, track_index, mask_index, feature).
        candidates: List[Tuple[float, int, int, np.ndarray]] = []
        for t_idx, tr in enumerate(self._tracks):
            prev_bbox = tr.bbox_history.get(tr.last_frame)
            if prev_bbox is None:
                continue
            plx1, ply1, plx2, ply2 = prev_bbox
            ref_w = max(1, int(plx2 - plx1))
            ref_h = max(1, int(ply2 - ply1))
            for m_idx, _ in enumerate(masks):
                curr_bbox = mask_bboxes[m_idx]
                if curr_bbox is None:
                    continue
                if m_idx < len(class_names) and tr.label != class_names[m_idx]:
                    continue
                lcx = 0.5 * (plx1 + plx2)
                lcy = 0.5 * (ply1 + ply2)
                cx, cy = mask_centers[m_idx]
                dx = cx - lcx
                dy = cy - lcy
                dist = (dx * dx + dy * dy) ** 0.5
                if dist > self.max_center_distance:
                    continue
                patch = self._masked_patch_for_feature(
                    image=image,
                    mask=masks[m_idx],
                    bbox=curr_bbox,
                    reference_size=(ref_w, ref_h),
                )
                if patch is None:
                    continue
                feat = _compute_patch_feature(patch)
                sim = _similarity(tr.feature, feat)
                if sim <= 0.0:
                    continue
                candidates.append((sim, t_idx, m_idx, feat))

        candidates.sort(key=lambda t: t[0], reverse=True)
        assigned_tracks: Set[int] = set()
        assigned_masks: Set[int] = set()
        mask_to_track: Dict[int, int] = {}

        for sim, t_idx, m_idx, feat in candidates:
            if sim < self.similarity_threshold:
                break
            if t_idx in assigned_tracks or m_idx in assigned_masks:
                continue
            tr = self._tracks[t_idx]
            bbox = mask_bboxes[m_idx]
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox
            tr.feature = feat
            tr.last_frame = frame_index
            tr.bbox_history[frame_index] = (x1, y1, x2, y2)
            assigned_tracks.add(t_idx)
            assigned_masks.add(m_idx)
            mask_to_track[m_idx] = tr.track_id

        # Create new tracks for remaining masks that are not well-explained.
        for m_idx, mask in enumerate(masks):
            if m_idx in assigned_masks:
                continue
            bbox = mask_bboxes[m_idx]
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox
            patch = self._masked_patch_for_feature(
                image=image,
                mask=mask,
                bbox=bbox,
                reference_size=None,
            )
            if patch is None:
                continue
            feat = _compute_patch_feature(patch)
            best_sim = 0.0
            for tr in self._tracks:
                if m_idx < len(class_names) and tr.label != class_names[m_idx]:
                    continue
                s = _similarity(tr.feature, feat)
                if s > best_sim:
                    best_sim = s
            if best_sim >= self.similarity_threshold:
                continue
            track_id = self._next_track_id
            self._next_track_id += 1
            vel0 = np.zeros(3, dtype=np.float32)
            label = class_names[m_idx] if m_idx < len(class_names) else "Unknown"
            new_tr = TrackedObject(
                track_id=track_id,
                label=label,
                feature=feat,
                first_frame=frame_index,
                last_frame=frame_index,
                last_center_3d=np.asarray([np.nan, np.nan, np.nan], dtype=np.float32),
                velocity_3d=vel0,
                prev_velocity_3d=vel0.copy(),
                bbox_history={frame_index: (x1, y1, x2, y2)},
            )
            self._tracks.append(new_tr)
            mask_to_track[m_idx] = track_id

        sample_index = meta.get("sample_index", frame_index)
        scene_objects: List[Dict[str, Any]] = []
        for tr in self._tracks:
            bbox = tr.bbox_history.get(frame_index)
            scene_objects.append(
                {
                    "track_id": tr.track_id,
                    "label": tr.label,
                    "appears": bbox is not None,
                    "bbox_xyxy": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])] if bbox is not None else None,
                }
            )
        self._frame_tracking_2d[frame_index] = {
            "frame_index": frame_index,
            "sample_index": sample_index,
            "objects": scene_objects,
        }

        return mask_to_track

    def track_on_image_sam2(
        self,
        frame_index: int,
        image: np.ndarray,
        masks: List[np.ndarray],
        class_names: List[str],
        meta: Dict[str, Any],
        sam_integration: SAMIntegration,
    ) -> Dict[int, int]:
        """
        Associate per-frame pipeline masks to tracks using SAM2 with a bbox prompt
        from each track's last known box, then mask–mask IoU against current masks.

        Same return contract as ``track_on_image`` (mask index -> track_id) and
        the same ``_frame_tracking_2d`` / ``TrackedObject`` updates.
        """
        if not sam_integration.model_type.startswith("sam2"):
            raise ValueError(
                "track_on_image_sam2 requires a SAM2 model on SAMIntegration, "
                f"got {sam_integration.model_type!r}"
            )

        print(
            f"[tracking] track_on_image_sam2 frame_index={frame_index}, "
            f"n_masks={len(masks)}, n_class_names={len(class_names)}"
        )

        frame_entry = self._frames.get(frame_index)
        if frame_entry is None:
            frame_entry = {
                "frame_index": frame_index,
                "meta": meta,
                "annotations": [],
            }
            self._frames[frame_index] = frame_entry

        mask_bboxes: List[Optional[Tuple[int, int, int, int]]] = []
        for mask in masks:
            mask_bboxes.append(_clamped_bbox_from_mask(mask, image.shape))

        propagated: List[Tuple[TrackedObject, np.ndarray]] = []
        for tr in self._tracks:
            prev_bbox = tr.bbox_history.get(tr.last_frame)
            if prev_bbox is None:
                continue
            sam_xyxy = _bbox_history_to_sam_xyxy(prev_bbox)
            sam_mask = sam_integration.get_mask_from_bbox(image, sam_xyxy)
            if np.sum(sam_mask > 0) == 0:
                continue
            propagated.append((tr, sam_mask))

        candidates: List[Tuple[float, int, int]] = []
        for p_idx, (tr, sam_mask) in enumerate(propagated):
            for m_idx, mask in enumerate(masks):
                if m_idx < len(class_names) and tr.label != class_names[m_idx]:
                    continue
                iou = _binary_mask_iou(sam_mask, mask)
                if iou >= self.sam2_track_iou_threshold:
                    candidates.append((iou, p_idx, m_idx))

        candidates.sort(key=lambda t: t[0], reverse=True)
        assigned_prop: Set[int] = set()
        assigned_masks: Set[int] = set()
        mask_to_track: Dict[int, int] = {}

        for iou, p_idx, m_idx in candidates:
            if p_idx in assigned_prop or m_idx in assigned_masks:
                continue
            tr, _ = propagated[p_idx]
            bbox = mask_bboxes[m_idx]
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox
            patch = self._masked_patch_for_feature(
                image=image,
                mask=masks[m_idx],
                bbox=bbox,
                reference_size=None,
            )
            if patch is None:
                continue
            feat = _compute_patch_feature(patch)
            tr.feature = feat
            tr.last_frame = frame_index
            tr.bbox_history[frame_index] = (x1, y1, x2, y2)
            assigned_prop.add(p_idx)
            assigned_masks.add(m_idx)
            mask_to_track[m_idx] = tr.track_id

        for m_idx, mask in enumerate(masks):
            if m_idx in assigned_masks:
                continue
            bbox = mask_bboxes[m_idx]
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox
            patch = self._masked_patch_for_feature(
                image=image,
                mask=mask,
                bbox=bbox,
                reference_size=None,
            )
            if patch is None:
                continue
            feat = _compute_patch_feature(patch)
            best_sim = 0.0
            for tr in self._tracks:
                if m_idx < len(class_names) and tr.label != class_names[m_idx]:
                    continue
                s = _similarity(tr.feature, feat)
                if s > best_sim:
                    best_sim = s
            if best_sim >= self.similarity_threshold:
                continue
            track_id = self._next_track_id
            self._next_track_id += 1
            vel0 = np.zeros(3, dtype=np.float32)
            label = class_names[m_idx] if m_idx < len(class_names) else "Unknown"
            new_tr = TrackedObject(
                track_id=track_id,
                label=label,
                feature=feat,
                first_frame=frame_index,
                last_frame=frame_index,
                last_center_3d=np.asarray([np.nan, np.nan, np.nan], dtype=np.float32),
                velocity_3d=vel0,
                prev_velocity_3d=vel0.copy(),
                bbox_history={frame_index: (x1, y1, x2, y2)},
            )
            self._tracks.append(new_tr)
            mask_to_track[m_idx] = track_id

        sample_index = meta.get("sample_index", frame_index)
        scene_objects: List[Dict[str, Any]] = []
        for tr in self._tracks:
            bbox = tr.bbox_history.get(frame_index)
            scene_objects.append(
                {
                    "track_id": tr.track_id,
                    "label": tr.label,
                    "appears": bbox is not None,
                    "bbox_xyxy": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])] if bbox is not None else None,
                }
            )
        self._frame_tracking_2d[frame_index] = {
            "frame_index": frame_index,
            "sample_index": sample_index,
            "objects": scene_objects,
        }

        return mask_to_track

    def match_tracks_with_3d_detections(
        self,
        frame_index: int,
        detected_cuboids: List[Dict[str, Any]],
        masks: List[np.ndarray],
        class_names: List[str],
        mask_to_track: Dict[int, int],
        meta: Dict[str, Any],
    ) -> None:
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
            if not (0 <= int(mask_idx) < len(masks)):
                print(f"[tracking] det[{det_idx}] mask_idx {mask_idx} out of range, skipping")
                continue

            if not (0 <= int(mask_idx) < len(class_names)):
                label = det.get("category", "Unknown")
            else:
                label = class_names[int(mask_idx)]

            track_id = mask_to_track.get(int(mask_idx))
            if track_id is None:
                print(f"[tracking] det[{det_idx}] mask_idx {mask_idx} has no assigned track, skipping")
                continue

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

    def apply_external_image_tracks(
        self,
        frame_index: int,
        image: np.ndarray,
        masks: List[np.ndarray],
        class_names: List[str],
        meta: Dict[str, Any],
        mask_to_track: Dict[int, int],
    ) -> None:
        """
        Update image-space tracker state using externally produced mask->track ids.
        """
        frame_entry = self._frames.get(frame_index)
        if frame_entry is None:
            frame_entry = {
                "frame_index": frame_index,
                "meta": meta,
                "annotations": [],
            }
            self._frames[frame_index] = frame_entry

        track_map: Dict[int, TrackedObject] = {tr.track_id: tr for tr in self._tracks}
        mask_bboxes: List[Optional[Tuple[int, int, int, int]]] = []
        for mask in masks:
            mask_bboxes.append(_clamped_bbox_from_mask(mask, image.shape))

        for m_idx, track_id in mask_to_track.items():
            if not (0 <= int(m_idx) < len(masks)):
                continue
            bbox = mask_bboxes[int(m_idx)]
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox
            label = class_names[int(m_idx)] if int(m_idx) < len(class_names) else "Unknown"
            tr = track_map.get(int(track_id))
            if tr is None:
                patch = self._masked_patch_for_feature(
                    image=image,
                    mask=masks[int(m_idx)],
                    bbox=bbox,
                    reference_size=None,
                )
                if patch is None:
                    continue
                feat = _compute_patch_feature(patch)
                vel0 = np.zeros(3, dtype=np.float32)
                new_tr = TrackedObject(
                    track_id=int(track_id),
                    label=label,
                    feature=feat,
                    first_frame=frame_index,
                    last_frame=frame_index,
                    last_center_3d=np.asarray([np.nan, np.nan, np.nan], dtype=np.float32),
                    velocity_3d=vel0,
                    prev_velocity_3d=vel0.copy(),
                    bbox_history={frame_index: (x1, y1, x2, y2)},
                )
                self._tracks.append(new_tr)
                track_map[int(track_id)] = new_tr
                if int(track_id) >= self._next_track_id:
                    self._next_track_id = int(track_id) + 1
                continue

            prev_bbox = tr.bbox_history.get(tr.last_frame)
            reference_size: Optional[Tuple[int, int]] = None
            if prev_bbox is not None:
                plx1, ply1, plx2, ply2 = prev_bbox
                reference_size = (max(1, plx2 - plx1), max(1, ply2 - ply1))
            patch = self._masked_patch_for_feature(
                image=image,
                mask=masks[int(m_idx)],
                bbox=bbox,
                reference_size=reference_size,
            )
            if patch is None:
                continue
            feat = _compute_patch_feature(patch)
            tr.label = label
            tr.feature = feat
            tr.last_frame = frame_index
            tr.bbox_history[frame_index] = (x1, y1, x2, y2)

        sample_index = meta.get("sample_index", frame_index)
        scene_objects: List[Dict[str, Any]] = []
        for tr in self._tracks:
            bbox = tr.bbox_history.get(frame_index)
            scene_objects.append(
                {
                    "track_id": tr.track_id,
                    "label": tr.label,
                    "appears": bbox is not None,
                    "bbox_xyxy": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])] if bbox is not None else None,
                }
            )
        self._frame_tracking_2d[frame_index] = {
            "frame_index": frame_index,
            "sample_index": sample_index,
            "objects": scene_objects,
        }

    def update_for_frame(
        self,
        frame_index: int,
        image: np.ndarray,
        detected_cuboids: List[Dict[str, Any]],
        masks: List[np.ndarray],
        class_names: List[str],
        meta: Dict[str, Any],
        image_track_mode: str = "appearance",
        sam_integration: Optional[SAMIntegration] = None,
    ) -> None:
        if image_track_mode == "sam2_bbox":
            if sam_integration is None:
                raise ValueError("update_for_frame(sam2_bbox) requires sam_integration")
            mask_to_track = self.track_on_image_sam2(
                frame_index=frame_index,
                image=image,
                masks=masks,
                class_names=class_names,
                meta=meta,
                sam_integration=sam_integration,
            )
        else:
            mask_to_track = self.track_on_image(
                frame_index=frame_index,
                image=image,
                masks=masks,
                class_names=class_names,
                meta=meta,
            )
        self.match_tracks_with_3d_detections(
            frame_index=frame_index,
            detected_cuboids=detected_cuboids,
            masks=masks,
            class_names=class_names,
            mask_to_track=mask_to_track,
            meta=meta,
        )

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

        sorted_frame_indices = sorted(self._frames.keys())

        track_frames: Dict[int, List[int]] = {}
        track_positions: Dict[int, List[np.ndarray]] = {}
        for frame_id, frame_index in enumerate(sorted_frame_indices):
            frame_entry = self._frames[frame_index]
            annotations = frame_entry["annotations"]
            for ann in annotations:
                track_id = ann["track_id"]
                if track_id not in track_frames:
                    track_frames[track_id] = []
                track_frames[track_id].append(frame_id)
                if track_id not in track_positions:
                    track_positions[track_id] = []
                track_positions[track_id].append(np.asarray(ann["position"], dtype=np.float32))

        track_keyframes: Dict[int, Set[int]] = {}
        for track_id, frames in track_frames.items():
            if not frames:
                continue

            # Ensure we have matching 3D positions for every annotated frame in this track.
            positions = track_positions.get(track_id, [])
            if len(positions) != len(frames):
                continue

            # Sort frames and positions together by frame index so that motion
            # vectors are computed in temporal order.
            sorted_pairs = sorted(zip(frames, positions), key=lambda p: p[0])
            sorted_frames = [p[0] for p in sorted_pairs]
            sorted_positions = [p[1] for p in sorted_pairs]

            first_f = sorted_frames[0]
            last_f = sorted_frames[-1]
            keyframes_for_track: Set[int] = set()

            if first_f == last_f:
                keyframes_for_track.add(first_f)
                track_keyframes[track_id] = keyframes_for_track
                continue

            # Always treat the first and last appearances of a track as
            # keyframes so CVAT interpolation has fixed endpoints.
            keyframes_for_track.add(first_f)
            keyframes_for_track.add(last_f)

            # Use 3D motion direction to insert additional keyframes whenever
            # the object turns sharply relative to its previous motion.
            prev_vec: Optional[np.ndarray] = None
            for idx in range(1, len(sorted_frames)):
                p_prev = sorted_positions[idx - 1]
                p_curr = sorted_positions[idx]
                curr_vec = p_curr - p_prev

                curr_len = float(np.linalg.norm(curr_vec))
                if curr_len < self.direction_change_min_step:
                    # Ignore almost-stationary motion segments.
                    continue

                if prev_vec is not None:
                    prev_len = float(np.linalg.norm(prev_vec))
                    if prev_len >= self.direction_change_min_step:
                        denom = prev_len * curr_len
                        if denom > 0.0:
                            cos_angle = float(np.dot(prev_vec, curr_vec) / denom)
                            # Clamp numerically in case of tiny floating point excursions.
                            if cos_angle < -1.0:
                                cos_angle = -1.0
                            elif cos_angle > 1.0:
                                cos_angle = 1.0

                            if cos_angle < self.direction_change_cos_threshold:
                                # Direction changed significantly between frames
                                # sorted_frames[idx - 1] and sorted_frames[idx].
                                # Mark both as keyframes so that CVAT does not
                                # interpolate a straight path across a sharp turn.
                                keyframes_for_track.add(sorted_frames[idx - 1])
                                keyframes_for_track.add(sorted_frames[idx])

                prev_vec = curr_vec

            track_keyframes[track_id] = keyframes_for_track

        items: List[Dict[str, Any]] = []

        for frame_id, frame_index in enumerate(sorted_frame_indices):
            frame_entry = self._frames[frame_index]
            meta = frame_entry["meta"]
            annotations = frame_entry["annotations"]

            item_annotations: List[Dict[str, Any]] = []
            for ann_id, ann in enumerate(annotations):
                label = ann["label"]
                label_id = label_to_id.get(label, 0)
                track_id = ann["track_id"]
                keyframe = False
                if track_id in track_keyframes:
                    keyframe = frame_id in track_keyframes[track_id]
                item_annotations.append(
                    {
                        "id": ann_id,
                        "type": "cuboid_3d",
                        "attributes": {
                            "track_id": track_id,
                            "keyframe": keyframe,
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

        return {
            "info": {},
            "categories": categories,
            "items": items,
        }

    def build_2d_tracking_history(self) -> Dict[str, Any]:
        """
        Build complete per-object 2D tracking history across processed frames.

        Output includes, for every tracked object and every frame in the batch:
        - whether the object appears in that frame
        - bbox corners (x1, y1, x2, y2) when present
        - sample index for frame-level traceability
        """
        frame_indices = sorted(self._frame_tracking_2d.keys())
        frames_summary = [self._frame_tracking_2d[frame_index] for frame_index in frame_indices]

        object_frames: Dict[int, List[Dict[str, Any]]] = {}
        object_labels: Dict[int, str] = {}
        for scene_entry in frames_summary:
            frame_index = int(scene_entry["frame_index"])
            sample_index = scene_entry["sample_index"]
            for obj in scene_entry["objects"]:
                track_id = int(obj["track_id"])
                if track_id not in object_frames:
                    object_frames[track_id] = []
                object_frames[track_id].append(
                    {
                        "frame_index": frame_index,
                        "sample_index": sample_index,
                        "appears": bool(obj["appears"]),
                        "bbox_xyxy": obj["bbox_xyxy"],
                    }
                )
                if track_id not in object_labels:
                    object_labels[track_id] = str(obj["label"])

        objects: List[Dict[str, Any]] = []
        for tr in self._tracks:
            frames_for_track = object_frames.get(tr.track_id, [])
            objects.append(
                {
                    "track_id": tr.track_id,
                    "label": object_labels.get(tr.track_id, tr.label),
                    "tracking_begin_frame": tr.first_frame,
                    "tracking_end_frame": tr.last_frame,
                    "frames": frames_for_track,
                }
            )

        return {
            "summary": {
                "num_frames": len(frame_indices),
                "num_tracks": len(self._tracks),
            },
            "frames": frames_summary,
            "objects": objects,
        }

