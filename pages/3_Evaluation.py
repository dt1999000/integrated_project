"""
Evaluation Page
Evaluate detection results against ground truth (primarily for KITTI).
Supports both single-sample and batch processing modes.
"""
import streamlit as st
import numpy as np
import pandas as pd
import cv2
import io
import json
import copy
import importlib.util
from pathlib import Path
from typing import Any, List, Dict, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
import plotly.graph_objects as go

from components.core.evaluation import (
    compute_3d_iou,
    compute_batch_statistics,
    compute_batch_azimuth_bin_metrics,
    compute_batch_ap,
    compute_kitti_difficulty_ap,
    compute_frame_metrics_at_iou,
    greedy_iou_match,
    _normalize_gt_cuboids,
)
from components.utils.visualization_helper import (
    draw_2d_boxes_on_image,
    draw_projected_cuboid_bboxes,
    create_comparison_plot,
    create_evaluation_mask_wireframe_figure,
    generate_distinct_colors,
    overlay_masks_on_image,
    render_point_cloud_plot as shared_render_point_cloud_plot,
)


def _canonical_eval_category(category: Any) -> str:
    """Canonicalize category labels for evaluation-time class matching."""
    label = str(category).strip()
    if label.lower() in {"person", "pedestrian"}:
        return "pedestrian"
    return label


def _apply_eval_category_aliases(batch_results: List[Dict]) -> List[Dict]:
    """
    Apply evaluation-time class aliases so equivalent labels are matched together.
    """
    aliased_results: List[Dict] = []
    for sample in batch_results:
        sample_copy = dict(sample)

        gt_raw = sample_copy.get("ground_truth_cuboids")
        if gt_raw is not None:
            gt_aliased: List[Dict] = []
            for gt in gt_raw or []:
                gt_copy = dict(gt)
                resolved = _canonical_eval_category(
                    gt_copy.get("category", gt_copy.get("class", "Unknown"))
                )
                gt_copy["category"] = resolved
                gt_aliased.append(gt_copy)
            sample_copy["ground_truth_cuboids"] = gt_aliased

        det_raw = sample_copy.get("detected_cuboids")
        if det_raw is not None:
            det_aliased: List[Dict] = []
            for det in det_raw or []:
                det_copy = dict(det)
                det_copy["category"] = _canonical_eval_category(
                    det_copy.get("category", "Unknown")
                )
                det_aliased.append(det_copy)
            sample_copy["detected_cuboids"] = det_aliased

        aliased_results.append(sample_copy)

    return aliased_results


def _normalize_gt_for_eval(raw_ground_truth: List[Dict]) -> List[Dict]:
    """Normalize mixed GT formats to min/max cuboid format used by evaluation helpers."""
    normalized = []
    for gt in raw_ground_truth:
        if all(k in gt for k in ["min_x", "min_y", "min_z", "max_x", "max_y", "max_z"]):
            normalized.append(gt)
            continue

        translation = gt.get("translation")
        size = gt.get("size")
        if translation is None or size is None or len(translation) != 3 or len(size) != 3:
            continue

        center = np.asarray(translation, dtype=np.float64)
        size_np = np.asarray(size, dtype=np.float64)
        half_size = size_np / 2.0
        bbox_min = center - half_size
        bbox_max = center + half_size

        norm_gt = dict(gt)
        norm_gt["min_x"] = float(bbox_min[0])
        norm_gt["min_y"] = float(bbox_min[1])
        norm_gt["min_z"] = float(bbox_min[2])
        norm_gt["max_x"] = float(bbox_max[0])
        norm_gt["max_y"] = float(bbox_max[1])
        norm_gt["max_z"] = float(bbox_max[2])
        norm_gt["category"] = _canonical_eval_category(
            norm_gt.get("category", norm_gt.get("class", "Person"))
        )
        normalized.append(norm_gt)
    return normalized


def _to_gt_table_rows(ground_truth_boxes: List[Dict]) -> List[Dict]:
    rows = []
    for idx, gt in enumerate(ground_truth_boxes):
        rows.append({
            "GT Index": idx,
            "Category": gt.get("category", gt.get("class", "Unknown")),
            "Track ID": gt.get("track_id", ""),
            "Center": gt.get("translation", ""),
            "Size": gt.get("size", ""),
            "Points in Box": gt.get("num_points", ""),
            "Has 2D BBox": gt.get("bbox_2d") is not None,
        })
    return rows


def get_ground_truth_objects_for_class_per_scene(
    batch_results: List[Dict],
    target_class: str,
) -> Dict[str, List[Dict]]:
    """
    Collect ground truth cuboids for a specific class for each scene.

    Returns a mapping from sample/scene identifier to the list of GT cuboids
    whose category matches `target_class`.  GT cuboids are normalized via
    `_normalize_gt_cuboids` to ensure a consistent min/max format.
    """
    per_scene_gt: Dict[str, List[Dict]] = {}
    for frame_idx, sample in enumerate(batch_results):
        raw_gt = sample.get("ground_truth_cuboids")
        if raw_gt is None:
            continue

        gt_all = _normalize_gt_cuboids(raw_gt or [])
        gt_cls = [
            g
            for g in gt_all
            if _canonical_eval_category(g.get("category", g.get("class", "Unknown")))
            == _canonical_eval_category(target_class)
        ]

        sample_id = str(sample.get("metadata", {}).get("sample_index", frame_idx))
        per_scene_gt[sample_id] = gt_cls

    return per_scene_gt


def _extract_detection_classes(
    batch_results: List[Dict], max_classes: Optional[int] = None
) -> List[str]:
    """
    Extract target detection classes from batch metadata (pipeline params),
    falling back to detected cuboid categories.
    """
    ordered: List[str] = []
    seen = set()

    for sample in batch_results:
        params = sample.get("metadata", {}).get("pipeline_params", {}) or {}
        class_names = params.get("class_names", []) or []
        for cls in class_names:
            cls_norm = _canonical_eval_category(cls)
            if cls_norm and cls_norm not in seen:
                ordered.append(cls_norm)
                seen.add(cls_norm)
                if max_classes is not None and len(ordered) >= max_classes:
                    return ordered

    for sample in batch_results:
        for det in sample.get("detected_cuboids", []) or []:
            cls_norm = _canonical_eval_category(det.get("category", det.get("class", "Unknown")))
            if cls_norm and cls_norm not in seen:
                ordered.append(cls_norm)
                seen.add(cls_norm)
                if max_classes is not None and len(ordered) >= max_classes:
                    return ordered

    return ordered


def _prepare_batch_results_for_eval(batch_results: List[Dict]) -> Tuple[List[Dict], int]:
    """
    Normalize batch samples for evaluation.

    For annotated datasets, missing GT is treated as an empty GT scene so that:
    - no detections => correct empty frame
    - any detections => false positives
    """
    annotated_datasets = {"kitti", "sim", "sunrgbd", "scannet"}
    prepared: List[Dict] = []
    n_inferred_empty = 0

    for sample in batch_results:
        sample_copy = dict(sample)
        metadata = dict(sample_copy.get("metadata", {}) or {})
        dataset_type = str(metadata.get("dataset_type", "")).lower()
        gt_cuboids = sample_copy.get("ground_truth_cuboids")

        if gt_cuboids is None and dataset_type in annotated_datasets:
            sample_copy["ground_truth_cuboids"] = []
            metadata["_gt_inferred_empty"] = True
            n_inferred_empty += 1
        else:
            metadata["_gt_inferred_empty"] = bool(metadata.get("_gt_inferred_empty", False))

        sample_copy["metadata"] = metadata
        prepared.append(sample_copy)

    return prepared, n_inferred_empty


def _batch_sample_stub_from_import(sample: Dict, frame_ordinal: int) -> Dict:
    """Synthetic batch-queue row so batch metrics queues match uploaded exports."""
    meta = dict(sample.get("metadata") or {})
    ix = meta.get("sample_index")
    return {
        "sample_index": ix if ix is not None else frame_ordinal,
        "dataset_path": meta.get("dataset_path", ""),
        "dataset_type": meta.get("dataset_type", ""),
        "image_path": meta.get("image_path", ""),
        "point_cloud_path": meta.get("point_cloud_path", ""),
    }


def _parse_detection_export_document(
    data: Any,
    source_name: str,
) -> Tuple[List[Dict], bool, Optional[str]]:
    """
    Parse Export-page compatible JSON into sample dicts.

    Returns ``(samples, batch_tracking_flag_from_envelope, error_message)``.
    Supported shapes:
      - Batch envelope ``{"samples": [...], optional batch_tracking_enabled}``
      - Single ``det3d_*.json`` object (metadata + detected_cuboids)
      - JSON array of sample objects ``[{...}, {...}]``

    Datumaro ``items`` exports are rejected with guidance to use cuboid JSON.
    """
    bte_flag = False
    if isinstance(data, list):
        samples_out: List[Dict] = []
        for elem in data:
            if isinstance(elem, dict) and isinstance(elem.get("detected_cuboids"), list):
                samples_out.append(dict(elem))
            else:
                return [], False, f"{source_name}: array elements must be sample objects with `detected_cuboids`"
        return samples_out, False, None
    if not isinstance(data, dict):
        return [], False, f"{source_name}: root must be JSON object or array"

    datumaro_like = isinstance(data.get("items"), list) and isinstance(data.get("categories"), dict)
    if datumaro_like:
        return (
            [],
            False,
            "Detected Datumaro/CVAT-style JSON (has `items` + `categories`). "
            "For evaluation please upload **Save 3D cuboids to JSON** output (`det3d_*.json`) "
            "or a batch `{ \"samples\": [...] }` payload from detection runs.",
        )
    samples_field = data.get("samples")
    if isinstance(samples_field, list):
        bte_flag = bool(data.get("batch_tracking_enabled"))
        out_samples: List[Dict] = []
        for elem in samples_field:
            if isinstance(elem, dict) and isinstance(elem.get("detected_cuboids"), list):
                out_samples.append(dict(elem))
            else:
                return (
                    [],
                    False,
                    f"{source_name}: invalid `samples` entry (needs `detected_cuboids` list)",
                )
        return out_samples, bte_flag, None
    det_list = data.get("detected_cuboids")
    meta = data.get("metadata")
    if isinstance(det_list, list) and isinstance(meta, dict):
        return [dict(data)], False, None
    return [], False, f"{source_name}: unrecognized layout (need `samples`, or `metadata` + `detected_cuboids`)"


def _merge_uploaded_detection_files(uploaded_files: List[Any]) -> Tuple[List[Dict], bool, List[str]]:
    """Merge uploaded JSON exports into one sample list."""
    merged: List[Dict] = []
    any_bte = False
    errs: List[str] = []
    for uf in uploaded_files:
        name = uf.name or "upload.json"
        raw = uf.read()
        uf.seek(0)
        decoded = raw.decode("utf-8").strip()
        if not decoded:
            errs.append(f"{name}: empty file")
            continue
        parsed = json.loads(decoded)
        chunk, chunk_bte, err = _parse_detection_export_document(parsed, name)
        any_bte = any_bte or chunk_bte
        if err is not None:
            errs.append(err)
            continue
        merged.extend(chunk)
    return merged, any_bte, errs


def _compute_per_class_per_frame_tables(
    batch_results: List[Dict],
    eval_classes: List[str],
) -> Dict[str, Dict[str, List[Dict]]]:
    """
    Build per-class, per-frame metrics at IoU 0.5 and 0.25.
    """
    out: Dict[str, Dict[str, List[Dict]]] = {}
    for cls in eval_classes:
        rows_50: List[Dict] = []
        rows_25: List[Dict] = []
        for frame_idx, sample in enumerate(batch_results):
            gt_all = _normalize_gt_cuboids(sample.get("ground_truth_cuboids", []) or [])
            det_all = sample.get("detected_cuboids", []) or []

            gt_cls = [
                g
                for g in gt_all
                if _canonical_eval_category(g.get("category", g.get("class", "Unknown")))
                == _canonical_eval_category(cls)
            ]
            det_cls = [
                d
                for d in det_all
                if _canonical_eval_category(d.get("category", d.get("class", "Unknown")))
                == _canonical_eval_category(cls)
            ]

            m50 = compute_frame_metrics_at_iou(
                gt_cls, det_cls, iou_threshold=0.5, match_by_category=True
            )
            m25 = compute_frame_metrics_at_iou(
                gt_cls, det_cls, iou_threshold=0.25, match_by_category=True
            )
            sample_id = sample.get("metadata", {}).get("sample_index", str(frame_idx))

            rows_50.append({
                "Frame": frame_idx,
                "Sample": sample_id,
                "Class": cls,
                "GT": m50["n_gt"],
                "Det": m50["n_det"],
                "TP": m50["TP"],
                "FP": m50["FP"],
                "FN": m50["FN"],
                "Precision": f"{m50['precision'] * 100:.1f}%",
                "Recall": f"{m50['recall'] * 100:.1f}%",
                "F1": f"{m50['f1'] * 100:.1f}%",
            })
            rows_25.append({
                "Frame": frame_idx,
                "Sample": sample_id,
                "Class": cls,
                "GT": m25["n_gt"],
                "Det": m25["n_det"],
                "TP": m25["TP"],
                "FP": m25["FP"],
                "FN": m25["FN"],
                "Precision": f"{m25['precision'] * 100:.1f}%",
                "Recall": f"{m25['recall'] * 100:.1f}%",
                "F1": f"{m25['f1'] * 100:.1f}%",
            })

        out[cls] = {"iou_50": rows_50, "iou_25": rows_25}
    return out


def _sum_per_class_across_frames(
    class_tables: Dict[str, Dict[str, List[Dict]]],
    iou_key: str = "iou_50",
) -> List[Dict]:
    """Aggregate per-class per-frame rows into one batch-level row per class."""
    rows: List[Dict] = []
    for cls, payload in class_tables.items():
        frame_rows = payload.get(iou_key, []) or []
        gt_sum = int(sum(int(r.get("GT", 0)) for r in frame_rows))
        det_sum = int(sum(int(r.get("Det", 0)) for r in frame_rows))
        tp_sum = int(sum(int(r.get("TP", 0)) for r in frame_rows))
        fp_sum = int(sum(int(r.get("FP", 0)) for r in frame_rows))
        fn_sum = int(sum(int(r.get("FN", 0)) for r in frame_rows))
        prec = float(tp_sum / (tp_sum + fp_sum)) if (tp_sum + fp_sum) > 0 else 0.0
        rec = float(tp_sum / (tp_sum + fn_sum)) if (tp_sum + fn_sum) > 0 else 0.0
        f1 = float((2.0 * prec * rec) / (prec + rec)) if (prec + rec) > 0.0 else 0.0
        rows.append(
            {
                "Class": cls,
                "Scenes": len(frame_rows),
                "GT (sum)": gt_sum,
                "Det (sum)": det_sum,
                "TP (sum)": tp_sum,
                "FP (sum)": fp_sum,
                "FN (sum)": fn_sum,
                "Precision": f"{prec * 100:.1f}%",
                "Recall": f"{rec * 100:.1f}%",
                "F1": f"{f1 * 100:.1f}%",
            }
        )
    return rows


def _compute_kitti_difficulty_ap_tables(batch_results: List[Dict]) -> Dict[str, pd.DataFrame]:
    """
    Compute per-class KITTI AP tables for Easy/Moderate/Hard.
    AP is shown both as PR-AUC and 11-point interpolated AP (R11).
    """
    tables: Dict[str, pd.DataFrame] = {}
    for iou_value in (0.5, 0.25):
        diff_stats = compute_kitti_difficulty_ap(
            batch_results,
            iou_threshold=float(iou_value),
            match_by_category=True,
        )
        per_diff = diff_stats.get("per_difficulty_per_class", {}) or {}
        rows: List[Dict] = []
        for difficulty_label in ("easy", "moderate", "hard"):
            per_class = per_diff.get(difficulty_label, {}) or {}
            all_classes = sorted(per_class.keys())
            for cls in all_classes:
                cls_stats = per_class.get(cls, {}) or {}
                rows.append(
                    {
                        "Difficulty": difficulty_label.capitalize(),
                        "Class": cls,
                        "GT count": int(cls_stats.get("n_gt", 0)),
                        "AP (PR AUC)": f"{float(cls_stats.get('ap_pr', 0.0)) * 100:.1f}%",
                        "AP_R11 (11-point)": f"{float(cls_stats.get('ap_r11', 0.0)) * 100:.1f}%",
                    }
                )
        tables[f"iou_{str(iou_value).replace('.', '')}"] = pd.DataFrame(rows)
    return tables


def _json_sanitize(obj: Any) -> Any:
    """Recursively convert values to JSON-serializable Python types (e.g. numpy)."""
    if obj is None:
        return None
    mod = str(type(obj).__module__)
    name = type(obj).__name__
    if name == "ndarray" and mod.startswith("numpy"):
        return _json_sanitize(obj.tolist())
    if isinstance(obj, np.ndarray):
        return _json_sanitize(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    shape = getattr(obj, "shape", None)
    tolist = getattr(obj, "tolist", None)
    if (
        shape is not None
        and callable(tolist)
        and getattr(obj, "__array__", None) is not None
        and not isinstance(obj, (str, bytes, bytearray, dict, list, tuple, set, frozenset))
    ):
        return _json_sanitize(obj.tolist())
    if isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (bytes, bytearray)):
        return obj.decode("utf-8", errors="replace")
    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


class _NumpySafeJSONEncoder(json.JSONEncoder):
    """
    JSON encoder that never falls through to the stdlib base ``default`` for
    numpy / array-like values (including ndarrays from a different numpy build
    where ``isinstance(o, np.ndarray)`` can be false).
    """

    def default(self, o: Any) -> Any:
        mod = str(type(o).__module__)
        name = type(o).__name__
        if name == "ndarray" and mod.startswith("numpy"):
            return o.tolist()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.generic):
            return o.item()
        shape = getattr(o, "shape", None)
        tolist = getattr(o, "tolist", None)
        if (
            shape is not None
            and callable(tolist)
            and getattr(o, "__array__", None) is not None
            and not isinstance(o, (str, bytes, bytearray, dict, list, tuple, set, frozenset))
        ):
            return tolist()
        if isinstance(o, (bytes, bytearray)):
            return o.decode("utf-8", errors="replace")
        if isinstance(o, Path):
            return str(o)
        return str(o)


def _cuboid_position_payload(cuboid: Dict) -> Dict:
    min_x = float(cuboid.get("min_x", 0.0))
    min_y = float(cuboid.get("min_y", 0.0))
    min_z = float(cuboid.get("min_z", 0.0))
    max_x = float(cuboid.get("max_x", 0.0))
    max_y = float(cuboid.get("max_y", 0.0))
    max_z = float(cuboid.get("max_z", 0.0))
    return {
        "min": [min_x, min_y, min_z],
        "max": [max_x, max_y, max_z],
        "center": [
            (min_x + max_x) / 2.0,
            (min_y + max_y) / 2.0,
            (min_z + max_z) / 2.0,
        ],
        "size": [
            max_x - min_x,
            max_y - min_y,
            max_z - min_z,
        ],
    }


def _cuboid_center_xyz(cuboid: Dict) -> np.ndarray:
    return np.asarray(
        [
            (float(cuboid.get("min_x", 0.0)) + float(cuboid.get("max_x", 0.0))) / 2.0,
            (float(cuboid.get("min_y", 0.0)) + float(cuboid.get("max_y", 0.0))) / 2.0,
            (float(cuboid.get("min_z", 0.0)) + float(cuboid.get("max_z", 0.0))) / 2.0,
        ],
        dtype=np.float64,
    )


def _trim_far_ground_truth_by_mask_capacity(
    batch_results: List[Dict],
    max_masks_per_image: int,
) -> Tuple[List[Dict], int, int]:
    """
    Keep at most `max_masks_per_image` GT boxes per frame by dropping GT cuboids
    farthest from detections (based on nearest detection center distance).
    """
    if max_masks_per_image <= 0:
        return batch_results, 0, 0

    trimmed_results: List[Dict] = []
    n_dropped = 0
    n_frames_affected = 0

    for sample in batch_results:
        gt_raw = sample.get("ground_truth_cuboids")
        if gt_raw is None:
            trimmed_results.append(sample)
            continue

        gt_pairs: List[Tuple[int, Dict]] = []
        for raw_idx, raw_gt in enumerate(gt_raw or []):
            normalized = _normalize_gt_for_eval([raw_gt])
            if normalized:
                gt_pairs.append((raw_idx, normalized[0]))
        gt_norm = [p[1] for p in gt_pairs]
        det_list = sample.get("detected_cuboids", []) or []
        if len(det_list) == 0:
            dropped_here = len(gt_norm)
            if dropped_here > 0:
                n_dropped += dropped_here
                n_frames_affected += 1
            sample_copy = dict(sample)
            sample_copy["ground_truth_cuboids"] = []
            trimmed_results.append(sample_copy)
            continue

        if len(gt_norm) <= max_masks_per_image:
            trimmed_results.append(sample)
            continue

        det_centers = np.asarray([_cuboid_center_xyz(det) for det in det_list], dtype=np.float64)
        gt_distance_rows: List[Tuple[float, int]] = []
        for gi, gt in enumerate(gt_norm):
            gt_center = _cuboid_center_xyz(gt)
            nearest = float(np.min(np.linalg.norm(det_centers - gt_center, axis=1)))
            gt_distance_rows.append((nearest, gi))

        keep_indices = {
            gi for _, gi in sorted(gt_distance_rows, key=lambda row: row[0])[:max_masks_per_image]
        }
        keep_raw_indices = {gt_pairs[gi][0] for gi in keep_indices}
        gt_trimmed = [gt for gi, gt in enumerate(gt_raw or []) if gi in keep_raw_indices]
        dropped_here = len(gt_norm) - len(gt_trimmed)
        if dropped_here > 0:
            n_dropped += dropped_here
            n_frames_affected += 1

        sample_copy = dict(sample)
        sample_copy["ground_truth_cuboids"] = gt_trimmed
        trimmed_results.append(sample_copy)

    return trimmed_results, n_dropped, n_frames_affected


def _build_mismatch_export_payload(
    batch_results: List[Dict],
    iou_threshold: float,
    match_by_category: bool = True,
) -> Dict:
    export_items: List[Dict] = []
    total_fn = 0
    total_fp = 0

    for frame_idx, sample in enumerate(batch_results):
        gt_raw = sample.get("ground_truth_cuboids")
        if gt_raw is None:
            continue

        gt_raw = gt_raw or []
        gt_pairs: List[Tuple[int, Dict, Dict]] = []
        for raw_idx, raw_gt in enumerate(gt_raw):
            normalized = _normalize_gt_for_eval([raw_gt])
            if normalized:
                gt_pairs.append((raw_idx, raw_gt, normalized[0]))

        gt_norm = [p[2] for p in gt_pairs]
        det = sample.get("detected_cuboids", []) or []
        _, unmatched_gt, unmatched_det = greedy_iou_match(
            gt_norm,
            det,
            iou_threshold=iou_threshold,
            match_by_category=match_by_category,
        )

        sample_meta = sample.get("metadata", {}) or {}
        sample_record = {
            "frame_index": frame_idx,
            "sample_index": sample_meta.get("sample_index", frame_idx),
            "dataset_type": sample_meta.get("dataset_type", ""),
            "dataset_path": sample_meta.get("dataset_path", ""),
            "false_negatives": [],
            "false_positives": [],
        }

        for gt_idx in sorted(unmatched_gt):
            raw_idx, raw_gt, norm_gt = gt_pairs[gt_idx]
            sample_record["false_negatives"].append({
                "gt_raw_index": raw_idx,
                "category": norm_gt.get("category", norm_gt.get("class", "Unknown")),
                "position": _cuboid_position_payload(norm_gt),
                "annotation_loaded_format": raw_gt,
            })
            total_fn += 1

        for det_idx in sorted(unmatched_det):
            det_box = det[det_idx]
            det_cat = det_box.get("category", "Unknown")

            best_iou_any = 0.0
            best_iou_same_class = 0.0
            best_gt_any = None
            best_gt_same_class = None
            for gi, gt_box in enumerate(gt_norm):
                iou = float(compute_3d_iou(gt_box, det_box))
                if iou > best_iou_any:
                    best_iou_any = iou
                    best_gt_any = gt_pairs[gi][0]
                gt_cat = gt_box.get("category", gt_box.get("class", "Unknown"))
                if gt_cat == det_cat and iou > best_iou_same_class:
                    best_iou_same_class = iou
                    best_gt_same_class = gt_pairs[gi][0]

            sample_record["false_positives"].append({
                "det_raw_index": det_idx,
                "category": det_cat,
                "position": _cuboid_position_payload(det_box),
                "best_iou_any_gt": best_iou_any,
                "best_iou_same_class_gt": best_iou_same_class,
                "best_gt_raw_index_any": best_gt_any,
                "best_gt_raw_index_same_class": best_gt_same_class,
                "annotation_loaded_format": det_box,
            })
            total_fp += 1

        if sample_record["false_negatives"] or sample_record["false_positives"]:
            export_items.append(sample_record)

    return {
        "format_version": "mismatch_export_v1",
        "iou_threshold": iou_threshold,
        "match_by_category": match_by_category,
        "summary": {
            "samples_with_mismatch": len(export_items),
            "false_negatives": total_fn,
            "false_positives": total_fp,
        },
        "samples": export_items,
    }


def _render_point_cloud_plot(
    fig,
    export_basename: str,
    *,
    show_legend: bool = True,
    use_container_width: bool = False,
) -> None:
    """Thin wrapper over shared point-cloud renderer."""
    shared_render_point_cloud_plot(
        fig=fig,
        export_basename=export_basename,
        use_container_width=use_container_width,
        show_legend=show_legend,
    )


# ---------------------------------------------------------------------------
# Single-sample evaluation (existing logic)
# ---------------------------------------------------------------------------

def _render_single_sample_eval():
    """Render the per-sample evaluation panel."""
    sample = st.session_state.sample
    sample_meta_data = sample['sample_meta_data']
    detected_cuboids = st.session_state.cuboids

    ground_truth_boxes: List[Dict] = []
    if 'export_results' in st.session_state and 'ground_truth_cuboids' in st.session_state.export_results:
        ground_truth_boxes = st.session_state.export_results['ground_truth_cuboids']
    elif 'ground_truth_annotations' in st.session_state:
        ground_truth_boxes = st.session_state.ground_truth_annotations
    else:
        ground_truth_boxes = sample_meta_data.get('ground_truth_boxes', [])

    ground_truth_boxes = _normalize_gt_for_eval(ground_truth_boxes)
    if not ground_truth_boxes:
        st.warning("⚠️ No ground truth boxes available for this sample.")
        st.info("Evaluation requires ground truth annotations from dataset extraction.")
        return

    with st.expander("📦 Ground Truth Annotations", expanded=False):
        st.dataframe(pd.DataFrame(_to_gt_table_rows(ground_truth_boxes)))

    point_cloud_obj = None
    if 'pipeline_state' in st.session_state:
        step_1_result = st.session_state.pipeline_state.get('step_1', {}).get('result')
        if step_1_result:
            point_cloud_obj = step_1_result.get('point_cloud_obj')

    st.subheader("📈 Detection Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Ground Truth Objects", len(ground_truth_boxes))
    with col2:
        st.metric("Detected Objects", len(detected_cuboids))
    with col3:
        detection_rate = len(detected_cuboids) / len(ground_truth_boxes) * 100 if ground_truth_boxes else 0
        st.metric("Detection Rate", f"{detection_rate:.1f}%")

    st.subheader("📊 3D IoU Matching Statistics")
    st.markdown("""
    **Matching Logic:** Each detected cuboid is matched to the ground truth box using `source_bbox_idx`
    which corresponds to the mask index matched to the bounding box.
    """)

    matching_results = []
    for detected in detected_cuboids:
        gt_idx = detected.get('source_bbox_idx')
        if gt_idx is not None and gt_idx < len(ground_truth_boxes):
            gt_box = ground_truth_boxes[gt_idx]
            iou_3d = compute_3d_iou(detected, gt_box)
            matching_results.append({
                'GT Index': gt_idx,
                'Category': detected.get('category', 'Unknown'),
                'GT Category': gt_box.get('category', 'Unknown'),
                '3D IoU': iou_3d,
                '2D IoU': detected.get('iou', None),
            })

    if matching_results:
        iou_3d_values = [r['3D IoU'] for r in matching_results]
        iou_2d_values = [r['2D IoU'] for r in matching_results if r['2D IoU'] is not None]

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Matched Pairs", len(matching_results))
        with col2:
            st.metric("Mean 3D IoU", f"{np.mean(iou_3d_values):.3f}")
        with col3:
            st.metric("Min 3D IoU", f"{np.min(iou_3d_values):.3f}")
        with col4:
            st.metric("Max 3D IoU", f"{np.max(iou_3d_values):.3f}")

        st.markdown("**Detection Quality by 3D IoU Threshold:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            n_above_50 = sum(1 for iou in iou_3d_values if iou >= 0.5)
            st.metric("IoU ≥ 0.5", f"{n_above_50}/{len(iou_3d_values)}")
        with col2:
            n_above_25 = sum(1 for iou in iou_3d_values if iou >= 0.25)
            st.metric("IoU ≥ 0.25", f"{n_above_25}/{len(iou_3d_values)}")
        with col3:
            n_above_10 = sum(1 for iou in iou_3d_values if iou >= 0.1)
            st.metric("IoU ≥ 0.1", f"{n_above_10}/{len(iou_3d_values)}")

        with st.expander("📋 Per-Object Matching Details", expanded=True):
            df_matching = pd.DataFrame(matching_results)
            df_matching['3D IoU'] = df_matching['3D IoU'].apply(lambda x: f"{x:.3f}")
            df_matching['2D IoU'] = df_matching['2D IoU'].apply(lambda x: f"{x:.3f}" if x is not None else "N/A")
            st.dataframe(df_matching)

    st.subheader("📷 2D Visualization")
    image = sample['image']

    gt_boxes_2d = [box for box in ground_truth_boxes if box.get("bbox_2d") is not None]
    if gt_boxes_2d:
        img_with_gt = draw_2d_boxes_on_image(image.copy(), gt_boxes_2d)
        st.image(img_with_gt, caption="Image with Ground Truth Boxes")
    else:
        st.info("No 2D GT bbox available for this sample.")

    cuboids_with_projection = [c for c in detected_cuboids if c.get("projected_bbox_2d") is not None]
    if cuboids_with_projection:
        st.subheader("📐 Reprojected Cuboid Bounding Boxes")
        img_proj = draw_projected_cuboid_bboxes(image.copy(), cuboids_with_projection, gt_boxes_2d)
        st.image(img_proj, caption="Reprojected 3D Cuboids to 2D")

    pipeline_state = st.session_state.get("pipeline_state")
    step_3_result = (
        pipeline_state.get("step_3", {}).get("result") if pipeline_state else None
    )
    sam_masks_eval = (
        (step_3_result or {}).get("sam_masks") or []
        if step_3_result
        else []
    )
    if sam_masks_eval:
        st.subheader("🎭 Segmentation Masks (same palette as Detection)")
        mask_colors_eval = generate_distinct_colors(len(sam_masks_eval))
        img_with_masks = overlay_masks_on_image(
            image.copy(), sam_masks_eval, mask_colors_eval, alpha=0.5
        )
        mask_bboxes_eval = (step_3_result or {}).get("mask_bboxes", []) or []
        detected_class_names_eval = (step_3_result or {}).get("class_names", []) or []
        confidences_eval = (step_3_result or {}).get("confidences", []) or []

        st.markdown("#### 2D Mask Visualization")
        fig_masks, ax_masks = plt.subplots(1, 1, figsize=(12, 8))
        ax_masks.imshow(img_with_masks)
        ax_masks.axis("off")
        for i, (bbox, class_name, confidence) in enumerate(
            zip(mask_bboxes_eval, detected_class_names_eval, confidences_eval)
        ):
            if bbox and len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                rect = patches.Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    linewidth=2,
                    edgecolor=mask_colors_eval[i],
                    facecolor="none",
                )
                ax_masks.add_patch(rect)
                label = (
                    f"{class_name}: {confidence:.2f}"
                    if confidence is not None
                    else class_name
                )
                ax_masks.text(
                    x1,
                    y1 - 5,
                    label,
                    color=mask_colors_eval[i],
                    fontsize=10,
                    bbox=dict(
                        boxstyle="round,pad=0.3", facecolor="black", alpha=0.7
                    ),
                )
        ax_masks.set_title(
            "Detected Objects with Masks, Bounding Boxes, and Confidence Scores"
        )
        st.pyplot(fig_masks)
        plt.close(fig_masks)

    if point_cloud_obj:
        st.subheader("🎯 3D comparison — dense point cloud + solid cuboids (GT vs detected)")
        st.caption(
            "Downloads as HTML with large axes/legend fonts for comparing against ground-truth cubes."
        )
        fig_unified = create_comparison_plot(point_cloud_obj, ground_truth_boxes, detected_cuboids)
        _render_point_cloud_plot(fig_unified, "evaluation_3d_comparison")

        step_2_result = (
            pipeline_state.get("step_2", {}).get("result") if pipeline_state else None
        )
        step_4_result = (
            pipeline_state.get("step_4", {}).get("result") if pipeline_state else None
        )
        sparse_eval = (
            (step_2_result or {}).get("colored_sparse_points")
            if step_2_result
            else None
        )
        assign_eval = (
            (step_3_result or {}).get("mask_assignments")
            if step_3_result
            else None
        )
        best_ix_eval = (
            (step_4_result or {}).get("best_cluster_sparse_indices")
            if step_4_result
            else None
        )

        if (
            sparse_eval is not None
            and len(sparse_eval) > 0
            and assign_eval is not None
            and len(assign_eval) == len(sparse_eval)
            and sam_masks_eval
        ):
            st.subheader("🎯 3D comparison — cuboid edges + mask-colored sparse depth")
            st.caption(
                "Detected boxes as wireframes (mask color); sparse points: best-cluster emphasis, "
                "other in-mask points fainter; points outside masks in light grey. "
                "Export HTML separately to compare against the dense view above."
            )
            wf_colors = generate_distinct_colors(len(sam_masks_eval))
            fig_wire = create_evaluation_mask_wireframe_figure(
                sparse_points=np.asarray(sparse_eval, dtype=np.float64),
                mask_assignments=np.asarray(assign_eval).reshape(-1),
                detected_cuboids=detected_cuboids,
                mask_colors=wf_colors,
                best_cluster_sparse_indices=best_ix_eval,
            )
            _render_point_cloud_plot(fig_wire, "evaluation_3d_mask_wireframe_sparse")
        elif pipeline_state:
            with st.expander("ℹ️ Mask / wireframe 3D view unavailable", expanded=False):
                reasons = []
                if sparse_eval is None or len(sparse_eval) == 0:
                    reasons.append("Step 2 `colored_sparse_points` missing or empty.")
                if assign_eval is None:
                    reasons.append("Step 3 `mask_assignments` missing.")
                elif sparse_eval is not None and (
                    len(assign_eval) != len(sparse_eval)
                ):
                    reasons.append(
                        "`mask_assignments` length does not match sparse points (re-run Detection steps 2–4)."
                    )
                if not sam_masks_eval:
                    reasons.append("No `sam_masks` from Step 3.")
                if reasons:
                    for r in reasons:
                        st.markdown(f"- {r}")
                else:
                    st.markdown("- Unknown reason.")


def _render_sim_azimuth_iou_block(
    sim_only: List[Dict],
    iou_threshold: float,
    n_bins: int,
    angular_mode: str,
    angle_offset_deg: float,
    angle_span_deg: float,
    file_tag: str,
    title_md: str,
) -> None:
    """Table + bar chart + figure download for one IoU threshold (used in Sim azimuth section)."""
    st.markdown(title_md)
    bin_stats = compute_batch_azimuth_bin_metrics(
        sim_only,
        iou_threshold=float(iou_threshold),
        n_bins=int(n_bins),
        match_by_category=True,
        angular_mode=angular_mode,
        angle_offset_deg=float(angle_offset_deg),
        angle_span_deg=float(angle_span_deg),
    )
    bin_rows = []
    for b in bin_stats["bins"]:
        bin_rows.append({
            "Bin": b["label"],
            "TP": b["TP"],
            "FP": b["FP"],
            "FN": b["FN"],
            "Precision": f"{b['precision'] * 100:.1f}%",
        })
    st.dataframe(pd.DataFrame(bin_rows), width="stretch")
    prec_vals = [b["precision"] * 100.0 for b in bin_stats["bins"]]
    labels = [b["label"] for b in bin_stats["bins"]]
    fig_az, ax_az = plt.subplots(figsize=(10, 4), dpi=120)
    xpos = np.arange(len(labels))
    ax_az.bar(xpos, prec_vals, color="#4C78A8")
    ax_az.set_xticks(xpos)
    ax_az.set_xticklabels(labels, rotation=35, ha="right")
    ax_az.set_ylabel("Precision (%)")
    ax_az.set_xlabel("Azimuth bin")
    ax_az.set_title(
        f"Sim — precision by bin (IoU ≥ {bin_stats['iou_threshold']}; "
        f"mode={bin_stats['angular_mode']}, span={bin_stats['binned_span_deg']:.0f}°)"
    )
    ax_az.set_ylim(0, 105)
    ax_az.grid(True, axis="y", linestyle="--", alpha=0.35)
    st.pyplot(fig_az, width="stretch")
    _make_download_buttons(fig_az, file_tag)


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

def _render_batch_eval():
    """Render the batch evaluation panel with IoU-threshold F1/precision metrics and per-frame stats."""
    batch_export = st.session_state.batch_export_results
    batch_results: List[Dict] = batch_export.get("samples", [])
    eval_batch_results, n_inferred_empty_gt = _prepare_batch_results_for_eval(batch_results)
    eval_batch_results = _apply_eval_category_aliases(eval_batch_results)
    max_masks_hint = int(st.session_state.get("eval_mask_capacity_max", 0) or 0)
    trim_gt_by_mask_capacity = bool(
        st.session_state.get("eval_trim_gt_by_mask_capacity_enabled", True)
    )
    if trim_gt_by_mask_capacity:
        eval_batch_results, n_trimmed_gt, n_trimmed_frames = _trim_far_ground_truth_by_mask_capacity(
            eval_batch_results,
            max_masks_per_image=max_masks_hint,
        )
    else:
        n_trimmed_gt = 0
        n_trimmed_frames = 0
    st.caption(
        "GT trim mode: "
        + (
            f"ON (mask capacity = {max_masks_hint})"
            if trim_gt_by_mask_capacity
            else "OFF (using original GT counts)"
        )
    )
    n_batch_loaded = len(batch_results)
    queued_from_runner = len(st.session_state.get("batch_samples", []))
    total_queued = max(queued_from_runner, n_batch_loaded)

    st.subheader("📦 Batch Sample Overview")

    n_processed = len(eval_batch_results)
    n_failed = max(0, total_queued - n_processed)
    # A sample is evaluable when the 'ground_truth_cuboids' key is present (even if the list
    # is empty — empty means a valid annotated scene with no objects, whose FP detections
    # must still count against precision).  A missing key means GT was not available at all.
    results_with_gt = [r for r in eval_batch_results if r.get("ground_truth_cuboids") is not None]
    n_evaluable = len(results_with_gt)
    n_no_gt = n_processed - n_evaluable
    # Frames that passed the filter but have zero GT annotations (empty scenes)
    n_empty_scenes = sum(
        1 for r in results_with_gt if len(r.get("ground_truth_cuboids") or []) == 0
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Queued", total_queued)
    with col2:
        st.metric(
            "Processed (kept)",
            n_processed,
            delta=None if total_queued == 0 else f"{n_processed / total_queued * 100:.0f}%",
        )
    with col3:
        st.metric("Failed / Skipped", n_failed, delta=f"-{n_failed}" if n_failed else None)
    with col4:
        st.metric("With GT (evaluable)", n_evaluable)

    if n_empty_scenes > 0:
        st.info(
            f"{n_empty_scenes} evaluable frame(s) have zero GT annotations (empty scenes). "
            "These frames are included in metric computation — all detections in them count as FP."
        )
    if n_inferred_empty_gt > 0:
        st.info(
            f"{n_inferred_empty_gt} frame(s) from annotated datasets had missing GT and were "
            "treated as empty scenes for evaluation."
        )
    if n_trimmed_gt > 0:
        st.info(
            f"Applied mask-capacity GT trimming on {n_trimmed_frames} frame(s): "
            f"dropped {n_trimmed_gt} far-away GT cuboid(s) using max masks per image = {max_masks_hint}."
        )

    if n_no_gt > 0:
        st.info(
            f"{n_no_gt} processed sample(s) have no GT source (e.g. rosbag / nuScenes) "
            "and are excluded from AP computation."
        )

    if n_evaluable == 0:
        st.warning(
            "⚠️ None of the processed samples contain ground truth cuboids. "
            "Evaluation metrics cannot be computed. Ground truth is available for KITTI and sim batches."
        )
        _render_batch_detection_only(eval_batch_results)
        return

    # ------------------------------------------------------------------
    # Compute statistics
    # ------------------------------------------------------------------
    omni3d_progress = st.progress(
        0,
        text="Omni3D class-agnostic mAP: preparing threshold sweep",
    )

    def _on_omni3d_progress(done: int, total: int, threshold: float) -> None:
        progress_ratio = float(done) / float(total) if total > 0 else 1.0
        omni3d_progress.progress(
            min(100, int(round(progress_ratio * 100.0))),
            text=(
                "Omni3D class-agnostic mAP: "
                f"{done}/{total} thresholds complete (latest IoU {threshold:.2f})"
            ),
        )

    with st.spinner("Computing batch metrics…"):
        stats = compute_batch_statistics(
            eval_batch_results,
            total_queued=max(total_queued, len(eval_batch_results)),
            match_by_category=True,
            omni3d_progress_callback=_on_omni3d_progress,
        )
    omni3d_progress.progress(100, text="Omni3D class-agnostic mAP: complete")

    ap_50 = stats["ap_50"]
    ap_25 = stats["ap_25"]

    # ------------------------------------------------------------------
    # Top-level AP metrics
    # ------------------------------------------------------------------
    st.subheader("🏆 Benchmark Metrics")
    st.caption(
        "**Heuristic-score mAP@0.5** / **mAP@0.25** are the mean of per-class AP, each AP equal to the area under the "
        "precision–recall curve (COCO-style interpolated envelope). Detections are ranked by "
        "the exported heuristic `confidence` score (computed before GT matching). "
        "**Macro-F1** is the unweighted mean of per-class F1 from greedy IoU matching at one threshold. "
        "**Micro-F1** is the harmonic mean of pooled micro precision and recall at that threshold."
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("mAP@0.5 (PR AUC)", f"{ap_50.get('map_pr', 0.0) * 100:.1f}%")
    with col2:
        st.metric("mAP@0.25 (PR AUC)", f"{ap_25.get('map_pr', 0.0) * 100:.1f}%")
    with col3:
        st.metric("Macro-F1 @0.5", f"{ap_50['macro_f1'] * 100:.1f}%")
    with col4:
        st.metric("Macro-F1 @0.25", f"{ap_25['macro_f1'] * 100:.1f}%")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Micro AP (PR, pooled) @0.5", f"{ap_50.get('micro_ap_pr', 0.0) * 100:.1f}%")
    with col2:
        st.metric("Micro AP (PR, pooled) @0.25", f"{ap_25.get('micro_ap_pr', 0.0) * 100:.1f}%")
    with col3:
        st.metric("Micro-F1 @0.5", f"{ap_50['f1'] * 100:.1f}%")
    with col4:
        st.metric("Micro-F1 @0.25", f"{ap_25['f1'] * 100:.1f}%")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("AP@11 (pooled) @0.5", f"{ap_50.get('micro_ap_r11', 0.0) * 100:.1f}%")
    with col2:
        st.metric("AP@11 (pooled) @0.25", f"{ap_25.get('micro_ap_r11', 0.0) * 100:.1f}%")
    with col3:
        st.metric("Micro precision @0.25", f"{ap_25['precision'] * 100:.1f}%")
    with col4:
        st.metric("Micro recall @0.25", f"{ap_25['recall'] * 100:.1f}%")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total GT objects", stats["total_ground_truth"])
    with col2:
        st.metric("Total detections", stats["total_detections"])

    omni3d_stats = stats.get("omni3d_class_agnostic", {}) or {}
    omni3d_ap_per_threshold = omni3d_stats.get("ap_per_threshold", {}) or {}
    if omni3d_ap_per_threshold:
        st.subheader("🥊 Omni3D-Style Class-Agnostic 3D mAP (secondary)")
        st.caption(
            "Secondary comparison metric aligned with Omni3D protocol: all labels are treated as one "
            "\"object\" class, AP is computed at IoU thresholds 0.05..0.50 (step 0.05), and mAP is "
            "the mean over those thresholds."
        )
        st.metric(
            "Class-agnostic mAP (IoU 0.05:0.50)",
            f"{float(omni3d_stats.get('map', 0.0)) * 100:.1f}%",
        )
        omni3d_rows = []
        for thr_str in sorted(omni3d_ap_per_threshold.keys(), key=lambda v: float(v)):
            omni3d_rows.append(
                {
                    "IoU threshold": thr_str,
                    "AP": f"{float(omni3d_ap_per_threshold.get(thr_str, 0.0)) * 100:.1f}%",
                }
            )
        st.dataframe(pd.DataFrame(omni3d_rows), width="stretch")

    st.subheader("📈 Precision-Recall Curves")
    _ds_slug = _batch_primary_dataset_slug(eval_batch_results)
    sun_note = ""
    if _ds_slug == "sunrgbd":
        sun_note = (
            "**SUN RGB-D**: micro-pooled PR is for sanity checks — paper **mAP@0.25 / mAP@0.5** "
            "is the mean of **per-class** AP in the benchmark table."
        )
    st.caption(
        "Separate figures per IoU threshold (micro-pooled, heuristic-score ranking). "
        "Plots end at **the attained recall** — the curve is not extrapolated horizontally to recall 1 "
        "(avoids misleading plateaus under `step` plotting). PNG/SVG per figure."
        + (f" {sun_note}" if sun_note else "")
    )
    _render_pr_curve_for_thesis(ap_25, ap_50)

    # ------------------------------------------------------------------
    # Per-class breakdown
    # ------------------------------------------------------------------
    eval_classes = _extract_detection_classes(eval_batch_results, max_classes=None)
    if eval_classes:
        st.caption(f"Evaluating configured detection classes: {', '.join(eval_classes)}")

    if ap_50["per_class"]:
        st.subheader("📊 Per-Class AP (reportable)")
        all_cats = sorted(set(list(ap_50["per_class"]) + list(ap_25["per_class"])))
        if eval_classes:
            all_cats = [c for c in all_cats if c in eval_classes]
        rows = []
        for cat in all_cats:
            ap_pr_50 = ap_50.get("per_class_pr_ap", {}).get(cat)
            ap_pr_25 = ap_25.get("per_class_pr_ap", {}).get(cat)
            rows.append({
                "Category": cat,
                "AP_PR@0.5": f"{ap_pr_50 * 100:.1f}%" if ap_pr_50 is not None else "—",
                "AP_PR@0.25": f"{ap_pr_25 * 100:.1f}%" if ap_pr_25 is not None else "—",
                "GT count": int(ap_25.get("n_gt_per_class_pr", {}).get(cat, 0)),
            })
        if rows:
            st.dataframe(pd.DataFrame(rows), width="stretch")
            _render_per_class_pr_curves(ap_25, ap_50, all_cats)
        else:
            st.info("No per-class metrics available for the configured detection classes.")

    dataset_slug = _batch_primary_dataset_slug(eval_batch_results)
    if dataset_slug == "kitti":
        st.subheader("🚗 KITTI Difficulty AP Tables")
        st.caption(
            "Per-class AP by KITTI difficulty (Easy/Moderate/Hard) using KITTI rules: "
            "Easy: h>=40px, occlusion<=0, truncation<=0.15; "
            "Moderate: h>=25px, occlusion<=1, truncation<=0.30; "
            "Hard: h>=25px, occlusion<=2, truncation<=0.50. "
            "Both PR-AUC AP and 11-point AP_R11 are shown."
        )
        kitti_tables = _compute_kitti_difficulty_ap_tables(eval_batch_results)
        st.markdown("**IoU >= 0.25**")
        st.dataframe(kitti_tables["iou_025"], width="stretch")
        st.markdown("**IoU >= 0.50**")
        st.dataframe(kitti_tables["iou_05"], width="stretch")

    st.subheader("⬇️ Export Mismatched Annotations")
    st.caption(
        "Exports false negatives (missed GT) and false positives (wrong detections) "
        "with sample ownership, GT/detection cuboid positions, and original annotation dicts."
    )
    ex1, ex2 = st.columns([1, 1])
    with ex1:
        mismatch_iou = st.selectbox(
            "Mismatch export IoU threshold",
            [0.5, 0.25],
            index=0,
            key="mismatch_export_iou",
        )
    with ex2:
        st.write("")
        st.write("")
    mismatch_payload = _build_mismatch_export_payload(
        results_with_gt,
        iou_threshold=float(mismatch_iou),
        match_by_category=True,
    )
    st.download_button(
        "⬇️ Download mismatched GT/detections (JSON)",
        data=json.dumps(
            _json_sanitize(mismatch_payload),
            indent=2,
            cls=_NumpySafeJSONEncoder,
        ),
        file_name=f"mismatched_annotations_iou_{str(mismatch_iou).replace('.', '_')}.json",
        mime="application/json",
        width="stretch",
    )

    # ------------------------------------------------------------------
    # Sim: azimuth (FOV) bin precision — study camera–LiDAR misalignment vs angle
    # ------------------------------------------------------------------
    results_with_gt = [r for r in eval_batch_results if r.get("ground_truth_cuboids") is not None]
    sim_only = [
        r for r in results_with_gt
        if str(r.get("metadata", {}).get("dataset_type", "")).lower() == "sim"
    ]
    if sim_only:
        st.subheader("🧭 Sim: precision by azimuth bin (LiDAR xy)")
        st.caption(
            "TP/FN use GT cuboid centers; FP uses detection centers. **Precision** (per bin) = "
            "TP / (TP + FP). Two IoU cutoffs (0.25 and 0.5) are shown in separate tabs — **0.25 first**, "
            "aligned with AP@0.25-style matching elsewhere."
        )
        ctrl1, ctrl2, ctrl3 = st.columns(3)
        with ctrl1:
            n_bins = st.number_input(
                "Number of azimuth bins",
                min_value=4,
                max_value=36,
                value=8,
                step=1,
                key="sim_n_bins",
            )
        with ctrl2:
            angular_mode = st.selectbox(
                "Angle layout",
                options=["rot_clip", "full360", "abs180"],
                format_func=lambda m: {
                    "rot_clip": "Rotated 0–180° window (offset + span; default sim FOV)",
                    "full360": "Full 360° (wrap, optional offset)",
                    "abs180": "Symmetric |atan2| on 0–180°",
                }[m],
                index=0,
                key="sim_azimuth_mode",
            )
        with ctrl3:
            st.write("")
            st.write("")
        if angular_mode == "rot_clip":
            off1, off2 = st.columns(2)
            with off1:
                angle_offset_deg = st.number_input(
                    "Rotate bins (°) subtracted after wrap to [0,360)",
                    min_value=0.0,
                    max_value=360.0,
                    value=90.0,
                    step=1.0,
                    key="sim_azimuth_offset",
                    help="Use offset 90° with span 180° to map the original 90°–270° sector into a 0°–180° axis.",
                )
            with off2:
                angle_span_deg = st.number_input(
                    "Binned span (°) after rotation",
                    min_value=30.0,
                    max_value=360.0,
                    value=180.0,
                    step=1.0,
                    key="sim_azimuth_span",
                )
        elif angular_mode == "full360":
            angle_offset_deg = st.number_input(
                "Rotate full circle (°)",
                min_value=0.0,
                max_value=360.0,
                value=0.0,
                step=1.0,
                key="sim_azimuth_offset_full",
            )
            angle_span_deg = 360.0
        else:
            angle_offset_deg = 0.0
            angle_span_deg = 180.0

        tab_az25, tab_az50 = st.tabs(["IoU ≥ 0.25 (AP@0.25-style)", "IoU ≥ 0.5 (AP@0.5-style)"])
        with tab_az25:
            _render_sim_azimuth_iou_block(
                sim_only,
                0.25,
                n_bins,
                angular_mode,
                angle_offset_deg,
                angle_span_deg,
                "sim_azimuth_precision_iou25",
                "**IoU threshold 0.25** — bin counts / precision use the same greedy 3D IoU matching as batch AP@0.25.",
            )
        with tab_az50:
            _render_sim_azimuth_iou_block(
                sim_only,
                0.5,
                n_bins,
                angular_mode,
                angle_offset_deg,
                angle_span_deg,
                "sim_azimuth_precision_iou50",
                "**IoU threshold 0.5** — same bins as the 0.25 tab; only TP/FP/FN assignment changes.",
            )

    # ------------------------------------------------------------------
    # Per-frame breakdown
    # ------------------------------------------------------------------
    with st.expander("📋 Per-Frame Metrics  (IoU ≥ 0.5)", expanded=False):
        frame_rows = []
        for fm in ap_50["per_frame_metrics"]:
            frame_rows.append({
                "Frame": fm["frame_index"],
                "Sample": fm["sample_index"],
                "GT": fm["n_gt"],
                "Det": fm["n_det"],
                "TP": fm["TP"],
                "FP": fm["FP"],
                "FN": fm["FN"],
                "Precision": f"{fm['precision'] * 100:.1f}%",
                "Recall": f"{fm['recall'] * 100:.1f}%",
                "F1": f"{fm['f1'] * 100:.1f}%",
            })
        if frame_rows:
            st.dataframe(pd.DataFrame(frame_rows), width="stretch")

    with st.expander("📋 Per-Frame Metrics  (IoU ≥ 0.25)", expanded=False):
        frame_rows_25 = []
        for fm in ap_25["per_frame_metrics"]:
            frame_rows_25.append({
                "Frame": fm["frame_index"],
                "Sample": fm["sample_index"],
                "GT": fm["n_gt"],
                "Det": fm["n_det"],
                "TP": fm["TP"],
                "FP": fm["FP"],
                "FN": fm["FN"],
                "Precision": f"{fm['precision'] * 100:.1f}%",
                "Recall": f"{fm['recall'] * 100:.1f}%",
                "F1": f"{fm['f1'] * 100:.1f}%",
            })
        if frame_rows_25:
            st.dataframe(pd.DataFrame(frame_rows_25), width="stretch")

    # ------------------------------------------------------------------
    # Per-class per-frame breakdown (requested for SUNRGBD / class-aware eval)
    # ------------------------------------------------------------------
    if eval_classes:
        st.subheader("🧪 Per-Class Per-Frame Metrics")
        st.caption(
            "Each table isolates a single configured detection class so you can "
            "inspect frame-by-frame TP/FP/FN against GT for that class only."
        )
        class_tables = _compute_per_class_per_frame_tables(eval_batch_results, eval_classes)
        st.markdown("**Batch-level sum across scenes (per class, IoU >= 0.25, primary)**")
        class_batch_rows_25 = _sum_per_class_across_frames(class_tables, iou_key="iou_25")
        if class_batch_rows_25:
            st.dataframe(pd.DataFrame(class_batch_rows_25), width="stretch")
        st.markdown("**Batch-level sum across scenes (per class, IoU >= 0.5, reference)**")
        class_batch_rows_50 = _sum_per_class_across_frames(class_tables, iou_key="iou_50")
        if class_batch_rows_50:
            st.dataframe(pd.DataFrame(class_batch_rows_50), width="stretch")
        class_tabs = st.tabs([f"Class: {cls}" for cls in eval_classes])
        for tab, cls in zip(class_tabs, eval_classes):
            with tab:
                st.markdown(f"**{cls} — IoU ≥ 0.25**")
                st.dataframe(pd.DataFrame(class_tables[cls]["iou_25"]), width="stretch")
                st.markdown(f"**{cls} — IoU ≥ 0.5**")
                st.dataframe(pd.DataFrame(class_tables[cls]["iou_50"]), width="stretch")

    # ------------------------------------------------------------------
    # Sample list with keep/skip status
    # ------------------------------------------------------------------
    with st.expander("🗂️ Sample Status (kept vs. failed)", expanded=False):
        queued_samples = st.session_state.get("batch_samples", [])
        processed_indices = {
            str(r.get("metadata", {}).get("sample_index", ""))
            for r in batch_results
        }
        status_rows = []
        for s in queued_samples:
            idx = str(s.get("sample_index", ""))
            matched = [
                r for r in eval_batch_results
                if str(r.get("metadata", {}).get("sample_index", "")) == idx
            ]
            if matched:
                gt_cuboids = matched[0].get("ground_truth_cuboids")
                has_gt = gt_cuboids is not None
                inferred_empty = bool(matched[0].get("metadata", {}).get("_gt_inferred_empty", False))
                n_det = len(matched[0].get("detected_cuboids", []))
                n_gt = len(gt_cuboids) if gt_cuboids else 0
                if not has_gt:
                    status = "✅ Processed (no GT source)"
                elif n_gt == 0:
                    if inferred_empty:
                        status = "✅ Processed + GT (empty scene, inferred)"
                    else:
                        status = "✅ Processed + GT (empty scene)"
                else:
                    status = f"✅ Processed + GT ({n_gt} obj)"
            else:
                n_det = 0
                n_gt = 0
                status = "❌ Failed / Skipped"
            status_rows.append({
                "Sample Index": idx,
                "Dataset Type": s.get("dataset_type", ""),
                "Status": status,
                "Detections": n_det,
                "GT Objects": n_gt,
            })
        if status_rows:
            st.dataframe(pd.DataFrame(status_rows), width="stretch")


def _render_batch_detection_only(batch_results: List[Dict]):
    """Show basic detection counts when GT is not available."""
    st.subheader("📈 Detection Summary (no GT)")
    rows = []
    for i, r in enumerate(batch_results):
        rows.append({
            "Frame": i,
            "Sample": r.get("metadata", {}).get("sample_index", str(i)),
            "Detections": len(r.get("detected_cuboids", [])),
        })
    st.dataframe(pd.DataFrame(rows), width="stretch")


def _infer_scene_bucket(sample_meta: Dict) -> str:
    dataset_type = str(sample_meta.get("dataset_type", "")).lower()
    if dataset_type in {"sunrgbd", "scannet"}:
        return "indoor"
    if dataset_type in {"kitti", "nuscenes", "rosbag", "waymo", "sim"}:
        return "outdoor"
    return "unknown"


def _compute_eval_stats_with_per_class(batch_results: List[Dict], total_queued: int) -> Tuple[Dict, List[Dict]]:
    stats = compute_batch_statistics(
        batch_results,
        total_queued=total_queued,
        match_by_category=False,
    )
    m25 = stats["ap_25"]
    m50 = stats["ap_50"]
    overall = {
        "macro_f1_25": m25["macro_f1"],
        "macro_f1_50": m50["macro_f1"],
        "f150": m50["f1"],
        "precision25": m25["precision"],
        "precision50": m50["precision"],
        "recall25": m25["recall"],
        "recall50": m50["recall"],
        "f125": m25["f1"],
        "n_samples": len(batch_results),
    }
    classes = sorted(set(m25.get("per_class", {}).keys()) | set(m50.get("per_class", {}).keys()))
    per_class_rows: List[Dict] = []
    for cls in classes:
        c25 = m25.get("per_class", {}).get(cls, {})
        c50 = m50.get("per_class", {}).get(cls, {})
        per_class_rows.append({
            "class": cls,
            "precision25": float(c25.get("precision", 0.0)),
            "recall25": float(c25.get("recall", 0.0)),
            "f125": float(c25.get("f1", 0.0)),
            "precision50": float(c50.get("precision", 0.0)),
            "recall50": float(c50.get("recall", 0.0)),
            "f150": float(c50.get("f1", 0.0)),
        })
    return overall, per_class_rows


@st.cache_resource
def _load_detection_page_module():
    page_path = Path(__file__).parent / "2_Detection.py"
    spec = importlib.util.spec_from_file_location("detection_page_for_eval_ablation", str(page_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_batch_with_params(
    detection_mod,
    batch_samples: List[Dict],
    params: Dict,
    preloaded_bbox_data: Optional[Dict],
) -> List[Dict]:
    st.session_state.params = params
    results: List[Dict] = []
    total = max(1, len(batch_samples))
    progress = st.progress(0.0)
    for i, sample_desc in enumerate(batch_samples):
        progress.progress((i + 1) / total)
        export_res = detection_mod._run_pipeline_for_batch_sample(
            dataset_path=sample_desc["dataset_path"],
            dataset_type=sample_desc.get("dataset_type", "kitti"),
            sample_index=sample_desc["sample_index"],
            tracker=None,
            frame_index=i,
            prev_image=None,
            preloaded_bbox_data=preloaded_bbox_data,
        )
        if export_res is not None:
            results.append(export_res)
    progress.empty()
    return results


def _make_download_buttons(fig: plt.Figure, base_name: str):
    png_buf = io.BytesIO()
    svg_buf = io.BytesIO()
    fig.savefig(png_buf, format="png", dpi=400, bbox_inches="tight")
    fig.savefig(svg_buf, format="svg", bbox_inches="tight")
    png_buf.seek(0)
    svg_buf.seek(0)
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "⬇️ Download PNG (400 DPI)",
            data=png_buf.getvalue(),
            file_name=f"{base_name}.png",
            mime="image/png",
            width="stretch",
        )
    with col2:
        st.download_button(
            "⬇️ Download SVG (vector)",
            data=svg_buf.getvalue(),
            file_name=f"{base_name}.svg",
            mime="image/svg+xml",
            width="stretch",
        )


def _batch_primary_dataset_slug(batch_results: List[Dict]) -> Optional[str]:
    """Most common ``dataset_type`` in batch exports (metadata or sample root key)."""
    if not batch_results:
        return None
    counts: Dict[str, int] = {}
    for r in batch_results:
        meta = r.get("metadata") or {}
        slug = meta.get("dataset_type") if meta.get("dataset_type") is not None else r.get("dataset_type")
        slug_s = str(slug).strip().lower() if slug is not None else "unknown"
        if slug_s == "":
            slug_s = "unknown"
        counts[slug_s] = counts.get(slug_s, 0) + 1
    return max(counts.keys(), key=lambda k: counts[k])


def _coco_style_precision_envelope_inplace(precision: np.ndarray) -> None:
    """Backward max envelope on precision (aligned with detector AP bookkeeping). Mutates precision."""
    p = precision
    if p.size <= 1:
        return
    for i in range(len(p) - 2, -1, -1):
        p[i] = float(max(float(p[i]), float(p[i + 1])))


def _micro_pr_curve_stair_xy(recalls: np.ndarray, precisions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a finite staircase polyline (x, y) for micro PR that does **not**
    extrapolate precision past the largest observed cumulative recall — unlike
    ``step(..., where='post')`` with ``xlim=(0,1)``, which extends the last bin to 1.0.

    Opens at recall 0 with the precision after the envelope at the first operating point.
    """
    if recalls.size == 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    r_raw = recalls.astype(np.float64)
    p_mono = precisions.astype(np.float64).copy()
    _coco_style_precision_envelope_inplace(p_mono)
    xr: List[float] = [0.0, float(r_raw[0])]
    yr: List[float] = [float(p_mono[0]), float(p_mono[0])]
    r_prev = float(r_raw[0])
    p_prev = float(p_mono[0])
    for i in range(1, len(r_raw)):
        ri = float(r_raw[i])
        pi = float(p_mono[i])
        if abs(ri - r_prev) < 1e-15:
            if abs(pi - p_prev) > 1e-15:
                xr.extend([ri, ri]); yr.extend([p_prev, pi]); p_prev = pi
            continue
        xr.extend([ri, ri]); yr.extend([p_prev, pi])
        r_prev, p_prev = ri, pi
    return np.asarray(xr, dtype=np.float64), np.asarray(yr, dtype=np.float64)


def _pr_fill_under_stair(ax, xr: np.ndarray, yr: np.ndarray, *, color: str, alpha: float = 0.10) -> None:
    """Closed polygon under staircase down to baseline 0."""
    if xr.size < 2:
        return
    r_last = float(xr[-1])
    verts = [(0.0, 0.0)]
    verts.extend([(float(rx), float(ry)) for rx, ry in zip(xr, yr)])
    verts.append((r_last, 0.0))
    ax.add_patch(
        Polygon(verts, closed=True, facecolor=color, edgecolor="none", alpha=alpha)
    )


def _render_single_pr_curve_figure(
    ap_bundle: Dict,
    *,
    iou_label: str,
    line_color: str,
    file_tag: str,
) -> None:
    """One PR figure ending at attained recall only; avoids horizontal tail artifacts."""
    rec = np.asarray(ap_bundle.get("pr_recalls_micro", []) or [], dtype=np.float64)
    prec = np.asarray(ap_bundle.get("pr_precisions_micro", []) or [], dtype=np.float64)
    if rec.size == 0:
        st.info(f"No PR points available for IoU {iou_label}.")
        return
    xr, yr = _micro_pr_curve_stair_xy(rec, prec)
    ap_val = float(ap_bundle.get("micro_ap_pr", 0.0))
    fig, ax = plt.subplots(figsize=(8.2, 5.2), dpi=180)
    ax.set_facecolor("#FBFBFD")
    if xr.size > 0 and yr.size > 0:
        _pr_fill_under_stair(ax, xr, yr, color=line_color, alpha=0.12)
        ax.plot(
            xr,
            yr,
            linestyle="-",
            linewidth=2.4,
            color=line_color,
            label=f"Pooled AP={ap_val:.3f}",
            marker="o",
            markersize=3.0,
        )
    else:
        ax.plot([], [], linestyle="-", linewidth=2.4, color=line_color, label=f"Pooled AP={ap_val:.3f}")
    r_max = float(np.max(rec))
    if r_max <= 1e-9:
        # Degenerate PR case (typically no TP): keep full axis so the curve
        # does not visually collapse to an "empty" plot at x ~= 0.
        ax.set_xlim(0.0, 1.0)
        if xr.size > 0 and yr.size > 0:
            ax.scatter([0.0], [float(yr[-1])], color=line_color, s=28, zorder=5)
            ax.text(
                0.02,
                min(0.98, float(yr[-1]) + 0.03),
                "Recall stays at 0",
                color=line_color,
                fontsize=9,
            )
    else:
        x_pad = max(1e-4, 0.02 * max(r_max, 1e-4))
        ax.set_xlim(0.0, min(1.0, r_max + x_pad))
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(
        f"Precision–Recall ({iou_label}; micro-pooled, heuristic-score ranking)",
        fontsize=13,
    )
    ax.grid(True, linestyle="--", alpha=0.30)
    ax.legend(loc="lower left", frameon=True, fontsize=10)
    st.pyplot(fig, width="stretch")
    _make_download_buttons(fig, file_tag)
    plt.close(fig)


def _render_pr_curve_for_thesis(ap_25: Dict, ap_50: Dict) -> None:
    """Render separate IoU PR figures — each ends at attained recall only (SUN RGB-D safe)."""
    rec25 = np.asarray(ap_25.get("pr_recalls_micro", []) or [], dtype=np.float64)
    rec50 = np.asarray(ap_50.get("pr_recalls_micro", []) or [], dtype=np.float64)

    if rec25.size == 0 and rec50.size == 0:
        st.info("No PR points available to plot.")
        return

    _render_single_pr_curve_figure(
        ap_25,
        iou_label="IoU ≥ 0.25",
        line_color="#1f77b4",
        file_tag="precision_recall_micro_iou025",
    )
    _render_single_pr_curve_figure(
        ap_50,
        iou_label="IoU ≥ 0.50",
        line_color="#d62728",
        file_tag="precision_recall_micro_iou050",
    )

    rows = []
    n25 = max(len(rec25), len(np.asarray(ap_25.get("pr_precisions_micro", []) or [])))
    prec25 = np.asarray(ap_25.get("pr_precisions_micro", []) or [], dtype=np.float64)
    n50 = max(len(rec50), len(np.asarray(ap_50.get("pr_precisions_micro", []) or [])))
    prec50 = np.asarray(ap_50.get("pr_precisions_micro", []) or [], dtype=np.float64)
    n = max(n25, n50)
    for i in range(n):
        rows.append({
            "idx": i,
            "recall_025_raw": float(rec25[i]) if i < len(rec25) else np.nan,
            "precision_025_raw": float(prec25[i]) if i < len(prec25) else np.nan,
            "recall_050_raw": float(rec50[i]) if i < len(rec50) else np.nan,
            "precision_050_raw": float(prec50[i]) if i < len(prec50) else np.nan,
        })
    df = pd.DataFrame(rows)
    st.download_button(
        "⬇️ Download PR curve points (CSV)",
        data=df.to_csv(index=False),
        file_name="precision_recall_curve_points.csv",
        mime="text/csv",
        width="stretch",
    )


def _render_per_class_pr_curves(ap_25: Dict, ap_50: Dict, classes: List[str]) -> None:
    """Render per-class PR curves (IoU 0.25 / 0.50) and export raw points."""
    if not classes:
        return
    st.subheader("🧩 Per-Class PR Curves")
    st.caption(
        "Each class curve uses only detections/GT of that class. "
        "AP is computed from the same per-class PR points shown here."
    )
    curves25 = ap_25.get("per_class_pr_curves", {}) or {}
    curves50 = ap_50.get("per_class_pr_curves", {}) or {}
    tabs = st.tabs(classes)
    for tab, cat in zip(tabs, classes):
        with tab:
            c25 = curves25.get(cat, {})
            c50 = curves50.get(cat, {})
            rec25 = np.asarray(c25.get("recalls", []) or [], dtype=np.float64)
            prec25 = np.asarray(c25.get("precisions", []) or [], dtype=np.float64)
            rec50 = np.asarray(c50.get("recalls", []) or [], dtype=np.float64)
            prec50 = np.asarray(c50.get("precisions", []) or [], dtype=np.float64)

            col_l, col_r = st.columns(2)
            with col_l:
                fig25, ax25 = plt.subplots(figsize=(5.3, 4.0), dpi=170)
                ax25.set_facecolor("#FBFBFD")
                if rec25.size > 0:
                    x25, y25 = _micro_pr_curve_stair_xy(rec25, prec25)
                    _pr_fill_under_stair(ax25, x25, y25, color="#1f77b4", alpha=0.12)
                    ax25.plot(
                        x25,
                        y25,
                        linewidth=2.1,
                        color="#1f77b4",
                        label=f"AP={float(ap_25.get('per_class_pr_ap', {}).get(cat, 0.0)):.3f}",
                    )
                    xlim_max_25 = min(1.0, float(np.max(rec25)) + max(1e-4, 0.02 * max(float(np.max(rec25)), 1e-4)))
                    ax25.set_xlim(0.0, xlim_max_25)
                else:
                    ax25.set_xlim(0.0, 1.0)
                    ax25.text(0.5, 0.5, "No PR points", ha="center", va="center", transform=ax25.transAxes)
                ax25.set_ylim(0.0, 1.02)
                ax25.set_xlabel("Recall")
                ax25.set_ylabel("Precision")
                ax25.set_title(f"{cat} — IoU ≥ 0.25")
                ax25.grid(True, linestyle="--", alpha=0.30)
                if rec25.size > 0:
                    ax25.legend(loc="lower left", fontsize=9, frameon=True)
                st.pyplot(fig25, width="stretch")
                plt.close(fig25)

            with col_r:
                fig50, ax50 = plt.subplots(figsize=(5.3, 4.0), dpi=170)
                ax50.set_facecolor("#FBFBFD")
                if rec50.size > 0:
                    x50, y50 = _micro_pr_curve_stair_xy(rec50, prec50)
                    _pr_fill_under_stair(ax50, x50, y50, color="#d62728", alpha=0.12)
                    ax50.plot(
                        x50,
                        y50,
                        linewidth=2.1,
                        color="#d62728",
                        label=f"AP={float(ap_50.get('per_class_pr_ap', {}).get(cat, 0.0)):.3f}",
                    )
                    xlim_max_50 = min(1.0, float(np.max(rec50)) + max(1e-4, 0.02 * max(float(np.max(rec50)), 1e-4)))
                    ax50.set_xlim(0.0, xlim_max_50)
                else:
                    ax50.set_xlim(0.0, 1.0)
                    ax50.text(0.5, 0.5, "No PR points", ha="center", va="center", transform=ax50.transAxes)
                ax50.set_ylim(0.0, 1.02)
                ax50.set_xlabel("Recall")
                ax50.set_ylabel("Precision")
                ax50.set_title(f"{cat} — IoU ≥ 0.50")
                ax50.grid(True, linestyle="--", alpha=0.30)
                if rec50.size > 0:
                    ax50.legend(loc="lower left", fontsize=9, frameon=True)
                st.pyplot(fig50, width="stretch")
                plt.close(fig50)

            n_raw = max(len(rec25), len(prec25), len(rec50), len(prec50))
            rows = []
            for i in range(n_raw):
                rows.append(
                    {
                        "idx": i,
                        "recall_025_raw": float(rec25[i]) if i < len(rec25) else np.nan,
                        "precision_025_raw": float(prec25[i]) if i < len(prec25) else np.nan,
                        "recall_050_raw": float(rec50[i]) if i < len(rec50) else np.nan,
                        "precision_050_raw": float(prec50[i]) if i < len(prec50) else np.nan,
                    }
                )
            st.download_button(
                f"⬇️ Download raw PR points — {cat} (CSV)",
                data=pd.DataFrame(rows).to_csv(index=False),
                file_name=f"pr_curve_points_{cat.replace(' ', '_')}.csv",
                mime="text/csv",
                width="stretch",
                key=f"dl_pr_curve_{cat}",
            )


def _render_ablation_study_runner():
    st.subheader("🧪 Ablation Study Runner")
    batch_samples = st.session_state.get("batch_samples", [])
    if not batch_samples:
        st.info("Load a batch first from `1_Dataset_Extraction`.")
        return

    scope1, scope2 = st.columns(2)
    with scope1:
        mini_size = st.number_input(
            "Mini-batch size (0 = full batch)",
            min_value=0,
            max_value=len(batch_samples),
            value=0,
            step=1,
        )
    with scope2:
        mini_start = st.number_input(
            "Start index",
            min_value=0,
            max_value=max(0, len(batch_samples) - 1),
            value=0,
            step=1,
        )
    if mini_size == 0:
        selected_batch = batch_samples
    else:
        end_idx = min(len(batch_samples), int(mini_start) + int(mini_size))
        selected_batch = batch_samples[int(mini_start):end_idx]
    st.caption(f"Selected samples for rerun: **{len(selected_batch)}**")

    cfg1, cfg2 = st.columns(2)
    with cfg1:
        primary_metric = st.selectbox(
            "Primary metric",
            ["Micro-F1@0.25", "Micro-F1@0.50", "Macro-F1@0.25", "Macro-F1@0.50"],
            index=0,
        )
    with cfg2:
        eps_raw = st.text_input("Adaptive DBSCAN eps sweep", "0.20,0.30,0.40,0.50,0.70,0.90")
    run_mode = st.radio(
        "Ablation run mode",
        ["Ground removal only", "DBSCAN epsilon only", "Run both"],
        horizontal=True,
    )
    report_cols = st.multiselect(
        "Also report",
        ["Precision@0.25", "Recall@0.25", "Precision@0.50", "Recall@0.50"],
        default=["Precision@0.25", "Recall@0.25"],
    )

    try:
        eps_values = [float(v.strip()) for v in eps_raw.split(",") if v.strip()]
    except ValueError:
        st.error("Epsilon list must be numeric (comma-separated).")
        return
    if len(eps_values) == 0:
        st.warning("Provide at least one epsilon value.")
        return

    if not st.button("🚀 Run Selected Ablation", type="primary"):
        payload = st.session_state.get("ablation_study_payload")
        per_class_payload = st.session_state.get("ablation_study_per_class_payload")
        if payload:
            st.caption("Cached ablation results available below.")
            st.dataframe(pd.DataFrame(payload), width="stretch")
        if per_class_payload:
            st.caption("Cached per-class ablation results available below.")
            st.dataframe(pd.DataFrame(per_class_payload), width="stretch")
        return

    detection_mod = _load_detection_page_module()
    if "params" in st.session_state and st.session_state.params:
        base_params = copy.deepcopy(st.session_state.params)
        detection_mod.ensure_detection_params(base_params)
    else:
        base_params = detection_mod.default_detection_params()
        detection_mod.ensure_detection_params(base_params)
    base_original = copy.deepcopy(base_params)
    preloaded_bbox_data = st.session_state.get("_batch_bbox_data")
    total_queued = len(selected_batch)

    rows: List[Dict] = []
    per_class_rows: List[Dict] = []
    run_ground = run_mode in {"Ground removal only", "Run both"}
    run_dbscan = run_mode in {"DBSCAN epsilon only", "Run both"}

    if run_ground:
        ground_variants: List[Tuple[str, Dict]] = []
        p_auto = copy.deepcopy(base_original)
        ground_variants.append(("scene-aware-ground", p_auto))
        p_outdoor = copy.deepcopy(base_original)
        p_outdoor["pipeline_indoor"] = copy.deepcopy(p_outdoor["pipeline"])
        ground_variants.append(("single-outdoor-ground", p_outdoor))
        p_indoor = copy.deepcopy(base_original)
        p_indoor["pipeline"] = copy.deepcopy(p_indoor["pipeline_indoor"])
        ground_variants.append(("single-indoor-ground", p_indoor))

        st.markdown("**Running ground-removal ablation...**")
        for variant_name, params_variant in ground_variants:
            results = _run_batch_with_params(detection_mod, selected_batch, params_variant, preloaded_bbox_data)
            stats_all, class_all = _compute_eval_stats_with_per_class(results, total_queued=total_queued)
            rows.append({"study": "ground_removal", "variant": variant_name, "scene": "all", **stats_all})
            per_class_rows.extend(
                [{"study": "ground_removal", "variant": variant_name, "scene": "all", **r} for r in class_all]
            )
            for scene in ["indoor", "outdoor"]:
                subset = [r for r in results if _infer_scene_bucket(r.get("metadata", {})) == scene]
                if subset:
                    scene_stats, scene_class = _compute_eval_stats_with_per_class(subset, len(subset))
                    rows.append({"study": "ground_removal", "variant": variant_name, "scene": scene, **scene_stats})
                    per_class_rows.extend(
                        [{"study": "ground_removal", "variant": variant_name, "scene": scene, **r} for r in scene_class]
                    )

    if run_dbscan:
        st.markdown("**Running adaptive DBSCAN epsilon ablation...**")
        for eps in eps_values:
            p_eps = copy.deepcopy(base_original)
            p_eps["clustering"]["clustering_algorithm"] = "adaptive_dbscan"
            p_eps["clustering"]["adaptive_dbscan_base_eps"] = float(eps)
            results = _run_batch_with_params(detection_mod, selected_batch, p_eps, preloaded_bbox_data)
            stats_all, class_all = _compute_eval_stats_with_per_class(results, total_queued=total_queued)
            rows.append({"study": "dbscan_eps", "variant": f"eps={eps:.2f}", "eps": float(eps), "scene": "all", **stats_all})
            per_class_rows.extend(
                [{"study": "dbscan_eps", "variant": f"eps={eps:.2f}", "eps": float(eps), "scene": "all", **r} for r in class_all]
            )
            for scene in ["indoor", "outdoor"]:
                subset = [r for r in results if _infer_scene_bucket(r.get("metadata", {})) == scene]
                if subset:
                    scene_stats, scene_class = _compute_eval_stats_with_per_class(subset, len(subset))
                    rows.append({"study": "dbscan_eps", "variant": f"eps={eps:.2f}", "eps": float(eps), "scene": scene, **scene_stats})
                    per_class_rows.extend(
                        [{"study": "dbscan_eps", "variant": f"eps={eps:.2f}", "eps": float(eps), "scene": scene, **r} for r in scene_class]
                    )

    st.session_state.params = base_original
    st.session_state.ablation_study_payload = rows
    st.session_state.ablation_study_per_class_payload = per_class_rows
    df = pd.DataFrame(rows)
    st.dataframe(df, width="stretch")
    st.download_button(
        "⬇️ Download ablation metrics (CSV)",
        data=df.to_csv(index=False),
        file_name="ablation_metrics.csv",
        mime="text/csv",
        width="stretch",
    )
    per_class_df = pd.DataFrame(per_class_rows)
    if not per_class_df.empty:
        st.markdown("**Per-class ablation metrics**")
        st.dataframe(per_class_df, width="stretch")
        st.download_button(
            "⬇️ Download per-class ablation metrics (CSV)",
            data=per_class_df.to_csv(index=False),
            file_name="ablation_metrics_per_class.csv",
            mime="text/csv",
            width="stretch",
        )

    if primary_metric == "Micro-F1@0.25":
        metric_col = "f125"
    elif primary_metric == "Micro-F1@0.50":
        metric_col = "f150"
    elif primary_metric == "Macro-F1@0.25":
        metric_col = "macro_f1_25"
    else:
        metric_col = "macro_f1_50"

    display_col_map = {
        "Precision@0.25": "precision25",
        "Recall@0.25": "recall25",
        "Precision@0.50": "precision50",
        "Recall@0.50": "recall50",
    }
    metric_and_reports = [metric_col] + [display_col_map[k] for k in report_cols if k in display_col_map]
    metric_and_reports = list(dict.fromkeys(metric_and_reports))
    if metric_and_reports:
        report_df = df.copy()
        for col in metric_and_reports:
            if col in report_df.columns:
                report_df[col] = report_df[col].astype(float) * 100.0
        st.markdown("**Selected metric report (%)**")
        st.dataframe(
            report_df[["study", "variant", "scene", *metric_and_reports, "n_samples"]],
            width="stretch",
        )
    ground_df = df[df["study"] == "ground_removal"]
    if not ground_df.empty:
        fig1, ax1 = plt.subplots(figsize=(10, 5), dpi=140)
        variants = ["scene-aware-ground", "single-outdoor-ground", "single-indoor-ground"]
        scenes = [s for s in ["indoor", "outdoor", "all"] if s in set(ground_df["scene"].tolist())]
        x = np.arange(len(variants))
        w = 0.24
        for i, scene in enumerate(scenes):
            ys = []
            for v in variants:
                rr = ground_df[(ground_df["variant"] == v) & (ground_df["scene"] == scene)]
                ys.append(float(rr.iloc[0][metric_col]) * 100 if not rr.empty else 0.0)
            ax1.bar(x + (i - 1) * w, ys, width=w, label=scene)
        ax1.set_xticks(x)
        ax1.set_xticklabels(variants, rotation=10)
        ax1.set_ylabel(f"{primary_metric} (%)")
        ax1.set_title("Ground Removal Ablation")
        ax1.grid(True, axis="y", linestyle="--", alpha=0.35)
        ax1.legend()
        st.pyplot(fig1, width="stretch")
        _make_download_buttons(fig1, "ablation_ground_removal")

    eps_df = df[df["study"] == "dbscan_eps"]
    if not eps_df.empty:
        fig2, ax2 = plt.subplots(figsize=(10, 5), dpi=140)
        for scene in [s for s in ["indoor", "outdoor", "all"] if s in set(eps_df["scene"].tolist())]:
            dd = eps_df[eps_df["scene"] == scene].sort_values("eps")
            ax2.plot(dd["eps"].to_numpy(dtype=float), dd[metric_col].to_numpy(dtype=float) * 100, marker="o", linewidth=2.0, label=scene)
        ax2.set_xlabel("Adaptive DBSCAN base epsilon")
        ax2.set_ylabel(f"{primary_metric} (%)")
        ax2.set_title("Adaptive DBSCAN Epsilon Sensitivity")
        ax2.grid(True, linestyle="--", alpha=0.35)
        ax2.legend()
        st.pyplot(fig2, width="stretch")
        _make_download_buttons(fig2, "ablation_dbscan_epsilon")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Main evaluation page function"""
    st.set_page_config(
        page_title="Evaluation",
        page_icon="📊",
        layout="wide"
    )

    st.header("📊 Detection Evaluation")
    st.markdown(
        "Evaluate detection results against ground truth. "
        "Computes 3D IoU, micro/macro-F1 at IoU 0.5 and 0.25, and **heuristic-score mAP@0.5 / mAP@0.25** "
        "as mean per-class AP (area under the PR curve; ranking uses exported heuristic `confidence`)."
    )

    has_batch = (
        'batch_export_results' in st.session_state
        and st.session_state.batch_export_results
        and st.session_state.batch_export_results.get("samples")
    )
    has_batch_selection = bool(st.session_state.get("batch_samples"))
    has_single = (
        'sample' in st.session_state
        and st.session_state.sample is not None
        and 'cuboids' in st.session_state
        and st.session_state.cuboids
    )

    if "eval_trim_gt_by_mask_capacity_enabled" not in st.session_state:
        st.session_state["eval_trim_gt_by_mask_capacity_enabled"] = True

    def _on_eval_trim_toggle_change() -> None:
        # Explicit rerender/version bump for all GT-dependent sections.
        st.session_state["eval_gt_trim_toggle_version"] = int(
            st.session_state.get("eval_gt_trim_toggle_version", 0)
        ) + 1

    with st.sidebar:
        st.subheader("Batch Evaluation")
        st.checkbox(
            "Apply GT mask-capacity trimming",
            key="eval_trim_gt_by_mask_capacity_enabled",
            on_change=_on_eval_trim_toggle_change,
            help=(
                "When enabled, per-frame GT can be reduced based on 2D mask capacity "
                "(including zeroing GT when no 2D detections exist)."
            ),
        )

        up_files = st.file_uploader(
            "Load exported detection JSON (from **4_Export**)",
            type=["json"],
            accept_multiple_files=True,
            key="eval_import_detection_files",
            help=(
                "**Save 3D cuboids to JSON** (`det3d_*.json`) or a `{ \"samples\": [...] }` batch payload. "
                "Multiple files concatenate in upload order."
            ),
        )
        load_btn_disabled = up_files is None or len(up_files) == 0
        load_clicked = st.button(
            "Use uploaded JSON as batch evaluation",
            key="eval_apply_detection_import",
            disabled=load_btn_disabled,
        )
        if load_clicked:
            merged, any_tracking, import_errs = _merge_uploaded_detection_files(up_files)
            if import_errs and not merged:
                for err_line in import_errs:
                    st.error(err_line)
            else:
                for err_line in import_errs:
                    st.warning(err_line)
            if merged:
                st.session_state.batch_export_results = {
                    "samples": merged,
                    "batch_tracking_enabled": any_tracking,
                }
                stubs = [_batch_sample_stub_from_import(s, idx) for idx, s in enumerate(merged)]
                st.session_state.batch_samples = stubs
                st.sidebar.success(f"Loaded **{len(merged)}** exported sample(s) for evaluation.")

    if not has_batch and not has_batch_selection and not has_single:
        st.info(
            "Use **Load exported detection JSON** in the sidebar (from **4_Export**, e.g. `det3d_*.json`), "
            "or load a dataset and run detection on **2_Detection**."
        )
        return

    if has_batch and has_batch_selection and has_single:
        tab_batch, tab_ablation, tab_single = st.tabs(["📚 Batch Evaluation", "🧪 Ablation Study", "🔬 Single Sample"])
        with tab_batch:
            _render_batch_eval()
        with tab_ablation:
            _render_ablation_study_runner()
        with tab_single:
            _render_single_sample_eval()
    elif has_batch and has_single:
        tab_batch, tab_single = st.tabs(["📚 Batch Evaluation", "🔬 Single Sample"])
        with tab_batch:
            _render_batch_eval()
        with tab_single:
            _render_single_sample_eval()
    elif has_batch and has_batch_selection:
        tab_batch, tab_ablation = st.tabs(["📚 Batch Evaluation", "🧪 Ablation Study"])
        with tab_batch:
            _render_batch_eval()
        with tab_ablation:
            _render_ablation_study_runner()
    elif has_batch:
        _render_batch_eval()
    elif has_batch_selection and has_single:
        tab_ablation, tab_single = st.tabs(["🧪 Ablation Study", "🔬 Single Sample"])
        with tab_ablation:
            _render_ablation_study_runner()
        with tab_single:
            _render_single_sample_eval()
    elif has_batch_selection:
        _render_ablation_study_runner()
    else:
        _render_single_sample_eval()


if __name__ == "__main__":
    main()
