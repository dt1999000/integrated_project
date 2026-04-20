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

from components.core.evaluation import (
    compute_3d_iou,
    compute_batch_statistics,
    compute_batch_azimuth_bin_metrics,
    compute_frame_metrics_at_iou,
    greedy_iou_match,
    _normalize_gt_cuboids,
)
from components.utils.visualization_helper import (
    draw_2d_boxes_on_image,
    draw_projected_cuboid_bboxes,
    create_comparison_plot,
)


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
        norm_gt["category"] = norm_gt.get("category", norm_gt.get("class", "Person"))
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
            if g.get("category", g.get("class", "Unknown")) == target_class
        ]

        sample_id = str(sample.get("metadata", {}).get("sample_index", frame_idx))
        per_scene_gt[sample_id] = gt_cls

    return per_scene_gt


def _extract_detection_classes(batch_results: List[Dict], max_classes: int = 3) -> List[str]:
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
            cls_norm = str(cls).strip()
            if cls_norm and cls_norm not in seen:
                ordered.append(cls_norm)
                seen.add(cls_norm)
                if len(ordered) >= max_classes:
                    return ordered

    for sample in batch_results:
        for det in sample.get("detected_cuboids", []) or []:
            cls_norm = str(det.get("category", "Unknown")).strip()
            if cls_norm and cls_norm not in seen:
                ordered.append(cls_norm)
                seen.add(cls_norm)
                if len(ordered) >= max_classes:
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

            gt_cls = [g for g in gt_all if g.get("category", g.get("class", "Unknown")) == cls]
            det_cls = [d for d in det_all if d.get("category", "Unknown") == cls]

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

    if point_cloud_obj:
        st.subheader("🎯 3D Comparison Visualization")
        fig_unified = create_comparison_plot(point_cloud_obj, ground_truth_boxes, detected_cuboids)
        st.plotly_chart(fig_unified, width="stretch")


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
    total_queued: int = len(st.session_state.get("batch_samples", []))

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
    with st.spinner("Computing batch metrics…"):
        stats = compute_batch_statistics(
            eval_batch_results,
            total_queued=total_queued,
            match_by_category=True,
        )

    ap_50 = stats["ap_50"]
    ap_25 = stats["ap_25"]

    # ------------------------------------------------------------------
    # Top-level AP metrics
    # ------------------------------------------------------------------
    st.subheader("🏆 Benchmark Metrics")
    st.caption(
        "**Macro-F1** is the unweighted mean of per-class F1 (each class F1 uses pooled TP/FP/FN). "
        "**Micro-F1** is the harmonic mean of micro precision and recall over all objects. "
        "These are not COCO-style AP (no PR curve without confidence scores)."
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Macro-F1 @0.5", f"{ap_50['macro_f1'] * 100:.1f}%")
    with col2:
        st.metric("Macro-F1 @0.25", f"{ap_25['macro_f1'] * 100:.1f}%")
    with col3:
        st.metric("Micro precision @0.25", f"{ap_25['precision'] * 100:.1f}%")
    with col4:
        st.metric("Micro recall @0.25", f"{ap_25['recall'] * 100:.1f}%")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Micro-F1 @0.5", f"{ap_50['f1'] * 100:.1f}%")
    with col2:
        st.metric("Micro-F1 @0.25", f"{ap_25['f1'] * 100:.1f}%")
    with col3:
        st.metric("Total GT objects", stats["total_ground_truth"])
    with col4:
        st.metric("Total detections", stats["total_detections"])

    # ------------------------------------------------------------------
    # Per-class breakdown
    # ------------------------------------------------------------------
    eval_classes = _extract_detection_classes(eval_batch_results, max_classes=3)
    if eval_classes:
        st.caption(f"Evaluating configured detection classes: {', '.join(eval_classes)}")

    if ap_50["per_class"]:
        st.subheader("📊 Per-Class Metrics")
        all_cats = sorted(set(list(ap_50["per_class"]) + list(ap_25["per_class"])))
        if eval_classes:
            all_cats = [c for c in all_cats if c in eval_classes]
        rows = []
        for cat in all_cats:
            c50 = ap_50["per_class"].get(cat, {})
            c25 = ap_25["per_class"].get(cat, {})
            rows.append({
                "Category": cat,
                "TP@0.5": c50.get("TP", 0),
                "FP@0.5": c50.get("FP", 0),
                "FN@0.5": c50.get("FN", 0),
                "Prec@0.5": f"{c50.get('precision', 0) * 100:.1f}%",
                "Rec@0.5": f"{c50.get('recall', 0) * 100:.1f}%",
                "F1@0.5": f"{c50.get('f1', 0) * 100:.1f}%",
                "F1@0.25": f"{c25.get('f1', 0) * 100:.1f}%",
            })
        if rows:
            st.dataframe(pd.DataFrame(rows), width="stretch")
        else:
            st.info("No per-class metrics available for the configured detection classes.")

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
                    help="With span 180°, offset 90° maps the former 90°–270° arc onto 0°–180°.",
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
        class_tabs = st.tabs([f"Class: {cls}" for cls in eval_classes])
        for tab, cls in zip(class_tabs, eval_classes):
            with tab:
                st.markdown(f"**{cls} — IoU ≥ 0.5**")
                st.dataframe(pd.DataFrame(class_tables[cls]["iou_50"]), width="stretch")
                st.markdown(f"**{cls} — IoU ≥ 0.25**")
                st.dataframe(pd.DataFrame(class_tables[cls]["iou_25"]), width="stretch")

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


def _compute_eval_stats(batch_results: List[Dict], total_queued: int) -> Dict:
    stats = compute_batch_statistics(
        batch_results,
        total_queued=total_queued,
        match_by_category=False,
    )
    m25 = stats["ap_25"]
    m50 = stats["ap_50"]
    return {
        "macro_f1_25": m25["macro_f1"],
        "macro_f1_50": m50["macro_f1"],
        "precision25": m25["precision"],
        "recall25": m25["recall"],
        "f125": m25["f1"],
        "n_samples": len(batch_results),
    }


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
            ["Micro-F1@0.25", "Macro-F1@0.25", "Macro-F1@0.50"],
            index=0,
        )
    with cfg2:
        eps_raw = st.text_input("Adaptive DBSCAN eps sweep", "0.20,0.30,0.40,0.50,0.70,0.90")

    try:
        eps_values = [float(v.strip()) for v in eps_raw.split(",") if v.strip()]
    except ValueError:
        st.error("Epsilon list must be numeric (comma-separated).")
        return
    if len(eps_values) == 0:
        st.warning("Provide at least one epsilon value.")
        return

    if not st.button("🚀 Run Ablation (rerun selected batch)", type="primary"):
        payload = st.session_state.get("ablation_study_payload")
        if payload:
            st.caption("Cached ablation results available below.")
            st.dataframe(pd.DataFrame(payload), width="stretch")
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
        stats_all = _compute_eval_stats(results, total_queued=total_queued)
        rows.append({"study": "ground_removal", "variant": variant_name, "scene": "all", **stats_all})
        for scene in ["indoor", "outdoor"]:
            subset = [r for r in results if _infer_scene_bucket(r.get("metadata", {})) == scene]
            if subset:
                rows.append({"study": "ground_removal", "variant": variant_name, "scene": scene, **_compute_eval_stats(subset, len(subset))})

    st.markdown("**Running adaptive DBSCAN epsilon ablation...**")
    for eps in eps_values:
        p_eps = copy.deepcopy(base_original)
        p_eps["clustering"]["clustering_algorithm"] = "adaptive_dbscan"
        p_eps["clustering"]["adaptive_dbscan_base_eps"] = float(eps)
        results = _run_batch_with_params(detection_mod, selected_batch, p_eps, preloaded_bbox_data)
        stats_all = _compute_eval_stats(results, total_queued=total_queued)
        rows.append({"study": "dbscan_eps", "variant": f"eps={eps:.2f}", "eps": float(eps), "scene": "all", **stats_all})
        for scene in ["indoor", "outdoor"]:
            subset = [r for r in results if _infer_scene_bucket(r.get("metadata", {})) == scene]
            if subset:
                rows.append({"study": "dbscan_eps", "variant": f"eps={eps:.2f}", "eps": float(eps), "scene": scene, **_compute_eval_stats(subset, len(subset))})

    st.session_state.params = base_original
    st.session_state.ablation_study_payload = rows
    df = pd.DataFrame(rows)
    st.dataframe(df, width="stretch")
    st.download_button(
        "⬇️ Download ablation metrics (CSV)",
        data=df.to_csv(index=False),
        file_name="ablation_metrics.csv",
        mime="text/csv",
        width="stretch",
    )

    if primary_metric == "Micro-F1@0.25":
        metric_col = "f125"
    elif primary_metric == "Macro-F1@0.25":
        metric_col = "macro_f1_25"
    else:
        metric_col = "macro_f1_50"
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
        "Computes 3D IoU and micro/macro-F1 metrics at IoU 0.5 and 0.25 "
        "(true AP requires per-detection scores, which are not used here)."
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

    if not has_batch and not has_batch_selection and not has_single:
        if 'sample' not in st.session_state or st.session_state.sample is None:
            st.info("👈 Please load a sample from **1_Dataset_Extraction** page first.")
        else:
            st.info("👈 Please run the detection pipeline on **2_Detection** page first.")
        return

    if has_batch and has_batch_selection and has_single:
        tab_batch, tab_ablation, tab_single = st.tabs(["📚 Batch Evaluation", "🧪 Ablation Study", "🔬 Single Sample"])
        with tab_batch:
            _render_batch_eval()
        with tab_ablation:
            _render_ablation_study_runner()
        with tab_single:
            _render_single_sample_eval()
    elif has_batch and has_batch_selection:
        tab_batch, tab_ablation = st.tabs(["📚 Batch Evaluation", "🧪 Ablation Study"])
        with tab_batch:
            _render_batch_eval()
        with tab_ablation:
            _render_ablation_study_runner()
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
