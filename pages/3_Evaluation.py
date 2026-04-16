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
import copy
import importlib.util
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import matplotlib.pyplot as plt

from components.core.evaluation import (
    compute_3d_iou,
    compute_batch_statistics,
    compute_frame_metrics_at_iou,
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
        st.plotly_chart(fig_unified)


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

def _render_batch_eval():
    """Render the batch evaluation panel with AP_50, AP_25 and per-frame stats."""
    batch_export = st.session_state.batch_export_results
    batch_results: List[Dict] = batch_export.get("samples", [])
    total_queued: int = len(st.session_state.get("batch_samples", []))

    st.subheader("📦 Batch Sample Overview")

    n_processed = len(batch_results)
    n_failed = max(0, total_queued - n_processed)
    # A sample is evaluable when the 'ground_truth_cuboids' key is present (even if the list
    # is empty — empty means a valid annotated scene with no objects, whose FP detections
    # must still count against precision).  A missing key means GT was not available at all.
    results_with_gt = [r for r in batch_results if r.get("ground_truth_cuboids") is not None]
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
            "These frames are included in AP computation — all detections in them count as FP."
        )

    if n_no_gt > 0:
        st.info(
            f"{n_no_gt} processed sample(s) have no GT source (e.g. rosbag / nuScenes) "
            "and are excluded from AP computation."
        )

    if n_evaluable == 0:
        st.warning(
            "⚠️ None of the processed samples contain ground truth cuboids. "
            "AP metrics cannot be computed.  Ground truth is available for KITTI and sim batches."
        )
        _render_batch_detection_only(batch_results)
        return

    # ------------------------------------------------------------------
    # Compute statistics
    # ------------------------------------------------------------------
    with st.spinner("Computing batch metrics…"):
        stats = compute_batch_statistics(
            batch_results,
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
        "AP is computed as the mean per-class F1 score at the given IoU threshold, "
        "aggregated across all evaluable frames.  "
        "AP@0.5 (AP_50) is the KITTI/ScanNet standard; AP@0.25 (AP_25) is common in indoor benchmarks."
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("AP_50  (IoU ≥ 0.5)", f"{ap_50['ap'] * 100:.1f}%")
    with col2:
        st.metric("AP_25  (IoU ≥ 0.25)", f"{ap_25['ap'] * 100:.1f}%")
    with col3:
        st.metric("Precision@0.25", f"{ap_25['precision'] * 100:.1f}%")
    with col4:
        st.metric("Recall@0.25", f"{ap_25['recall'] * 100:.1f}%")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("F1@0.5", f"{ap_50['f1'] * 100:.1f}%")
    with col2:
        st.metric("F1@0.25", f"{ap_25['f1'] * 100:.1f}%")
    with col3:
        st.metric("Total GT objects", stats["total_ground_truth"])
    with col4:
        st.metric("Total detections", stats["total_detections"])

    # ------------------------------------------------------------------
    # Per-class breakdown
    # ------------------------------------------------------------------
    eval_classes = _extract_detection_classes(batch_results, max_classes=3)
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
                "AP@0.5": f"{c50.get('ap', 0) * 100:.1f}%",
                "AP@0.25": f"{c25.get('ap', 0) * 100:.1f}%",
            })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            st.info("No per-class metrics available for the configured detection classes.")

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
            st.dataframe(pd.DataFrame(frame_rows), use_container_width=True)

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
            st.dataframe(pd.DataFrame(frame_rows_25), use_container_width=True)

    # ------------------------------------------------------------------
    # Per-class per-frame breakdown (requested for SUNRGBD / class-aware eval)
    # ------------------------------------------------------------------
    if eval_classes:
        st.subheader("🧪 Per-Class Per-Frame Metrics")
        st.caption(
            "Each table isolates a single configured detection class so you can "
            "inspect frame-by-frame TP/FP/FN against GT for that class only."
        )
        class_tables = _compute_per_class_per_frame_tables(batch_results, eval_classes)
        class_tabs = st.tabs([f"Class: {cls}" for cls in eval_classes])
        for tab, cls in zip(class_tabs, eval_classes):
            with tab:
                st.markdown(f"**{cls} — IoU ≥ 0.5**")
                st.dataframe(pd.DataFrame(class_tables[cls]["iou_50"]), use_container_width=True)
                st.markdown(f"**{cls} — IoU ≥ 0.25**")
                st.dataframe(pd.DataFrame(class_tables[cls]["iou_25"]), use_container_width=True)

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
                r for r in batch_results
                if str(r.get("metadata", {}).get("sample_index", "")) == idx
            ]
            if matched:
                gt_cuboids = matched[0].get("ground_truth_cuboids")
                has_gt = gt_cuboids is not None
                n_det = len(matched[0].get("detected_cuboids", []))
                n_gt = len(gt_cuboids) if gt_cuboids else 0
                if not has_gt:
                    status = "✅ Processed (no GT source)"
                elif n_gt == 0:
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
            st.dataframe(pd.DataFrame(status_rows), use_container_width=True)


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
    st.dataframe(pd.DataFrame(rows), use_container_width=True)


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
    ap25 = stats["ap_25"]
    ap50 = stats["ap_50"]
    return {
        "ap25": ap25["ap"],
        "ap50": ap50["ap"],
        "precision25": ap25["precision"],
        "recall25": ap25["recall"],
        "f125": ap25["f1"],
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
            use_container_width=True,
        )
    with col2:
        st.download_button(
            "⬇️ Download SVG (vector)",
            data=svg_buf.getvalue(),
            file_name=f"{base_name}.svg",
            mime="image/svg+xml",
            use_container_width=True,
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
        primary_metric = st.selectbox("Primary metric", ["F1@0.25", "AP@0.25", "AP@0.50"], index=0)
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
            st.dataframe(pd.DataFrame(payload), use_container_width=True)
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
    st.dataframe(df, use_container_width=True)
    st.download_button(
        "⬇️ Download ablation metrics (CSV)",
        data=df.to_csv(index=False),
        file_name="ablation_metrics.csv",
        mime="text/csv",
        use_container_width=True,
    )

    metric_col = "f125" if primary_metric == "F1@0.25" else ("ap25" if primary_metric == "AP@0.25" else "ap50")
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
        st.pyplot(fig1, use_container_width=True)
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
        st.pyplot(fig2, use_container_width=True)
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
        "Computes 3D IoU, AP_50 and AP_25 benchmark metrics."
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
