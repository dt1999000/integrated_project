"""
Export Page
Export batch detection results (tracklet XML) and dataset annotations (CVAT format).
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import streamlit as st

from components.dataset_loaders.dataset_loader import LinkedDataHandler


def _to_serializable(obj: Any) -> Any:
    """
    Convert numpy types in results to JSON-serializable Python types
    while keeping the structure as close as possible to the original.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_serializable(v) for v in obj]
    return obj


def _batch_to_tracklet_xml(samples: List[Dict]) -> str:
    """Build tracklet-style XML from batch_export_results['samples']."""
    lines = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        "<!DOCTYPE boost_serialization>",
        '<boost_serialization version="9" signature="serialization::archive">',
        '<tracklets version="0" tracking_level="0" class_id="0">',
    ]
    all_items = []
    for frame_idx, export_res in enumerate(samples):
        cuboids = export_res.get("detected_cuboids", [])
        for c in cuboids:
            center = c.get("center", [0, 0, 0])
            if hasattr(center, "tolist"):
                center = center.tolist()
            yaw = float(c.get("yaw", 0))
            l = float(c.get("length", 1))
            w = float(c.get("width", 1))
            h = float(c.get("height", 1))
            cat = c.get("category", "Unknown")
            all_items.append((frame_idx, cat, h, w, l, center[0], center[1], center[2], yaw))
    lines.append(f"  <count>{len(all_items)}</count>")
    lines.append("  <item_version>1</item_version>")
    for frame_idx, cat, h, w, l, tx, ty, tz, rz in all_items:
        lines.append('  <item version="1" tracking_level="0" class_id="1">')
        lines.append(f"    <objectType>{cat}</objectType>")
        lines.append(f"    <h>{h:.2f}</h>")
        lines.append(f"    <w>{w:.2f}</w>")
        lines.append(f"    <l>{l:.2f}</l>")
        lines.append(f"    <first_frame>{frame_idx}</first_frame>")
        lines.append('    <poses version="0" tracking_level="0" class_id="2">')
        lines.append("      <count>1</count>")
        lines.append("      <item_version>0</item_version>")
        lines.append('      <item version="1" tracking_level="0" class_id="3">')
        lines.append(f"        <tx>{tx:.2f}</tx>")
        lines.append(f"        <ty>{ty:.2f}</ty>")
        lines.append(f"        <tz>{tz:.2f}</tz>")
        lines.append("        <rx>0.0</rx>")
        lines.append("        <ry>0.0</ry>")
        lines.append(f"        <rz>{rz:.2f}</rz>")
        lines.append("        <state>2</state>")
        lines.append("        <occlusion>0</occlusion>")
        lines.append("        <occlusion_kf>0</occlusion_kf>")
        lines.append("        <truncation>0</truncation>")
        lines.append("        <amt_occlusion>-1</amt_occlusion>")
        lines.append("        <amt_border_l>-1</amt_border_l>")
        lines.append("        <amt_border_r>-1</amt_border_r>")
        lines.append("        <amt_occlusion_kf>-1</amt_occlusion_kf>")
        lines.append("        <amt_border_kf>-1</amt_border_kf>")
        lines.append("      </item>")
        lines.append("    </poses>")
        lines.append("    <finished>1</finished>")
        lines.append("  </item>")
    lines.append("</tracklets>")
    lines.append("</boost_serialization>")
    return "\n".join(lines)


def main():
    """Main export page function"""
    st.set_page_config(
        page_title="Export Results",
        page_icon="💾",
        layout="wide"
    )
    
    st.header("💾 Export Results")

    # ------------------------------------------------------------------
    # 1) Batch tracklet XML export (from 2_Detection batch processing)
    # ------------------------------------------------------------------
    batch_results = st.session_state.get("batch_export_results")
    if batch_results and batch_results.get("samples"):
        samples = batch_results["samples"]
        st.subheader("📦 Batch Tracklet XML Export")
        st.metric("Samples in batch", len(samples))
        total_detections = sum(len(s.get("detected_cuboids", [])) for s in samples)
        st.metric("Total detections", total_detections)

        output_root = st.session_state.get("output_root_dir", "")
        if not output_root:
            st.warning("Output folder is not set. Please open **1_Dataset_Extraction** and set the Output Directory.")
        else:
            st.info(f"Tracklet XML will be saved under: `{output_root}` as `tracklet_labels.xml`.")
            if st.button("💾 Save batch as Tracklet XML", key="export_tracklet_xml_btn"):
                try:
                    out_dir = Path(output_root).expanduser()
                    out_dir.mkdir(parents=True, exist_ok=True)
                    xml_str = _batch_to_tracklet_xml(samples)
                    out_file = out_dir / "tracklet_labels.xml"
                    out_file.write_text(xml_str, encoding="utf-8")
                    st.success(f"✅ Saved tracklet XML to **{out_file}**")
                except Exception as e:
                    st.error(f"Could not save tracklet XML: {e}")

        st.markdown("---")

    # ------------------------------------------------------------------
    # 2) Dataset annotations (CVAT) using LinkedDataHandler
    # ------------------------------------------------------------------
    st.subheader("📤 Export dataset annotations (CVAT)")
    st.markdown(
        "Export all subsets to CVAT-style JSON files (uses `LinkedDataHandler.exportAnnotations`). "
        "The dataset root comes from **1_Dataset_Extraction** (`Dataset Path`)."
    )

    # Use dataset root from session state (set on 1_Dataset_Extraction)
    cvat_root = st.session_state.get("dataset_path", "")
    if not cvat_root:
        st.warning("Dataset root is not set. Please open **1_Dataset_Extraction** and select a dataset first.")
        return
    else:
        st.info(f"Using dataset root: `{cvat_root}`")

    key_frame_steps = st.number_input(
        "Keyframe every N frames",
        min_value=1,
        value=10,
        key="export_cvat_keyframe_steps",
        help="keyFrameSteps passed to exportAnnotations",
    )
    if st.button("💾 Export annotations to CVAT JSON", key="export_cvat_btn"):
        root = Path(cvat_root.strip())
        if not root.exists():
            st.error(f"Path does not exist: {root}")
        elif not (root / "dataset.json").exists():
            st.error(
                f"dataset.json not found in {root}. Use a dataset root that has dataset.json (sim/custom format)."
            )
        else:
            try:
                handler = LinkedDataHandler(root_dir=str(root), load_dataset=True)
                handler.exportAnnotations(keyFrameSteps=int(key_frame_steps))
                subsets = handler.list_subsets()
                files = [f"{s}_cvat.json" for s in subsets]
                st.success(f"✅ Exported {len(files)} file(s) to {root}: {', '.join(files)}")
            except Exception as e:
                st.error(f"Export failed: {e}")
                import traceback
                st.code(traceback.format_exc())

    st.markdown("---")

    # ------------------------------------------------------------------
    # 3) Per-sample detection export (3D cuboids + 2D image annotations)
    # ------------------------------------------------------------------
    st.subheader("🧊 Export detection results to JSON")
    st.markdown(
        "Export the last detection result from **2_Detection** as JSON. "
        "3D cuboid annotations and 2D image annotations are saved as separate files."
    )

    export_results = st.session_state.get("export_results")
    if not export_results:
        st.info("No detection results found. Run the pipeline on **2_Detection** first.")
        return

    meta = export_results.get("metadata", {})
    dataset_type = meta.get("dataset_type", "unknown")
    sample_index = meta.get("sample_index", "unknown")
    image_path = meta.get("image_path", "unknown")

    col_meta_1, col_meta_2, col_meta_3 = st.columns(3)
    with col_meta_1:
        st.metric("Dataset", str(dataset_type).upper())
    with col_meta_2:
        st.metric("Sample Index", str(sample_index))
    with col_meta_3:
        st.metric("Image Path", str(image_path))

    output_root = st.session_state.get("output_root_dir", "")
    if not output_root:
        st.warning("Output folder is not set. Please open **1_Dataset_Extraction** and set the Output Directory.")
        return

    out_dir = Path(output_root).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------- 3D cuboid export -------------------------
    st.markdown("#### 3D cuboid annotations (JSON)")
    st.caption("Exports the detected 3D cuboids (and ground-truth cuboids if available) as JSON.")

    if st.button("💾 Save 3D cuboids to JSON", key="export_3d_cuboids_json"):
        try:
            cuboid_payload: Dict[str, Any] = {
                "metadata": _to_serializable(meta),
                "detected_cuboids": _to_serializable(export_results.get("detected_cuboids", [])),
            }
            if "ground_truth_cuboids" in export_results:
                cuboid_payload["ground_truth_cuboids"] = _to_serializable(
                    export_results.get("ground_truth_cuboids", [])
                )

            fname_3d = f"det3d_{dataset_type}_{sample_index}.json"
            out_file_3d = out_dir / fname_3d
            out_file_3d.write_text(json.dumps(cuboid_payload, indent=2), encoding="utf-8")
            st.success(f"✅ Saved 3D cuboid annotations to **{out_file_3d}**")
        except Exception as e:
            st.error(f"Could not save 3D cuboid JSON: {e}")

    st.markdown("#### 2D image annotations (JSON)")
    st.caption(
        "Exports the image-space annotations from Step 3 (SAM segmentation) with both masks and 2D bounding boxes. "
        "Structure is kept as close as possible to the original Step 3 result."
    )

    pipeline_state = st.session_state.get("pipeline_state", {})
    step_3_state = pipeline_state.get("step_3") if isinstance(pipeline_state, dict) else None
    step_3_result = step_3_state.get("result") if step_3_state else None

    if not step_3_result:
        st.warning(
            "Step 3 SAM segmentation results not found in session. "
            "Run Step 3 or the full pipeline on **2_Detection** before exporting 2D annotations."
        )
        return

    if st.button("💾 Save 2D image annotations to JSON", key="export_2d_image_json"):
        try:
            # Masks are numpy arrays of shape (H, W); convert to lists of ints (0/1) to stay close to original.
            raw_masks = step_3_result.get("sam_masks", []) or []
            masks_serializable: List[Any] = []
            for m in raw_masks:
                if isinstance(m, np.ndarray):
                    # Ensure binary int mask for JSON
                    m_bin = (m > 0).astype(np.uint8)
                    masks_serializable.append(m_bin.tolist())
                else:
                    masks_serializable.append(_to_serializable(m))

            image_annotations: Dict[str, Any] = {
                # Keep key names aligned with the original Step 3 result where possible
                "masks": masks_serializable,
                "mask_bboxes": _to_serializable(step_3_result.get("mask_bboxes", [])),
                "class_names": _to_serializable(step_3_result.get("class_names", [])),
                "confidences": _to_serializable(step_3_result.get("confidences", [])),
                "n_masks": _to_serializable(step_3_result.get("n_masks", len(raw_masks))),
            }

            image_payload: Dict[str, Any] = {
                "metadata": _to_serializable(meta),
                "image_annotations": image_annotations,
            }

            fname_2d = f"det2d_{dataset_type}_{sample_index}.json"
            out_file_2d = out_dir / fname_2d
            out_file_2d.write_text(json.dumps(image_payload, indent=2), encoding="utf-8")
            st.success(f"✅ Saved 2D image annotations to **{out_file_2d}**")
        except Exception as e:
            st.error(f"Could not save 2D image annotation JSON: {e}")


if __name__ == "__main__":
    main()

