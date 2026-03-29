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
from components.utils.export_utils import Export


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


def _datumaro_to_tracklet_xml(tracking_state: Dict) -> str:
    """
    Build KITTI-style tracklet XML from datumaro_tracking state.
    Uses 3D cuboid tracks (track_id) across frames instead of per-frame detections.
    """
    items = tracking_state.get("items", [])
    categories = tracking_state.get("categories", {})
    label_cat = categories.get("label", {})
    labels = label_cat.get("labels", [])

    def _label_name(label_id: int) -> str:
        if 0 <= label_id < len(labels):
            return labels[label_id].get("name", "Unknown")
        return "Unknown"

    # Aggregate per-track information
    tracks: Dict[int, Dict[str, Any]] = {}
    for item in items:
        frame_id = int(item.get("attr", {}).get("frame", 0))
        for ann in item.get("annotations", []):
            if ann.get("type") != "cuboid_3d":
                continue
            attrs = ann.get("attributes", {})
            track_id = int(attrs.get("track_id", -1))
            if track_id < 0:
                continue

            label_id = int(ann.get("label_id", 0))
            label = _label_name(label_id)

            position = ann.get("position", [0.0, 0.0, 0.0])
            rotation = ann.get("rotation", [0.0, 0.0, 0.0])
            scale = ann.get("scale", [1.0, 1.0, 1.0])

            # Datumaro uses (length, width, height) for scale
            length = float(scale[0])
            width = float(scale[1])
            height = float(scale[2])

            tx = float(position[0])
            ty = float(position[1])
            tz = float(position[2])
            rz = float(rotation[2])

            tr = tracks.get(track_id)
            if tr is None:
                tr = {
                    "label": label,
                    "length": length,
                    "width": width,
                    "height": height,
                    "first_frame": frame_id,
                    "poses": [],
                }
                tracks[track_id] = tr

            # Keep earliest first_frame
            if frame_id < tr["first_frame"]:
                tr["first_frame"] = frame_id

            tr["poses"].append(
                {
                    "frame": frame_id,
                    "tx": tx,
                    "ty": ty,
                    "tz": tz,
                    "rz": rz,
                }
            )

    lines = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        "<!DOCTYPE boost_serialization>",
        '<boost_serialization version="9" signature="serialization::archive">',
        '<tracklets version="0" tracking_level="0" class_id="0">',
    ]

    track_items = list(tracks.values())
    lines.append(f"  <count>{len(track_items)}</count>")
    lines.append("  <item_version>1</item_version>")

    for tr in track_items:
        poses = sorted(tr["poses"], key=lambda p: p["frame"])
        lines.append('  <item version="1" tracking_level="0" class_id="1">')
        lines.append(f"    <objectType>{tr['label']}</objectType>")
        lines.append(f"    <h>{tr['height']:.2f}</h>")
        lines.append(f"    <w>{tr['width']:.2f}</w>")
        lines.append(f"    <l>{tr['length']:.2f}</l>")
        lines.append(f"    <first_frame>{poses[0]['frame']}</first_frame>")
        lines.append('    <poses version="0" tracking_level="0" class_id="2">')
        lines.append(f"      <count>{len(poses)}</count>")
        lines.append("      <item_version>0</item_version>")
        for p in poses:
            lines.append('      <item version="1" tracking_level="0" class_id="3">')
            lines.append(f"        <tx>{p['tx']:.2f}</tx>")
            lines.append(f"        <ty>{p['ty']:.2f}</ty>")
            lines.append(f"        <tz>{p['tz']:.2f}</tz>")
            lines.append("        <rx>0.0</rx>")
            lines.append("        <ry>0.0</ry>")
            lines.append(f"        <rz>{p['rz']:.2f}</rz>")
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

    output_root = st.session_state.get("output_root_dir", "")
    batch_export_results = st.session_state.get("batch_export_results") or {}
    batch_samples_list = batch_export_results.get("samples") or []
    _bte = batch_export_results.get("batch_tracking_enabled")
    _ts_legacy = st.session_state.get("datumaro_tracking")
    if _bte is None:
        batch_tracking_enabled = bool(_ts_legacy and _ts_legacy.get("items"))
    else:
        batch_tracking_enabled = bool(_bte)
    tracking_state = st.session_state.get("datumaro_tracking")
    show_batch_tracking_exports = batch_tracking_enabled and bool(
        tracking_state and tracking_state.get("items")
    )

    st.subheader("🚗 KITTI format exports")
    st.caption(
        "Exports in this section follow KITTI-style conventions. "
        "Note: **CVAT's KITTI-compatible import currently does not support tracking**; "
        "for tracked batch exports use the Datumaro-style block (shown when batch tracking was enabled on **2_Detection**)."
    )

    # ------------------------------------------------------------------
    # 1) Tracklet XML export from tracking (Datumaro state)
    # ------------------------------------------------------------------
    if show_batch_tracking_exports:
        st.subheader("📦 Tracklet XML Export (KITTI-style from tracking)")
        n_frames = len(tracking_state.get("items", []))
        st.metric("Frames in batch", n_frames)

        # Approximate number of tracklets as unique track_ids in annotations
        track_ids = set()
        for item in tracking_state.get("items", []):
            for ann in item.get("annotations", []):
                attrs = ann.get("attributes", {})
                if "track_id" in attrs:
                    track_ids.add(int(attrs["track_id"]))
        st.metric("Tracklets", len(track_ids))

        if not output_root:
            st.warning("Output folder is not set. Please open **1_Dataset_Extraction** and set the Output Directory.")
        else:
            st.info(f"Tracklet XML will be saved under: `{output_root}` as `tracklet_labels.xml`.")
            if st.button("💾 Save tracking as Tracklet XML", key="export_tracklet_xml_btn"):
                out_dir = Path(output_root).expanduser()
                out_dir.mkdir(parents=True, exist_ok=True)
                xml_str = _datumaro_to_tracklet_xml(tracking_state)
                out_file = out_dir / "tracklet_labels.xml"
                out_file.write_text(xml_str, encoding="utf-8")
                st.success(f"✅ Saved tracklet XML to **{out_file}**")

            # Complete 2D tracking history JSON export from ObjectTracker state.
            if st.button("💾 Save complete 2D tracking history (JSON)", key="export_tracking_2d_history_json"):
                tracking_2d_json = st.session_state.get("tracking_2d_history")
                if tracking_2d_json is None:
                    st.warning("No cached 2D tracking history found. Run batch tracking first.")
                else:
                    out_dir = Path(output_root).expanduser()
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_file = out_dir / "tracking_2d_history.json"
                    out_file.write_text(json.dumps(tracking_2d_json, indent=2), encoding="utf-8")
                    st.success(f"✅ Saved complete 2D tracking history to **{out_file}**")

        st.markdown("---")

    # ------------------------------------------------------------------
    # 4) Datumaro-style export with tracking
    # ------------------------------------------------------------------
    if show_batch_tracking_exports:
        st.subheader("📦 Datumaro-style export (with tracking)")
        st.caption(
            "Exports batch detections in a Datumaro/CVAT-compatible JSON format that "
            "includes 3D cuboid tracks (track_id, keyframes, occlusion flags). "
            "Uses the tracking state built during batch processing on **2_Detection**."
        )

        if not output_root:
            st.warning(
                "Output folder is not set. Please open **1_Dataset_Extraction** and set the Output Directory."
            )
        elif st.button("💾 Save Datumaro-style tracking JSON", key="export_datumaro_tracking_json"):
            export_root = Path(output_root).expanduser()
            export_root.mkdir(parents=True, exist_ok=True)

            datumaro_json = tracking_state
            out_file = export_root / "detections_datumaro_tracking.json"
            out_file.write_text(json.dumps(datumaro_json, indent=2), encoding="utf-8")
            st.success(f"✅ Saved Datumaro-style tracking annotations to **{out_file}**")

        st.markdown("---")
    elif batch_samples_list and batch_export_results.get("batch_tracking_enabled") is False:
        st.markdown("---")
    elif batch_samples_list and not show_batch_tracking_exports:
        st.subheader("📦 Datumaro-style export (with tracking)")
        st.info(
            "No tracking state in session. Run batch processing on **2_Detection** "
            "with tracking enabled before exporting in Datumaro style."
        )
        st.markdown("---")
    elif not batch_samples_list:
        st.subheader("📦 Datumaro-style export (with tracking)")
        st.caption(
            "Exports batch detections in a Datumaro/CVAT-compatible JSON format that "
            "includes 3D cuboid tracks (track_id, keyframes, occlusion flags). "
            "Uses the tracking state built during batch processing on **2_Detection**."
        )
        st.info(
            "No batch results yet. Run batch processing on **2_Detection** "
            "(with tracking enabled if you need track-based exports)."
        )
        st.markdown("---")

    # ------------------------------------------------------------------
    # 2) Dataset annotations (CVAT) using LinkedDataHandler
    # ------------------------------------------------------------------
    st.markdown("### 📤 Export dataset annotations (CVAT, KITTI-style)")
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
    st.subheader("🧊 Export detection results to JSON (KITTI-style)")
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
            "segmentation_debug": _to_serializable(step_3_result.get("segmentation_debug", {})),
        }

        image_payload: Dict[str, Any] = {
            "metadata": _to_serializable(meta),
            "image_annotations": image_annotations,
        }

        fname_2d = f"det2d_{dataset_type}_{sample_index}.json"
        out_file_2d = out_dir / fname_2d
        out_file_2d.write_text(json.dumps(image_payload, indent=2), encoding="utf-8")
        st.success(f"✅ Saved 2D image annotations to **{out_file_2d}**")

    st.markdown("---")


if __name__ == "__main__":
    main()

