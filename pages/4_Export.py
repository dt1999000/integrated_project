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
    st.markdown("Export all subsets to CVAT-style JSON files (uses `LinkedDataHandler.exportAnnotations`). "
                "The dataset root comes from **1_Dataset_Extraction** (`Dataset Path`).")

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
        help="keyFrameSteps passed to exportAnnotations"
    )
    if st.button("💾 Export annotations to CVAT JSON", key="export_cvat_btn"):
        root = Path(cvat_root.strip())
        if not root.exists():
            st.error(f"Path does not exist: {root}")
        elif not (root / "dataset.json").exists():
            st.error(f"dataset.json not found in {root}. Use a dataset root that has dataset.json (sim/custom format).")
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


if __name__ == "__main__":
    main()

