"""
Export Page
Export dataset annotations (CVAT format) using LinkedDataHandler.
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import streamlit as st

from components.dataset_loaders.dataset_loader import LinkedDataHandler


def main():
    """Main export page function"""
    st.set_page_config(
        page_title="Export Results",
        page_icon="💾",
        layout="wide"
    )
    
    st.header("💾 Export Dataset Annotations (CVAT)")
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

