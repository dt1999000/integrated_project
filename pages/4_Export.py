"""
Export Page
Export detection results and dataset annotations (CVAT format).
"""
import streamlit as st
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict

from components.dataset_loaders.dataset_loader import LinkedDataHandler


def main():
    """Main export page function"""
    st.set_page_config(
        page_title="Export Results",
        page_icon="💾",
        layout="wide"
    )
    
    st.header("💾 Export Detection Results")
    st.markdown("""
    Export detected cuboids to various formats for downstream use.
    """)
    
    # Check if detection results are available
    if 'export_results' not in st.session_state or not st.session_state.export_results:
        # Fallback to old format
        if 'cuboids' not in st.session_state or not st.session_state.cuboids:
            st.info("👈 Please run the detection pipeline on **2_Detection** page first.")
            return
        # Create export_results from old format for backward compatibility
        detected_cuboids = st.session_state.cuboids
        export_results = {
            'detected_cuboids': detected_cuboids,
            'metadata': {
                'dataset_type': 'unknown',
                'sample_index': 'unknown',
                'image_path': 'unknown',
                'timestamp': datetime.now().isoformat(),
                'n_detections': len(detected_cuboids)
            }
        }
        st.session_state.export_results = export_results
    else:
        export_results = st.session_state.export_results
        detected_cuboids = export_results['detected_cuboids']
    
    st.subheader("📦 Detection Results Summary")
    st.metric("Number of Detections", len(detected_cuboids))
    
    if detected_cuboids:
        # Display detection table
        cuboid_data = []
        for i, cuboid in enumerate(detected_cuboids):
            cuboid_data.append({
                'ID': i + 1,
                'Category': cuboid.get('category', 'Unknown'),
                'Center X': f"{cuboid['center'][0]:.2f}",
                'Center Y': f"{cuboid['center'][1]:.2f}",
                'Center Z': f"{cuboid['center'][2]:.2f}",
                'Yaw (deg)': f"{np.degrees(cuboid['yaw']):.1f}",
                'Length': f"{cuboid['length']:.2f}",
                'Width': f"{cuboid['width']:.2f}",
                'Height': f"{cuboid['height']:.2f}",
            })
        df = pd.DataFrame(cuboid_data)
        st.dataframe(df, use_container_width=True)
    
    # Export options
    st.subheader("📤 Export Options")
    
    # Output directory (similar to dataset path in Dataset Extraction)
    st.markdown("**Output directory**")
    output_dir = st.text_input(
        "Export output directory",
        value="",
        key="export_output_dir",
        help="Optional. Directory to save the annotation file directly (e.g. ./exports or C:/data/exports). Leave empty to use download only."
    )
    
    export_format = st.selectbox(
        "Export Format",
        options=['JSON (Custom)', 'KITTI Format', 'COCO Format'],
        help="Choose the export format"
    )
    
    # Get metadata from export_results (already includes all necessary info)
    metadata = export_results.get('metadata', {})
    
    # Export button
    if st.button("💾 Export Results", type="primary"):
        sample_id = metadata.get('sample_index', 'unknown')
        
        if export_format == 'JSON (Custom)':
            # Create JSON file
            # Include ground truth cuboids if available
            export_data = {
                'metadata': metadata,
                'timestamp': metadata.get('timestamp', datetime.now().isoformat()),
                'n_detections': len(detected_cuboids),
                'detections': [
                    {
                        'category': c.get('category', 'Unknown'),
                        'center': c['center'].tolist() if isinstance(c['center'], np.ndarray) else c['center'],
                        'yaw': float(c['yaw']),
                        'dimensions': {
                            'length': float(c['length']),
                            'width': float(c['width']),
                            'height': float(c['height'])
                        },
                        'corners': c['corners'].tolist() if isinstance(c['corners'], np.ndarray) else c['corners'],
                        'bounds': {
                            'min_x': float(c['min_x']),
                            'max_x': float(c['max_x']),
                            'min_y': float(c['min_y']),
                            'max_y': float(c['max_y']),
                            'min_z': float(c['min_z']),
                            'max_z': float(c['max_z'])
                        },
                        'score': float(c.get('score', 0.0)),
                        'method': c.get('method', 'unknown'),
                        'n_points': int(c.get('n_points', 0))
                    }
                    for c in detected_cuboids
                ]
            }
            
            # Add ground truth cuboids if available
            if 'ground_truth_cuboids' in export_results:
                export_data['ground_truth_cuboids'] = [
                    {
                        'category': gt.get('category', 'Unknown'),
                        'corners': gt['corners'].tolist() if isinstance(gt.get('corners'), np.ndarray) else gt.get('corners'),
                        'bbox_2d': gt.get('bbox_2d'),
                        'bounds': {
                            'min_x': float(gt.get('min_x', 0)),
                            'max_x': float(gt.get('max_x', 0)),
                            'min_y': float(gt.get('min_y', 0)),
                            'max_y': float(gt.get('max_y', 0)),
                            'min_z': float(gt.get('min_z', 0)),
                            'max_z': float(gt.get('max_z', 0))
                        }
                    }
                    for gt in export_results['ground_truth_cuboids']
                ]
                export_data['n_ground_truth'] = len(export_data['ground_truth_cuboids'])
            
            json_str = json.dumps(export_data, indent=2)
            file_name = f"detections_{sample_id}.json"
            
            # Write to output directory if specified
            if output_dir and output_dir.strip():
                out_path = Path(output_dir.strip())
                try:
                    out_path.mkdir(parents=True, exist_ok=True)
                    out_file = out_path / file_name
                    with open(out_file, 'w') as f:
                        f.write(json_str)
                    st.success(f"✅ Saved to **{out_file}**")
                except Exception as e:
                    st.error(f"Could not write to output directory: {e}")
            
            st.download_button(
                label="📥 Download JSON",
                data=json_str,
                file_name=file_name,
                mime="application/json"
            )
        
        elif export_format == 'KITTI Format':
            # Create KITTI format text file
            lines = []
            for cuboid in detected_cuboids:
                category = cuboid.get('category', 'Unknown')
                center = cuboid['center']
                yaw = cuboid['yaw']
                length = cuboid['length']
                width = cuboid['width']
                height = cuboid['height']
                
                # Simplified KITTI format
                line = f"{category} 0.00 0 0.00 0.00 0.00 0.00 0.00 {height:.2f} {width:.2f} {length:.2f} {center[0]:.2f} {center[1]:.2f} {center[2]:.2f} {yaw:.2f}"
                lines.append(line)
            
            kitti_str = '\n'.join(lines)
            file_name = f"detections_{sample_id}.txt"
            
            # Write to output directory if specified
            if output_dir and output_dir.strip():
                out_path = Path(output_dir.strip())
                try:
                    out_path.mkdir(parents=True, exist_ok=True)
                    out_file = out_path / file_name
                    with open(out_file, 'w') as f:
                        f.write(kitti_str)
                    st.success(f"✅ Saved to **{out_file}**")
                except Exception as e:
                    st.error(f"Could not write to output directory: {e}")
            
            st.download_button(
                label="📥 Download KITTI Format",
                data=kitti_str,
                file_name=file_name,
                mime="text/plain"
            )
        
        elif export_format == 'COCO Format':
            st.info("COCO format export coming soon...")

    # Export dataset annotations (CVAT) using LinkedDataHandler
    st.subheader("📤 Export dataset annotations (CVAT)")
    st.markdown("Export all subsets to CVAT-style JSON files (uses `LinkedDataHandler.exportAnnotations`). Requires a dataset root that contains `dataset.json` (sim/custom format).")
    cvat_root = st.text_input(
        "Dataset root path (CVAT export)",
        value=st.session_state.get("dataset_path", ""),
        key="export_cvat_dataset_path",
        help="Path to dataset root containing dataset.json (e.g. folder with subset folders and dataset.json)"
    )
    key_frame_steps = st.number_input(
        "Keyframe every N frames",
        min_value=1,
        value=10,
        key="export_cvat_keyframe_steps",
        help="keyFrameSteps passed to exportAnnotations"
    )
    if st.button("💾 Export annotations to CVAT JSON", key="export_cvat_btn"):
        if not cvat_root or not cvat_root.strip():
            st.warning("Please enter a dataset root path.")
        else:
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

