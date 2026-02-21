"""
Export Page
Export detection results to JSON format.
"""
import streamlit as st
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import List, Dict


def export_to_kitti_format(cuboids: List[Dict], output_path: str):
    """
    Export cuboids to KITTI format.
    
    KITTI format: type truncated occluded alpha bbox2d(4) dim(3) loc(3) rotation_y
    """
    lines = []
    for cuboid in cuboids:
        category = cuboid.get('category', 'Unknown')
        center = cuboid['center']
        yaw = cuboid['yaw']
        length = cuboid['length']
        width = cuboid['width']
        height = cuboid['height']
        
        # Convert to KITTI format
        # Note: This is a simplified export - full KITTI format requires more fields
        line = f"{category} 0.00 0 0.00 "
        
        # 2D bbox (if available)
        if 'bbox_2d' in cuboid:
            bbox = cuboid['bbox_2d']
            line += f"{bbox['left']:.2f} {bbox['top']:.2f} {bbox['right']:.2f} {bbox['bottom']:.2f} "
        else:
            line += "0.00 0.00 0.00 0.00 "
        
        # Dimensions (height, width, length in KITTI)
        line += f"{height:.2f} {width:.2f} {length:.2f} "
        
        # Location (x, y, z in camera coordinates - simplified)
        line += f"{center[0]:.2f} {center[1]:.2f} {center[2]:.2f} "
        
        # Rotation y
        line += f"{yaw:.2f}"
        
        lines.append(line)
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))


def export_to_json_format(cuboids: List[Dict], output_path: str, metadata: Dict = None):
    """
    Export cuboids to custom JSON format with metadata.
    """
    export_data = {
        'metadata': metadata or {},
        'timestamp': datetime.now().isoformat(),
        'n_detections': len(cuboids),
        'detections': []
    }
    
    for cuboid in cuboids:
        detection = {
            'category': cuboid.get('category', 'Unknown'),
            'center': cuboid['center'].tolist() if isinstance(cuboid['center'], np.ndarray) else cuboid['center'],
            'yaw': float(cuboid['yaw']),
            'dimensions': {
                'length': float(cuboid['length']),
                'width': float(cuboid['width']),
                'height': float(cuboid['height'])
            },
            'corners': cuboid['corners'].tolist() if isinstance(cuboid['corners'], np.ndarray) else cuboid['corners'],
            'bounds': {
                'min_x': float(cuboid['min_x']),
                'max_x': float(cuboid['max_x']),
                'min_y': float(cuboid['min_y']),
                'max_y': float(cuboid['max_y']),
                'min_z': float(cuboid['min_z']),
                'max_z': float(cuboid['max_z'])
            },
            'score': float(cuboid.get('score', 0.0)),
            'method': cuboid.get('method', 'unknown'),
            'n_points': int(cuboid.get('n_points', 0))
        }
        
        export_data['detections'].append(detection)
    
    with open(output_path, 'w') as f:
        json.dump(export_data, f, indent=2)


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
    if 'cuboids' not in st.session_state or not st.session_state.cuboids:
        st.info("👈 Please run the detection pipeline on **2_Detection** page first.")
        return
    
    detected_cuboids = st.session_state.cuboids
    
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
    
    export_format = st.selectbox(
        "Export Format",
        options=['JSON (Custom)', 'KITTI Format', 'COCO Format'],
        help="Choose the export format"
    )
    
    # Get metadata
    metadata = {}
    if 'sample' in st.session_state:
        sample_meta_data = st.session_state.sample['sample_meta_data']
        metadata = {
            'dataset_type': sample_meta_data.get('dataset_type', 'unknown'),
            'sample_index': sample_meta_data.get('sample_index', 'unknown'),
            'image_path': sample_meta_data.get('image_path', 'unknown')
        }
    
    if 'pipeline_state' in st.session_state:
        metadata['pipeline_params'] = st.session_state.params
    
    # Export button
    if st.button("💾 Export Results", type="primary"):
        if export_format == 'JSON (Custom)':
            # Create JSON file
            json_str = json.dumps({
                'metadata': metadata,
                'timestamp': datetime.now().isoformat(),
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
            }, indent=2)
            
            st.download_button(
                label="📥 Download JSON",
                data=json_str,
                file_name=f"detections_{metadata.get('sample_index', 'unknown')}.json",
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
            st.download_button(
                label="📥 Download KITTI Format",
                data=kitti_str,
                file_name=f"detections_{metadata.get('sample_index', 'unknown')}.txt",
                mime="text/plain"
            )
        
        elif export_format == 'COCO Format':
            st.info("COCO format export coming soon...")


if __name__ == "__main__":
    main()

