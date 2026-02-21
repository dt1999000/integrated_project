"""
Evaluation Page
Evaluate detection results against ground truth (primarily for KITTI).
"""
import streamlit as st
import numpy as np
import pandas as pd
import cv2
from typing import List, Dict

from components.core.evaluation import compute_3d_iou
from components.utils.visualization_helper import (
    draw_2d_boxes_on_image,
    draw_projected_cuboid_bboxes,
    create_comparison_plot,
)


def main():
    """Main evaluation page function"""
    st.set_page_config(
        page_title="Evaluation",
        page_icon="📊",
        layout="wide"
    )
    
    st.header("📊 Detection Evaluation")
    st.markdown("""
    Evaluate detection results against ground truth.
    Computes 3D IoU and 2D IoU metrics for comparison.
    """)
    
    # Check if sample and detection results are available
    if 'sample' not in st.session_state or st.session_state.sample is None:
        st.info("👈 Please load a sample from **1_Dataset_Extraction** page first.")
        return
    
    if 'cuboids' not in st.session_state or not st.session_state.cuboids:
        st.info("👈 Please run the detection pipeline on **2_Detection** page first.")
        return
    
    sample = st.session_state.sample
    sample_meta_data = sample['sample_meta_data']
    detected_cuboids = st.session_state.cuboids
    ground_truth_boxes = sample_meta_data.get('ground_truth_boxes', [])
    
    if not ground_truth_boxes:
        st.warning("⚠️ No ground truth boxes available for this sample.")
        st.info("Evaluation requires ground truth annotations (typically available for KITTI dataset).")
        return
    
    # Get point cloud for visualization
    point_cloud_obj = None
    if 'pipeline_state' in st.session_state:
        step_1_result = st.session_state.pipeline_state.get('step_1', {}).get('result')
        if step_1_result:
            point_cloud_obj = step_1_result.get('point_cloud_obj')
    
    # Display statistics
    st.subheader("📈 Detection Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Ground Truth Objects", len(ground_truth_boxes))
    with col2:
        st.metric("Detected Objects", len(detected_cuboids))
    with col3:
        detection_rate = len(detected_cuboids) / len(ground_truth_boxes) * 100 if ground_truth_boxes else 0
        st.metric("Detection Rate", f"{detection_rate:.1f}%")
    
    # Compute 3D IoU for matched pairs
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
        # Summary metrics
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
        
        # IoU threshold counts
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
        
        # Detailed table
        with st.expander("📋 Per-Object Matching Details", expanded=True):
            df_matching = pd.DataFrame(matching_results)
            df_matching['3D IoU'] = df_matching['3D IoU'].apply(lambda x: f"{x:.3f}")
            df_matching['2D IoU'] = df_matching['2D IoU'].apply(lambda x: f"{x:.3f}" if x is not None else "N/A")
            st.dataframe(df_matching, use_container_width=True)
    
    # 2D Visualization
    st.subheader("📷 2D Visualization")
    image = sample['image']
    
    # Show image with ground truth boxes
    img_with_gt = draw_2d_boxes_on_image(image.copy(), ground_truth_boxes)
    st.image(img_with_gt, caption="Image with Ground Truth Boxes", use_container_width=True)
    
    # Show reprojected cuboid bboxes if available
    if detected_cuboids:
        st.subheader("📐 Reprojected Cuboid Bounding Boxes")
        img_proj = draw_projected_cuboid_bboxes(image.copy(), detected_cuboids, ground_truth_boxes)
        st.image(img_proj, caption="Reprojected 3D Cuboids to 2D", use_container_width=True)
    
    # 3D Comparison Visualization
    if point_cloud_obj:
        st.subheader("🎯 3D Comparison Visualization")
        fig_unified = create_comparison_plot(point_cloud_obj, ground_truth_boxes, detected_cuboids)
        st.plotly_chart(fig_unified, use_container_width=True)


if __name__ == "__main__":
    main()

