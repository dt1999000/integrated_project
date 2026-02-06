"""
Page module - extracted from app.py
"""
import streamlit as st
import numpy as np
import pandas as pd
import cv2
import time
import matplotlib.pyplot as plt
from typing import Dict, List, Optional

from components.utils.visualization_helper import (
    draw_2d_boxes_on_image,
    draw_projected_cuboid_bboxes,
    add_frustums_to_figure,
    add_cuboids_to_figure,
    create_3d_scatter_plot,
    create_comparison_plot,
)
from components.core.frustum_manager import FrustumManager
from components.core.evaluation import compute_3d_iou, run_pipeline_on_sample
from components.core.clustering_manager import ClusteringManager
from components.core.pointcloud_projection import filter_points_in_frustum

def kitti_groundtruth_page():
    """KITTI Ground Truth Comparison page - Uses same pipeline as other pages"""
    st.header("🎯 KITTI Ground Truth Comparison")

    # Check if we have KITTI data loaded
    if 'sample_data' not in st.session_state or st.session_state.get('current_dataset') != 'kitti':
        st.info("👈 Switch to KITTI dataset in the sidebar and load a sample to compare with ground truth")
        st.markdown("""
        ### How to use:
        1. In the **Data Controls** sidebar, select **Dataset: KITTI**
        2. Choose a sample index (0-7480)
        3. Click **Load Sample**
        4. Run clustering on other tabs (DBSCAN, BIRCH, etc.)
        5. Return here to see ground truth vs detected objects

        The ground truth 3D cuboids from KITTI will be shown in **green**,
        while your pipeline's detected clusters will be shown in **red**.
        """)
        return

    sample_data = st.session_state.sample_data
    point_cloud = st.session_state.point_cloud

    # Get ground truth boxes for visualization
    ground_truth_boxes = sample_data.get('ground_truth_boxes', [])

    # Sidebar controls for 2D visualization
    with st.sidebar.expander("Visualization Settings", expanded=True):
        show_2d_boxes = st.checkbox("Show 2D Bounding Boxes", value=True, key="show_2d_boxes_kitti")
        match_distance_threshold = st.slider("Match Distance Threshold (m)", min_value=1.0, max_value=10.0,
                                             value=5.0, step=0.5, key="match_dist_kitti",
                                             help="Maximum distance to match detected cuboid to GT")

    # Note: Frustum-based clustering is now automatic
    st.sidebar.info("Frustum filtering is automatic when running clustering on KITTI data. "
                    "Select a clustering algorithm in the sidebar to update results.")


    # Display camera image with optional 2D bounding boxes
    st.subheader("📷 Camera Image")
    try:
        img = cv2.imread(sample_data['image_path'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Draw 2D bounding boxes if enabled
        if show_2d_boxes and ground_truth_boxes:
            img = draw_2d_boxes_on_image(img, ground_truth_boxes)
            caption = f"KITTI Sample {sample_data.get('sample_index', 0)} - With 2D Ground Truth Boxes"
        else:
            caption = f"KITTI Sample {sample_data.get('sample_index', 0)}"

        st.image(img, caption=caption, width='stretch')

        # Show 2D bbox statistics
        if show_2d_boxes and ground_truth_boxes:
            n_boxes_with_2d = sum(1 for box in ground_truth_boxes if box.get('bbox_2d') is not None)
            st.caption(f"Showing {n_boxes_with_2d} 2D bounding boxes from KITTI annotations")

    except Exception as e:
        st.warning(f"Could not load image: {str(e)}")

    # Show reprojected cuboid bboxes if cuboids are available
    if 'cuboids' in st.session_state and st.session_state.cuboids:
        print(f'show reprojected cuboid bounding boxes')
        detected_cuboids = st.session_state.cuboids
        # Check if any cuboids have projected_bbox_2d (from find_best_cuboid)
        cuboids_with_proj = [c for c in detected_cuboids]
        if cuboids_with_proj:
            st.subheader("📐 Reprojected Cuboid Bounding Boxes")
            st.markdown("""
            **Color Legend:**
            - **Thin boxes (green/blue/etc.)**: Original 2D ground truth bounding boxes
            - **Thick boxes (orange tones)**: Reprojected 2D bboxes from detected 3D cuboids
            - **IoU**: Intersection over Union between reprojected and original bbox
            """)

            try:
                img_proj = cv2.imread(sample_data['image_path'])
                img_proj = cv2.cvtColor(img_proj, cv2.COLOR_BGR2RGB)

                # Draw both original GT boxes and reprojected cuboid bboxes
                img_proj = draw_projected_cuboid_bboxes(img_proj, cuboids_with_proj, ground_truth_boxes)

                st.image(img_proj, caption=f"Reprojected 3D Cuboids to 2D - {len(cuboids_with_proj)} cuboids",
                         use_container_width=True)

                # Show IoU statistics
                ious = [c.get('iou', 0) for c in cuboids_with_proj if c.get('iou') is not None]
                if ious:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Cuboids with Proj", len(cuboids_with_proj))
                    with col2:
                        st.metric("Mean IoU", f"{np.mean(ious):.3f}")
                    with col3:
                        st.metric("Min IoU", f"{np.min(ious):.3f}")
                    with col4:
                        st.metric("Max IoU", f"{np.max(ious):.3f}")

            except Exception as e:
                st.warning(f"Could not render reprojected bboxes: {str(e)}")

    # Display ground truth info
    if ground_truth_boxes:
        st.subheader("📦 Ground Truth Objects")
        categories = [box['category'] for box in ground_truth_boxes]
        category_counts = pd.DataFrame({'Category': categories}).value_counts().reset_index()
        category_counts.columns = ['Category', 'Count']

        col1, col2 = st.columns([1, 2])
        with col1:
            st.dataframe(category_counts, width='stretch')
        with col2:
            st.markdown(f"""
            **Total Ground Truth Objects**: {len(ground_truth_boxes)}

            These are the human-annotated 3D bounding boxes from the KITTI dataset.
            They will be visualized in **green** in the comparison plots below.
            """)

    else:
        st.warning("No ground truth boxes available for this sample")

    # Check if user has run clustering
    st.subheader("🔴 Pipeline Detection Results")
    detected_cuboids = st.session_state.get('cuboids', [])
    available_algos = [
        algo for algo in ['hdbscan', 'dbscan', 'birch', 'agglomerative', 'optics', 'frustum_filtered']
        if algo in st.session_state.clustering_results
    ]

    if not available_algos and not detected_cuboids:
        st.info("Run a clustering algorithm to see pipeline detection results.")
        return

    clustering_result = None
    algo_name = None
    if available_algos:
        algo_labels = {
            'hdbscan': 'HDBSCAN',
            'dbscan': 'DBSCAN',
            'birch': 'BIRCH',
            'agglomerative': 'Agglomerative',
            'optics': 'OPTICS',
            'frustum_filtered': 'Frustum Filtered'
        }
        label_to_algo = {algo_labels[algo]: algo for algo in available_algos}
        selected_label = st.selectbox("Clustering Result", list(label_to_algo.keys()))
        algo_name = label_to_algo[selected_label]
        clustering_result = st.session_state.clustering_results.get(algo_name)

    labels = clustering_result.get('labels') if clustering_result else None
    is_frustum_filtered = False
    if clustering_result:
        is_frustum_filtered = clustering_result.get('is_frustum_based', False) or algo_name == 'frustum_filtered'

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Ground Truth Objects", len(ground_truth_boxes))
    with col2:
        if detected_cuboids:
            detected_count = len(detected_cuboids)
        elif labels is not None:
            unique_labels = np.unique(labels)
            detected_count = len(unique_labels) - (1 if -1 in unique_labels else 0)
        else:
            detected_count = 0
        st.metric("Detected Clusters", detected_count)
    with col3:
        if is_frustum_filtered and clustering_result:
            bbox_results = clustering_result.get('bbox_results', [])
            n_successful = sum(1 for r in bbox_results if r['status'] == 'success')
            st.metric("BBoxes with Clusters", n_successful)
        elif labels is not None:
            n_noise = np.sum(labels == -1) if -1 in labels else 0
            st.metric("Noise Points", n_noise)
        elif clustering_result:
            st.metric("Method", "Frustum-based")
        else:
            st.metric("Method", "Cuboids only")

    if not detected_cuboids:
        st.info("No cuboids available for IoU matching yet.")
        return

    # 3D IoU Matching Statistics Section
    st.subheader("📊 3D IoU Matching Statistics")
    st.markdown("""
    **Matching Logic:** Each detected cuboid is matched to the ground truth box using `source_bbox_idx`
    which corresponds to the frustum index. The 3D IoU measures volumetric overlap between
    detected and ground truth cuboids.
    """)

    # Compute 3D IoU for each matched pair
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
                'Need Review': detected.get('need_review', False)
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

        # Count by IoU thresholds
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

        # Detailed per-object table
        with st.expander("📋 Per-Object Matching Details", expanded=True):
            df_matching = pd.DataFrame(matching_results)
            # Format IoU columns
            df_matching['3D IoU'] = df_matching['3D IoU'].apply(lambda x: f"{x:.3f}")
            df_matching['2D IoU'] = df_matching['2D IoU'].apply(lambda x: f"{x:.3f}" if x is not None else "N/A")
            st.dataframe(df_matching, use_container_width=True)

        # Compare 2D vs 3D IoU if both available
        if iou_2d_values and len(iou_2d_values) == len(iou_3d_values):
            st.markdown("**2D vs 3D IoU Comparison:**")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Mean 2D IoU", f"{np.mean(iou_2d_values):.3f}")
            with col2:
                correlation = np.corrcoef(iou_2d_values, iou_3d_values)[0, 1]
                st.metric("2D-3D Correlation", f"{correlation:.3f}")

    else:
        st.info("No matched pairs found. Ensure detected cuboids have `source_bbox_idx` set.")

    # Unified comparison view
    st.subheader("🎯 Unified Comparison View")
    st.markdown("""
    **Color Legend:** Ground truth cuboids use lighter shades, detected cuboids use darker shades of the same color (by category).
    Yellow lines connect matched pairs.
    """)

    fig_unified = create_comparison_plot(point_cloud, ground_truth_boxes, detected_cuboids)
    st.plotly_chart(fig_unified, width='stretch', key='kitti_comparison_chart')

