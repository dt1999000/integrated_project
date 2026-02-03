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

from visualization_helper import (
    draw_2d_boxes_on_image,
    draw_projected_cuboid_bboxes,
    add_frustums_to_figure,
    add_cuboids_to_figure,
    create_3d_scatter_plot,
    create_comparison_plot,
)
from frustum_manager import FrustumManager
from evaluation import compute_3d_iou, run_pipeline_on_sample
from clustering_manager import ClusteringManager
from pointcloud_projection import filter_points_in_frustum

def project_segmentation_mask_on_pointcloud_page(sample_data, point_cloud):
    st.header("🎯 Projection & Frustum Visualization")

    # Check if we're using KITTI dataset - show frustum visualization instead of segmentation
    if st.session_state.get('current_dataset') == 'kitti':
        st.subheader("KITTI 2D→3D Frustum Projection")

        sample_data_full = st.session_state.get('sample_data', {})
        ground_truth_boxes = sample_data_full.get('ground_truth_boxes', [])
        has_2d_bboxes = any(box.get('bbox_2d') is not None for box in ground_truth_boxes)

        if not has_2d_bboxes:
            st.warning("No 2D bounding boxes available for this sample.")
            return

        # Frustum parameters
        with st.sidebar.expander("Frustum Parameters", expanded=True):
            frustum_depth = st.slider("Frustum Depth (m)", min_value=5, max_value=100, value=100, step=5,
                                      key="frustum_depth_projection",
                                      help="How far to project the 2D bounding boxes into 3D space")
            frustum_opacity = st.slider("Frustum Opacity", min_value=0.05, max_value=0.5, value=0.2, step=0.05,
                                        key="frustum_opacity_projection")

        # Compute frustums using FrustumManager
        fm = FrustumManager(
            sample_data_full['camera_intrinsic'],
            sample_data_full['camera_to_lidar_transform']
        )
        frustums = fm.create_frustums_from_bboxes(ground_truth_boxes, depth=frustum_depth)
        st.session_state.frustums = frustums

        # Show camera image with 2D bboxes
        st.subheader("Camera Image with 2D Bounding Boxes")
        try:
            img = cv2.imread(sample_data_full['image_path'])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = draw_2d_boxes_on_image(img, ground_truth_boxes)
            st.image(img, caption=f"KITTI Sample - {len(frustums)} 2D Bounding Boxes", width="stretch")
        except Exception as e:
            st.warning(f"Could not load image: {str(e)}")

        # Show frustum statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total 2D Bboxes", len([b for b in ground_truth_boxes if b.get('bbox_2d')]))
        with col2:
            st.metric("Frustums Generated", len(frustums))
        with col3:
            categories = [f.category for f in frustums]
            unique_cats = len(set(categories))
            st.metric("Object Categories", unique_cats)

        # Show frustums in 3D
        st.subheader("3D Point Cloud with Frustums")
        point_cloud_obj = st.session_state.get('point_cloud')
        points_in_frustums = np.array([]).reshape(0, 3)
        for frustum in frustums:
            points_in_frustum, mask = filter_points_in_frustum(point_cloud_obj.point_cloud_plane_removed, frustum.camera_origin, frustum.base_corners) 
            points_in_frustums = np.concatenate([points_in_frustums, points_in_frustum])
        
        if point_cloud_obj is not None:
            fig = create_3d_scatter_plot(
                points=point_cloud_obj,
                labels=None,
                mask_points=None,
                cuboids=None,
                rays=None,
                points_in_frustums=points_in_frustums,
                title="Point Cloud with 2D→3D Frustum Projections"
            )

            # Add frustums to the figure
            camera_intrinsic = sample_data_full.get('camera_intrinsic')
            camera_to_lidar = sample_data_full.get('camera_to_lidar_transform')
            if camera_intrinsic is not None and camera_to_lidar is not None:
                add_frustums_to_figure(
                    fig, ground_truth_boxes,
                    camera_intrinsic, camera_to_lidar,
                    depth=frustum_depth, opacity=frustum_opacity
                )

            st.plotly_chart(fig, width='stretch', key='frustum_projection_chart')

        # Per-frustum info table
        if frustums:
            st.subheader("Frustum Details")
            frustum_data = []
            for f in frustums:
                frustum_data.append({
                    'Index': f.idx,
                    'Category': f.category,
                    'BBox Left': f.bbox_2d['left'],
                    'BBox Top': f.bbox_2d['top'],
                    'BBox Right': f.bbox_2d['right'],
                    'BBox Bottom': f.bbox_2d['bottom']
                })
            df = pd.DataFrame(frustum_data)
            st.dataframe(df, use_container_width=True)

        st.info("Select a clustering algorithm in the sidebar to see per-frustum clustering results. "
                "Clusters are automatically filtered by these frustums when using KITTI data.")
        return

