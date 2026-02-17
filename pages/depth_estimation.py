"""
Page module - extracted from app.py
Combines depth estimation and depth completion functionality.
"""
import streamlit as st
import numpy as np
import pandas as pd
import cv2
import time
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from typing import Dict, List, Optional

from components.utils.visualization_helper import create_3d_scatter_plot
from components.core.frustum_manager import FrustumManager
from components.core.evaluation import compute_3d_iou, run_pipeline_on_sample
from components.core.clustering_manager import ClusteringManager
from components.core.pointcloud_projection import filter_points_in_frustum

def depth_estimation_page():
    """Depth Estimation and 3D Reconstruction page"""
    st.header("🔍 Sparse Depth Backprojection")
    
    # Check if data is loaded
    if 'sample_data' not in st.session_state or st.session_state.sample_data is None:
        st.info("👈 Load a sample from the sidebar to see sparse depth backprojection")
        st.markdown("""
        ### How to use:
        1. Load a KITTI sample using the sidebar controls
        2. The sparse depth map is automatically created by back-projecting 3D LiDAR points onto the 2D image
        3. View the sparse depth map and backprojected 3D point cloud below
        
        **Sparse Depth Backprojection:**
        - When LiDAR point cloud is loaded, the pipeline automatically creates a sparse depth map
        - The sparse depth map is created by back-projecting 3D LiDAR points onto the 2D image
        - Each valid depth pixel is backprojected to 3D with its corresponding image color
        - This allows validation that backprojected points are correctly assigned to segmentation masks
        """)
        
        return
    
    sample_data = st.session_state.sample_data
    
    # Focus on sparse depth map and backprojection (no Marigold reconstruction)
    has_sparse_depth = st.session_state.get('sparse_depth_map') is not None
    has_colored_sparse = (
        st.session_state.get('colored_sparse_points') is not None and 
        len(st.session_state.get('colored_sparse_points', [])) > 0
    )
    
    if not has_sparse_depth:
        st.warning("⚠️ No sparse depth map available. Please load a sample first.")
        return
    
    # Display statistics
    st.subheader("📊 Sparse Depth Statistics")
    
    sparse_depth_map = st.session_state.sparse_depth_map
    n_sparse = np.sum(sparse_depth_map > 0)
    coverage = 100 * n_sparse / (sparse_depth_map.shape[0] * sparse_depth_map.shape[1])
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Sparse Depth Points", f"{n_sparse:,}")
    with col2:
        st.metric("Coverage", f"{coverage:.2f}%")
    with col3:
        if has_colored_sparse:
            colored_points = st.session_state.colored_sparse_points
            st.metric("Backprojected Points", f"{len(colored_points):,}")
        else:
            st.metric("Backprojected Points", "0")
    
    # Display depth map visualization
    st.subheader("🗺️ Sparse Depth Map Visualization")
    
    img = cv2.imread(sample_data['image_path'])
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    sparse_depth = st.session_state.sparse_depth_map
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Original Image**")
        st.image(img_rgb, use_container_width=True)
    
    with col2:
        st.markdown("**Sparse Depth Map**")
        fig_sparse, ax_sparse = plt.subplots(figsize=(8, 6))
        vmax = sparse_depth[sparse_depth > 0].max() if np.sum(sparse_depth > 0) > 0 else 100.0
        im_sparse = ax_sparse.imshow(sparse_depth, cmap='viridis', vmin=0, vmax=vmax)
        ax_sparse.set_title("Sparse Depth (from LiDAR)")
        ax_sparse.axis('off')
        plt.colorbar(im_sparse, ax=ax_sparse, fraction=0.046, pad=0.04, label="Depth (m)")
        st.pyplot(fig_sparse)
        plt.close()
    
    # 3D Visualization of backprojected sparse depth points
    st.subheader("🎯 3D Backprojected Sparse Depth Points")
    
    if not has_colored_sparse:
        st.info("No backprojected sparse depth points available. They are computed automatically when loading a sample.")
        return
    
    colored_points = st.session_state.colored_sparse_points
    colors = st.session_state.colored_sparse_colors
    
    # Options for visualization
    col1, col2 = st.columns(2)
    with col1:
        show_lidar = st.checkbox("Show Original LiDAR", value=True, key="show_lidar_depth")
    with col2:
        show_backprojected = st.checkbox("Show Backprojected Points", value=True, key="show_backprojected_depth")
    
    # Create 3D visualization
    point_cloud_obj = st.session_state.point_cloud
    fig = go.Figure()
    
    # Add LiDAR background if requested
    if show_lidar and point_cloud_obj is not None:
        lidar_points = point_cloud_obj.point_cloud_plane_removed
        if len(lidar_points) > 0:
            sample_size = min(10000, len(lidar_points))
            sample_indices = np.random.choice(len(lidar_points), sample_size, replace=False)
            sampled_lidar = lidar_points[sample_indices]
            
            fig.add_trace(go.Scatter3d(
                x=sampled_lidar[:, 0],
                y=sampled_lidar[:, 1],
                z=sampled_lidar[:, 2],
                mode='markers',
                marker=dict(size=1, color='rgb(200, 200, 200)', opacity=0.3),
                name='LiDAR Points'
            ))
    
    # Add backprojected sparse depth points with their image colors
    if show_backprojected and len(colored_points) > 0:
        colors_rgb = [f'rgb({int(c[0])},{int(c[1])},{int(c[2])})' for c in colors]
        
        fig.add_trace(go.Scatter3d(
            x=colored_points[:, 0],
            y=colored_points[:, 1],
            z=colored_points[:, 2],
            mode='markers',
            marker=dict(
                size=3,
                color=colors_rgb,
                opacity=0.9
            ),
            name='Backprojected Sparse Depth Points'
        ))
    
    fig.update_layout(
        title="3D Backprojected Sparse Depth Points (Colored by Image)",
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            aspectmode='data'
        ),
        width=1000,
        height=800
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistics
    st.subheader("📈 Point Cloud Statistics")
    if st.session_state.point_cloud is not None:
        lidar_points = st.session_state.point_cloud.point_cloud_plane_removed
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("LiDAR Points", f"{len(lidar_points):,}")
            if len(lidar_points) > 0:
                st.caption(f"X: [{lidar_points[:, 0].min():.1f}, {lidar_points[:, 0].max():.1f}]")
                st.caption(f"Y: [{lidar_points[:, 1].min():.1f}, {lidar_points[:, 1].max():.1f}]")
                st.caption(f"Z: [{lidar_points[:, 2].min():.1f}, {lidar_points[:, 2].max():.1f}]")
        
        with col2:
            st.metric("Backprojected Points", f"{len(colored_points):,}")
            if len(colored_points) > 0:
                st.caption(f"X: [{colored_points[:, 0].min():.1f}, {colored_points[:, 0].max():.1f}]")
                st.caption(f"Y: [{colored_points[:, 1].min():.1f}, {colored_points[:, 1].max():.1f}]")
                st.caption(f"Z: [{colored_points[:, 2].min():.1f}, {colored_points[:, 2].max():.1f}]")

