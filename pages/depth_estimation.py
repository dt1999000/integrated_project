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

def depth_estimation_page():
    """Depth Estimation and 3D Reconstruction page"""
    st.header("🔍 Depth Estimation & 3D Reconstruction")
    
    # Check if data is loaded
    if 'sample_data' not in st.session_state or st.session_state.sample_data is None:
        st.info("👈 Load a sample from the sidebar and click 'Estimate Depth' to get started")
        st.markdown("""
        ### How to use:
        1. Load a KITTI sample using the sidebar controls
        2. Select depth estimation model in the sidebar (Marigold or Depth Anything)
        3. Click the **🔍 Estimate Depth** button in the sidebar
        4. Wait for the depth estimation to complete (may take 10-30 seconds)
        5. View the depth map and reconstructed 3D point cloud below
        
        **Available Models:**
        - **Marigold**: High-quality metric depth estimation (requires GPU/CUDA for best performance)
        - **Marigold-DC with Sparse Prior**: If LiDAR points are available, automatically uses them as sparse depth guidance for improved accuracy
        - **Depth Anything**: Fast depth estimation, works on CPU
        
        **Sparse Depth Prior (Automatic):**
        - When LiDAR point cloud is loaded, the pipeline automatically creates a sparse depth map
        - Uses Marigold-DC with sparse depth guidance instead of regular Marigold
        - Provides better accuracy by incorporating ground truth depth measurements from LiDAR
        - The sparse depth map is created by back-projecting 3D LiDAR points onto the 2D image
        
        The tool reconstructs a dense 3D point cloud from the depth map that can be combined with LiDAR data.
        """)
        
        # Show system information
        try:
            from depth_estimation import MARIGOLD_AVAILABLE, DEPTH_ANYTHING_AVAILABLE
            with st.expander("📋 Model Availability"):
                col1, col2 = st.columns(2)
                with col1:
                    if MARIGOLD_AVAILABLE:
                        st.success("✅ Marigold Available")
                    else:
                        st.warning("⚠️ Marigold Unavailable")
                        st.caption("Install with: pip install diffusers")
                with col2:
                    if DEPTH_ANYTHING_AVAILABLE:
                        st.success("✅ Depth Anything V2 Available")
                    else:
                        st.info("ℹ️ Using Depth Anything Small (HF)")
        except:
            pass
        
        return
    
    sample_data = st.session_state.sample_data
    
    # Check if depth has been estimated
    if st.session_state.depth_map is None:
        st.warning("⚠️ No depth map available. Click '🔍 Estimate Depth' in the sidebar to run depth estimation.")
        return
    
    depth_map = st.session_state.depth_map
    reconstructed_points = st.session_state.reconstructed_points
    
    # Display statistics
    st.subheader("📊 Depth Estimation Statistics")
    
    # Check if sparse depth prior was used
    used_sparse_prior = st.session_state.get('sparse_depth_map') is not None
    sparse_info = ""
    if used_sparse_prior:
        sparse_depth_prior = st.session_state.sparse_depth_map
        n_sparse = np.sum(sparse_depth_prior > 0)
        coverage = 100 * n_sparse / (sparse_depth_prior.shape[0] * sparse_depth_prior.shape[1])
        sparse_info = f" (with {n_sparse:,} sparse depth points, {coverage:.2f}% coverage)"
        st.info(f"✅ Used sparse depth prior from LiDAR{sparse_info}")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Depth Map Shape", f"{depth_map.shape[0]}×{depth_map.shape[1]}")
    with col2:
        st.metric("Min Depth", f"{depth_map.min():.2f}m")
    with col3:
        st.metric("Max Depth", f"{depth_map.max():.2f}m")
    with col4:
        st.metric("Reconstructed Points", f"{len(reconstructed_points):,}")
    
    # Display depth map visualization
    st.subheader("🗺️ Depth Map Visualization")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Original Image**")
        img = cv2.imread(sample_data['image_path'])
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        st.image(img_rgb, use_container_width=True)
    
    with col2:
        st.markdown("**Estimated Depth Map**")
        # Create colorized depth map
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(depth_map, cmap='viridis')
        ax.set_title("Depth Map (meters)")
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        st.pyplot(fig)
        plt.close()
    
    # 3D Visualization of reconstructed points
    st.subheader("🎯 3D Reconstructed Point Cloud")
    
    # Options for visualization
    col1, col2, col3 = st.columns(3)
    with col1:
        show_lidar = st.checkbox("Show Original LiDAR", value=True, key="show_lidar_depth")
    with col2:
        show_reconstructed = st.checkbox("Show Reconstructed Points", value=True, key="show_reconstructed_depth")
    with col3:
        color_by_depth = st.checkbox("Color by Depth", value=True, key="color_by_depth")
    
    # Create 3D visualization using the adapted helper function
    point_cloud_obj = st.session_state.point_cloud
    fig = create_3d_scatter_plot(
        points=point_cloud_obj,
        labels=None,
        mask_points=None,
        cuboids=None,
        rays=None,
        points_in_frustums=None,
        reconstructed_points=reconstructed_points if show_reconstructed else None,
        show_lidar=show_lidar,
        show_reconstructed=show_reconstructed,
        color_by_depth=color_by_depth,
        title="3D Point Cloud Comparison"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistics comparison
    st.subheader("📈 Point Cloud Comparison")
    if st.session_state.point_cloud is not None:
        lidar_points = st.session_state.point_cloud.point_cloud_plane_removed
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("LiDAR Points", f"{len(lidar_points):,}")
            st.caption(f"X: [{lidar_points[:, 0].min():.1f}, {lidar_points[:, 0].max():.1f}]")
            st.caption(f"Y: [{lidar_points[:, 1].min():.1f}, {lidar_points[:, 1].max():.1f}]")
            st.caption(f"Z: [{lidar_points[:, 2].min():.1f}, {lidar_points[:, 2].max():.1f}]")
        
        with col2:
            st.metric("Reconstructed Points", f"{len(reconstructed_points):,}")
            st.caption(f"X: [{reconstructed_points[:, 0].min():.1f}, {reconstructed_points[:, 0].max():.1f}]")
            st.caption(f"Y: [{reconstructed_points[:, 1].min():.1f}, {reconstructed_points[:, 1].max():.1f}]")
            st.caption(f"Z: [{reconstructed_points[:, 2].min():.1f}, {reconstructed_points[:, 2].max():.1f}]")
        
        with col3:
            combined_count = len(lidar_points) + len(reconstructed_points)
            st.metric("Combined Total", f"{combined_count:,}")
            density_increase = (len(reconstructed_points) / len(lidar_points)) * 100
            st.caption(f"Density increase: +{density_increase:.1f}%")
    
    # Option to add reconstructed points to point cloud
    st.subheader("🔧 Point Cloud Integration")
    st.markdown("""
    You can add the reconstructed points to the current point cloud for use in clustering algorithms.
    This will create a denser point cloud by combining LiDAR and depth-based reconstruction.
    """)
    
    if st.button("➕ Add Reconstructed Points to Point Cloud", key="add_reconstructed"):
        if st.session_state.point_cloud is not None:
            st.session_state.point_cloud.add_projected_points(reconstructed_points)
            st.success(f"✅ Added {len(reconstructed_points):,} reconstructed points to point cloud!")
            st.info("The combined point cloud is now available for clustering algorithms.")
        else:
            st.error("No point cloud loaded. Load a sample first.")
    
    # Export options
    with st.expander("💾 Export Options"):
        st.markdown("**Export reconstructed point cloud**")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Download as .npy", key="export_npy"):
                np.save('reconstructed_points.npy', reconstructed_points)
                st.success("Saved to reconstructed_points.npy")
        with col2:
            if st.button("Download depth map", key="export_depth"):
                np.save('depth_map.npy', depth_map)
                st.success("Saved to depth_map.npy")

