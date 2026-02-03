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

def depth_completion_page():
    """Depth Completion page using Marigold-DC"""
    st.header("🎯 Depth Completion (Marigold-DC)")
    
    # Check if data is loaded
    if 'sample_data' not in st.session_state or st.session_state.sample_data is None:
        st.info("👈 Load a sample from the sidebar and click '🎯 Complete Depth' to get started")
        st.markdown("""
        ### How to use:
        1. Load a KITTI sample using the sidebar controls
        2. Initialize depth estimator (click '🔍 Estimate Depth' or it will auto-initialize)
        3. Adjust Marigold-DC parameters in the sidebar (expand 'Marigold-DC Parameters')
        4. Click the **🎯 Complete Depth** button in the sidebar
        5. Wait for depth completion to finish (may take 30-120 seconds depending on settings)
        6. View the sparse depth map, completed depth map, and comparison below
        
        **What is Depth Completion?**
        - Takes sparse depth measurements (from LiDAR) and completes them to dense depth maps
        - Uses Marigold-DC diffusion model to fill in missing depth values
        - Combines the accuracy of LiDAR with the density of monocular depth estimation
        
        **Parameters:**
        - **Inference Steps**: More steps = better quality but slower (10-100)
        - **Ensemble Size**: Multiple predictions averaged together (1-4)
        - **Processing Resolution**: Higher resolution = better quality but slower (256-1024)
        """)
        
        # Show system information
        try:
            from depth_estimation import MARIGOLD_AVAILABLE
            with st.expander("📋 Model Availability"):
                if MARIGOLD_AVAILABLE:
                    st.success("✅ Marigold-DC Available")
                else:
                    st.warning("⚠️ Marigold-DC Unavailable")
                    st.caption("Install with: pip install diffusers")
        except:
            pass
        
        return
    
    sample_data = st.session_state.sample_data
    
    # Check if depth completion has been run
    if st.session_state.completed_depth_map is None:
        st.warning("⚠️ No completed depth map available. Click '🎯 Complete Depth' in the sidebar to run depth completion.")
        
        # Show info about what will happen
        if st.session_state.point_cloud is not None:
            point_cloud = st.session_state.point_cloud.point_cloud_plane_removed
            st.info(f"Ready to process: {len(point_cloud):,} LiDAR points will be projected to create sparse depth map.")
        return
    
    sparse_depth = st.session_state.sparse_depth_map
    completed_depth = st.session_state.completed_depth_map
    
    # Display statistics
    st.subheader("📊 Depth Completion Statistics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Sparse Points", f"{np.sum(sparse_depth > 0):,}")
        coverage = 100 * np.sum(sparse_depth > 0) / (sparse_depth.shape[0] * sparse_depth.shape[1])
        st.caption(f"Coverage: {coverage:.2f}%")
    with col2:
        st.metric("Completed Pixels", f"{np.sum(completed_depth > 0):,}")
        st.caption(f"Coverage: 100%")
    with col3:
        if np.sum(sparse_depth > 0) > 0:
            st.metric("Sparse Depth Range", f"{sparse_depth[sparse_depth>0].min():.1f}-{sparse_depth[sparse_depth>0].max():.1f}m")
        else:
            st.metric("Sparse Depth Range", "N/A")
    with col4:
        st.metric("Completed Depth Range", f"{completed_depth.min():.1f}-{completed_depth.max():.1f}m")
    
    # Display visualizations
    st.subheader("🖼️ Depth Map Comparison")
    
    # Load original image
    img = cv2.imread(sample_data['image_path'])
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Original Image**")
        st.image(img_rgb, use_container_width=True)
    
    with col2:
        st.markdown("**Sparse Depth Map**")
        fig_sparse, ax_sparse = plt.subplots(figsize=(8, 6))
        im_sparse = ax_sparse.imshow(sparse_depth, cmap='viridis', vmin=0, vmax=completed_depth.max())
        ax_sparse.set_title("Sparse Depth (from LiDAR)")
        ax_sparse.axis('off')
        plt.colorbar(im_sparse, ax=ax_sparse, fraction=0.046, pad=0.04, label="Depth (m)")
        st.pyplot(fig_sparse)
        plt.close()
    
    with col3:
        st.markdown("**Completed Depth Map**")
        fig_completed, ax_completed = plt.subplots(figsize=(8, 6))
        im_completed = ax_completed.imshow(completed_depth, cmap='viridis', vmin=0, vmax=completed_depth.max())
        ax_completed.set_title("Completed Depth (Marigold-DC)")
        ax_completed.axis('off')
        plt.colorbar(im_completed, ax=ax_completed, fraction=0.046, pad=0.04, label="Depth (m)")
        st.pyplot(fig_completed)
        plt.close()
    
    # Side-by-side comparison
    st.subheader("📈 Detailed Comparison")
    
    # Create overlay visualization
    fig_overlay, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Sparse depth overlay
    axes[0].imshow(img_rgb)
    sparse_overlay = axes[0].imshow(sparse_depth, cmap='viridis', alpha=0.6, vmin=0, vmax=completed_depth.max())
    axes[0].set_title("Sparse Depth Overlay")
    axes[0].axis('off')
    plt.colorbar(sparse_overlay, ax=axes[0], fraction=0.046, pad=0.04, label="Depth (m)")
    
    # Completed depth overlay
    axes[1].imshow(img_rgb)
    completed_overlay = axes[1].imshow(completed_depth, cmap='viridis', alpha=0.6, vmin=0, vmax=completed_depth.max())
    axes[1].set_title("Completed Depth Overlay")
    axes[1].axis('off')
    plt.colorbar(completed_overlay, ax=axes[1], fraction=0.046, pad=0.04, label="Depth (m)")
    
    plt.tight_layout()
    st.pyplot(fig_overlay)
    plt.close()
    
    # Depth difference analysis
    st.subheader("🔍 Depth Analysis")
    
    # Compare sparse and completed depths where sparse exists
    sparse_mask = sparse_depth > 0
    if np.sum(sparse_mask) > 0:
        sparse_values = sparse_depth[sparse_mask]
        completed_values_at_sparse = completed_depth[sparse_mask]
        
        depth_diff = completed_values_at_sparse - sparse_values
        depth_diff_pct = 100 * depth_diff / (sparse_values + 1e-6)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean Absolute Error", f"{np.mean(np.abs(depth_diff)):.3f}m")
        with col2:
            st.metric("Mean Relative Error", f"{np.mean(np.abs(depth_diff_pct)):.2f}%")
        with col3:
            st.metric("RMSE", f"{np.sqrt(np.mean(depth_diff**2)):.3f}m")
        with col4:
            st.metric("Max Error", f"{np.max(np.abs(depth_diff)):.3f}m")
        
        # Error distribution
        fig_error, ax_error = plt.subplots(figsize=(10, 4))
        ax_error.hist(depth_diff, bins=50, alpha=0.7, edgecolor='black')
        ax_error.axvline(0, color='red', linestyle='--', label='Zero Error')
        ax_error.set_xlabel("Depth Error (m)")
        ax_error.set_ylabel("Frequency")
        ax_error.set_title("Depth Completion Error Distribution")
        ax_error.legend()
        ax_error.grid(True, alpha=0.3)
        st.pyplot(fig_error)
        plt.close()
    
    # 3D Visualization option
    st.subheader("🎯 3D Visualization")
    
    col1, col2 = st.columns(2)
    with col1:
        show_sparse_3d = st.checkbox("Show Sparse 3D Points", value=True, key="show_sparse_3d")
    with col2:
        show_completed_3d = st.checkbox("Show Completed 3D Points", value=True, key="show_completed_3d")
    
    if show_sparse_3d or show_completed_3d:
        # Reconstruct 3D points from completed depth
        if show_completed_3d:
            # Ensure camera parameters are set
            if st.session_state.depth_estimator.camera_intrinsic is None:
                st.session_state.depth_estimator.set_camera_params(
                    camera_intrinsic=sample_data['camera_intrinsic'],
                    camera_to_lidar_transform=sample_data['camera_to_lidar_transform']
                )
            
            completed_points = st.session_state.depth_estimator.reconstruct_points_from_depth(
                depth_map=completed_depth,
                stride=2  # Subsample for visualization
            )
        else:
            completed_points = None
        
        # Create 3D visualization
        fig_3d = go.Figure()
        
        # Add sparse points (from original point cloud)
        if show_sparse_3d and st.session_state.point_cloud is not None:
            sparse_points_3d = st.session_state.point_cloud.point_cloud_plane_removed
            fig_3d.add_trace(go.Scatter3d(
                x=sparse_points_3d[:, 0],
                y=sparse_points_3d[:, 1],
                z=sparse_points_3d[:, 2],
                mode='markers',
                marker=dict(size=2, color='blue', opacity=0.5),
                name='Sparse LiDAR Points'
            ))
        
        # Add completed points
        if show_completed_3d and completed_points is not None:
            depths_completed = np.linalg.norm(completed_points, axis=1)
            fig_3d.add_trace(go.Scatter3d(
                x=completed_points[:, 0],
                y=completed_points[:, 1],
                z=completed_points[:, 2],
                mode='markers',
                marker=dict(
                    size=1,
                    color=depths_completed,
                    colorscale='Viridis',
                    opacity=0.8,
                    colorbar=dict(title="Depth (m)")
                ),
                name='Completed Depth Points'
            ))
        
        fig_3d.update_layout(
            scene=dict(
                xaxis_title='X (m)',
                yaxis_title='Y (m)',
                zaxis_title='Z (m)',
                aspectmode='data'
            ),
            height=600,
            title="3D Point Cloud Comparison"
        )
        
        st.plotly_chart(fig_3d, use_container_width=True)
    
    # Export options
    with st.expander("💾 Export Options"):
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Download Sparse Depth", key="export_sparse"):
                np.save('sparse_depth.npy', sparse_depth)
                st.success("Saved to sparse_depth.npy")
        with col2:
            if st.button("Download Completed Depth", key="export_completed"):
                np.save('completed_depth.npy', completed_depth)
                st.success("Saved to completed_depth.npy")
        with col3:
            if st.button("Download Both", key="export_both"):
                np.save('sparse_depth.npy', sparse_depth)
                np.save('completed_depth.npy', completed_depth)
                st.success("Saved both files")

