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
    st.header("🔍 Depth Estimation & 3D Reconstruction")
    
    # Check if data is loaded
    if 'sample_data' not in st.session_state or st.session_state.sample_data is None:
        st.info("👈 Load a sample from the sidebar and click 'Reconstruct Points' to get started")
        st.markdown("""
        ### How to use:
        1. Load a KITTI sample using the sidebar controls
        2. Select depth estimation model in the sidebar (Marigold or Depth Anything)
        3. Click the **🔧 Reconstruct Points** button in the sidebar
        4. Wait for the reconstruction to complete (may take 10-30 seconds)
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
        st.warning("⚠️ No depth map available. Click '🔧 Reconstruct Points' in the sidebar to run reconstruction.")
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
    
    # Show sparse depth map if available
    show_sparse = st.session_state.get('sparse_depth_map') is not None
    if show_sparse:
        col1, col2, col3 = st.columns(3)
        sparse_depth = st.session_state.sparse_depth_map
    else:
        col1, col2 = st.columns(2)
        sparse_depth = None
    
    with col1:
        st.markdown("**Original Image**")
        img = cv2.imread(sample_data['image_path'])
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        st.image(img_rgb, use_container_width=True)
    
    if show_sparse:
        with col2:
            st.markdown("**Sparse Depth Map**")
            fig_sparse, ax_sparse = plt.subplots(figsize=(8, 6))
            completed_depth = st.session_state.get('completed_depth_map', depth_map)
            im_sparse = ax_sparse.imshow(sparse_depth, cmap='viridis', vmin=0, vmax=completed_depth.max())
            ax_sparse.set_title("Sparse Depth (from LiDAR)")
            ax_sparse.axis('off')
            plt.colorbar(im_sparse, ax=ax_sparse, fraction=0.046, pad=0.04, label="Depth (m)")
            st.pyplot(fig_sparse)
            plt.close()
        
        with col3:
            # Show completed depth if available, otherwise show estimated depth
            if st.session_state.get('completed_depth_map') is not None:
                st.markdown("**Completed Depth Map**")
                completed_depth = st.session_state.completed_depth_map
                fig, ax = plt.subplots(figsize=(8, 6))
                im = ax.imshow(completed_depth, cmap='viridis', vmin=0, vmax=completed_depth.max())
                ax.set_title("Completed Depth (Marigold-DC)")
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Depth (m)")
            else:
                st.markdown("**Estimated Depth Map**")
                fig, ax = plt.subplots(figsize=(8, 6))
                im = ax.imshow(depth_map, cmap='viridis')
                ax.set_title("Depth Map (meters)")
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            st.pyplot(fig)
            plt.close()
    else:
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
    
    # Show depth completion section if completed depth is available
    if st.session_state.get('completed_depth_map') is not None:
        st.subheader("🎯 Depth Completion Results")
        completed_depth = st.session_state.completed_depth_map
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Sparse Depth Map**")
            fig_sparse, ax_sparse = plt.subplots(figsize=(8, 6))
            sparse_d = st.session_state.sparse_depth_map if st.session_state.get('sparse_depth_map') is not None else np.zeros_like(completed_depth)
            im_sparse = ax_sparse.imshow(sparse_d, cmap='viridis', vmin=0, vmax=completed_depth.max())
            ax_sparse.set_title("Sparse Depth (from LiDAR)")
            ax_sparse.axis('off')
            plt.colorbar(im_sparse, ax=ax_sparse, fraction=0.046, pad=0.04, label="Depth (m)")
            st.pyplot(fig_sparse)
            plt.close()
        
        with col2:
            st.markdown("**Completed Depth Map**")
            fig_completed, ax_completed = plt.subplots(figsize=(8, 6))
            im_completed = ax_completed.imshow(completed_depth, cmap='viridis', vmin=0, vmax=completed_depth.max())
            ax_completed.set_title("Completed Depth (Marigold-DC)")
            ax_completed.axis('off')
            plt.colorbar(im_completed, ax=ax_completed, fraction=0.046, pad=0.04, label="Depth (m)")
            st.pyplot(fig_completed)
            plt.close()
        
        with col3:
            st.markdown("**Comparison**")
            fig_overlay, ax_overlay = plt.subplots(figsize=(8, 6))
            ax_overlay.imshow(img_rgb)
            overlay = ax_overlay.imshow(completed_depth, cmap='viridis', alpha=0.6, vmin=0, vmax=completed_depth.max())
            ax_overlay.set_title("Completed Depth Overlay")
            ax_overlay.axis('off')
            plt.colorbar(overlay, ax=ax_overlay, fraction=0.046, pad=0.04, label="Depth (m)")
            st.pyplot(fig_overlay)
            plt.close()
        
        # Depth completion statistics
        if sparse_d is not None and np.sum(sparse_d > 0) > 0:
            sparse_mask = sparse_d > 0
            sparse_values = sparse_d[sparse_mask]
            completed_values_at_sparse = completed_depth[sparse_mask]
            
            depth_diff = completed_values_at_sparse - sparse_values
            depth_diff_pct = 100 * depth_diff / (sparse_values + 1e-6)
            
            st.subheader("🔍 Depth Completion Analysis")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean Absolute Error", f"{np.mean(np.abs(depth_diff)):.3f}m")
            with col2:
                st.metric("Mean Relative Error", f"{np.mean(np.abs(depth_diff_pct)):.2f}%")
            with col3:
                st.metric("RMSE", f"{np.sqrt(np.mean(depth_diff**2)):.3f}m")
            with col4:
                st.metric("Max Error", f"{np.max(np.abs(depth_diff)):.3f}m")
    
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

