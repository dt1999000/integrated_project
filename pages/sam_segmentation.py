"""
Page module for SAM segmentation visualization.
Shows 2D segmentation masks and 3D reconstructed points colored by mask assignment.
"""
import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import List, Optional, Tuple

from components.utils.visualization_helper import (
    create_3d_scatter_plot,
    generate_distinct_colors,
    overlay_masks_on_image,
    create_3d_mask_assignment_figure,
)
from components.core.pointcloud_projection import Projection
from components.core.sam_integration import assign_points_to_masks


def sam_segmentation_page():
    """SAM Segmentation Visualization page"""
    st.header("🎨 SAM Segmentation Visualization")
    
    # Check if data is loaded
    if 'sample_data' not in st.session_state or st.session_state.sample_data is None:
        st.info("👈 Load a sample from the sidebar and click 'Generate SAM Masks' to see segmentation results")
        st.markdown("""
        ### How to use:
        1. Load a KITTI sample using the sidebar controls
        2. Select SAM model (SAM2 or SAM3) in the sidebar
        3. Click **🔧 Generate SAM Masks** in the sidebar to generate masks
        4. View the segmentation results and backprojection here
        
        **Features:**
        - **2D Visualization**: See segmentation masks overlaid on the original image
        - **3D Visualization**: See backprojected sparse depth points colored by their assigned mask
        - **Mask Statistics**: View information about each segmented object and point assignments
        """)
        return
    
    # Check if SAM masks are available
    sam_masks = st.session_state.get('sam_masks')
    if sam_masks is None or len(sam_masks) == 0:
        st.warning("⚠️ No SAM masks available. Please click 'Generate SAM Masks' in the sidebar first.")
        st.info("Make sure you have:")
        st.markdown("""
        - Selected a SAM model (SAM2 or SAM3) in the sidebar
        - Loaded a KITTI sample with ground truth bounding boxes
        - Clicked 'Generate SAM Masks' to generate masks
        """)
        return
    
    sample_data = st.session_state.sample_data
    
    # Load image
    try:
        img = cv2.imread(sample_data['image_path'])
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
    except Exception as e:
        st.error(f"Could not load image: {str(e)}")
        return
    
    # Generate distinct colors for each mask
    n_masks = len(sam_masks)
    colors = generate_distinct_colors(n_masks)
    
    # Sidebar controls
    st.sidebar.markdown("### Visualization Settings")
    mask_alpha = st.sidebar.slider(
        "Mask Overlay Opacity",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.1,
        key="sam_mask_alpha",
        help="Transparency of mask overlay on image"
    )
    
    show_individual_masks = st.sidebar.checkbox(
        "Show Individual Masks",
        value=False,
        key="show_individual_sam_masks",
        help="Show each mask separately instead of combined overlay"
    )
    
    # Display 2D segmentation results
    st.subheader("📷 2D Segmentation Results")
    
    if show_individual_masks:
        # Show each mask separately
        cols = st.columns(min(3, n_masks))
        for i, (mask, color) in enumerate(zip(sam_masks, colors)):
            if mask is not None:
                with cols[i % 3]:
                    # Create overlay for this mask only
                    overlay = overlay_masks_on_image(img_rgb, [mask], [color], alpha=mask_alpha)
                    st.image(overlay, caption=f"Mask {i+1}", use_container_width=True)
                    
                    # Show mask statistics
                    mask_area = np.sum(mask > 0)
                    mask_percent = 100 * mask_area / (h * w)
                    st.caption(f"Area: {mask_area:,} px ({mask_percent:.1f}%)")
    else:
        # Show combined overlay
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Original Image**")
            st.image(img_rgb, use_container_width=True)
        
        with col2:
            st.markdown("**Segmentation Masks Overlay**")
            overlay = overlay_masks_on_image(img_rgb, sam_masks, colors, alpha=mask_alpha)
            st.image(overlay, use_container_width=True)
    
    # Mask statistics
    st.subheader("📊 Mask Statistics")
    mask_stats = []
    for i, (mask, color) in enumerate(zip(sam_masks, colors)):
        if mask is not None:
            mask_area = np.sum(mask > 0)
            mask_percent = 100 * mask_area / (h * w)
            
            # Get bounding box of mask
            coords = np.column_stack(np.where(mask > 0))
            if len(coords) > 0:
                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0)
                bbox_width = x_max - x_min
                bbox_height = y_max - y_min
            else:
                bbox_width = bbox_height = 0
            
            mask_stats.append({
                'Mask ID': i + 1,
                'Area (px)': f"{mask_area:,}",
                'Area (%)': f"{mask_percent:.2f}",
                'BBox Width': f"{bbox_width}",
                'BBox Height': f"{bbox_height}",
                'Color': f"RGB({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)})"
            })
    
    if mask_stats:
        import pandas as pd
        df = pd.DataFrame(mask_stats)
        st.dataframe(df, use_container_width=True)
    
    # 3D visualization with colored backprojected sparse depth points
    st.subheader("🎯 3D Backprojected Sparse Depth Points (Colored by Mask)")

    point_cloud_obj = st.session_state.get('point_cloud')
    if point_cloud_obj is None:
        st.warning("⚠️ No point cloud available. Please load a sample first.")
        return

    # Get backprojected sparse depth points
    has_colored_sparse = (
        st.session_state.get('colored_sparse_points') is not None and 
        len(st.session_state.get('colored_sparse_points', [])) > 0
    )
    
    if not has_colored_sparse:
        st.warning("⚠️ No backprojected sparse depth points available. Please load a sample first.")
        return
    
    backprojected_points = st.session_state.colored_sparse_points
    backprojected_colors = st.session_state.colored_sparse_colors
    
    # Create projection object for 3D to 2D mapping (to find which mask each point belongs to)
    projection = Projection(
        camera_intrinsic=sample_data['camera_intrinsic'],
        camera_extrinsic=sample_data.get('camera_extrinsic', np.eye(4)),
        camera_to_lidar_transform=sample_data['camera_to_lidar_transform'],
        point_cloud=backprojected_points,
    )

    # Assign backprojected points to masks
    with st.spinner("Mapping backprojected points to segmentation masks..."):
        mask_assignments = assign_points_to_masks(
            backprojected_points, sam_masks, projection, (h, w)
        )

    n_points = len(backprojected_points)
    n_assigned = int(np.sum(mask_assignments >= 0))
    n_unassigned = n_points - n_assigned
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Backprojected Points", f"{n_points:,}")
    with col2:
        st.metric("Points in Masks", f"{n_assigned:,}", 
                 delta=f"{100 * n_assigned / n_points:.1f}%")
    with col3:
        st.metric("Unassigned Points", f"{n_unassigned:,}",
                 delta=f"{100 * n_unassigned / n_points:.1f}%")
    
    # Per-mask point counts
    if n_masks > 0:
        st.markdown("**Points per Mask:**")
        mask_counts = []
        for i in range(n_masks):
            count = np.sum(mask_assignments == i)
            if count > 0:
                mask_counts.append(f"Mask {i+1}: {count:,} points")
        if mask_counts:
            st.text(", ".join(mask_counts))
    
    # Create 3D visualization showing backprojected points colored by mask
    lidar_background = point_cloud_obj.point_cloud_plane_removed if point_cloud_obj is not None else None
    fig = create_3d_mask_assignment_figure(
        points_3d=backprojected_points,
        mask_assignments=mask_assignments,
        mask_colors=colors,
        lidar_points=lidar_background,
        show_unassigned=True,
        title="3D Backprojected Sparse Depth Points Colored by Segmentation Mask",
    )

    st.plotly_chart(fig, use_container_width=True)
    
    # Show legend mapping
    st.subheader("🎨 Color Legend")
    legend_cols = st.columns(min(5, n_masks))
    for i, (color, mask) in enumerate(zip(colors, sam_masks)):
        if mask is not None:
            with legend_cols[i % 5]:
                # Create a small color swatch
                swatch = np.zeros((50, 50, 3), dtype=np.uint8)
                swatch[:, :] = [int(c * 255) for c in color]
                st.image(swatch, caption=f"Mask {i+1}", width=100)
                
                # Show point count
                point_count = np.sum(mask_assignments == i)
                st.caption(f"{point_count:,} points")

