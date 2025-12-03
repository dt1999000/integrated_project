"""
Visualization helper functions for 3D point cloud and 2D image visualization.
Extracted from app.py for better code organization.
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import cv2
from typing import Dict, List, Optional

from element import (
    cuboid_from_minmax,
    cuboid_from_corners,
    frustum_from_camera_and_corners
)
from pointcloud_projection import Projection2DTo3D


# =============================================================================
# 2D Image Visualization
# =============================================================================

def draw_2d_boxes_on_image(image: np.ndarray, boxes: List[Dict]) -> np.ndarray:
    """
    Draw 2D bounding boxes on a camera image.

    Args:
        image: RGB image array (H, W, 3)
        boxes: List of box dictionaries containing 'bbox_2d' and 'category'

    Returns:
        Image with 2D bounding boxes drawn
    """
    img_with_boxes = image.copy()

    # Color mapping for different categories (BGR format for cv2)
    colors = {
        'Car': (0, 255, 0),        # Green
        'Pedestrian': (255, 0, 0),  # Red (actually BGR: Blue)
        'Cyclist': (0, 0, 255),     # Blue (actually BGR: Red)
        'Van': (255, 255, 0),       # Cyan
        'Truck': (0, 255, 255),     # Yellow
        'Person_sitting': (255, 0, 255),  # Magenta
        'Tram': (128, 128, 0),      # Olive
        'Misc': (128, 0, 128),      # Purple
    }

    for box in boxes:
        bbox = box.get('bbox_2d')
        if bbox is None:
            continue

        category = box.get('category', 'Unknown')
        color = colors.get(category, (255, 255, 0))  # Default: yellow

        # Draw rectangle
        left, top = int(bbox['left']), int(bbox['top'])
        right, bottom = int(bbox['right']), int(bbox['bottom'])
        cv2.rectangle(img_with_boxes, (left, top), (right, bottom), color, 2)

        # Draw label
        label = category
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        # Background for text
        cv2.rectangle(img_with_boxes,
                      (left, top - text_height - 5),
                      (left + text_width, top),
                      color, -1)
        cv2.putText(img_with_boxes, label, (left, top - 5),
                    font, font_scale, (255, 255, 255), thickness)

    return img_with_boxes


# =============================================================================
# 3D Figure Modifier Functions
# =============================================================================

def add_frustums_to_figure(fig: go.Figure,
                           ground_truth_boxes: List[Dict],
                           camera_intrinsic: np.ndarray,
                           camera_to_lidar_transform: np.ndarray,
                           depth: float = 30.0,
                           opacity: float = 0.15) -> go.Figure:
    """
    Add frustum pyramids to a Plotly figure for each 2D bounding box.

    Args:
        fig: Plotly figure to add frustums to
        ground_truth_boxes: List of box dicts with 'bbox_2d' and 'category'
        camera_intrinsic: 3x3 camera intrinsic matrix
        camera_to_lidar_transform: 4x4 camera to LiDAR transform
        depth: Frustum depth in meters
        opacity: Frustum transparency

    Returns:
        Figure with frustum meshes added
    """
    # Color mapping for different categories
    colors = {
        'Car': 'green',
        'Pedestrian': 'red',
        'Cyclist': 'blue',
        'Van': 'cyan',
        'Truck': 'orange',
        'Person_sitting': 'magenta',
        'Tram': 'olive',
        'Misc': 'purple',
    }

    # Create a Projection2DTo3D instance for projection calculations
    # We use a dummy point cloud and extrinsic since we only need the projection method
    dummy_point_cloud = np.zeros((1, 3))
    projection = Projection2DTo3D(
        camera_intrinsic=camera_intrinsic,
        camera_extrinsic=np.eye(4),
        camera_to_lidar_transform=camera_to_lidar_transform,
        point_cloud=dummy_point_cloud
    )

    for i, box in enumerate(ground_truth_boxes):
        bbox_2d = box.get('bbox_2d')
        if bbox_2d is None:
            continue

        category = box.get('category', 'Unknown')

        # Project 2D bbox corners to 3D using the class method
        camera_origin, base_corners = projection.project_bbox_corners_to_3d(
            bbox_2d, depth=depth
        )

        # Create frustum mesh
        color = colors.get(category, 'yellow')
        frustum = frustum_from_camera_and_corners(
            camera_origin, base_corners,
            color=color, opacity=opacity,
            name=f"Frustum: {category} #{i+1}"
        )
        fig.add_trace(frustum)

    return fig


def add_cuboids_to_figure(fig: go.Figure, cuboids: List[Dict],
                          color: str = 'green',
                          opacity: float = 0.2, name_prefix: str = "") -> go.Figure:
    """
    Add cuboids to a plotly figure, supporting both corner-based and min/max formats.

    Args:
        fig: Plotly figure to add cuboids to
        cuboids: List of cuboid dictionaries with either 'corners' or min/max keys
        color: Color for cuboids
        opacity: Opacity of cuboids (0.0 to 1.0)
        name_prefix: Prefix for cuboid names (e.g., "GT: " or "Detected: ")

    Returns:
        Modified figure with cuboids added
    """
    for cuboid in cuboids:
        category = cuboid.get('category', 'Unknown')
        cuboid_name = f"{name_prefix}{category}" if name_prefix else category

        # Use corner-based visualization if corners are available (preserves rotation)
        if 'corners' in cuboid and cuboid['corners'] is not None:
            fig.add_trace(cuboid_from_corners(
                cuboid['corners'],
                color=color,
                opacity=opacity,
                name=cuboid_name
            ))
        else:
            # Fallback to min/max format
            mesh = cuboid_from_minmax(
                cuboid['min_x'], cuboid['min_y'], cuboid['min_z'],
                cuboid['max_x'], cuboid['max_y'], cuboid['max_z'],
                color=color,
                opacity=opacity
            )
            mesh.name = cuboid_name
            fig.add_trace(mesh)

    return fig


# =============================================================================
# 3D Figure Creators (Complete Visualizations)
# =============================================================================

def create_3d_scatter_plot(points, labels: Optional[np.ndarray] = None,
                            mask_points: Optional[Dict[int, np.ndarray]] = None,
                            cuboids: Optional[List[Dict]] = None,
                            rays: Optional[Dict[int, np.ndarray]] = None,
                            title: str = "3D Point Cloud") -> go.Figure:
    """Create a 3D scatter plot using Plotly for web compatibility"""
    fig = go.Figure()

    # Convert PointCloud object to numpy array if needed
    if hasattr(points, 'point_cloud_plane_removed'):
        # It's a PointCloud object
        point_array = points.point_cloud_plane_removed
    else:
        # It's already a numpy array
        point_array = points

    if labels is None:
        # Single color for all points
        fig.add_trace(go.Scatter3d(
            x=point_array[:, 0],
            y=point_array[:, 1],
            z=point_array[:, 2],
            mode='markers',
            marker=dict(size=2, color='lightblue'),
            name='Points'
        ))
    else:
        # Color by cluster
        unique_labels = np.unique(labels)
        colors = px.colors.qualitative.Plotly[:len(unique_labels)]

        for i, label in enumerate(unique_labels):
            if label == -1:  # Noise points
                mask = labels == label
                fig.add_trace(go.Scatter3d(
                    x=point_array[mask, 0],
                    y=point_array[mask, 1],
                    z=point_array[mask, 2],
                    mode='markers',
                    marker=dict(size=2, color='gray'),
                    name='Noise'
                ))
            else:
                mask = labels == label
                fig.add_trace(go.Scatter3d(
                    x=point_array[mask, 0],
                    y=point_array[mask, 1],
                    z=point_array[mask, 2],
                    mode='markers',
                    marker=dict(size=2, color=colors[i % len(colors)]),
                    name=f'Cluster {label}'
                ))

    if cuboids is not None:
        add_cuboids_to_figure(fig, cuboids, color='green', opacity=0.2, name_prefix="Cuboid: ")

    if mask_points is not None:
        # Use different colors for different masks
        colors = px.colors.qualitative.Plotly
        for i, (mask_id, mask_point) in enumerate(mask_points.items()):
            color = colors[i % len(colors)]
            fig.add_trace(go.Scatter3d(
                x=mask_point[:, 0],
                y=mask_point[:, 1],
                z=mask_point[:, 2],
                mode='markers',
                marker=dict(size=2, color=color),
                name=f'Mask {mask_id}'
            ))

        if rays is not None:
            # Handle nested dictionary structure where rays is {mask_id: {'origins': array, 'directions': array}}
            for mask_id, ray_data in rays.items():
                if 'origins' in ray_data and 'directions' in ray_data:
                    origin = ray_data['origins'][0]  # All origins are the same for a mask
                    directions = ray_data['directions']

                    # Get the corresponding mask points if available
                    mask_points_for_id = mask_points.get(mask_id, []) if mask_points is not None else []

                    for i in range(len(directions)):
                        direction = directions[i]
                        if len(mask_points_for_id) > i:
                            projected = mask_points_for_id[i]
                        else:
                            projected = origin + direction * 20.0
                        fig.add_trace(go.Scatter3d(
                            x=[origin[0], projected[0]],
                            y=[origin[1], projected[1]],
                            z=[origin[2], projected[2]],
                            mode='lines',
                            line=dict(color='blue'),
                            name=f'Ray {mask_id}'
                        ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z'),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        height=600
    )

    return fig


def create_comparison_plot(point_cloud, ground_truth_cuboids, detected_cuboids):
    """Create overlay plot with both GT and detected cuboids"""
    fig = go.Figure()

    # Add point cloud
    pc_array = point_cloud.point_cloud_plane_removed
    fig.add_trace(go.Scatter3d(
        x=pc_array[:, 0], y=pc_array[:, 1], z=pc_array[:, 2],
        mode='markers',
        marker=dict(size=1, color='lightgray'),
        name='Point Cloud'
    ))

    # Add ground truth cuboids (green) using helper function
    add_cuboids_to_figure(fig, ground_truth_cuboids, color='green', opacity=0.3, name_prefix="GT: ")

    # Add detected cuboids (red) using helper function
    add_cuboids_to_figure(fig, detected_cuboids, color='red', opacity=0.3, name_prefix="Detected: ")

    fig.update_layout(
        title="Ground Truth (Green) vs Detected (Red)",
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z'),
            aspectmode='data'
        ),
        height=700
    )

    return fig
