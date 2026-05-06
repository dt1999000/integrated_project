"""
Visualization helper functions for 3D point cloud and 2D image visualization.
Contains both primitive mesh creators (cuboids, frustums) and higher-level visualization functions.
"""

import json
import textwrap
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import cv2
import streamlit as st
from typing import Any, Dict, List, Optional, Tuple

from ..core.pointcloud_projection import Projection


# =============================================================================
# Primitive Mesh Creators (from element.py)
# =============================================================================

def cuboid_from_minmax(min_x, min_y, min_z, max_x, max_y, max_z,
                       color="blue", opacity=0.2):
    """
    Create a 3D cuboid mesh from min/max bounds.

    Args:
        min_x, min_y, min_z: Minimum corner coordinates
        max_x, max_y, max_z: Maximum corner coordinates
        color: Color of the cuboid
        opacity: Opacity of the cuboid (0.0 to 1.0)

    Returns:
        go.Mesh3d: Plotly Mesh3d object
    """
    x0, y0, z0 = min_x, min_y, min_z
    x1, y1, z1 = max_x, max_y, max_z

    vertices = [
        [x0, y0, z0],  # 0
        [x1, y0, z0],  # 1
        [x1, y1, z0],  # 2
        [x0, y1, z0],  # 3
        [x0, y0, z1],  # 4
        [x1, y0, z1],  # 5
        [x1, y1, z1],  # 6
        [x0, y1, z1],  # 7
    ]

    x, y, z = zip(*vertices)

    # 12 triangles (2 per face)
    i = [0, 0, 0, 1, 1, 2, 4, 4, 4, 5, 5, 6]
    j = [1, 2, 4, 2, 5, 3, 5, 7, 0, 6, 1, 7]
    k = [2, 3, 5, 3, 6, 0, 6, 4, 3, 7, 2, 4]

    cuboid = go.Mesh3d(
        x=x, y=y, z=z,
        i=i, j=j, k=k,
        color=color,
        opacity=opacity,
        flatshading=True,
        showscale=False,
        name="cuboid",
    )
    return cuboid


def cuboid_from_corners(corners, color="blue", opacity=0.2, name="cuboid"):
    """
    Create a 3D cuboid mesh from 8 corner points.
    Preserves rotation and exact box shape from corner coordinates.

    Args:
        corners: np.ndarray of shape (8, 3) representing 8 corner positions [x, y, z]
        color: Color of the cuboid
        opacity: Opacity of the cuboid (0.0 to 1.0)
        name: Name for the mesh trace

    Returns:
        go.Mesh3d: Plotly Mesh3d object representing the cuboid

    Note:
        Corner ordering is expected to match KITTI format after transformation:
        - Corners 0-3: one face of the box
        - Corners 4-7: opposite face of the box
    """
    if corners.shape != (8, 3):
        raise ValueError(f"Expected corners shape (8, 3), got {corners.shape}")

    x = corners[:, 0].tolist()
    y = corners[:, 1].tolist()
    z = corners[:, 2].tolist()

    # Define triangles for 6 faces (2 triangles per face = 12 triangles)
    i = [0, 0, 4, 4, 0, 0, 2, 2, 1, 1, 3, 3]
    j = [1, 3, 5, 7, 1, 3, 3, 6, 2, 5, 7, 2]
    k = [2, 2, 6, 6, 5, 7, 7, 7, 6, 6, 4, 0]

    cuboid = go.Mesh3d(
        x=x, y=y, z=z,
        i=i, j=j, k=k,
        color=color,
        opacity=opacity,
        flatshading=True,
        showscale=False,
        name=name,
    )
    return cuboid

def cuboid_kitti_format(center, yaw, length, width, height, color="blue", opacity=0.2, name="cuboid"):
    """
    Create a 3D cuboid mesh from KITTI format parameters (center, yaw, dimensions).
    
    Args:
        center: np.ndarray (3,) - Center position [x, y, z] in LiDAR coordinates
        yaw: float - Rotation angle around Z-axis (in radians)
        length: float - Length of cuboid along X-axis (forward direction)
        width: float - Width of cuboid along Y-axis (lateral direction)
        height: float - Height of cuboid along Z-axis (vertical direction)
        color: Color of the cuboid
        opacity: Opacity of the cuboid (0.0 to 1.0)
        name: Name for the mesh trace
    
    Returns:
        go.Mesh3d: Plotly Mesh3d object representing the cuboid
    """
    center = np.asarray(center).flatten()
    if center.shape != (3,):
        raise ValueError(f"Expected center shape (3,), got {center.shape}")
    
    # Half-dimensions
    l_half = length / 2.0
    w_half = width / 2.0
    h_half = height / 2.0
    
    # Create 8 corners in local coordinate system (centered at origin, axis-aligned)
    # Order: bottom face first (z=-h_half), then top face (z=+h_half)
    # Each face: front-left, front-right, back-right, back-left
    corners_local = np.array([
        [-l_half, -w_half, -h_half],  # 0: bottom front-left
        [ l_half, -w_half, -h_half],  # 1: bottom front-right
        [ l_half,  w_half, -h_half],  # 2: bottom back-right
        [-l_half,  w_half, -h_half],  # 3: bottom back-left
        [-l_half, -w_half,  h_half],  # 4: top front-left
        [ l_half, -w_half,  h_half],  # 5: top front-right
        [ l_half,  w_half,  h_half],  # 6: top back-right
        [-l_half,  w_half,  h_half],  # 7: top back-left
    ])
    
    # Rotation matrix around Z-axis (yaw)
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    R_z = np.array([
        [cos_yaw, -sin_yaw, 0],
        [sin_yaw,  cos_yaw, 0],
        [0,        0,       1]
    ])
    
    # Rotate corners and translate to center
    corners_rotated = (R_z @ corners_local.T).T
    corners = corners_rotated + center
    
    # Use cuboid_from_corners to create the mesh
    return cuboid_from_corners(corners, color=color, opacity=opacity, name=name)

def frustum_from_camera_and_corners(camera_origin: np.ndarray,
                                     base_corners: np.ndarray,
                                     color: str = "blue",
                                     opacity: float = 0.2,
                                     name: str = "frustum") -> go.Mesh3d:
    """
    Create a frustum/pyramid mesh from camera origin (apex) to 4 base corners.
    Used to visualize 2D bounding box projection onto 3D point cloud.

    Args:
        camera_origin: np.ndarray (3,) - Camera center in LiDAR coords (apex of pyramid)
        base_corners: np.ndarray (4, 3) - 4 corner points in LiDAR coords (base of pyramid)
                      Order: [top-left, top-right, bottom-right, bottom-left]
        color: Mesh color
        opacity: Mesh transparency (0.0 to 1.0)
        name: Trace name for the mesh

    Returns:
        go.Mesh3d: Plotly mesh object for the frustum pyramid
    """
    camera_origin = np.asarray(camera_origin).flatten()
    if camera_origin.shape != (3,):
        raise ValueError(f"Expected camera_origin shape (3,), got {camera_origin.shape}")

    base_corners = np.asarray(base_corners)
    if base_corners.shape != (4, 3):
        raise ValueError(f"Expected base_corners shape (4, 3), got {base_corners.shape}")

    # 5 vertices: apex (camera) + 4 base corners
    vertices = np.vstack([camera_origin.reshape(1, 3), base_corners])
    x = vertices[:, 0].tolist()
    y = vertices[:, 1].tolist()
    z = vertices[:, 2].tolist()

    # Triangle faces: 4 side triangles + 2 base triangles
    i = [0, 0, 0, 0, 1, 1]
    j = [1, 2, 3, 4, 2, 3]
    k = [2, 3, 4, 1, 3, 4]

    frustum = go.Mesh3d(
        x=x, y=y, z=z,
        i=i, j=j, k=k,
        color=color,
        opacity=opacity,
        flatshading=True,
        showscale=False,
        name=name,
    )
    return frustum


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

    for index, box in enumerate(boxes):
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
        label = category + f"{index+1}"
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


def generate_distinct_colors(n: int) -> List[Tuple[float, float, float]]:
    """
    Generate n distinct RGB colors in [0, 1] for visualization.

    Uses Plotly's qualitative color set and cycles if n exceeds palette size.
    """
    if n <= 0:
        return []

    base_colors = px.colors.qualitative.Plotly
    colors: List[Tuple[float, float, float]] = []

    for i in range(n):
        color_str = base_colors[i % len(base_colors)]
        color_str = color_str.strip()

        # Handle hex colors like "#636efa" or "#fff"
        if color_str.startswith("#"):
            hex_str = color_str.lstrip("#")
            length = len(hex_str)
            if length == 6:
                r = int(hex_str[0:2], 16)
                g = int(hex_str[2:4], 16)
                b = int(hex_str[4:6], 16)
            elif length == 3:
                r = int(hex_str[0] * 2, 16)
                g = int(hex_str[1] * 2, 16)
                b = int(hex_str[2] * 2, 16)
            else:
                # Fallback to a neutral color if format is unexpected
                r = 128
                g = 128
                b = 128
        else:
            # Handle "rgb(r,g,b)" or "rgba(r,g,b,a)" strings
            cleaned = color_str.lower().replace("rgba", "rgb")
            cleaned = cleaned.replace("rgb(", "").replace(")", "")
            parts = cleaned.split(",")
            if len(parts) >= 3:
                r = float(parts[0])
                g = float(parts[1])
                b = float(parts[2])
            else:
                r = 128.0
                g = 128.0
                b = 128.0

        colors.append((r / 255.0, g / 255.0, b / 255.0))

    return colors


def overlay_masks_on_image(
    image: np.ndarray,
    masks: List[np.ndarray],
    colors: List[Tuple[float, float, float]],
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Overlay segmentation masks on image with different colors.

    Args:
        image: Input image (H, W, 3) in RGB uint8 format
        masks: List of binary masks (H, W)
        colors: List of RGB colors (0-1 range) for each mask
        alpha: Transparency of mask overlay (0-1)

    Returns:
        Image with masks overlaid (uint8 RGB)
    """
    if image.dtype != np.uint8:
        img_float = np.clip(image.astype(np.float32), 0.0, 255.0)
        base = img_float / 255.0
    else:
        base = image.astype(np.float32) / 255.0

    result = base.copy()

    for mask, color in zip(masks, colors):
        if mask is None:
            continue

        mask_binary = (mask > 0).astype(np.float32)
        if mask_binary.ndim == 2:
            mask_binary = mask_binary[:, :, np.newaxis]

        colored_mask = np.zeros_like(result)
        colored_mask[:, :, 0] = color[0]
        colored_mask[:, :, 1] = color[1]
        colored_mask[:, :, 2] = color[2]

        result = result * (1.0 - alpha * mask_binary) + colored_mask * (alpha * mask_binary)

    result_uint8 = (np.clip(result, 0.0, 1.0) * 255.0).astype(np.uint8)
    return result_uint8


def draw_projected_cuboid_bboxes(image: np.ndarray, cuboids: List[Dict],
                                  original_boxes: Optional[List[Dict]] = None) -> np.ndarray:
    """
    Draw reprojected 2D bounding boxes from 3D cuboids on a camera image.

    Shows the projected_bbox_2d from each cuboid (result of find_best_cuboid)
    overlayed with original 2D bounding boxes for comparison.

    Args:
        image: RGB image array (H, W, 3)
        cuboids: List of cuboid dictionaries containing 'projected_bbox_2d', 'category',
                 'source_bbox_idx', and optionally 'iou'
        original_boxes: Optional list of original ground truth boxes to draw first

    Returns:
        Image with reprojected bounding boxes drawn
    """
    img_with_boxes = image.copy()

    # Color mapping for different categories (BGR format for cv2)
    gt_colors = {
        'Car': (0, 255, 0),        # Green
        'Pedestrian': (255, 0, 0),  # Blue
        'Cyclist': (0, 0, 255),     # Red
        'Van': (255, 255, 0),       # Cyan
        'Truck': (0, 255, 255),     # Yellow
        'Person_sitting': (255, 0, 255),  # Magenta
        'Tram': (128, 128, 0),      # Olive
        'Misc': (128, 0, 128),      # Purple
    }

    # Projected bbox uses orange/red tones to distinguish from GT
    proj_colors = {
        'Car': (0, 165, 255),        # Orange
        'Pedestrian': (255, 100, 100),  # Light blue
        'Cyclist': (100, 100, 255),     # Light red
        'Van': (255, 200, 0),       # Light cyan
        'Truck': (0, 200, 255),     # Light yellow
        'Person_sitting': (255, 100, 255),  # Light magenta
        'Tram': (150, 150, 0),      # Light olive
        'Misc': (150, 0, 150),      # Light purple
    }

    # First draw original GT boxes (solid line)
    if original_boxes:
        for index, box in enumerate(original_boxes):
            bbox = box.get('bbox_2d')
            if bbox is None:
                continue

            category = box.get('category', 'Unknown')
            color = gt_colors.get(category, (255, 255, 0))

            left, top = int(bbox['left']), int(bbox['top'])
            right, bottom = int(bbox['right']), int(bbox['bottom'])
            cv2.rectangle(img_with_boxes, (left, top), (right, bottom), color, 2)

            # Draw label
            label = f"GT{index}: {category}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            thickness = 1
            (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
            cv2.rectangle(img_with_boxes,
                          (left, top - text_height - 5),
                          (left + text_width, top),
                          color, -1)
            cv2.putText(img_with_boxes, label, (left, top - 5),
                        font, font_scale, (255, 255, 255), thickness)

    # Then draw projected cuboid bboxes (dashed effect using thicker line)
    for cuboid in cuboids:
        proj_bbox = cuboid['projected_bbox_2d']
        print(f"proj_bbox: {proj_bbox}")

        category = cuboid.get('category', 'Unknown')
        frustum_idx = cuboid.get('source_bbox_idx', '?')
        iou = cuboid.get('iou')
        color = proj_colors.get(category, (0, 165, 255))  # Default: orange

        left, top = int(proj_bbox['left']), int(proj_bbox['top'])
        right, bottom = int(proj_bbox['right']), int(proj_bbox['bottom'])

        # Draw with thicker line to distinguish from GT
        cv2.rectangle(img_with_boxes, (left, top), (right, bottom), color, 3)

        # Draw label with IoU if available
        if iou is not None:
            label = f"F{frustum_idx}: {category} IoU:{iou:.2f}"
        else:
            label = f"F{frustum_idx}: {category}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        thickness = 1
        (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)

        # Position label at bottom of bbox to avoid overlap with GT label
        cv2.rectangle(img_with_boxes,
                      (left, bottom),
                      (left + text_width, bottom + text_height + 5),
                      color, -1)
        cv2.putText(img_with_boxes, label, (left, bottom + text_height),
                    font, font_scale, (0, 0, 0), thickness)

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

    # Create a Projection instance for projection calculations
    # We use a dummy point cloud and extrinsic since we only need the projection method
    dummy_point_cloud = np.zeros((1, 3))
    projection = Projection(
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
        frustum_idx = cuboid.get('source_bbox_idx')
        iou = cuboid.get('iou')

        # Build cuboid name with frustum index and IoU if available
        if frustum_idx is not None:
            cuboid_name = f"{name_prefix}F{frustum_idx}: {category}"
            if iou is not None:
                cuboid_name += f" (IoU: {iou:.2f})"
        else:
            cuboid_name = f"{name_prefix}{category}" if name_prefix else category

        # Use corner-based visualization if corners are available (preserves rotation)
        if 'corners' in cuboid and cuboid['corners'] is not None:
            fig.add_trace(cuboid_from_corners(
                cuboid['corners'],
                color=color,
                opacity=opacity,
                name=cuboid_name
            ))
        elif cuboid.get('format') == 'kitti' or ('center' in cuboid and 'yaw' in cuboid and 
                                                  'length' in cuboid and 'width' in cuboid and 'height' in cuboid):
            # KITTI format: use cuboid_kitti_format
            mesh = cuboid_kitti_format(
                center=cuboid['center'],
                yaw=cuboid['yaw'],
                length=cuboid['length'],
                width=cuboid['width'],
                height=cuboid['height'],
                color=color,
                opacity=opacity,
                name=cuboid_name
            )
            fig.add_trace(mesh)
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
                            points_in_frustums: Optional[List[np.ndarray]] = None,
                            reconstructed_points: Optional[np.ndarray] = None,
                            show_lidar: bool = True,
                            show_reconstructed: bool = True,
                            color_by_depth: bool = False,
                            title: str = "3D Point Cloud") -> go.Figure:
    """
    Create a 3D scatter plot using Plotly for web compatibility.
    
    Args:
        points: Point cloud data (PointCloud object or numpy array)
        labels: Optional cluster labels for coloring points
        mask_points: Optional dict mapping mask IDs to point arrays (legacy parameter)
        cuboids: Optional list of cuboid dictionaries to visualize
        rays: Optional dict mapping mask IDs to ray data (used with mask_points)
        points_in_frustums: Optional array of points within frustums
        reconstructed_points: Optional array of reconstructed points from depth estimation
        show_lidar: Whether to show original LiDAR points (default: True)
        show_reconstructed: Whether to show reconstructed points (default: True)
        color_by_depth: Whether to color reconstructed points by depth (default: False)
        title: Plot title
    
    Returns:
        go.Figure: Plotly figure object
    """
    fig = go.Figure()

    # Convert PointCloud object to numpy array if needed
    point_array = None
    if points is not None:
        if hasattr(points, 'point_cloud_plane_removed'):
            # It's a PointCloud object
            point_array = points.point_cloud_plane_removed
        else:
            # It's already a numpy array
            point_array = points

    # Add original LiDAR points if enabled
    if show_lidar and point_array is not None and len(point_array) > 0:
        if labels is None:
            # Single color for all points
            fig.add_trace(go.Scatter3d(
                x=point_array[:, 0],
                y=point_array[:, 1],
                z=point_array[:, 2],
                mode='markers',
                marker=dict(size=1, color='lightblue', opacity=0.3),
                name='LiDAR Points'
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

    # Add reconstructed points from depth estimation if enabled
    if show_reconstructed and reconstructed_points is not None and len(reconstructed_points) > 0:
        if color_by_depth:
            # Calculate depth from origin
            depths = np.linalg.norm(reconstructed_points, axis=1)
            fig.add_trace(go.Scatter3d(
                x=reconstructed_points[:, 0],
                y=reconstructed_points[:, 1],
                z=reconstructed_points[:, 2],
                mode='markers',
                marker=dict(
                    size=2,
                    color=depths,
                    colorscale='Viridis',
                    opacity=0.8,
                    colorbar=dict(title="Depth (m)")
                ),
                name='Reconstructed Points'
            ))
        else:
            fig.add_trace(go.Scatter3d(
                x=reconstructed_points[:, 0],
                y=reconstructed_points[:, 1],
                z=reconstructed_points[:, 2],
                mode='markers',
                marker=dict(size=2, color='red', opacity=0.8),
                name='Reconstructed Points'
            ))

    if points_in_frustums is not None:
        fig.add_trace(go.Scatter3d(
            x=points_in_frustums[:, 0],
            y=points_in_frustums[:, 1],
            z=points_in_frustums[:, 2],
            mode='markers',
            marker=dict(size=2, color='red'),
            name='Points in Frustum'
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
            xaxis=dict(title='X (m)'),
            yaxis=dict(title='Y (m)'),
            zaxis=dict(title='Z (m)'),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        height=600
    )

    return fig


def create_3d_mask_assignment_figure(
    points_3d: np.ndarray,
    mask_assignments: np.ndarray,
    mask_colors: List[Tuple[float, float, float]],
    lidar_points: Optional[np.ndarray] = None,
    show_unassigned: bool = True,
    unassigned_color: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    title: str = "3D Points Colored by Segmentation Mask",
) -> go.Figure:
    """
    Create a 3D scatter figure where each point is colored by its assigned mask.

    This is a reusable building block for visualizing segmentation outputs
    on top of LiDAR or reconstructed 3D points.
    """
    fig = go.Figure()

    n_points = len(points_3d)
    if n_points == 0:
        fig.update_layout(title=title)
        return fig

    n_masks = len(mask_colors)

    for mask_idx in range(n_masks):
        mask_points = points_3d[mask_assignments == mask_idx]
        if len(mask_points) == 0:
            continue

        color = mask_colors[mask_idx]
        color_str = f"rgb({int(color[0] * 255)}, {int(color[1] * 255)}, {int(color[2] * 255)})"

        fig.add_trace(
            go.Scatter3d(
                x=mask_points[:, 0],
                y=mask_points[:, 1],
                z=mask_points[:, 2],
                mode="markers",
                marker=dict(size=2, color=color_str, opacity=0.8),
                name=f"Mask {mask_idx + 1}",
                showlegend=True,
            )
        )

    if show_unassigned:
        unassigned_points = points_3d[mask_assignments == -1]
        if len(unassigned_points) > 0:
            ua_color_str = (
                f"rgb({int(unassigned_color[0] * 255)}, "
                f"{int(unassigned_color[1] * 255)}, "
                f"{int(unassigned_color[2] * 255)})"
            )
            fig.add_trace(
                go.Scatter3d(
                    x=unassigned_points[:, 0],
                    y=unassigned_points[:, 1],
                    z=unassigned_points[:, 2],
                    mode="markers",
                    marker=dict(size=2, color=ua_color_str, opacity=0.5),
                    name="Unassigned",
                    showlegend=True,
                )
            )

    if lidar_points is not None and len(lidar_points) > 0:
        sample_size = min(10000, len(lidar_points))
        indices = np.random.choice(len(lidar_points), sample_size, replace=False)
        sampled_lidar = lidar_points[indices]

        fig.add_trace(
            go.Scatter3d(
                x=sampled_lidar[:, 0],
                y=sampled_lidar[:, 1],
                z=sampled_lidar[:, 2],
                mode="markers",
                marker=dict(size=1, color="rgb(200,200,200)", opacity=0.3),
                name="LiDAR Points",
                showlegend=True,
            )
        )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        height=700,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
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


def render_point_cloud_plot(
    fig: go.Figure,
    export_basename: str,
    use_container_width: bool = False,
) -> None:
    """Render a point-cloud plot with Streamlit display + structured HTML export."""

    def _build_structured_pointcloud_html(plot_fig: go.Figure, plot_name: str) -> str:
        fig_dict = plot_fig.to_dict()
        data_json = json.dumps(fig_dict.get("data", []), indent=2)
        layout_json = json.dumps(fig_dict.get("layout", {}), indent=2)
        config_json = json.dumps({"responsive": True, "displaylogo": False}, indent=2)
        return textwrap.dedent(
            f"""\
            <!doctype html>
            <html lang="en">
            <head>
              <meta charset="utf-8" />
              <meta name="viewport" content="width=device-width, initial-scale=1" />
              <title>{plot_name}</title>
              <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
            </head>
            <body style="margin:0;">
              <div id="pointcloud_plot" style="width:100vw; height:100vh;"></div>
              <script>
                // =========================================================================
                // POINT CLOUD DATA (all traces / 3D points)
                // =========================================================================
                const pointCloudData = {data_json};

                // =========================================================================
                // LEGEND + CAPTION/TITLE + AXES (layout styling lives here)
                // =========================================================================
                const pointCloudLayout = {layout_json};

                // =========================================================================
                // EXPORT + INTERACTION OPTIONS
                // =========================================================================
                const pointCloudConfig = {config_json};

                Plotly.newPlot("pointcloud_plot", pointCloudData, pointCloudLayout, pointCloudConfig);
              </script>
            </body>
            </html>
            """
        )

    # Streamlit in-app display (from Detection-page version)
    display_legend_font_size = 12
    display_caption_font_size = 14
    display_axis_title_font_size = 12
    display_axis_tick_font_size = 11
    # Downloaded HTML / high-res export (from Detection-page version)
    export_legend_font_size = 28
    export_caption_font_size = 36
    export_axis_title_font_size = 26
    export_axis_tick_font_size = 18

    axis_tick_settings = {
        "x": {"nticks": 2, "dtick": None},
        "y": {"nticks": 2, "dtick": None},
        "z": {"nticks": 2, "dtick": None},
    }
    axis_range_overrides = {
        "x": None,
        "y": None,
        "z": None,
    }

    def _scene_axis(
        axis_key: str,
        title_text: str,
        range_override: Optional[List[float]],
        axis_title_font_size: int,
        axis_tick_font_size: int,
    ) -> Dict[str, Any]:
        axis_cfg: Dict[str, Any] = {
            "title": {"text": title_text, "font": {"size": axis_title_font_size}},
            "tickfont": {"size": axis_tick_font_size},
        }
        tick_cfg = axis_tick_settings.get(axis_key, {})
        axis_cfg["nticks"] = tick_cfg.get("nticks", 8)
        if tick_cfg.get("dtick") is not None:
            axis_cfg["dtick"] = tick_cfg["dtick"]
        if range_override is not None:
            axis_cfg["range"] = range_override
        return axis_cfg

    fig.update_layout(
        legend=dict(
            font=dict(size=display_legend_font_size),
            title=dict(font=dict(size=display_legend_font_size)),
            itemsizing="constant",
        ),
        title=dict(font=dict(size=display_caption_font_size)),
        scene=dict(
            xaxis=_scene_axis(
                "x",
                "X (m)",
                axis_range_overrides["x"],
                display_axis_title_font_size,
                display_axis_tick_font_size,
            ),
            yaxis=_scene_axis(
                "y",
                "Y (m)",
                axis_range_overrides["y"],
                display_axis_title_font_size,
                display_axis_tick_font_size,
            ),
            zaxis=_scene_axis(
                "z",
                "Z (m)",
                axis_range_overrides["z"],
                display_axis_title_font_size,
                display_axis_tick_font_size,
            ),
        ),
    )
    export_config = {
        "toImageButtonOptions": {
            "format": "png",
            "filename": export_basename,
            "width": 2200,
            "height": 1400,
            "scale": 2,
        },
        "displaylogo": False,
    }

    if use_container_width:
        st.plotly_chart(fig, use_container_width=True, config=export_config)
    else:
        st.plotly_chart(fig, config=export_config)

    export_fig = go.Figure(fig)
    export_fig.update_layout(
        width=1920,
        height=1080,
        autosize=True,
        margin=dict(l=10, r=10, t=55, b=10),
        legend=dict(
            font=dict(size=export_legend_font_size),
            title=dict(font=dict(size=export_legend_font_size)),
            itemsizing="constant",
        ),
        title=dict(
            font=dict(size=export_caption_font_size),
            pad=dict(t=6, b=2),
        ),
        scene=dict(
            domain=dict(x=[0.0, 1.0], y=[0.0, 1.0]),
            xaxis=_scene_axis(
                "x",
                "X (m)",
                axis_range_overrides["x"],
                export_axis_title_font_size,
                export_axis_tick_font_size,
            ),
            yaxis=_scene_axis(
                "y",
                "Y (m)",
                axis_range_overrides["y"],
                export_axis_title_font_size,
                export_axis_tick_font_size,
            ),
            zaxis=_scene_axis(
                "z",
                "Z (m)",
                axis_range_overrides["z"],
                export_axis_title_font_size,
                export_axis_tick_font_size,
            ),
        ),
    )

    st.download_button(
        "⬇️ Download interactive HTML (high quality)",
        data=_build_structured_pointcloud_html(export_fig, export_basename),
        file_name=f"{export_basename}.html",
        mime="text/html",
        key=f"pointcloud_html_export_{export_basename}",
        width="stretch",
    )
