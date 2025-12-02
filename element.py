import plotly.graph_objects as go
import numpy as np

def cuboid_from_minmax(min_x, min_y, min_z, max_x, max_y, max_z,
                       color="blue", opacity=0.2):
    # 8 vertices
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

    # Extract x, y, z coordinates
    x = corners[:, 0].tolist()
    y = corners[:, 1].tolist()
    z = corners[:, 2].tolist()

    # Define triangles for 6 faces (2 triangles per face = 12 triangles)
    # Face connectivity assuming corner ordering:
    #     4-------5
    #    /|      /|
    #   / |     / |
    #  7-------6  |
    #  |  0----|--1
    #  | /     | /
    #  |/      |/
    #  3-------2

    i = [0, 0, 4, 4, 0, 0, 2, 2, 1, 1, 3, 3]  # First vertex of each triangle
    j = [1, 3, 5, 7, 1, 3, 3, 6, 2, 5, 7, 2]  # Second vertex of each triangle
    k = [2, 2, 6, 6, 5, 7, 7, 7, 6, 6, 4, 0]  # Third vertex of each triangle

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