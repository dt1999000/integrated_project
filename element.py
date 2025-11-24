import plotly.graph_objects as go

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