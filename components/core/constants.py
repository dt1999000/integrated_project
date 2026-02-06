"""
Constants for 3D Object Detection Pipeline
"""

# =============================================================================
# KITTI Cuboid Templates - Based on ground truth dimension statistics
# Dimensions: (length, width, height) in meters
# Length = X dimension (forward), Width = Y dimension (lateral), Height = Z dimension
# =============================================================================
KITTI_CUBOID_TEMPLATES = {
    'Car': {'length': 3.64, 'width': 1.86, 'height': 1.58},  # Using median width
    'Pedestrian': {'length': 0.88, 'width': 0.90, 'height': 1.77},
    'Cyclist': {'length': 1.68, 'width': 0.75, 'height': 1.76},  # Using median width
    'Van': {'length': 4.76, 'width': 2.22, 'height': 2.27},  # Using median width
    'Truck': {'length': 9.82, 'width': 2.99, 'height': 3.38},  # Using median width
    'Tram': {'length': 15.59, 'width': 3.66, 'height': 3.73},  # Using median width
    'Misc': {'length': 2.56, 'width': 1.91, 'height': 1.68},  # Using median values
    'Person_sitting': {'length': 0.72, 'width': 0.80, 'height': 1.29},  # Using median width
    # Default fallback template
    'Unknown': {'length': 2.0, 'width': 1.5, 'height': 1.5},
}

