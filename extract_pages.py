"""
Script to extract page functions from app.py into separate files in pages/ folder.
"""
import re
import os

def extract_function(file_path, function_name):
    """Extract a function definition and its body from app.py"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find function definition
    pattern = rf'def {function_name}\([^)]*\):.*?(?=\n\ndef |\nif __name__|$)'
    match = re.search(pattern, content, re.DOTALL)
    
    if match:
        return match.group(0)
    return None

def get_common_imports():
    """Get common imports needed by all pages"""
    return '''"""
Page module - extracted from app.py
"""
import streamlit as st
import numpy as np
import pandas as pd
import cv2
import time
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

'''

# Page functions to extract
page_functions = [
    'project_segmentation_mask_on_pointcloud_page',
    'dbscan_page',
    'optics_page',
    'birch_page',
    'agglomerative_page',
    'hdbscan_page',
    'kitti_groundtruth_page',
    'statistics_page',
    'depth_estimation_page',
    'depth_completion_page',
]

if __name__ == '__main__':
    app_path = 'app.py'
    
    for func_name in page_functions:
        func_code = extract_function(app_path, func_name)
        if func_code:
            # Create filename
            filename = func_name.replace('_page', '.py')
            if not filename.startswith('pages/'):
                filename = f'pages/{filename}'
            
            # Write file
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(get_common_imports())
                f.write(func_code)
                f.write('\n')
            
            print(f"Extracted {func_name} to {filename}")
        else:
            print(f"Could not find {func_name}")

