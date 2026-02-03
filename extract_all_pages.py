"""Extract all page functions from app.py to pages/ folder"""
import os

# Function name to line range mapping (start line, end line inclusive)
FUNCTIONS = {
    'project_segmentation_mask_on_pointcloud_page': (217, 319),
    'dbscan_page': (320, 562),
    'optics_page': (563, 787),
    'birch_page': (788, 980),
    'agglomerative_page': (981, 1177),
    'hdbscan_page': (1178, 1392),
    'kitti_groundtruth_page': (1393, 1660),
    'statistics_page': (1661, 2083),
    'depth_estimation_page': (2084, 2269),
    'depth_completion_page': (2270, 2523),
}

COMMON_IMPORTS = '''"""
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

'''

def extract_pages():
    """Extract all page functions to separate files"""
    with open('app.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    os.makedirs('pages', exist_ok=True)
    
    for func_name, (start, end) in FUNCTIONS.items():
        # Extract function code (convert to 0-based indexing)
        func_lines = lines[start-1:end]
        func_code = ''.join(func_lines)
        
        # Create filename
        filename = func_name.replace('_page', '.py')
        filepath = os.path.join('pages', filename)
        
        # Write file with imports
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(COMMON_IMPORTS)
            f.write(func_code)
        
        print(f"Extracted {func_name} to {filepath} ({end-start+1} lines)")

if __name__ == '__main__':
    extract_pages()

