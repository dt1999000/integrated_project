import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
import time
import sys
import os
from typing import Dict, List, Optional
import pandas as pd
import cv2
from segmentation_detection import SegmentationDetector
from bounding_boxes import BoundingBoxes
from segmentation_detection import SegmentationToPointCloud
from pointcloud_projection import Projection
import matplotlib.pyplot as plt
# Add the current directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our pipeline components
from kitti_dataset_loader import KITTIDatasetLoader
from pointcloud_projection import PointCloud, Projection
from clustering_manager import ClusteringManager
from segmentation_detection import SegmentationDetector

from pointcloud_projection import filter_points_in_frustum


# Configure Streamlit page
st.set_page_config(
    page_title="3D Object Detection & Clustering Pipeline",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .param-section {
        background-color: #fafafa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'point_cloud' not in st.session_state:
    st.session_state.point_cloud = None
if 'clustering_results' not in st.session_state:
    st.session_state.clustering_results = {}
if 'segmentation_masks' not in st.session_state:
    st.session_state.segmentation_masks = None
if 'projected_point_cloud' not in st.session_state:
    st.session_state.projected_point_cloud = None
if 'all_mask_points' not in st.session_state:
    st.session_state.all_mask_points = {}
if 'all_rays' not in st.session_state:
    st.session_state.all_rays = {}
if 'frustums' not in st.session_state:
    st.session_state.frustums = []  # List of (camera_origin, base_corners, category, bbox_2d) tuples
if 'per_frustum_clusters' not in st.session_state:
    st.session_state.per_frustum_clusters = []  # List of cluster results per frustum
if 'depth_map' not in st.session_state:
    st.session_state.depth_map = None
if 'depth_image' not in st.session_state:
    st.session_state.depth_image = None

# Initialize centralized parameters dict
if 'params' not in st.session_state:
    st.session_state.params = {
        # Pipeline hyperparameters
        'pipeline': {
            'distance_threshold': 0.3,
            'ransac_n': 3,
            'num_iterations': 1000,
            'filter_forward_only': True,
            'validate_overlap': True,
            'overlap_threshold': 0.7,
            'use_templates': True,
            'frustum_depth': 100
        },
        # HDBSCAN parameters
        'hdbscan': {
            'min_cluster_size': 5,
            'min_samples': 5,
            'cluster_selection_method': 'eom'
        },
        # DBSCAN parameters
        'dbscan': {
            'eps': 0.5,
            'min_samples': 10,
            'metric': 'euclidean',
            'algorithm': 'auto',
            'leaf_size': 30
        },
        # OPTICS parameters
        'optics': {
            'min_samples': 10,
            'max_eps': 1.0,
            'xi': 0.05,
            'min_cluster_size': 10,
            'metric': 'euclidean'
        },
        # BIRCH parameters
        'birch': {
            'threshold': 0.5,
            'branching_factor': 50,
            'n_clusters': 5
        },
        # Agglomerative parameters
        'agglomerative': {
            'n_clusters': 5,
            'linkage': 'ward'
        }
    }


def load_dataset_sample(sample_index: int = 0, distance_threshold: float = 0.3, ransac_n: int = 3, num_iterations: int = 1000, dataset: str = "kitti", filter_forward_only: bool = True):
    """
    Load a sample from KITTI dataset.

    Args:
        sample_index: Index of the sample to load
        distance_threshold: RANSAC distance threshold for ground plane removal
        ransac_n: RANSAC number of points
        num_iterations: RANSAC number of iterations
        dataset: 'kitti'
        filter_forward_only: Whether to keep only forward-facing points (x > 0)

    Returns:
        Tuple of (sample_data dict, PointCloud object with ground removed)
    """
    try:
        if dataset == "kitti":
            # Load KITTI data
            dataset_loader = KITTIDatasetLoader(dataroot='dataset/kitti', split='training')
            dataset_loader.load_dataset()

            # Load synchronized camera, LiDAR, and ground truth data
            sample_data = dataset_loader.load_kitti_data(sample_index)

        else:
            st.error(f"Unknown dataset: {dataset}")
            return None, None

        if sample_data is None:
            st.error(f"Failed to load sample {sample_index}")
            return None, None

        # Load point cloud and remove ground plane
        point_cloud = PointCloud(sample_data['point_cloud'])
        point_cloud.remove_ground_plane_ransac(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations,
            filter_forward_only=filter_forward_only
        )

        # Store ground_z at origin in session state for template cuboids
        ground_z = point_cloud.get_ground_z(x=0.0, y=0.0)
        st.session_state.ground_z = ground_z
        st.session_state.ground_plane_model = point_cloud.ground_plane_model
        if ground_z is not None:
            print(f"Ground plane z at origin: {ground_z:.3f}m")

        return sample_data, point_cloud

    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None, None

def project_segmentation_mask_on_pointcloud_page(sample_data, point_cloud):
    st.header("🎯 Projection & Frustum Visualization")

    # Check if we're using KITTI dataset - show frustum visualization instead of segmentation
    if st.session_state.get('current_dataset') == 'kitti':
        st.subheader("KITTI 2D→3D Frustum Projection")

        sample_data_full = st.session_state.get('sample_data', {})
        ground_truth_boxes = sample_data_full.get('ground_truth_boxes', [])
        has_2d_bboxes = any(box.get('bbox_2d') is not None for box in ground_truth_boxes)

        if not has_2d_bboxes:
            st.warning("No 2D bounding boxes available for this sample.")
            return

        # Frustum parameters
        with st.sidebar.expander("Frustum Parameters", expanded=True):
            frustum_depth = st.slider("Frustum Depth (m)", min_value=5, max_value=100, value=30, step=5,
                                      key="frustum_depth_projection",
                                      help="How far to project the 2D bounding boxes into 3D space")
            frustum_opacity = st.slider("Frustum Opacity", min_value=0.05, max_value=0.5, value=0.2, step=0.05,
                                        key="frustum_opacity_projection")

        # Compute frustums using FrustumManager
        fm = FrustumManager(
            sample_data_full['camera_intrinsic'],
            sample_data_full['camera_to_lidar_transform']
        )
        frustums = fm.create_frustums_from_bboxes(ground_truth_boxes, depth=frustum_depth)
        st.session_state.frustums = frustums

        # Show camera image with 2D bboxes
        st.subheader("Camera Image with 2D Bounding Boxes")
        try:
            img = cv2.imread(sample_data_full['image_path'])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = draw_2d_boxes_on_image(img, ground_truth_boxes)
            st.image(img, caption=f"KITTI Sample - {len(frustums)} 2D Bounding Boxes", width="stretch")
        except Exception as e:
            st.warning(f"Could not load image: {str(e)}")

        # Show frustum statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total 2D Bboxes", len([b for b in ground_truth_boxes if b.get('bbox_2d')]))
        with col2:
            st.metric("Frustums Generated", len(frustums))
        with col3:
            categories = [f.category for f in frustums]
            unique_cats = len(set(categories))
            st.metric("Object Categories", unique_cats)

        # Show frustums in 3D
        st.subheader("3D Point Cloud with Frustums")
        point_cloud_obj = st.session_state.get('point_cloud')
        points_in_frustums = np.array([]).reshape(0, 3)
        for frustum in frustums:
            points_in_frustum, mask = filter_points_in_frustum(point_cloud_obj.point_cloud_plane_removed, frustum.camera_origin, frustum.base_corners) 
            points_in_frustums = np.concatenate([points_in_frustums, points_in_frustum])
        
        if point_cloud_obj is not None:
            fig = create_3d_scatter_plot(point_cloud_obj, None, None, None, None, points_in_frustums,
                                         "Point Cloud with 2D→3D Frustum Projections")

            # Add frustums to the figure
            camera_intrinsic = sample_data_full.get('camera_intrinsic')
            camera_to_lidar = sample_data_full.get('camera_to_lidar_transform')
            if camera_intrinsic is not None and camera_to_lidar is not None:
                add_frustums_to_figure(
                    fig, ground_truth_boxes,
                    camera_intrinsic, camera_to_lidar,
                    depth=frustum_depth, opacity=frustum_opacity
                )

            st.plotly_chart(fig, width='stretch', key='frustum_projection_chart')

        # Per-frustum info table
        if frustums:
            st.subheader("Frustum Details")
            frustum_data = []
            for f in frustums:
                frustum_data.append({
                    'Index': f.idx,
                    'Category': f.category,
                    'BBox Left': f.bbox_2d['left'],
                    'BBox Top': f.bbox_2d['top'],
                    'BBox Right': f.bbox_2d['right'],
                    'BBox Bottom': f.bbox_2d['bottom']
                })
            df = pd.DataFrame(frustum_data)
            st.dataframe(df, use_container_width=True)

        st.info("Run any clustering algorithm on the other tabs to see per-frustum clustering results. "
                "Clusters are automatically filtered by these frustums when using KITTI data.")
        return

def dbscan_page(point_cloud):
    """DBSCAN algorithm parameter control and visualization page"""
    st.header("🎯 DBSCAN Clustering")

    # Parameter controls
    with st.sidebar.expander("DBSCAN Parameters", expanded=True):
        st.session_state.params['dbscan']['eps'] = st.slider(
            "Epsilon (eps)", min_value=0.1, max_value=2.0,
            value=st.session_state.params['dbscan']['eps'],
            step=0.05,
            help="Maximum distance between two samples for one to be considered as in the neighborhood of the other", key="eps")
        st.session_state.params['dbscan']['min_samples'] = st.slider(
            "Min Samples", min_value=2, max_value=50,
            value=st.session_state.params['dbscan']['min_samples'],
            step=1,
            help="Number of samples in a neighborhood for a point to be considered as a core point", key="min_samples_dbscan")
        metric_options = ['euclidean', 'manhattan', 'chebyshev', 'minkowski']
        st.session_state.params['dbscan']['metric'] = st.selectbox(
            "Distance Metric", options=metric_options,
            index=metric_options.index(st.session_state.params['dbscan']['metric']),
            help="Metric to use when calculating distance between instances", key="metric_dbscan")
        algorithm_options = ['auto', 'ball_tree', 'kd_tree', 'brute']
        st.session_state.params['dbscan']['algorithm'] = st.selectbox(
            "Algorithm", options=algorithm_options,
            index=algorithm_options.index(st.session_state.params['dbscan']['algorithm']),
            help="Algorithm used to compute the nearest neighbors", key="algorithm")
        st.session_state.params['dbscan']['leaf_size'] = st.slider(
            "Leaf Size", min_value=10, max_value=100,
            value=st.session_state.params['dbscan']['leaf_size'],
            step=5,
            help="Leaf size passed to BallTree or KDTree", key="leaf_size")

        # Extract to local variables for convenience
        eps = st.session_state.params['dbscan']['eps']
        min_samples = st.session_state.params['dbscan']['min_samples']
        metric = st.session_state.params['dbscan']['metric']
        algorithm = st.session_state.params['dbscan']['algorithm']
        leaf_size = st.session_state.params['dbscan']['leaf_size']

    # Run clustering button
    if st.sidebar.button("🚀 Run DBSCAN", key="run_dbscan"):
        # Check if KITTI 2D bboxes are available for frustum filtering
        sample_data = st.session_state.get('sample_data', {})
        is_kitti = st.session_state.get('current_dataset') == 'kitti'
        ground_truth_boxes = sample_data.get('ground_truth_boxes', [])
        has_2d_bboxes = any(box.get('bbox_2d') is not None for box in ground_truth_boxes)

        if is_kitti and has_2d_bboxes:
            # Use frustum-based clustering
            with st.spinner("Running frustum-based DBSCAN clustering..."):
                start_time = time.time()

                # Create FrustumManager and compute frustums
                fm = FrustumManager(
                    sample_data['camera_intrinsic'],
                    sample_data['camera_to_lidar_transform']
                )
                frustums = fm.create_frustums_from_bboxes(ground_truth_boxes, depth=100)
                st.session_state.frustums = frustums

                # Get points
                points = st.session_state.point_cloud.point_cloud_plane_removed

                # Build clustering params from UI sliders
                clustering_params = {
                    'dbscan': {
                        'eps': eps,
                        'min_samples': min_samples,
                        'metric': metric,
                        'algorithm': algorithm,
                        'leaf_size': leaf_size
                    }
                }

                # Run per-frustum clustering with overlap validation
                cuboids, per_frustum_results = fm.cluster_in_frustums(
                    points, frustums,
                    min_cluster_size=min_samples,
                    min_samples=min_samples,
                    algorithm='dbscan',
                    validate_overlap=st.session_state.validate_overlap,
                    overlap_threshold=st.session_state.overlap_threshold,
                    use_templates=st.session_state.use_templates,
                    clustering_params=clustering_params,
                    ground_plane_model=st.session_state.get('ground_plane_model')
                )
                bbox_results = FrustumManager.results_to_bbox_summary(per_frustum_results)

                # Store per-frustum results
                st.session_state.per_frustum_clusters = per_frustum_results
                st.session_state.cuboids = cuboids

                # Store results
                st.session_state.clustering_results['dbscan'] = {
                    'labels': None,  # No global labels for frustum-based
                    'per_frustum_clusters': per_frustum_results,
                    'bbox_results': bbox_results,
                    'is_frustum_based': True,
                    'params': {
                        'eps': eps,
                        'min_samples': min_samples,
                        'metric': metric,
                        'algorithm': algorithm,
                        'leaf_size': leaf_size,
                        'validate_overlap': st.session_state.validate_overlap,
                        'overlap_threshold': st.session_state.overlap_threshold,
                        'use_templates': st.session_state.use_templates
                    },
                    'runtime': time.time() - start_time
                }

                n_successful = sum(1 for r in bbox_results if r['status'] == 'success')
                st.success(f"Frustum-based DBSCAN completed in {time.time() - start_time:.2f}s. "
                          f"Processed {len(frustums)} frustums, {n_successful} with clusters, {len(cuboids)} cuboids found.")
        else:
            # Standard whole point cloud clustering
            with st.spinner("Running DBSCAN clustering..."):
                start_time = time.time()

                # Get points
                points = st.session_state.point_cloud.point_cloud_plane_removed

                # Initialize clustering manager
                clustering_manager = ClusteringManager(points)

                # Run DBSCAN
                labels = clustering_manager.run_dbscan(
                    eps=eps, min_samples=min_samples, metric=metric,
                    algorithm=algorithm, leaf_size=leaf_size
                )

                # Store results
                st.session_state.clustering_results['dbscan'] = {
                    'labels': labels,
                    'is_frustum_based': False,
                    'params': {
                        'eps': eps,
                        'min_samples': min_samples,
                        'metric': metric,
                        'algorithm': algorithm,
                        'leaf_size': leaf_size
                    },
                    'runtime': time.time() - start_time
                }

                # Generate and store cuboids
                clustering_manager = ClusteringManager(point_cloud.point_cloud_plane_removed)
                cuboids = clustering_manager.generate_cuboids_from_clusters(labels)
                st.session_state.cuboids = cuboids

                st.success(f"DBSCAN completed in {time.time() - start_time:.2f} seconds")

    # Display results if available
    if 'dbscan' in st.session_state.clustering_results:
        result = st.session_state.clustering_results['dbscan']
        is_frustum_based = result.get('is_frustum_based', False)

        if is_frustum_based:
            # Frustum-based metrics
            per_frustum_clusters = result.get('per_frustum_clusters', [])
            bbox_results = result.get('bbox_results', [])
            n_frustums = len(per_frustum_clusters)
            n_successful = sum(1 for r in bbox_results if r['status'] == 'success')
            n_cuboids = len(st.session_state.get('cuboids', []))

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Frustums", n_frustums)
            with col2:
                st.metric("With Clusters", n_successful)
            with col3:
                st.metric("Total Cuboids", n_cuboids)
            with col4:
                st.metric("Runtime", f"{result['runtime']:.2f}s")

            # Show per-frustum breakdown
            if bbox_results:
                st.subheader("Per-Frustum Results")
                df = pd.DataFrame(bbox_results)
                st.dataframe(df, use_container_width=True)

            # Show IoU info if overlap validation was used
            params = result.get('params', {})
            if params.get('validate_overlap'):
                st.info(f"Overlap validation enabled with IoU threshold: {params.get('overlap_threshold', 0.3):.2f}")
        else:
            # Standard metrics
            labels = result['labels']
            unique_labels = np.unique(labels)
            n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
            n_noise = np.sum(labels == -1)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Clusters", n_clusters)
            with col2:
                st.metric("Noise Points", n_noise)
            with col3:
                st.metric("Runtime", f"{result['runtime']:.2f}s")

        # 3D Visualization (show clusters with distinct colors)
        st.subheader("3D Visualization")
        if is_frustum_based:
            # For frustum-based: combine all frustum clusters with distinct colors
            per_frustum_clusters = result.get('per_frustum_clusters', [])
            cluster_points, cluster_labels = FrustumManager.combine_cluster_results(per_frustum_clusters)
            if cluster_points is not None:
                # Create custom figure showing only clustered points with colors
                fig = create_3d_scatter_plot(cluster_points, cluster_labels, None, st.session_state.cuboids,
                                             "Frustum-Based DBSCAN Results (Colored by Cluster)")
            else:
                fig = create_3d_scatter_plot(point_cloud, None, None, st.session_state.cuboids,
                                             "Frustum-Based DBSCAN Results")
        else:
            fig = create_3d_scatter_plot(point_cloud, result['labels'], None, st.session_state.cuboids,
                                         "DBSCAN Clustering Results")
        st.plotly_chart(fig, width='stretch', key='dbscan_clustering_chart')

        # Parameter summary
        st.subheader("Parameters Used")
        params = result['params']
        col1, col2 = st.columns(2)
        with col1:
            st.json({
                "eps": params['eps'],
                "min_samples": params['min_samples'],
                "metric": params['metric']
            })
        with col2:
            st.json({
                "algorithm": params['algorithm'],
                "leaf_size": params['leaf_size']
            })

def optics_page(point_cloud):
    """OPTICS algorithm parameter control and visualization page"""
    st.header("🔭 OPTICS Clustering")
    
    # Parameter controls
    with st.sidebar.expander("OPTICS Parameters", expanded=True):
        st.session_state.params['optics']['min_samples'] = st.slider(
            "Min Samples", min_value=2, max_value=50,
            value=st.session_state.params['optics']['min_samples'],
            step=1,
            help="Number of samples in a neighborhood for a point to be considered as a core point", key="min_samples_optics")
        st.session_state.params['optics']['max_eps'] = st.slider(
            "Max Epsilon", min_value=0.1, max_value=2.0,
            value=st.session_state.params['optics']['max_eps'],
            step=0.05,
            help="Maximum distance between two samples for one to be considered as in the neighborhood of the other", key="max_eps_optics")
        st.session_state.params['optics']['xi'] = st.slider(
            "Xi", min_value=0.01, max_value=0.5,
            value=st.session_state.params['optics']['xi'],
            step=0.01,
            help="Determines the minimum steepness on the reachability plot", key="xi_optics")
        st.session_state.params['optics']['min_cluster_size'] = st.slider(
            "Min Cluster Size", min_value=5, max_value=100,
            value=st.session_state.params['optics']['min_cluster_size'],
            step=1,
            help="Minimum number of points in a cluster", key="min_cluster_size_optics")
        metric_options = ['euclidean', 'manhattan', 'chebyshev', 'minkowski']
        st.session_state.params['optics']['metric'] = st.selectbox(
            "Distance Metric", options=metric_options,
            index=metric_options.index(st.session_state.params['optics']['metric']),
            help="Metric to use when calculating distance between instances", key="metric_optics")

        # Extract to local variables for convenience
        min_samples = st.session_state.params['optics']['min_samples']
        max_eps = st.session_state.params['optics']['max_eps']
        xi = st.session_state.params['optics']['xi']
        min_cluster_size = st.session_state.params['optics']['min_cluster_size']
        metric = st.session_state.params['optics']['metric']

    # Run clustering button
    if st.sidebar.button("🚀 Run OPTICS", key="run_optics"):
        # Check if KITTI 2D bboxes are available for frustum filtering
        sample_data = st.session_state.get('sample_data', {})
        is_kitti = st.session_state.get('current_dataset') == 'kitti'
        ground_truth_boxes = sample_data.get('ground_truth_boxes', [])
        has_2d_bboxes = any(box.get('bbox_2d') is not None for box in ground_truth_boxes)

        if is_kitti and has_2d_bboxes:
            # Use frustum-based clustering
            with st.spinner("Running frustum-based OPTICS clustering..."):
                start_time = time.time()

                # Create FrustumManager and compute frustums
                fm = FrustumManager(
                    sample_data['camera_intrinsic'],
                    sample_data['camera_to_lidar_transform']
                )
                frustums = fm.create_frustums_from_bboxes(ground_truth_boxes, depth=100)
                st.session_state.frustums = frustums

                # Get points
                points = st.session_state.point_cloud.point_cloud_plane_removed

                # Build clustering params from UI sliders
                clustering_params = {
                    'optics': {
                        'min_samples': min_samples,
                        'max_eps': max_eps,
                        'xi': xi,
                        'min_cluster_size': min_cluster_size,
                        'metric': metric
                    }
                }

                # Run per-frustum clustering with OPTICS
                cuboids, per_frustum_results = fm.cluster_in_frustums(
                    points, frustums,
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    algorithm='optics',
                    validate_overlap=st.session_state.validate_overlap,
                    overlap_threshold=st.session_state.overlap_threshold,
                    use_templates=st.session_state.use_templates,
                    clustering_params=clustering_params,
                    ground_plane_model=st.session_state.get('ground_plane_model')
                )
                bbox_results = FrustumManager.results_to_bbox_summary(per_frustum_results)

                # Store results
                st.session_state.per_frustum_clusters = per_frustum_results
                st.session_state.cuboids = cuboids

                st.session_state.clustering_results['optics'] = {
                    'labels': None,
                    'per_frustum_clusters': per_frustum_results,
                    'bbox_results': bbox_results,
                    'is_frustum_based': True,
                    'params': {
                        'min_samples': min_samples,
                        'max_eps': max_eps,
                        'xi': xi,
                        'min_cluster_size': min_cluster_size,
                        'metric': metric,
                        'validate_overlap': st.session_state.validate_overlap,
                        'overlap_threshold': st.session_state.overlap_threshold,
                        'use_templates': st.session_state.use_templates
                    },
                    'runtime': time.time() - start_time
                }

                n_successful = sum(1 for r in bbox_results if r['status'] == 'success')
                st.success(f"Frustum-based OPTICS completed in {time.time() - start_time:.2f}s. "
                          f"Processed {len(frustums)} frustums, {n_successful} with clusters, {len(cuboids)} cuboids found.")
        else:
            with st.spinner("Running OPTICS clustering..."):
                start_time = time.time()

                # Get points
                points = point_cloud.point_cloud_plane_removed

                # Initialize clustering manager
                clustering_manager = ClusteringManager(points)

                # Run OPTICS
                labels = clustering_manager.run_optics(
                    min_samples=min_samples, max_eps=max_eps, xi=xi,
                    min_cluster_size=min_cluster_size, metric=metric
                )

                # Store results
                st.session_state.clustering_results['optics'] = {
                    'labels': labels,
                    'is_frustum_based': False,
                    'params': {
                        'min_samples': min_samples,
                        'max_eps': max_eps,
                        'xi': xi,
                        'min_cluster_size': min_cluster_size,
                        'metric': metric
                    },
                    'runtime': time.time() - start_time
                }

                # Generate and store cuboids
                cuboids = clustering_manager.generate_cuboids_from_clusters(labels)
                st.session_state.cuboids = cuboids
                st.success(f"OPTICS completed in {time.time() - start_time:.2f} seconds")

    # Display results if available
    if 'optics' in st.session_state.clustering_results:
        result = st.session_state.clustering_results['optics']
        is_frustum_based = result.get('is_frustum_based', False)

        if is_frustum_based:
            # Frustum-based metrics
            per_frustum_clusters = result.get('per_frustum_clusters', [])
            bbox_results = result.get('bbox_results', [])
            n_frustums = len(per_frustum_clusters)
            n_successful = sum(1 for r in bbox_results if r['status'] == 'success')
            n_cuboids = len(st.session_state.get('cuboids', []))

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Frustums", n_frustums)
            with col2:
                st.metric("With Clusters", n_successful)
            with col3:
                st.metric("Total Cuboids", n_cuboids)
            with col4:
                st.metric("Runtime", f"{result['runtime']:.2f}s")

            # Show per-frustum breakdown
            if bbox_results:
                st.subheader("Per-Frustum Results")
                df = pd.DataFrame(bbox_results)
                st.dataframe(df, use_container_width=True)

            # Show IoU info if overlap validation was used
            params = result.get('params', {})
            if params.get('validate_overlap'):
                st.info(f"Overlap validation enabled with IoU threshold: {params.get('overlap_threshold', 0.3):.2f}")
        else:
            # Standard metrics
            labels = result['labels']
            unique_labels = np.unique(labels)
            n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
            n_noise = np.sum(labels == -1)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Clusters", n_clusters)
            with col2:
                st.metric("Noise Points", n_noise)
            with col3:
                st.metric("Runtime", f"{result['runtime']:.2f}s")

        # 3D Visualization
        st.subheader("3D Visualization")
        if is_frustum_based:
            per_frustum_clusters = result.get('per_frustum_clusters', [])
            cluster_points, cluster_labels = FrustumManager.combine_cluster_results(per_frustum_clusters)
            if cluster_points is not None:
                fig = create_3d_scatter_plot(cluster_points, cluster_labels, None, st.session_state.cuboids,
                                             "Frustum-Based OPTICS Results (Colored by Cluster)")
            else:
                fig = create_3d_scatter_plot(point_cloud, None, None, st.session_state.cuboids,
                                             "Frustum-Based OPTICS Results")
        else:
            fig = create_3d_scatter_plot(point_cloud, result['labels'], None, st.session_state.cuboids,
                                         "OPTICS Clustering Results")
        st.plotly_chart(fig, width='stretch', key='optics_clustering_chart')

        # Parameter summary
        st.subheader("Parameters Used")
        st.json(result['params'])

def birch_page(point_cloud):
    """BIRCH algorithm parameter control and visualization page"""
    st.header("🌳 BIRCH Clustering")

    # Parameter controls
    with st.sidebar.expander("BIRCH Parameters", expanded=True):
        st.session_state.params['birch']['threshold'] = st.slider(
            "Threshold", min_value=0.1, max_value=2.0,
            value=st.session_state.params['birch']['threshold'],
            step=0.05,
            help="The radius of the subcluster obtained by merging a new sample and the closest subcluster", key="threshold")
        st.session_state.params['birch']['branching_factor'] = st.slider(
            "Branching Factor", min_value=10, max_value=100,
            value=st.session_state.params['birch']['branching_factor'],
            step=5,
            help="Maximum number of CF subclusters in each node", key="branching_factor")
        st.session_state.params['birch']['n_clusters'] = st.slider(
            "Number of Clusters", min_value=2, max_value=50,
            value=st.session_state.params['birch']['n_clusters'],
            step=1,
            help="Number of clusters after clustering", key="n_clusters_birch")

        # Extract to local variables for convenience
        threshold = st.session_state.params['birch']['threshold']
        branching_factor = st.session_state.params['birch']['branching_factor']
        n_clusters = st.session_state.params['birch']['n_clusters']

    # Run clustering button
    if st.sidebar.button("🚀 Run BIRCH", key="run_birch"):
        # Check if KITTI 2D bboxes are available for frustum filtering
        sample_data = st.session_state.get('sample_data', {})
        is_kitti = st.session_state.get('current_dataset') == 'kitti'
        ground_truth_boxes = sample_data.get('ground_truth_boxes', [])
        has_2d_bboxes = any(box.get('bbox_2d') is not None for box in ground_truth_boxes)

        if is_kitti and has_2d_bboxes:
            # Use frustum-based clustering (with HDBSCAN for automatic cluster detection)
            with st.spinner("Running frustum-based clustering..."):
                start_time = time.time()

                # Create FrustumManager and compute frustums
                fm = FrustumManager(
                    sample_data['camera_intrinsic'],
                    sample_data['camera_to_lidar_transform']
                )
                frustums = fm.create_frustums_from_bboxes(ground_truth_boxes, depth=100)
                st.session_state.frustums = frustums

                # Get points
                points = st.session_state.point_cloud.point_cloud_plane_removed
                clustering_params = {'birch': {'threshold': threshold, 'branching_factor': branching_factor, 'n_clusters': n_clusters}}
                # Run per-frustum clustering
                cuboids, per_frustum_results = fm.cluster_in_frustums(
                    points, frustums, min_cluster_size=5, min_samples=3, algorithm='birch',
                    validate_overlap=st.session_state.validate_overlap,
                    overlap_threshold=st.session_state.overlap_threshold,
                    use_templates=st.session_state.use_templates,
                    clustering_params=clustering_params,
                    ground_plane_model=st.session_state.get('ground_plane_model')
                )
                bbox_results = FrustumManager.results_to_bbox_summary(per_frustum_results)

                # Store results
                st.session_state.per_frustum_clusters = per_frustum_results
                st.session_state.cuboids = cuboids

                st.session_state.clustering_results['birch'] = {
                    'labels': None,
                    'per_frustum_clusters': per_frustum_results,
                    'bbox_results': bbox_results,
                    'is_frustum_based': True,
                    'params': {
                        'threshold': threshold,
                        'branching_factor': branching_factor,
                        'n_clusters': n_clusters,
                        'validate_overlap': st.session_state.validate_overlap,
                        'overlap_threshold': st.session_state.overlap_threshold,
                        'use_templates': st.session_state.use_templates
                    },
                    'runtime': time.time() - start_time
                }

                n_successful = sum(1 for r in bbox_results if r['status'] == 'success')
                st.success(f"Frustum-based BIRCH completed in {time.time() - start_time:.2f}s. "
                          f"{n_successful} frustums with clusters, {len(cuboids)} cuboids found.")
        else:
            with st.spinner("Running BIRCH clustering..."):
                start_time = time.time()

                # Get points
                points = point_cloud.point_cloud_plane_removed

                # Initialize clustering manager
                clustering_manager = ClusteringManager(points)

                # Run BIRCH
                labels = clustering_manager.run_birch(
                    threshold=threshold, branching_factor=branching_factor, n_clusters=n_clusters
                )

                # Store results
                st.session_state.clustering_results['birch'] = {
                    'labels': labels,
                    'is_frustum_based': False,
                    'params': {
                        'threshold': threshold,
                        'branching_factor': branching_factor,
                        'n_clusters': n_clusters
                    },
                    'runtime': time.time() - start_time
                }

                # Generate and store cuboids
                clustering_manager = ClusteringManager(point_cloud.point_cloud_plane_removed)
                cuboids = clustering_manager.generate_cuboids_from_clusters(labels)
                st.session_state.cuboids = cuboids

                st.success(f"BIRCH completed in {time.time() - start_time:.2f} seconds")

    # Display results if available
    if 'birch' in st.session_state.clustering_results:
        result = st.session_state.clustering_results['birch']
        is_frustum_based = result.get('is_frustum_based', False)

        if is_frustum_based:
            # Frustum-based metrics
            bbox_results = result.get('bbox_results', [])
            n_frustums = len(bbox_results)
            n_successful = sum(1 for r in bbox_results if r['status'] == 'success')
            n_cuboids = len(st.session_state.get('cuboids', []))

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Frustums", n_frustums)
            with col2:
                st.metric("With Clusters", n_successful)
            with col3:
                st.metric("Total Cuboids", n_cuboids)
            with col4:
                st.metric("Runtime", f"{result['runtime']:.2f}s")

            if bbox_results:
                st.subheader("Per-Frustum Results")
                df = pd.DataFrame(bbox_results)
                st.dataframe(df, use_container_width=True)

            # Show IoU info if overlap validation was used
            params = result.get('params', {})
            if params.get('validate_overlap'):
                st.info(f"Overlap validation enabled with IoU threshold: {params.get('overlap_threshold', 0.3):.2f}")
        else:
            # Standard metrics
            labels = result['labels']
            unique_labels = np.unique(labels)
            n_clusters = len(unique_labels)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Clusters", n_clusters)
            with col2:
                st.metric("Noise Points", 0)
            with col3:
                st.metric("Runtime", f"{result['runtime']:.2f}s")

        # 3D Visualization (show clusters with distinct colors)
        st.subheader("3D Visualization")
        if is_frustum_based:
            per_frustum_clusters = result.get('per_frustum_clusters', [])
            cluster_points, cluster_labels = FrustumManager.combine_cluster_results(per_frustum_clusters)
            if cluster_points is not None:
                fig = create_3d_scatter_plot(cluster_points, cluster_labels, None, st.session_state.cuboids,
                                             "Frustum-Based BIRCH Results (Colored by Cluster)")
            else:
                fig = create_3d_scatter_plot(point_cloud, None, None, st.session_state.cuboids,
                                             "Frustum-Based BIRCH Results")
        else:
            fig = create_3d_scatter_plot(point_cloud, result['labels'], None, st.session_state.cuboids,
                                         "BIRCH Clustering Results")
        st.plotly_chart(fig, width='stretch', key='birch_clustering_chart')

        # Parameter summary
        st.subheader("Parameters Used")
        st.json(result['params'])

def agglomerative_page(point_cloud):
    """Agglomerative clustering algorithm parameter control and visualization page"""
    st.header("🔗 Agglomerative Clustering")

    # Parameter controls
    with st.sidebar.expander("Agglomerative Parameters", expanded=True):
        st.session_state.params['agglomerative']['n_clusters'] = st.slider(
            "Number of Clusters", min_value=2, max_value=50,
            value=st.session_state.params['agglomerative']['n_clusters'],
            step=1,
            help="Number of clusters to find", key="n_clusters_agglomerative")
        linkage_options = ['ward', 'complete', 'average', 'single']
        st.session_state.params['agglomerative']['linkage'] = st.selectbox(
            "Linkage", options=linkage_options,
            index=linkage_options.index(st.session_state.params['agglomerative']['linkage']),
            help="Linkage criterion to use", key="linkage_agglomerative")

        # Extract to local variables for convenience
        n_clusters = st.session_state.params['agglomerative']['n_clusters']
        linkage = st.session_state.params['agglomerative']['linkage']

    # Run clustering button
    if st.sidebar.button("🚀 Run Agglomerative", key="run_agglomerative"):
        # Check if KITTI 2D bboxes are available for frustum filtering
        sample_data = st.session_state.get('sample_data', {})
        is_kitti = st.session_state.get('current_dataset') == 'kitti'
        ground_truth_boxes = sample_data.get('ground_truth_boxes', [])
        has_2d_bboxes = any(box.get('bbox_2d') is not None for box in ground_truth_boxes)

        if is_kitti and has_2d_bboxes:
            # Use frustum-based clustering
            with st.spinner("Running frustum-based Agglomerative clustering..."):
                start_time = time.time()

                # Create FrustumManager and compute frustums
                fm = FrustumManager(
                    sample_data['camera_intrinsic'],
                    sample_data['camera_to_lidar_transform']
                )
                frustums = fm.create_frustums_from_bboxes(ground_truth_boxes, depth=100)
                st.session_state.frustums = frustums

                # Get points
                points = st.session_state.point_cloud.point_cloud_plane_removed

                # Build clustering params from UI sliders
                clustering_params = {
                    'agglomerative': {
                        'n_clusters': n_clusters,
                        'linkage': linkage
                    }
                }

                # Run per-frustum clustering with Agglomerative
                cuboids, per_frustum_results = fm.cluster_in_frustums(
                    points, frustums,
                    min_cluster_size=10,
                    min_samples=5,
                    algorithm='agglomerative',
                    validate_overlap=st.session_state.validate_overlap,
                    overlap_threshold=st.session_state.overlap_threshold,
                    use_templates=st.session_state.use_templates,
                    clustering_params=clustering_params,
                    ground_plane_model=st.session_state.get('ground_plane_model')
                )
                bbox_results = FrustumManager.results_to_bbox_summary(per_frustum_results)

                # Store results
                st.session_state.per_frustum_clusters = per_frustum_results
                st.session_state.cuboids = cuboids

                st.session_state.clustering_results['agglomerative'] = {
                    'labels': None,
                    'per_frustum_clusters': per_frustum_results,
                    'bbox_results': bbox_results,
                    'is_frustum_based': True,
                    'params': {
                        'n_clusters': n_clusters,
                        'linkage': linkage,
                        'validate_overlap': st.session_state.validate_overlap,
                        'overlap_threshold': st.session_state.overlap_threshold,
                        'use_templates': st.session_state.use_templates
                    },
                    'runtime': time.time() - start_time
                }

                n_successful = sum(1 for r in bbox_results if r['status'] == 'success')
                st.success(f"Frustum-based Agglomerative completed in {time.time() - start_time:.2f}s. "
                          f"{n_successful} frustums with clusters, {len(cuboids)} cuboids found.")
        else:
            with st.spinner("Running Agglomerative clustering..."):
                start_time = time.time()

                # Get points
                points = point_cloud.point_cloud_plane_removed

                # Initialize clustering manager
                clustering_manager = ClusteringManager(points)

                # Run Agglomerative
                labels = clustering_manager.run_agglomerative(
                    n_clusters=n_clusters, linkage=linkage
                )

                # Store results
                st.session_state.clustering_results['agglomerative'] = {
                    'labels': labels,
                    'is_frustum_based': False,
                    'params': {
                        'n_clusters': n_clusters,
                        'linkage': linkage
                    },
                    'runtime': time.time() - start_time
                }

                # Generate and store cuboids
                clustering_manager = ClusteringManager(point_cloud.point_cloud_plane_removed)
                cuboids = clustering_manager.generate_cuboids_from_clusters(labels)
                st.session_state.cuboids = cuboids

                st.success(f"Agglomerative completed in {time.time() - start_time:.2f} seconds")

    # Display results if available
    if 'agglomerative' in st.session_state.clustering_results:
        result = st.session_state.clustering_results['agglomerative']
        is_frustum_based = result.get('is_frustum_based', False)

        if is_frustum_based:
            # Frustum-based metrics
            bbox_results = result.get('bbox_results', [])
            n_frustums = len(bbox_results)
            n_successful = sum(1 for r in bbox_results if r['status'] == 'success')
            n_cuboids = len(st.session_state.get('cuboids', []))

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Frustums", n_frustums)
            with col2:
                st.metric("With Clusters", n_successful)
            with col3:
                st.metric("Total Cuboids", n_cuboids)
            with col4:
                st.metric("Runtime", f"{result['runtime']:.2f}s")

            if bbox_results:
                st.subheader("Per-Frustum Results")
                df = pd.DataFrame(bbox_results)
                st.dataframe(df, use_container_width=True)

            # Show IoU info if overlap validation was used
            params = result.get('params', {})
            if params.get('validate_overlap'):
                st.info(f"Overlap validation enabled with IoU threshold: {params.get('overlap_threshold', 0.3):.2f}")
        else:
            # Standard metrics
            labels = result['labels']
            unique_labels = np.unique(labels)
            n_clusters = len(unique_labels)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Clusters", n_clusters)
            with col2:
                st.metric("Noise Points", 0)
            with col3:
                st.metric("Runtime", f"{result['runtime']:.2f}s")

        # 3D Visualization (show clusters with distinct colors)
        st.subheader("3D Visualization")
        if is_frustum_based:
            per_frustum_clusters = result.get('per_frustum_clusters', [])
            cluster_points, cluster_labels = FrustumManager.combine_cluster_results(per_frustum_clusters)
            if cluster_points is not None:
                fig = create_3d_scatter_plot(cluster_points, cluster_labels, None, st.session_state.cuboids,
                                             "Frustum-Based Agglomerative Results (Colored by Cluster)")
            else:
                fig = create_3d_scatter_plot(point_cloud, None, None, st.session_state.cuboids,
                                             "Frustum-Based Agglomerative Results")
        else:
            fig = create_3d_scatter_plot(point_cloud, result['labels'], None, st.session_state.cuboids,
                                         "Agglomerative Clustering Results")
        st.plotly_chart(fig, width='stretch', key='agglomerative_clustering_chart')

        # Parameter summary
        st.subheader("Parameters Used")
        st.json(result['params'])

def hdbscan_page(point_cloud):
    """HDBSCAN algorithm parameter control and visualization page"""
    st.header("🔬 HDBSCAN Clustering")

    # Parameter controls
    with st.sidebar.expander("HDBSCAN Parameters", expanded=True):
        st.session_state.params['hdbscan']['min_cluster_size'] = st.slider(
            "Min Cluster Size", min_value=5, max_value=100,
            value=st.session_state.params['hdbscan']['min_cluster_size'],
            step=1,
            help="Minimum number of points in a cluster", key="min_cluster_size_hdbscan")
        st.session_state.params['hdbscan']['min_samples'] = st.slider(
            "Min Samples", min_value=1, max_value=50,
            value=st.session_state.params['hdbscan']['min_samples'],
            step=1,
            help="Number of samples in a neighborhood for a point to be considered as a core point",
            key="min_samples_hdbscan")
        st.session_state.params['hdbscan']['cluster_selection_method'] = st.selectbox(
            "Cluster Selection Method",
            options=['eom', 'leaf'],
            index=0 if st.session_state.params['hdbscan']['cluster_selection_method'] == 'eom' else 1,
            help="Method used to select clusters from the condensed tree",
            key="cluster_selection_method_hdbscan")

        # Extract to local variables for convenience
        min_cluster_size = st.session_state.params['hdbscan']['min_cluster_size']
        min_samples = st.session_state.params['hdbscan']['min_samples']
        cluster_selection_method = st.session_state.params['hdbscan']['cluster_selection_method']

    # Run clustering button
    if st.sidebar.button("🚀 Run HDBSCAN", key="run_hdbscan"):
        # Check if KITTI 2D bboxes are available for frustum filtering
        sample_data = st.session_state.get('sample_data', {})
        is_kitti = st.session_state.get('current_dataset') == 'kitti'
        ground_truth_boxes = sample_data.get('ground_truth_boxes', [])
        has_2d_bboxes = any(box.get('bbox_2d') is not None for box in ground_truth_boxes)

        if is_kitti and has_2d_bboxes:
            # Use frustum-based clustering with HDBSCAN
            with st.spinner("Running frustum-based HDBSCAN clustering..."):
                start_time = time.time()

                # Create FrustumManager and compute frustums
                fm = FrustumManager(
                    sample_data['camera_intrinsic'],
                    sample_data['camera_to_lidar_transform']
                )
                frustums = fm.create_frustums_from_bboxes(ground_truth_boxes, depth=100)
                st.session_state.frustums = frustums

                # Get points
                points = st.session_state.point_cloud.point_cloud_plane_removed

                # Build clustering params from UI sliders
                clustering_params = {
                    'hdbscan': {
                        'min_cluster_size': min_cluster_size,
                        'min_samples': min_samples,
                        'cluster_selection_method': cluster_selection_method
                    }
                }

                # Run per-frustum clustering with HDBSCAN and overlap validation
                cuboids, per_frustum_results = fm.cluster_in_frustums(
                    points, frustums,
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    algorithm='hdbscan',
                    validate_overlap=st.session_state.validate_overlap,
                    overlap_threshold=st.session_state.overlap_threshold,
                    use_templates=st.session_state.use_templates,
                    clustering_params=clustering_params,
                    ground_plane_model=st.session_state.get('ground_plane_model')
                )
                bbox_results = FrustumManager.results_to_bbox_summary(per_frustum_results)

                # Store per-frustum results
                st.session_state.per_frustum_clusters = per_frustum_results
                st.session_state.cuboids = cuboids

                # Store results
                st.session_state.clustering_results['hdbscan'] = {
                    'labels': None,
                    'per_frustum_clusters': per_frustum_results,
                    'bbox_results': bbox_results,
                    'is_frustum_based': True,
                    'params': {
                        'min_cluster_size': min_cluster_size,
                        'min_samples': min_samples,
                        'cluster_selection_method': cluster_selection_method,
                        'validate_overlap': st.session_state.validate_overlap,
                        'overlap_threshold': st.session_state.overlap_threshold,
                        'use_templates': st.session_state.use_templates
                    },
                    'runtime': time.time() - start_time
                }

                n_successful = sum(1 for r in bbox_results if r['status'] == 'success')
                st.success(f"Frustum-based HDBSCAN completed in {time.time() - start_time:.2f}s. "
                          f"Processed {len(frustums)} frustums, {n_successful} with clusters, {len(cuboids)} cuboids found.")
        else:
            # Standard whole point cloud clustering
            with st.spinner("Running HDBSCAN clustering..."):
                start_time = time.time()

                # Get points
                points = st.session_state.point_cloud.point_cloud_plane_removed

                # Initialize clustering manager
                clustering_manager = ClusteringManager(points)

                # Run HDBSCAN
                labels = clustering_manager.run_hdbscan(
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    cluster_selection_method=cluster_selection_method
                )

                # Store results
                st.session_state.clustering_results['hdbscan'] = {
                    'labels': labels,
                    'is_frustum_based': False,
                    'params': {
                        'min_cluster_size': min_cluster_size,
                        'min_samples': min_samples,
                        'cluster_selection_method': cluster_selection_method
                    },
                    'runtime': time.time() - start_time
                }

                # Generate and store cuboids
                cuboids = clustering_manager.generate_cuboids_from_clusters(labels)
                st.session_state.cuboids = cuboids

                st.success(f"HDBSCAN completed in {time.time() - start_time:.2f} seconds")

    # Display results if available
    if 'hdbscan' in st.session_state.clustering_results:
        result = st.session_state.clustering_results['hdbscan']
        is_frustum_based = result.get('is_frustum_based', False)

        if is_frustum_based:
            # Frustum-based metrics
            per_frustum_clusters = result.get('per_frustum_clusters', [])
            bbox_results = result.get('bbox_results', [])
            n_frustums = len(per_frustum_clusters)
            n_successful = sum(1 for r in bbox_results if r['status'] == 'success')
            n_cuboids = len(st.session_state.get('cuboids', []))

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Frustums", n_frustums)
            with col2:
                st.metric("With Clusters", n_successful)
            with col3:
                st.metric("Total Cuboids", n_cuboids)
            with col4:
                st.metric("Runtime", f"{result['runtime']:.2f}s")

            # Show per-frustum breakdown
            if bbox_results:
                st.subheader("Per-Frustum Results")
                df = pd.DataFrame(bbox_results)
                st.dataframe(df, use_container_width=True)

            # Show IoU info if overlap validation was used
            params = result.get('params', {})
            if params.get('validate_overlap'):
                st.info(f"Overlap validation enabled with IoU threshold: {params.get('overlap_threshold', 0.3):.2f}")
        else:
            # Standard metrics
            labels = result['labels']
            unique_labels = np.unique(labels)
            n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
            n_noise = np.sum(labels == -1)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Clusters", n_clusters)
            with col2:
                st.metric("Noise Points", n_noise)
            with col3:
                st.metric("Runtime", f"{result['runtime']:.2f}s")

        # 3D Visualization
        st.subheader("3D Visualization")
        if is_frustum_based:
            per_frustum_clusters = result.get('per_frustum_clusters', [])
            cluster_points, cluster_labels = FrustumManager.combine_cluster_results(per_frustum_clusters)
            if cluster_points is not None:
                fig = create_3d_scatter_plot(cluster_points, cluster_labels, None, st.session_state.cuboids,
                                             "Frustum-Based HDBSCAN Results (Colored by Cluster)")
            else:
                fig = create_3d_scatter_plot(point_cloud, None, None, st.session_state.cuboids,
                                             "Frustum-Based HDBSCAN Results")
        else:
            fig = create_3d_scatter_plot(point_cloud, result['labels'], None, st.session_state.cuboids,
                                         "HDBSCAN Clustering Results")
        st.plotly_chart(fig, width='stretch', key='hdbscan_clustering_chart')

        # Parameter summary
        st.subheader("Parameters Used")
        st.json(result['params'])


def kitti_groundtruth_page():
    """KITTI Ground Truth Comparison page - Uses same pipeline as other pages"""
    st.header("🎯 KITTI Ground Truth Comparison")

    # Check if we have KITTI data loaded
    if 'sample_data' not in st.session_state or st.session_state.get('current_dataset') != 'kitti':
        st.info("👈 Switch to KITTI dataset in the sidebar and load a sample to compare with ground truth")
        st.markdown("""
        ### How to use:
        1. In the **Data Controls** sidebar, select **Dataset: KITTI**
        2. Choose a sample index (0-7480)
        3. Click **Load Sample**
        4. Run clustering on other tabs (DBSCAN, BIRCH, etc.)
        5. Return here to see ground truth vs detected objects

        The ground truth 3D cuboids from KITTI will be shown in **green**,
        while your pipeline's detected clusters will be shown in **red**.
        """)
        return

    sample_data = st.session_state.sample_data
    point_cloud = st.session_state.point_cloud

    # Get ground truth boxes for visualization
    ground_truth_boxes = sample_data.get('ground_truth_boxes', [])

    # Sidebar controls for 2D visualization
    with st.sidebar.expander("Visualization Settings", expanded=True):
        show_2d_boxes = st.checkbox("Show 2D Bounding Boxes", value=True, key="show_2d_boxes_kitti")
        match_distance_threshold = st.slider("Match Distance Threshold (m)", min_value=1.0, max_value=10.0,
                                             value=5.0, step=0.5, key="match_dist_kitti",
                                             help="Maximum distance to match detected cuboid to GT")

    # Note: Frustum-based clustering is now automatic
    st.sidebar.info("Frustum filtering is automatic when running clustering on KITTI data. "
                    "Run DBSCAN, BIRCH, or Agglomerative on the other tabs.")


    # Display camera image with optional 2D bounding boxes
    st.subheader("📷 Camera Image")
    try:
        img = cv2.imread(sample_data['image_path'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Draw 2D bounding boxes if enabled
        if show_2d_boxes and ground_truth_boxes:
            img = draw_2d_boxes_on_image(img, ground_truth_boxes)
            caption = f"KITTI Sample {sample_data.get('sample_index', 0)} - With 2D Ground Truth Boxes"
        else:
            caption = f"KITTI Sample {sample_data.get('sample_index', 0)}"

        st.image(img, caption=caption, width='stretch')

        # Show 2D bbox statistics
        if show_2d_boxes and ground_truth_boxes:
            n_boxes_with_2d = sum(1 for box in ground_truth_boxes if box.get('bbox_2d') is not None)
            st.caption(f"Showing {n_boxes_with_2d} 2D bounding boxes from KITTI annotations")

    except Exception as e:
        st.warning(f"Could not load image: {str(e)}")

    # Show reprojected cuboid bboxes if cuboids are available
    if 'cuboids' in st.session_state and st.session_state.cuboids:
        print(f'show reprojected cuboid bounding boxes')
        detected_cuboids = st.session_state.cuboids
        # Check if any cuboids have projected_bbox_2d (from find_best_cuboid)
        cuboids_with_proj = [c for c in detected_cuboids]
        if cuboids_with_proj:
            st.subheader("📐 Reprojected Cuboid Bounding Boxes")
            st.markdown("""
            **Color Legend:**
            - **Thin boxes (green/blue/etc.)**: Original 2D ground truth bounding boxes
            - **Thick boxes (orange tones)**: Reprojected 2D bboxes from detected 3D cuboids
            - **IoU**: Intersection over Union between reprojected and original bbox
            """)

            try:
                img_proj = cv2.imread(sample_data['image_path'])
                img_proj = cv2.cvtColor(img_proj, cv2.COLOR_BGR2RGB)

                # Draw both original GT boxes and reprojected cuboid bboxes
                img_proj = draw_projected_cuboid_bboxes(img_proj, cuboids_with_proj, ground_truth_boxes)

                st.image(img_proj, caption=f"Reprojected 3D Cuboids to 2D - {len(cuboids_with_proj)} cuboids",
                         use_container_width=True)

                # Show IoU statistics
                ious = [c.get('iou', 0) for c in cuboids_with_proj if c.get('iou') is not None]
                if ious:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Cuboids with Proj", len(cuboids_with_proj))
                    with col2:
                        st.metric("Mean IoU", f"{np.mean(ious):.3f}")
                    with col3:
                        st.metric("Min IoU", f"{np.min(ious):.3f}")
                    with col4:
                        st.metric("Max IoU", f"{np.max(ious):.3f}")

            except Exception as e:
                st.warning(f"Could not render reprojected bboxes: {str(e)}")

    # Display ground truth info
    if ground_truth_boxes:
        st.subheader("📦 Ground Truth Objects")
        categories = [box['category'] for box in ground_truth_boxes]
        category_counts = pd.DataFrame({'Category': categories}).value_counts().reset_index()
        category_counts.columns = ['Category', 'Count']

        col1, col2 = st.columns([1, 2])
        with col1:
            st.dataframe(category_counts, width='stretch')
        with col2:
            st.markdown(f"""
            **Total Ground Truth Objects**: {len(ground_truth_boxes)}

            These are the human-annotated 3D bounding boxes from the KITTI dataset.
            They will be visualized in **green** in the comparison plots below.
            """)

    else:
        st.warning("No ground truth boxes available for this sample")

    # Check if user has run clustering
    st.subheader("🔴 Pipeline Detection Results")
    detected_cuboids = st.session_state.get('cuboids', [])
    available_algos = [
        algo for algo in ['hdbscan', 'dbscan', 'birch', 'agglomerative', 'optics', 'frustum_filtered']
        if algo in st.session_state.clustering_results
    ]

    if not available_algos and not detected_cuboids:
        st.info("Run a clustering algorithm to see pipeline detection results.")
        return

    clustering_result = None
    algo_name = None
    if available_algos:
        algo_labels = {
            'hdbscan': 'HDBSCAN',
            'dbscan': 'DBSCAN',
            'birch': 'BIRCH',
            'agglomerative': 'Agglomerative',
            'optics': 'OPTICS',
            'frustum_filtered': 'Frustum Filtered'
        }
        label_to_algo = {algo_labels[algo]: algo for algo in available_algos}
        selected_label = st.selectbox("Clustering Result", list(label_to_algo.keys()))
        algo_name = label_to_algo[selected_label]
        clustering_result = st.session_state.clustering_results.get(algo_name)

    labels = clustering_result.get('labels') if clustering_result else None
    is_frustum_filtered = False
    if clustering_result:
        is_frustum_filtered = clustering_result.get('is_frustum_based', False) or algo_name == 'frustum_filtered'

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Ground Truth Objects", len(ground_truth_boxes))
    with col2:
        if detected_cuboids:
            detected_count = len(detected_cuboids)
        elif labels is not None:
            unique_labels = np.unique(labels)
            detected_count = len(unique_labels) - (1 if -1 in unique_labels else 0)
        else:
            detected_count = 0
        st.metric("Detected Clusters", detected_count)
    with col3:
        if is_frustum_filtered and clustering_result:
            bbox_results = clustering_result.get('bbox_results', [])
            n_successful = sum(1 for r in bbox_results if r['status'] == 'success')
            st.metric("BBoxes with Clusters", n_successful)
        elif labels is not None:
            n_noise = np.sum(labels == -1) if -1 in labels else 0
            st.metric("Noise Points", n_noise)
        elif clustering_result:
            st.metric("Method", "Frustum-based")
        else:
            st.metric("Method", "Cuboids only")

    if not detected_cuboids:
        st.info("No cuboids available for IoU matching yet.")
        return

    # 3D IoU Matching Statistics Section
    st.subheader("📊 3D IoU Matching Statistics")
    st.markdown("""
    **Matching Logic:** Each detected cuboid is matched to the ground truth box using `source_bbox_idx`
    which corresponds to the frustum index. The 3D IoU measures volumetric overlap between
    detected and ground truth cuboids.
    """)

    # Compute 3D IoU for each matched pair
    matching_results = []
    for detected in detected_cuboids:
        gt_idx = detected.get('source_bbox_idx')
        if gt_idx is not None and gt_idx < len(ground_truth_boxes):
            gt_box = ground_truth_boxes[gt_idx]
            iou_3d = compute_3d_iou(detected, gt_box)
            matching_results.append({
                'GT Index': gt_idx,
                'Category': detected.get('category', 'Unknown'),
                'GT Category': gt_box.get('category', 'Unknown'),
                '3D IoU': iou_3d,
                '2D IoU': detected.get('iou', None),
                'Need Review': detected.get('need_review', False)
            })

    if matching_results:
        # Summary metrics
        iou_3d_values = [r['3D IoU'] for r in matching_results]
        iou_2d_values = [r['2D IoU'] for r in matching_results if r['2D IoU'] is not None]

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Matched Pairs", len(matching_results))
        with col2:
            st.metric("Mean 3D IoU", f"{np.mean(iou_3d_values):.3f}")
        with col3:
            st.metric("Min 3D IoU", f"{np.min(iou_3d_values):.3f}")
        with col4:
            st.metric("Max 3D IoU", f"{np.max(iou_3d_values):.3f}")

        # Count by IoU thresholds
        st.markdown("**Detection Quality by 3D IoU Threshold:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            n_above_50 = sum(1 for iou in iou_3d_values if iou >= 0.5)
            st.metric("IoU ≥ 0.5", f"{n_above_50}/{len(iou_3d_values)}")
        with col2:
            n_above_25 = sum(1 for iou in iou_3d_values if iou >= 0.25)
            st.metric("IoU ≥ 0.25", f"{n_above_25}/{len(iou_3d_values)}")
        with col3:
            n_above_10 = sum(1 for iou in iou_3d_values if iou >= 0.1)
            st.metric("IoU ≥ 0.1", f"{n_above_10}/{len(iou_3d_values)}")

        # Detailed per-object table
        with st.expander("📋 Per-Object Matching Details", expanded=True):
            df_matching = pd.DataFrame(matching_results)
            # Format IoU columns
            df_matching['3D IoU'] = df_matching['3D IoU'].apply(lambda x: f"{x:.3f}")
            df_matching['2D IoU'] = df_matching['2D IoU'].apply(lambda x: f"{x:.3f}" if x is not None else "N/A")
            st.dataframe(df_matching, use_container_width=True)

        # Compare 2D vs 3D IoU if both available
        if iou_2d_values and len(iou_2d_values) == len(iou_3d_values):
            st.markdown("**2D vs 3D IoU Comparison:**")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Mean 2D IoU", f"{np.mean(iou_2d_values):.3f}")
            with col2:
                correlation = np.corrcoef(iou_2d_values, iou_3d_values)[0, 1]
                st.metric("2D-3D Correlation", f"{correlation:.3f}")

    else:
        st.info("No matched pairs found. Ensure detected cuboids have `source_bbox_idx` set.")

    # Unified comparison view
    st.subheader("🎯 Unified Comparison View")
    st.markdown("""
    **Color Legend:** Ground truth cuboids use lighter shades, detected cuboids use darker shades of the same color (by category).
    Yellow lines connect matched pairs.
    """)

    fig_unified = create_comparison_plot(point_cloud, ground_truth_boxes, detected_cuboids)
    st.plotly_chart(fig_unified, width='stretch', key='kitti_comparison_chart')

def statistics_page():
    """Batch evaluation statistics page for KITTI dataset"""
    st.header("📊 Batch Evaluation Statistics")

    st.markdown("""
    Run the detection pipeline on a batch of random KITTI samples and evaluate performance.
    This page calculates **3D IoU**, **Precision**, **Recall**, and other metrics across multiple samples.

    **Note:** Uses the Overlap Validation settings from the sidebar (shared with other tabs).
    """)

    # Initialize session state for statistics parameters
    if 'stats_batch_size' not in st.session_state:
        st.session_state.stats_batch_size = 10
    if 'stats_algorithm' not in st.session_state:
        st.session_state.stats_algorithm = 'hdbscan'
    if 'stats_iou_threshold' not in st.session_state:
        st.session_state.stats_iou_threshold = 0.25
    if 'stats_results' not in st.session_state:
        st.session_state.stats_results = None

    # Settings in main content area (not sidebar to avoid duplicates)
    st.subheader("Batch Settings")
    col1, col2, col3 = st.columns(3)
    with col1:
        batch_size = st.number_input("Batch Size", min_value=1, max_value=500,
                                     value=st.session_state.stats_batch_size, step=5,
                                     key="stats_batch_input",
                                     help="Number of random KITTI samples to evaluate")
        st.session_state.stats_batch_size = batch_size
    with col2:
        random_seed = st.number_input("Random Seed", min_value=0, max_value=9999,
                                      value=42, step=1, key="stats_seed",
                                      help="Seed for reproducible random sample selection")
    with col3:
        iou_threshold = st.slider("3D IoU Threshold (TP/FP)", 0.0, 1.0,
                                  st.session_state.stats_iou_threshold, 0.05,
                                  key="stats_iou_thresh",
                                  help="3D IoU threshold for counting as True Positive")
        st.session_state.stats_iou_threshold = iou_threshold

    # Algorithm selection
    st.subheader("Algorithm Settings")
    col1, col2 = st.columns([1, 2])
    with col1:
        algorithm = st.selectbox("Clustering Algorithm",
                                options=['hdbscan', 'dbscan', 'optics', 'birch', 'agglomerative'],
                                index=['hdbscan', 'dbscan', 'optics', 'birch', 'agglomerative'].index(
                                    st.session_state.stats_algorithm),
                                key="stats_algo_select")
        st.session_state.stats_algorithm = algorithm

    # Algorithm-specific parameters in main area
    with col2:
        # Default clustering params
        clustering_params = {'min_cluster_size': 5, 'min_samples': 5}

        if algorithm == 'hdbscan':
            c1, c2, c3 = st.columns(3)
            with c1:
                min_cluster_size = st.number_input("Min Cluster Size", 5, 100, 5, key="stats_hdbscan_mcs")
            with c2:
                min_samples = st.number_input("Min Samples", 1, 50, 5, key="stats_hdbscan_ms")
            
            clustering_params = {
                'min_cluster_size': min_cluster_size,
                'min_samples': min_samples,
                'hdbscan': {
                    'min_cluster_size': min_cluster_size,
                    'min_samples': min_samples
                }
            }
        elif algorithm == 'dbscan':
            c1, c2 = st.columns(2)
            with c1:
                eps = st.number_input("Epsilon", 0.1, 5.0, 0.5, 0.1, key="stats_dbscan_eps")
            with c2:
                min_samples = st.number_input("Min Samples", 2, 50, 10, key="stats_dbscan_ms")
            clustering_params = {
                'min_cluster_size': min_samples,
                'min_samples': min_samples,
                'dbscan': {'eps': eps, 'min_samples': min_samples}
            }
        elif algorithm == 'optics':
            c1, c2, c3 = st.columns(3)
            with c1:
                min_samples = st.number_input("Min Samples", 2, 50, 5, key="stats_optics_ms")
            with c2:
                max_eps = st.number_input("Max Eps", 0.5, 20.0, 5.0, 0.5, key="stats_optics_maxeps")
            with c3:
                xi = st.number_input("Xi", 0.01, 0.5, 0.05, 0.01, key="stats_optics_xi")
            clustering_params = {
                'min_cluster_size': min_samples,
                'min_samples': min_samples,
                'optics': {'min_samples': min_samples, 'max_eps': max_eps, 'xi': xi}
            }
        elif algorithm == 'birch':
            c1, c2, c3 = st.columns(3)
            with c1:
                threshold = st.number_input("Threshold", 0.1, 2.0, 0.5, 0.1, key="stats_birch_thresh")
            with c2:
                branching_factor = st.number_input("Branching Factor", 10, 100, 50, key="stats_birch_bf")
            with c3:
                n_clusters = st.number_input("N Clusters", 1, 20, 5, key="stats_birch_nc")
            clustering_params = {
                'min_cluster_size': 5,
                'min_samples': 5,
                'birch': {'threshold': threshold, 'branching_factor': branching_factor, 'n_clusters': n_clusters}
            }
        elif algorithm == 'agglomerative':
            c1, c2 = st.columns(2)
            with c1:
                n_clusters = st.number_input("N Clusters", 1, 20, 5, key="stats_agg_nc")
            with c2:
                linkage = st.selectbox("Linkage", ['ward', 'complete', 'average', 'single'], key="stats_agg_link")
            clustering_params = {
                'min_cluster_size': 5,
                'min_samples': 5,
                'agglomerative': {'n_clusters': n_clusters, 'linkage': linkage}
            }

    # Build params dict for batch evaluation
    # Use shared pipeline settings from session state if available
    pipeline_params = st.session_state.params.get('pipeline', {
        'distance_threshold': 0.3,
        'ransac_n': 3,
        'num_iterations': 1000,
        'filter_forward_only': True,
        'validate_overlap': True,
        'overlap_threshold': 0.7,
        'use_templates': True,
        'frustum_depth': 100
    })

    # Show current settings
    st.info(f"**Current Settings:** Validate Overlap: {pipeline_params['validate_overlap']} | "
            f"2D IoU Threshold: {pipeline_params['overlap_threshold']} | Use Templates: {pipeline_params['use_templates']}")

    # Run button
    st.markdown("---")
    run_button = st.button("🚀 Run Batch Evaluation", type="primary", key="run_stats_batch", use_container_width=True)

    if run_button:
        # Generate random sample indices
        np.random.seed(int(random_seed))
        max_samples = 7480  # KITTI training set size
        batch_size_int = int(batch_size)
        sample_indices = np.random.choice(max_samples, size=min(batch_size_int, max_samples), replace=False)

        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        debug_container = st.empty()

        all_results = []
        all_3d_ious = []
        all_2d_ious = []
        total_gt = 0
        total_detected = 0
        total_tp = 0  # True positives (3D IoU >= threshold)
        total_fp = 0  # False positives
        total_fn = 0  # False negatives
        per_category_stats = {}
        failed_samples = 0

        for i, sample_idx in enumerate(sample_indices):
            status_text.text(f"Processing sample {sample_idx} ({i+1}/{len(sample_indices)})...")
            progress_bar.progress((i + 1) / len(sample_indices))

            # Build params dict for this run
            batch_params = {
                'pipeline': pipeline_params,
                algorithm: clustering_params.get(algorithm, {})
            }

            result = run_pipeline_on_sample(
                sample_index=int(sample_idx),
                algorithm=algorithm,
                params_dict=batch_params
            )

            if result is None:
                failed_samples += 1
                debug_container.text(f"Progress: {len(all_results)} processed, {failed_samples} failed, "
                                    f"{total_detected} detections, {total_gt} GT objects")
                continue

            detected_cuboids = result['detected_cuboids']
            ground_truth_boxes = result['ground_truth_boxes']

            # Track totals
            total_gt += len(ground_truth_boxes)
            total_detected += len(detected_cuboids)

            # Calculate 3D IoU for each detection matched to GT
            sample_matched_gt = set()
            for det in detected_cuboids:
                gt_idx = det.get('source_bbox_idx')
                if gt_idx is not None and gt_idx < len(ground_truth_boxes):
                    gt_box = ground_truth_boxes[gt_idx]
                    iou_3d = compute_3d_iou(det, gt_box)
                    if iou_3d == 0:
                        print(f"3D IoU is 0 for gt_idx {gt_idx+1} for sample {sample_idx}")
                    all_3d_ious.append(iou_3d)

                    # Track 2D IoU if available
                    if det.get('iou') is not None:
                        all_2d_ious.append(det['iou'])

                    # Count TP/FP based on 3D IoU threshold
                    if iou_3d >= iou_threshold:
                        total_tp += 1
                        sample_matched_gt.add(gt_idx)
                    else:
                        total_fp += 1

                    # Per-category stats
                    category = det.get('category', 'Unknown')
                    if category not in per_category_stats:
                        per_category_stats[category] = {'TP': 0, 'FP': 0, 'FN': 0, 'ious': []}
                    per_category_stats[category]['ious'].append(iou_3d)
                    if iou_3d >= iou_threshold:
                        per_category_stats[category]['TP'] += 1
                    else:
                        per_category_stats[category]['FP'] += 1

            # Count FN (GT boxes not matched)
            total_fn += len(ground_truth_boxes) - len(sample_matched_gt)
            for gt_idx, gt_box in enumerate(ground_truth_boxes):
                if gt_idx not in sample_matched_gt:
                    category = gt_box.get('category', 'Unknown')
                    if category not in per_category_stats:
                        per_category_stats[category] = {'TP': 0, 'FP': 0, 'FN': 0, 'ious': []}
                    per_category_stats[category]['FN'] += 1

            all_results.append(result)
            debug_container.text(f"Progress: {len(all_results)} processed, {failed_samples} failed, "
                                f"{total_detected} detections, {total_gt} GT objects")

        progress_bar.progress(1.0)
        status_text.text(f"Completed! {len(all_results)}/{batch_size_int} samples processed, "
                        f"{failed_samples} failed, {total_detected} detections from {total_gt} GT objects.")
        debug_container.empty()

        # Store results in session state
        st.session_state.stats_results = {
            'all_results': all_results,
            'all_3d_ious': all_3d_ious,
            'all_2d_ious': all_2d_ious,
            'total_gt': total_gt,
            'total_detected': total_detected,
            'total_tp': total_tp,
            'total_fp': total_fp,
            'total_fn': total_fn,
            'per_category_stats': per_category_stats,
            'failed_samples': failed_samples,
            'params': {
                'algorithm': algorithm,
                'batch_size': batch_size_int,
                'iou_threshold': iou_threshold,
                'overlap_threshold': pipeline_params['overlap_threshold'],
                'use_templates': pipeline_params['use_templates']
            }
        }

    # Display results
    if st.session_state.stats_results is not None:
        results = st.session_state.stats_results

        st.markdown("---")
        st.subheader("📈 Evaluation Results")

        params = results['params']
        failed = results.get('failed_samples', 0)
        st.info(f"**Algorithm:** {params['algorithm'].upper()} | "
                f"**Samples Processed:** {len(results['all_results'])}/{params['batch_size']} | "
                f"**Failed:** {failed} | "
                f"**3D IoU Threshold:** {params['iou_threshold']}")

        # Summary metrics
        st.markdown("### Overall Metrics")
        col1, col2, col3, col4 = st.columns(4)

        precision = results['total_tp'] / (results['total_tp'] + results['total_fp']) if (results['total_tp'] + results['total_fp']) > 0 else 0
        recall = results['total_tp'] / (results['total_tp'] + results['total_fn']) if (results['total_tp'] + results['total_fn']) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        with col1:
            st.metric("Precision", f"{precision:.3f}")
        with col2:
            st.metric("Recall", f"{recall:.3f}")
        with col3:
            st.metric("F1 Score", f"{f1:.3f}")
        with col4:
            detection_rate = results['total_detected'] / results['total_gt'] if results['total_gt'] > 0 else 0
            st.metric("Detection Rate", f"{detection_rate:.3f}")

        # Counts
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total GT Objects", results['total_gt'])
        with col2:
            st.metric("Total Detections", results['total_detected'])
        with col3:
            st.metric("True Positives", results['total_tp'])
        with col4:
            st.metric("False Positives", results['total_fp'])

        # 3D IoU Statistics
        st.markdown("### 3D IoU Statistics")
        if results['all_3d_ious']:
            ious = results['all_3d_ious']
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean 3D IoU", f"{np.mean(ious):.3f}")
            with col2:
                st.metric("Median 3D IoU", f"{np.median(ious):.3f}")
            with col3:
                st.metric("Min 3D IoU", f"{np.min(ious):.3f}")
            with col4:
                st.metric("Max 3D IoU", f"{np.max(ious):.3f}")

            # IoU distribution by threshold
            st.markdown("**Detection Quality by 3D IoU Threshold:**")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                n_above_50 = sum(1 for iou in ious if iou >= 0.5)
                st.metric("IoU ≥ 0.5", f"{n_above_50}/{len(ious)} ({100*n_above_50/len(ious):.1f}%)")
            with col2:
                n_above_25 = sum(1 for iou in ious if iou >= 0.25)
                st.metric("IoU ≥ 0.25", f"{n_above_25}/{len(ious)} ({100*n_above_25/len(ious):.1f}%)")
            with col3:
                n_above_10 = sum(1 for iou in ious if iou >= 0.1)
                st.metric("IoU ≥ 0.1", f"{n_above_10}/{len(ious)} ({100*n_above_10/len(ious):.1f}%)")
            with col4:
                n_zero = sum(1 for iou in ious if iou == 0)
                st.metric("IoU = 0", f"{n_zero}/{len(ious)} ({100*n_zero/len(ious):.1f}%)")

            # IoU histogram
            st.markdown("**3D IoU Distribution:**")
            fig_hist = go.Figure(data=[go.Histogram(x=ious, nbinsx=20, name='3D IoU')])
            fig_hist.update_layout(
                xaxis_title="3D IoU",
                yaxis_title="Count",
                height=300
            )
            fig_hist.add_vline(x=params['iou_threshold'], line_dash="dash",
                             annotation_text=f"Threshold ({params['iou_threshold']})")
            st.plotly_chart(fig_hist, use_container_width=True)

        # 2D vs 3D IoU comparison
        if results['all_2d_ious'] and len(results['all_2d_ious']) == len(results['all_3d_ious']):
            st.markdown("### 2D vs 3D IoU Comparison")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Mean 2D IoU", f"{np.mean(results['all_2d_ious']):.3f}")
            with col2:
                corr = np.corrcoef(results['all_2d_ious'], results['all_3d_ious'])[0, 1]
                st.metric("2D-3D Correlation", f"{corr:.3f}")

            # Scatter plot
            fig_scatter = go.Figure(data=[go.Scatter(
                x=results['all_2d_ious'],
                y=results['all_3d_ious'],
                mode='markers',
                marker=dict(size=5, opacity=0.6)
            )])
            fig_scatter.update_layout(
                xaxis_title="2D IoU",
                yaxis_title="3D IoU",
                height=400
            )
            fig_scatter.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                                 line=dict(color="red", dash="dash"))
            st.plotly_chart(fig_scatter, use_container_width=True)

        # Per-category statistics
        st.markdown("### Per-Category Statistics")
        per_cat = results['per_category_stats']
        if per_cat:
            cat_data = []
            for cat, stats in per_cat.items():
                tp, fp, fn = stats['TP'], stats['FP'], stats['FN']
                cat_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                cat_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                cat_f1 = 2 * cat_precision * cat_recall / (cat_precision + cat_recall) if (cat_precision + cat_recall) > 0 else 0
                mean_iou = np.mean(stats['ious']) if stats['ious'] else 0
                cat_data.append({
                    'Category': cat,
                    'TP': tp,
                    'FP': fp,
                    'FN': fn,
                    'Precision': f"{cat_precision:.3f}",
                    'Recall': f"{cat_recall:.3f}",
                    'F1': f"{cat_f1:.3f}",
                    'Mean IoU': f"{mean_iou:.3f}"
                })

            df_cat = pd.DataFrame(cat_data)
            st.dataframe(df_cat, use_container_width=True)

        # Export option
        with st.expander("📥 Export Results"):
            export_data = {
                'params': results['params'],
                'metrics': {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'detection_rate': detection_rate,
                    'mean_3d_iou': np.mean(results['all_3d_ious']) if results['all_3d_ious'] else 0,
                    'total_gt': results['total_gt'],
                    'total_detected': results['total_detected'],
                    'total_tp': results['total_tp'],
                    'total_fp': results['total_fp'],
                    'total_fn': results['total_fn']
                },
                'per_category': {cat: {k: v for k, v in stats.items() if k != 'ious'}
                                for cat, stats in per_cat.items()}
            }
            st.json(export_data)


def main():
    """Main application function"""
    # Header
    st.markdown('<h1 class="main-header">3D Object Detection & Clustering Pipeline</h1>', 
                unsafe_allow_html=True)
    
    # Introduction
    st.markdown("""
    This application provides an interactive interface for testing and visualizing different
    clustering algorithms on 3D point cloud data from autonomous driving datasets.

    ### Features:
    - **Multiple Clustering Algorithms**: DBSCAN, OPTICS, BIRCH, Agglomerative
    - **Real-time Parameter Tuning**: Adjust parameters and see immediate results
    - **3D Visualization**: Interactive point cloud visualization with cluster coloring
    - **Performance Metrics**: Comprehensive evaluation metrics for each algorithm
    - **Algorithm Comparison**: Compare multiple algorithms side-by-side

    ### Getting Started:
    1. Load a dataset sample using the controls in the sidebar
    2. Navigate to different clustering algorithm pages
    3. Adjust parameters and run clustering
    4. Analyze the results through visualizations and metrics
    """)
    
    # Sidebar controls
    st.sidebar.header("📂 Data Controls")

    # Dataset selection
    dataset = st.sidebar.selectbox(
        "Dataset",
        options=["kitti"],
        format_func=lambda x: "KITTI" if x == "kitti" else "KITTI",
        key="dataset_selector"
    )

    # Sample selection (max value depends on dataset)
    max_sample = 7480
    sample_index = st.sidebar.text_input(
        "Sample Index",
        value=0,
        key="sample_index",
        help=f"0-{max_sample} for {dataset.upper()}"
    )
    sample_index = int(sample_index)

    # RANSAC parameters for ground plane removal
    st.sidebar.markdown("### Ground Plane Removal")
    st.session_state.params['pipeline']['distance_threshold'] = st.sidebar.slider(
        "Distance Threshold", min_value=0.1, max_value=1.0,
        value=st.session_state.params['pipeline']['distance_threshold'],
        step=0.01, key="distance_threshold")
    st.session_state.params['pipeline']['ransac_n'] = st.sidebar.slider(
        "RANSAC N", min_value=3, max_value=10,
        value=st.session_state.params['pipeline']['ransac_n'],
        step=1, key="ransac_n")
    st.session_state.params['pipeline']['num_iterations'] = st.sidebar.slider(
        "Number of Iterations", min_value=100, max_value=1000,
        value=st.session_state.params['pipeline']['num_iterations'],
        step=100, key="num_iterations")
    st.session_state.params['pipeline']['filter_forward_only'] = st.sidebar.checkbox(
        "Forward-Facing Only",
        value=st.session_state.params['pipeline']['filter_forward_only'],
        key="filter_forward_only",
        help="Keep only points in front of vehicle (x > 0). Enable for forward-facing camera datasets like KITTI.")

    # Load data button
    if st.sidebar.button("🔄 Load Sample", key="load_sample"):
        with st.spinner(f"Loading {dataset.upper()} sample {sample_index}..."):
            sample_data, point_cloud = load_dataset_sample(
                sample_index,
                st.session_state.params['pipeline']['distance_threshold'],
                st.session_state.params['pipeline']['ransac_n'],
                st.session_state.params['pipeline']['num_iterations'],
                dataset=dataset,
                filter_forward_only=st.session_state.params['pipeline']['filter_forward_only']
            )

            if sample_data is not None:
                st.session_state.data_loaded = True
                st.session_state.point_cloud = point_cloud
                st.session_state.clustering_results = {}
                st.session_state.sample_data = sample_data
                st.session_state.current_dataset = dataset
                st.session_state.cuboids = []  # Clear previous cuboids

                # Show ground truth info for KITTI
                if dataset == "kitti" and 'ground_truth_boxes' in sample_data:
                    n_gt = len(sample_data['ground_truth_boxes'])
                    st.success(f"✅ {dataset.upper()} sample {sample_index} loaded! Ground truth: {n_gt} objects")
                else:
                    st.success(f"✅ {dataset.upper()} sample {sample_index} loaded!")
                st.rerun()

    if st.sidebar.button("Estimate Depth", key="estimate_depth"):
        sample_data = st.session_state.get("sample_data")
        if not sample_data:
            st.warning("Load a sample first to estimate depth.")
        else:
            pass
    # Navigation tabs
    point_cloud = st.session_state.point_cloud
    if point_cloud is None:
        st.error("No point cloud data available")
        return

    # Display point cloud info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Point Cloud Info")
    points = point_cloud.point_cloud_plane_removed
    st.sidebar.info(f"Points: {len(points):,}")

    if len(points) > 0:
        st.sidebar.info(f"X Range: [{points[:, 0].min():.1f}, {points[:, 0].max():.1f}]")
        st.sidebar.info(f"Y Range: [{points[:, 1].min():.1f}, {points[:, 1].max():.1f}]")
        st.sidebar.info(f"Z Range: [{points[:, 2].min():.1f}, {points[:, 2].max():.1f}]")
        ground_z = st.session_state.get('ground_z')
        if ground_z is not None:
            st.sidebar.info(f"Ground Z (at origin): {ground_z:.3f}m")

    # Clear results button
    if st.sidebar.button("🗑️ Clear All Results"):
        st.session_state.clustering_results = {}
        st.rerun()
    
    with st.sidebar.expander("Overlap Validation", expanded=True):
        st.session_state.params['pipeline']['validate_overlap'] = st.checkbox(
            "Validate with 2D Overlap",
            value=st.session_state.params['pipeline']['validate_overlap'],
            key="validate_overlap_hdbscan",
            help="Select best cuboid by projecting back to 2D and checking IoU with original bbox")
        st.session_state.params['pipeline']['overlap_threshold'] = st.slider(
            "Min IoU Threshold", min_value=0.0, max_value=1.0,
            value=st.session_state.params['pipeline']['overlap_threshold'],
            step=0.05,
            key="overlap_threshold_hdbscan",
            help="Minimum IoU required to accept a cuboid")
        st.session_state.params['pipeline']['use_templates'] = st.checkbox(
            "Use Template Cuboids",
            value=st.session_state.params['pipeline']['use_templates'],
            key="use_templates_hdbscan",
            help="Use class-specific cuboid templates based on KITTI statistics")
        # Keep legacy session state variables for backward compatibility
        st.session_state.validate_overlap = st.session_state.params['pipeline']['validate_overlap']
        st.session_state.overlap_threshold = st.session_state.params['pipeline']['overlap_threshold']
        st.session_state.use_templates = st.session_state.params['pipeline']['use_templates']
    # Main navigation
    tab1, tab2, tab3, tab4 = st.tabs([
        "SEGMENTATION AND PROJECTION", "CLUSTERING", "KITTI Ground Truth", "Statistics"
    ])

    with tab1:
        project_segmentation_mask_on_pointcloud_page(st.session_state.sample_data, points)
    with tab2:
        cluster_tab1, cluster_tab2, cluster_tab3, cluster_tab4, cluster_tab5 = st.tabs([
            "HDBSCAN", "DBSCAN", "BIRCH", "Agglomerative", "OPTICS"
        ])
        with cluster_tab1:
            hdbscan_page(point_cloud)
        with cluster_tab2:
            dbscan_page(point_cloud)
        with cluster_tab3:
            birch_page(point_cloud)
        with cluster_tab4:
            agglomerative_page(point_cloud)
        with cluster_tab5:
            optics_page(point_cloud)
    with tab3:
        kitti_groundtruth_page()
    with tab4:
        statistics_page()
        

if __name__ == "__main__":
    main()