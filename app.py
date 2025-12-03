import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from visualization_helper import (
    draw_2d_boxes_on_image,
    add_frustums_to_figure,
    add_cuboids_to_figure,
    create_3d_scatter_plot,
    create_comparison_plot
)
import time
import sys
import os
from typing import Dict, List, Optional
import pandas as pd
import cv2
from segmentation_detection import SegmentationDetector
from bounding_boxes import BoundingBoxes
from segmentation_detection import SegmentationToPointCloud
from pointcloud_projection import Projection2DTo3D
import matplotlib.pyplot as plt
# Add the current directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our pipeline components
from nuscenes_dataset_loader import NuScenesDatasetLoader
from kitti_dataset_loader import KITTIDatasetLoader
from pointcloud_projection import PointCloud, Projection2DTo3D
from clustering_manager import ClusteringManager
from segmentation_detection import SegmentationDetector

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


def load_dataset_sample(sample_index: int = 0, distance_threshold: float = 0.3, ransac_n: int = 3, num_iterations: int = 1000, dataset: str = "nuscenes", filter_forward_only: bool = True):
    """
    Load a sample from either NuScenes or KITTI dataset.

    Args:
        sample_index: Index of the sample to load
        distance_threshold: RANSAC distance threshold for ground plane removal
        ransac_n: RANSAC number of points
        num_iterations: RANSAC number of iterations
        dataset: 'nuscenes' or 'kitti'
        filter_forward_only: Whether to keep only forward-facing points (x > 0)

    Returns:
        Tuple of (sample_data dict, PointCloud object with ground removed)
    """
    try:
        if dataset == "nuscenes":
            # Load NuScenes data
            dataset_loader = NuScenesDatasetLoader(dataroot='dataset/nuscenes')
            dataset_loader.load_dataset()

            # Get sample token
            sample_token = dataset_loader.nusc.sample[sample_index]['token']

            # Load synchronized camera and LiDAR data
            sample_data = dataset_loader.load_nuscenes_data(sample_token)

        elif dataset == "kitti":
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

        return sample_data, point_cloud

    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None, None

def get_segmentation_mask(sample_data):
    # Check if we have NuScenes data (segmentation only works with NuScenes)
    if 'nusc' not in sample_data or 'sample_token' not in sample_data:
        return None

    image = cv2.imread(sample_data['image_path'])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    nusc = sample_data['nusc']
    bounding_boxes = BoundingBoxes(nusc=nusc, data_format="nuscenes")
    bounding_boxes.get_boxes_for_sample(sample_data['sample_token'], "CAM_FRONT")
    segmentation_detector = SegmentationDetector()
    segmentation_mask = segmentation_detector.get_segmentation_mask(image, bounding_boxes)
    return segmentation_mask

def project_segmentation_mask_on_pointcloud(sample_data, segmentation_mask, point_cloud, max_distance=100.0, distance_threshold=0.5):
    projection = Projection2DTo3D(
        camera_intrinsic=sample_data['camera_intrinsic'],
        camera_extrinsic=sample_data['camera_extrinsic'],
        camera_to_lidar_transform=sample_data['camera_to_lidar_transform'],
        point_cloud=point_cloud,
        image=sample_data['image_path']
    )
    segmentation_projection = SegmentationToPointCloud(projection)
    results = segmentation_projection.project_all_masks(segmentation_mask, max_distance=max_distance, distance_threshold=distance_threshold)
    rays = {}
    mask_points = {}
    for mask_id, result in results.items():
        rays[mask_id] = result['rays']
        mask_points[mask_id] = result['projected_points']
    return rays, mask_points

def project_segmentation_mask_on_pointcloud_page(sample_data, point_cloud):
    st.header("🎯 Project Segmentation Mask on Point Cloud")

    # Check if we're using KITTI dataset (segmentation only works with NuScenes)
    if st.session_state.get('current_dataset') == 'kitti':
        st.warning("⚠️ Segmentation projection is only available for NuScenes dataset. Please switch to NuScenes in the sidebar to use this feature.")
        st.info("💡 For KITTI dataset, please use the 'KITTI Ground Truth' tab to view ground truth annotations.")
        return

    st.subheader("Segmentation Mask")

    # Projection parameters in sidebar
    with st.sidebar.expander("Projection Parameters", expanded=True):
        max_distance = st.slider("Max Ray Distance", min_value=10.0, max_value=200.0, value=100.0, step=10.0,
                              help="Maximum ray extension distance for projection", key="max_distance")
        distance_threshold = st.slider("Distance Threshold", min_value=0.1, max_value=2.0, value=0.5, step=0.1,
                                    help="Maximum perpendicular distance to consider a point on the ray", key="ray_distance_threshold")

    # Run segmentation button
    if st.sidebar.button("🚀 Run Segmentation", key="run_segmentation"):
        with st.spinner("Running segmentation..."):
            segmentation_mask = get_segmentation_mask(sample_data)
            if segmentation_mask is None:
                st.error("Failed to generate segmentation mask. Please ensure NuScenes data is loaded.")
                return
            st.session_state.segmentation_masks = segmentation_mask
            
            # Show segmentation mask on image
            fig, ax = plt.subplots(1, 2, figsize=(15, 6))
            ax[0].imshow(cv2.imread(sample_data['image_path']))
            ax[0].set_title("Original Image")
            ax[0].axis('off')
            ax[1].imshow(cv2.imread(sample_data['image_path']))
            ax[1].imshow(segmentation_mask, cmap='rgb', alpha=0.7)
            ax[1].set_title("Segmentation Mask")
            ax[1].axis('off')
            st.pyplot(fig)
            
            # Get unique mask IDs
            unique_mask_ids = np.unique(segmentation_mask)
            unique_mask_ids = [id for id in unique_mask_ids if id != 0]  # Filter out background (0)
            st.session_state.unique_mask_ids = unique_mask_ids
            st.success(f"Found {len(unique_mask_ids)} segmentation masks")
    
        if st.session_state.segmentation_masks is not None:

            with st.spinner("Running projection..."):
                start_time = time.time()
                rays, mask_points = project_segmentation_mask_on_pointcloud(
                    sample_data, 
                    st.session_state.segmentation_masks, 
                    point_cloud,
                    max_distance=max_distance,
                    distance_threshold=distance_threshold
                )
                runtime = time.time() - start_time
                
                # Store results in session state
                st.session_state.all_rays = rays
                st.session_state.all_mask_points = mask_points
                st.session_state.projection_runtime = runtime
                
                # Save the projected points to the point cloud for further use
                if hasattr(point_cloud, 'copy') and hasattr(point_cloud, 'add_segmentation_projected_points'):
                    # It's a PointCloud object
                    projected_point_cloud = point_cloud.copy()
                    projected_point_cloud.add_segmentation_projected_points(mask_points)
                    st.session_state.projected_point_cloud = projected_point_cloud
                else:
                    # It's a numpy array
                    st.session_state.projected_point_cloud = point_cloud
                    st.session_state.all_mask_points = mask_points  # Just store the mask points separately
                
                st.success(f"Projection completed in {runtime:.2f} seconds")
            
            # Display results if available
            if 'all_mask_points' in st.session_state and st.session_state.all_mask_points:
                # Get all available mask IDs
                all_mask_ids = list(st.session_state.all_mask_points.keys())
                
                # Create mask selection slider
                st.subheader("Mask Selection")
                
                # Determine how many masks to show
                num_masks = len(all_mask_ids)
                if num_masks > 10:
                    # If more than 10 masks, create a slider to select a range
                    start_idx = st.slider("Start Mask Index", 0, max(0, num_masks - 10), 0, key="mask_start_idx")
                    end_idx = min(start_idx + 10, num_masks)
                    selected_mask_ids = all_mask_ids[start_idx:end_idx]
                    st.info(f"Showing masks {start_idx+1} to {end_idx} of {num_masks}")
                else:
                    # If 10 or fewer, show all masks
                    selected_mask_ids = all_mask_ids
                
                # Create multiselect for specific masks
                selected_masks = st.multiselect(
                    "Select Masks to Display",
                    options=selected_mask_ids,
                    default=selected_mask_ids[:min(3, len(selected_mask_ids))],
                    key="selected_masks"
                )
                
                # Filter mask_points and rays based on selection
                filtered_mask_points = {mask_id: st.session_state.all_mask_points[mask_id] 
                                    for mask_id in selected_masks if mask_id in st.session_state.all_mask_points}
                filtered_rays = {mask_id: st.session_state.all_rays[mask_id] 
                            for mask_id in selected_masks if mask_id in st.session_state.all_rays}
                
                # Create 3D visualization
                st.subheader("3D Visualization")
                if point_cloud is not None:
                    display_point_cloud = st.session_state.projected_point_cloud if st.session_state.projected_point_cloud is not None else point_cloud
                    fig = create_3d_scatter_plot(display_point_cloud, None, filtered_mask_points, None, filtered_rays, 
                                            "Projected Segmentation Mask on Point Cloud")
                    st.plotly_chart(fig, width='stretch')

def dbscan_page(point_cloud):
    """DBSCAN algorithm parameter control and visualization page"""
    st.header("🎯 DBSCAN Clustering")

    # Parameter controls
    with st.sidebar.expander("DBSCAN Parameters", expanded=True):
        eps = st.slider("Epsilon (eps)", min_value=0.1, max_value=2.0, value=0.5, step=0.05,
                     help="Maximum distance between two samples for one to be considered as in the neighborhood of the other", key="eps")
        min_samples = st.slider("Min Samples", min_value=2, max_value=50, value=10, step=1,
                              help="Number of samples in a neighborhood for a point to be considered as a core point", key="min_samples_dbscan")
        metric = st.selectbox("Distance Metric", options=['euclidean', 'manhattan', 'chebyshev', 'minkowski'],
                            index=0, help="Metric to use when calculating distance between instances", key="metric_dbscan")
        algorithm = st.selectbox("Algorithm", options=['auto', 'ball_tree', 'kd_tree', 'brute'],
                              index=0, help="Algorithm used to compute the nearest neighbors", key="algorithm")
        leaf_size = st.slider("Leaf Size", min_value=10, max_value=100, value=30, step=5,
                           help="Leaf size passed to BallTree or KDTree", key="leaf_size")

    # Run clustering button
    if st.sidebar.button("🚀 Run DBSCAN", key="run_dbscan"):
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

        # Metrics
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
        fig = create_3d_scatter_plot(point_cloud, labels, None, st.session_state.cuboids, "DBSCAN Clustering Results")
        st.plotly_chart(fig, width='stretch')

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
        min_samples = st.slider("Min Samples", min_value=2, max_value=50, value=10, step=1,
                              help="Number of samples in a neighborhood for a point to be considered as a core point", key="min_samples_optics")
        max_eps = st.slider("Max Epsilon", min_value=0.1, max_value=2.0, value=1.0, step=0.05,
                          help="Maximum distance between two samples for one to be considered as in the neighborhood of the other", key="max_eps_optics")
        xi = st.slider("Xi", min_value=0.01, max_value=0.5, value=0.05, step=0.01,
                     help="Determines the minimum steepness on the reachability plot", key="xi_optics")
        min_cluster_size = st.slider("Min Cluster Size", min_value=5, max_value=100, value=10, step=1,
                                  help="Minimum number of points in a cluster", key="min_cluster_size_optics")
        metric = st.selectbox("Distance Metric", options=['euclidean', 'manhattan', 'chebyshev', 'minkowski'],
                            index=0, help="Metric to use when calculating distance between instances", key="metric")

    # Run clustering button
    if st.sidebar.button("🚀 Run OPTICS", key="run_optics"):
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
            clustering_manager = ClusteringManager(point_cloud.point_cloud_plane_removed)
            cuboids = clustering_manager.generate_cuboids_from_clusters(labels)
            st.session_state.cuboids = cuboids
            st.success(f"OPTICS completed in {time.time() - start_time:.2f} seconds")

    # Display results if available
    if 'optics' in st.session_state.clustering_results:
        result = st.session_state.clustering_results['optics']
        
        # Metrics
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
        fig = create_3d_scatter_plot(point_cloud, labels, None, st.session_state.cuboids, "OPTICS Clustering Results")
        st.plotly_chart(fig, width='stretch')

        # Parameter summary
        st.subheader("Parameters Used")
        st.json(result['params'])

def birch_page(point_cloud):
    """BIRCH algorithm parameter control and visualization page"""
    st.header("🌳 BIRCH Clustering")

    # Parameter controls
    with st.sidebar.expander("BIRCH Parameters", expanded=True):
        threshold = st.slider("Threshold", min_value=0.1, max_value=2.0, value=0.5, step=0.05,
                           help="The radius of the subcluster obtained by merging a new sample and the closest subcluster", key="threshold")
        branching_factor = st.slider("Branching Factor", min_value=10, max_value=100, value=50, step=5,
                                 help="Maximum number of CF subclusters in each node", key="branching_factor")
        n_clusters = st.slider("Number of Clusters", min_value=2, max_value=50, value=5, step=1,
                             help="Number of clusters after clustering", key="n_clusters_birch")

    # Run clustering button
    if st.sidebar.button("🚀 Run BIRCH", key="run_birch"):
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
        
        # Metrics
        labels = result['labels']
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Clusters", n_clusters)
        with col2:
            st.metric("Noise Points", 0)  # BIRCH doesn't have noise points
        with col3:
            st.metric("Runtime", f"{result['runtime']:.2f}s")
        
        # 3D Visualization
        st.subheader("3D Visualization")
        fig = create_3d_scatter_plot(point_cloud, labels, None, st.session_state.cuboids, "BIRCH Clustering Results")
        st.plotly_chart(fig, width='stretch')

        # Parameter summary
        st.subheader("Parameters Used")
        st.json(result['params'])

def agglomerative_page(point_cloud):
    """Agglomerative clustering algorithm parameter control and visualization page"""
    st.header("🔗 Agglomerative Clustering")

    # Parameter controls
    with st.sidebar.expander("Agglomerative Parameters", expanded=True):
        n_clusters = st.slider("Number of Clusters", min_value=2, max_value=50, value=5, step=1,
                             help="Number of clusters to find", key="n_clusters_agglomerative")
        linkage = st.selectbox("Linkage", options=['ward', 'complete', 'average', 'single'],
                            index=0, help="Linkage criterion to use", key="linkage_agglomerative")
        affinity = st.selectbox("Affinity", options=['euclidean', 'manhattan', 'cosine', 'l1', 'l2'],
                             index=0, help="Metric used to compute the linkage", key="affinity_agglomerative")

    # Run clustering button
    if st.sidebar.button("🚀 Run Agglomerative", key="run_agglomerative"):
        with st.spinner("Running Agglomerative clustering..."):
            start_time = time.time()
            
            # Get points
            points = point_cloud.point_cloud_plane_removed
            
            # Initialize clustering manager
            clustering_manager = ClusteringManager(points)
            
            # Run Agglomerative
            labels = clustering_manager.run_agglomerative(
                n_clusters=n_clusters, linkage=linkage, affinity=affinity
            )
            
            # Store results
            st.session_state.clustering_results['agglomerative'] = {
                'labels': labels,
                'params': {
                    'n_clusters': n_clusters,
                    'linkage': linkage,
                    'affinity': affinity
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
        
        # Metrics
        labels = result['labels']
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Clusters", n_clusters)
        with col2:
            st.metric("Noise Points", 0)  # Agglomerative doesn't have noise points
        with col3:
            st.metric("Runtime", f"{result['runtime']:.2f}s")
        
        # 3D Visualization
        st.subheader("3D Visualization")
        fig = create_3d_scatter_plot(point_cloud, labels, None, st.session_state.cuboids, "Agglomerative Clustering Results")
        st.plotly_chart(fig, width='stretch')

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

    # Sidebar controls for 2D/3D visualization
    with st.sidebar.expander("2D/3D Projection Settings", expanded=True):
        show_2d_boxes = st.checkbox("Show 2D Bounding Boxes", value=True, key="show_2d_boxes_kitti")
        show_frustums = st.checkbox("Show 2D→3D Frustums", value=True, key="show_frustums_kitti")
        frustum_depth = st.slider("Frustum Depth (m)", min_value=5, max_value=100, value=30, step=5,
                                  key="frustum_depth_kitti",
                                  help="How far to project the 2D bounding box frustums into 3D space")
        frustum_opacity = st.slider("Frustum Opacity", min_value=0.05, max_value=0.5, value=0.15, step=0.05,
                                    key="frustum_opacity_kitti")

    # Frustum-based clustering controls
    with st.sidebar.expander("Frustum-Based Clustering", expanded=False):
        use_frustum_filtering = st.checkbox(
            "Filter Points by Frustums",
            value=False,
            key="use_frustum_filtering_kitti",
            help="Only cluster points inside the 2D bbox frustum projections"
        )
        frustum_cluster_depth = st.slider(
            "Filtering Depth (m)", min_value=5, max_value=100, value=30, step=5,
            key="frustum_cluster_depth_kitti",
            help="Depth of frustums for point filtering"
        )
        frustum_min_cluster_size = st.slider(
            "Min Cluster Size", min_value=5, max_value=100, value=15, step=5,
            key="frustum_min_cluster_size_kitti",
            help="Minimum points for a cluster when using frustum filtering"
        )
        frustum_min_samples = st.slider(
            "Min Samples", min_value=2, max_value=30, value=5, step=1,
            key="frustum_min_samples_kitti",
            help="Minimum samples for HDBSCAN core points"
        )

        # Run frustum-filtered clustering button
        run_frustum_clustering = st.button(
            "🎯 Run Frustum-Filtered Clustering",
            key="run_frustum_clustering_kitti",
            help="Re-run clustering only on points inside the 2D bbox frustums"
        )

    # Handle frustum-filtered clustering
    if run_frustum_clustering:
        if not ground_truth_boxes:
            st.sidebar.error("No ground truth boxes available for frustum filtering")
        else:
            camera_intrinsic = sample_data.get('camera_intrinsic')
            camera_to_lidar = sample_data.get('camera_to_lidar_transform')

            if camera_intrinsic is None or camera_to_lidar is None:
                st.sidebar.error("Camera calibration data not available")
            else:
                with st.spinner("Running frustum-filtered clustering..."):
                    start_time = time.time()

                    # Create projection instance
                    projection = Projection2DTo3D(
                        camera_intrinsic=camera_intrinsic,
                        camera_extrinsic=np.eye(4),
                        camera_to_lidar_transform=camera_to_lidar,
                        point_cloud=point_cloud.point_cloud_plane_removed
                    )

                    # Get frustums from ground truth 2D bboxes
                    frustums = projection.get_frustums_from_bboxes(
                        ground_truth_boxes, depth=frustum_cluster_depth
                    )

                    # Filter points by frustums
                    filtered_points, mask = point_cloud.filter_by_frustums(frustums)

                    if len(filtered_points) < frustum_min_cluster_size:
                        st.sidebar.warning(
                            f"Only {len(filtered_points)} points in frustums. "
                            f"Need at least {frustum_min_cluster_size} for clustering."
                        )
                    else:
                        # Run HDBSCAN on filtered points
                        cluster_manager = ClusteringManager(filtered_points)

                        try:
                            labels = cluster_manager.run_hdbscan(
                                min_cluster_size=frustum_min_cluster_size,
                                min_samples=frustum_min_samples
                            )
                        except Exception as e:
                            # Fallback to DBSCAN if HDBSCAN fails
                            st.sidebar.warning(f"HDBSCAN failed ({e}), using DBSCAN")
                            labels = cluster_manager.run_dbscan(
                                eps=0.5, min_samples=frustum_min_samples
                            )

                        # Generate cuboids from clusters
                        cuboids = cluster_manager.generate_cuboids_from_clusters(labels)

                        # Store results
                        st.session_state.clustering_results['frustum_filtered'] = {
                            'labels': labels,
                            'filtered_points': filtered_points,
                            'mask': mask,
                            'params': {
                                'depth': frustum_cluster_depth,
                                'min_cluster_size': frustum_min_cluster_size,
                                'min_samples': frustum_min_samples,
                                'n_frustums': len(frustums)
                            },
                            'runtime': time.time() - start_time
                        }
                        st.session_state.cuboids = cuboids

                        # Calculate stats
                        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                        n_noise = np.sum(labels == -1)

                        st.sidebar.success(
                            f"Frustum clustering complete!\n"
                            f"- Points in frustums: {len(filtered_points)}\n"
                            f"- Clusters found: {n_clusters}\n"
                            f"- Noise points: {n_noise}\n"
                            f"- Runtime: {time.time() - start_time:.2f}s"
                        )

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
    if 'cuboids' in st.session_state and st.session_state.cuboids:
        detected_cuboids = st.session_state.cuboids

        # Get clustering results
        clustering_result = None
        for algo in ['dbscan', 'optics', 'birch', 'agglomerative']:
            if algo in st.session_state.clustering_results:
                clustering_result = st.session_state.clustering_results[algo]
                labels = clustering_result['labels']
                break

        if clustering_result:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Ground Truth Objects", len(ground_truth_boxes))
            with col2:
                st.metric("Detected Clusters", len(detected_cuboids))
            with col3:
                n_noise = np.sum(labels == -1) if -1 in labels else 0
                st.metric("Noise Points", n_noise)

            # Side-by-side comparison
            st.subheader("📊 Side-by-Side Comparison")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**🟢 Ground Truth (KITTI)**")
                fig_gt = create_3d_scatter_plot(
                    point_cloud,
                    None,
                    None,
                    ground_truth_boxes,
                    None,
                    "Ground Truth"
                )

                # Add frustums to ground truth view if enabled
                if show_frustums and ground_truth_boxes:
                    camera_intrinsic = sample_data.get('camera_intrinsic')
                    camera_to_lidar = sample_data.get('camera_to_lidar_transform')
                    if camera_intrinsic is not None and camera_to_lidar is not None:
                        add_frustums_to_figure(
                            fig_gt, ground_truth_boxes,
                            camera_intrinsic, camera_to_lidar,
                            depth=frustum_depth, opacity=frustum_opacity
                        )

                st.plotly_chart(fig_gt, width='stretch')

            with col2:
                st.markdown("**🔴 Pipeline Detection**")
                fig_det = create_3d_scatter_plot(
                    point_cloud,
                    labels,
                    None,
                    detected_cuboids,
                    None,
                    "Detected Objects"
                )
                st.plotly_chart(fig_det, width='stretch')

            # Overlay view
            st.subheader("🎨 Overlay Comparison")
            st.markdown("**Green** = Ground Truth | **Red** = Detected | **Pyramids** = 2D→3D Frustums")
            fig_overlay = create_comparison_plot(
                point_cloud,
                ground_truth_boxes,
                detected_cuboids
            )

            # Add frustums to overlay view if enabled
            if show_frustums and ground_truth_boxes:
                camera_intrinsic = sample_data.get('camera_intrinsic')
                camera_to_lidar = sample_data.get('camera_to_lidar_transform')
                if camera_intrinsic is not None and camera_to_lidar is not None:
                    add_frustums_to_figure(
                        fig_overlay, ground_truth_boxes,
                        camera_intrinsic, camera_to_lidar,
                        depth=frustum_depth, opacity=frustum_opacity
                    )

            st.plotly_chart(fig_overlay, width='stretch')

        else:
            st.info("Run a clustering algorithm (DBSCAN, OPTICS, etc.) on other tabs to see detection comparison")
    else:
        st.info("Run a clustering algorithm (DBSCAN, OPTICS, etc.) on other tabs to see detection comparison")

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
        options=["nuscenes", "kitti"],
        format_func=lambda x: "NuScenes" if x == "nuscenes" else "KITTI",
        key="dataset_selector"
    )

    # Sample selection (max value depends on dataset)
    max_sample = 403 if dataset == "nuscenes" else 7480
    sample_index = st.sidebar.slider(
        "Sample Index",
        min_value=0,
        max_value=max_sample,
        value=0,
        step=1,
        key="sample_index",
        help=f"0-{max_sample} for {dataset.upper()}"
    )

    # RANSAC parameters for ground plane removal
    st.sidebar.markdown("### Ground Plane Removal")
    distance_threshold = st.sidebar.slider("Distance Threshold", min_value=0.1, max_value=1.0, value=0.3, step=0.01, key="distance_threshold")
    ransac_n = st.sidebar.slider("RANSAC N", min_value=3, max_value=10, value=3, step=1, key="ransac_n")
    num_iterations = st.sidebar.slider("Number of Iterations", min_value=100, max_value=1000, value=1000, step=100, key="num_iterations")
    filter_forward_only = st.sidebar.checkbox("Forward-Facing Only", value=True, key="filter_forward_only",
                                              help="Keep only points in front of vehicle (x > 0). Enable for forward-facing camera datasets like KITTI.")

    # Load data button
    if st.sidebar.button("🔄 Load Sample", key="load_sample"):
        with st.spinner(f"Loading {dataset.upper()} sample {sample_index}..."):
            sample_data, point_cloud = load_dataset_sample(
                sample_index,
                distance_threshold,
                ransac_n,
                num_iterations,
                dataset=dataset,
                filter_forward_only=filter_forward_only
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

    # Clear results button
    if st.sidebar.button("🗑️ Clear All Results"):
        st.session_state.clustering_results = {}
        st.rerun()

    # Main navigation
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "SEGMENTATION AND PROJECTION ", "DBSCAN", " BIRCH", " Agglomerative", " Comparison", "KITTI Ground Truth"
    ])

    with tab1:
        #project_segmask_on_pointcloud_page(segmentation_mask, point_cloud)
        project_segmentation_mask_on_pointcloud_page(st.session_state.sample_data, points)
    with tab2:
        dbscan_page(point_cloud)

    with tab3:
        birch_page(point_cloud)

    with tab4:
        agglomerative_page(point_cloud)

    with tab5:
        optics_page(point_cloud)

    with tab6:
        kitti_groundtruth_page()
        

if __name__ == "__main__":
    main()