import streamlit as st
import numpy as np
import sys
import os
import cv2
from depth_estimation import DepthEstimator
# Add the current directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# Import page functions from pages module
from pages.project_segmentation_mask_on_pointcloud import project_segmentation_mask_on_pointcloud_page
from pages.dbscan import dbscan_page
from pages.optics import optics_page
from pages.birch import birch_page
from pages.agglomerative import agglomerative_page
from pages.hdbscan import hdbscan_page
from pages.kitti_groundtruth import kitti_groundtruth_page
from pages.statistics import statistics_page
from pages.depth_estimation import depth_estimation_page
from pages.utils import load_dataset_sample

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
if 'depth_map' not in st.session_state:
    st.session_state.depth_map = None
if 'depth_image' not in st.session_state:
    st.session_state.depth_image = None
if 'reconstructed_points' not in st.session_state:
    st.session_state.reconstructed_points = None
if 'depth_estimator' not in st.session_state:
    st.session_state.depth_estimator = None
if 'sparse_depth_map' not in st.session_state:
    st.session_state.sparse_depth_map = None
if 'completed_depth_map' not in st.session_state:
    st.session_state.completed_depth_map = None

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
            'frustum_depth': 100,
            'clustering_algorithm': 'hdbscan'
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
            'min_samples': 5,
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
        },
        # Marigold-DC parameters
        'marigold_dc': {
            'num_inference_steps': 50,
            'ensemble_size': 1,
            'processing_resolution': 768,
            'seed': 2024,
            'use_full_precision': False,
            'use_tiny_vae': False
        }
    }


# Page functions are now imported from pages module

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
    3. Adjust parameters (clustering updates automatically)
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

    st.sidebar.markdown("### Clustering Algorithm")
    algorithm_options = ['hdbscan', 'dbscan', 'optics', 'birch', 'agglomerative']
    current_algorithm = st.session_state.params['pipeline'].get('clustering_algorithm', 'hdbscan')
    if current_algorithm not in algorithm_options:
        current_algorithm = 'hdbscan'
        st.session_state.params['pipeline']['clustering_algorithm'] = current_algorithm
    st.session_state.params['pipeline']['clustering_algorithm'] = st.sidebar.selectbox(
        "Algorithm",
        options=algorithm_options,
        index=algorithm_options.index(current_algorithm),
        key="pipeline_clustering_algorithm"
    )

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

    # Depth estimation settings
    st.sidebar.markdown("### Depth Estimation")
    use_marigold = st.sidebar.checkbox(
        "Use Marigold (better quality, slower)",
        value=True,
        key="use_marigold_checkbox",
        help="Marigold provides metric depth. Falls back to Depth Anything if unavailable."
    )
    
    # Pose estimation parameters
    st.sidebar.markdown("### Pose Estimation")
    pose_estimation_method = st.sidebar.selectbox(
        "Pose Estimation Method",
        options=['pca', 'l_shape'],
        index=0,
        key="pose_estimation_method",
        help="PCA: Fast, works well for dense point clouds. L-Shape: Robust to partial views, slower."
    )
    use_pose_estimation = st.sidebar.checkbox(
        "Use Pose Estimation",
        value=False,
        key="use_pose_estimation_checkbox",
        help="Use pose estimation (PCA/L-Shape) instead of template cuboids for better orientation"
    )
    
    # Marigold-DC parameters
    with st.sidebar.expander("Marigold-DC Parameters", expanded=False):
        st.session_state.params['marigold_dc'] = st.session_state.params.get('marigold_dc', {
            'num_inference_steps': 50,
            'ensemble_size': 1,
            'processing_resolution': 768,
            'seed': 2024,
            'use_full_precision': False,
            'use_tiny_vae': False
        })
        
        st.session_state.params['marigold_dc']['num_inference_steps'] = st.slider(
            "Inference Steps", min_value=10, max_value=100, 
            value=st.session_state.params['marigold_dc']['num_inference_steps'],
            step=5, key="dc_num_steps",
            help="Number of denoising steps (more = better quality, slower)")
        
        st.session_state.params['marigold_dc']['ensemble_size'] = st.slider(
            "Ensemble Size", min_value=1, max_value=4,
            value=st.session_state.params['marigold_dc']['ensemble_size'],
            step=1, key="dc_ensemble_size",
            help="Number of predictions to ensemble (more = better quality, slower)")
        
        st.session_state.params['marigold_dc']['processing_resolution'] = st.slider(
            "Processing Resolution", min_value=256, max_value=1024,
            value=st.session_state.params['marigold_dc']['processing_resolution'],
            step=64, key="dc_resolution",
            help="Resolution for processing (higher = better quality, slower)")
        
        st.session_state.params['marigold_dc']['seed'] = st.number_input(
            "Random Seed", min_value=0, max_value=9999,
            value=st.session_state.params['marigold_dc']['seed'],
            step=1, key="dc_seed",
            help="Random seed for reproducibility")
        
        st.session_state.params['marigold_dc']['use_full_precision'] = st.checkbox(
            "Use Full Precision (float32)",
            value=st.session_state.params['marigold_dc']['use_full_precision'],
            key="dc_full_precision",
            help="Use float32 instead of float16/bf16 (slower but more accurate)")
        
        st.session_state.params['marigold_dc']['use_tiny_vae'] = st.checkbox(
            "Use Tiny VAE",
            value=st.session_state.params['marigold_dc']['use_tiny_vae'],
            key="dc_tiny_vae",
            help="Use lightweight VAE for faster processing (lower quality)")
    
    if st.sidebar.button("🔍 Estimate Depth", key="estimate_depth"):
        sample_data = st.session_state.get("sample_data")
        if not sample_data:
            st.sidebar.warning("Load a sample first to estimate depth.")
        else:
            with st.spinner("Initializing depth estimation model..."):
                # Initialize depth estimator if not already done or settings changed
                dc_params = st.session_state.params.get('marigold_dc', {})
                use_full_precision = dc_params.get('use_full_precision', False)
                use_tiny_vae = dc_params.get('use_tiny_vae', False)
                
                # Initialize or update camera parameters
                needs_init = st.session_state.depth_estimator is None
                if needs_init:
                    try:
                        st.session_state.depth_estimator = DepthEstimator(
                            use_marigold=use_marigold,
                            use_full_precision=use_full_precision,
                            use_tiny_vae=use_tiny_vae,
                            camera_intrinsic=sample_data['camera_intrinsic'],
                            camera_to_lidar_transform=sample_data['camera_to_lidar_transform']
                        )
                    except Exception as e:
                        st.sidebar.error(f"Failed to initialize depth estimator: {str(e)}")
                        st.sidebar.info("Try unchecking 'Use Marigold' to use Depth Anything instead.")
                        st.stop()
                else:
                    # Update camera parameters if they changed
                    st.session_state.depth_estimator.set_camera_params(
                        camera_intrinsic=sample_data['camera_intrinsic'],
                        camera_to_lidar_transform=sample_data['camera_to_lidar_transform']
                    )
                
                # Load image
                try:
                    img = cv2.imread(sample_data['image_path'])
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Get LiDAR point cloud if available for sparse depth prior
                    # Use original point cloud (before ground removal) for sparse depth map
                    h, w = img_rgb.shape[:2]
                    
                    if st.session_state.point_cloud is not None:
                        # Use original point cloud for sparse depth creation
                        lidar_points_original = st.session_state.point_cloud.original_point_cloud
                        print(f"Using {len(lidar_points_original):,} original LiDAR points for sparse depth map")
                        
                        # Create sparse depth map from original point cloud
                        sparse_depth = st.session_state.depth_estimator.create_sparse_depth_map(
                            point_cloud=lidar_points_original,
                            image_shape=(h, w)
                        )
                        st.session_state.sparse_depth_map = sparse_depth
                        print(f"Created sparse depth map: {np.sum(sparse_depth > 0):,} valid depth points")
                        
                        # Get DC parameters
                        dc_params = st.session_state.params.get('marigold_dc', {})
                        
                        # Complete depth using Marigold-DC
                        print("Running depth completion with Marigold-DC...")
                        completed_depth = st.session_state.depth_estimator.complete_depth(
                            image=img_rgb,
                            sparse_depth=sparse_depth,
                            num_inference_steps=dc_params.get('num_inference_steps', 50),
                            ensemble_size=dc_params.get('ensemble_size', 1),
                            processing_resolution=dc_params.get('processing_resolution', 768),
                            seed=dc_params.get('seed', 2024)
                        )
                        st.session_state.completed_depth_map = completed_depth
                        st.session_state.depth_map = completed_depth
                        
                        # Reconstruct 3D points from completed depth
                        print("Reconstructing 3D points from completed depth...")
                        reconstructed_points = st.session_state.depth_estimator.reconstruct_points_from_depth(
                            depth_map=completed_depth,
                            stride=2,  # Subsample to reduce point count
                            depth_threshold_min=0.5,
                            depth_threshold_max=80.0
                        )
                        st.session_state.reconstructed_points = reconstructed_points
                        print(f"Reconstructed {len(reconstructed_points):,} points from completed depth")
                        
                        # Add reconstructed points to original point cloud
                        print(f"Adding {len(reconstructed_points):,} reconstructed points to original point cloud...")
                        st.session_state.point_cloud.original_point_cloud = np.vstack([
                            st.session_state.point_cloud.original_point_cloud,
                            reconstructed_points
                        ])
                        print(f"Combined point cloud now has {len(st.session_state.point_cloud.original_point_cloud):,} points")
                        
                        # Re-run ground plane removal on combined point cloud
                        print("Re-running ground plane removal on combined point cloud...")
                        st.session_state.point_cloud.remove_ground_plane_ransac(
                            distance_threshold=st.session_state.params['pipeline']['distance_threshold'],
                            ransac_n=st.session_state.params['pipeline']['ransac_n'],
                            num_iterations=st.session_state.params['pipeline']['num_iterations'],
                            filter_forward_only=st.session_state.params['pipeline']['filter_forward_only']
                        )
                        
                        # Update ground_z in session state
                        ground_z = st.session_state.point_cloud.get_ground_z(x=0.0, y=0.0)
                        st.session_state.ground_z = ground_z
                        st.session_state.ground_plane_model = st.session_state.point_cloud.ground_plane_model
                        
                        n_sparse = np.sum(sparse_depth > 0)
                        coverage = 100 * n_sparse / (h * w)
                        st.sidebar.success(f"✅ Depth completed and reconstructed! Coverage: {coverage:.1f}% → 100%. Added {len(reconstructed_points):,} points to point cloud.")
                    else:
                        # No point cloud available, use regular depth estimation
                        depth_map = st.session_state.depth_estimator.get_depth_map_marigold(img_rgb)
                        st.session_state.depth_map = depth_map
                        reconstructed_points = st.session_state.depth_estimator.reconstruct_points_from_depth(
                            depth_map=depth_map,
                            stride=2,
                            depth_threshold_min=0.5,
                            depth_threshold_max=80.0
                        )
                        st.session_state.reconstructed_points = reconstructed_points
                        st.sidebar.success(f"✅ Depth estimated! Reconstructed {len(reconstructed_points):,} points")
                    
                    st.rerun()
                except Exception as e:
                    st.sidebar.error(f"Depth estimation failed: {str(e)}")
                    import traceback
                    st.sidebar.code(traceback.format_exc())
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
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "SEGMENTATION AND PROJECTION", "DEPTH ESTIMATION", "CLUSTERING", "KITTI Ground Truth", "Statistics"
    ])

    with tab1:
        project_segmentation_mask_on_pointcloud_page(st.session_state.sample_data, points)
    with tab2:
        depth_estimation_page()
    with tab3:
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
    with tab4:
        kitti_groundtruth_page()
    with tab5:
        statistics_page()
        

if __name__ == "__main__":
    main()