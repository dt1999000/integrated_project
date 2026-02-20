import streamlit as st
import numpy as np
import sys
import os
import cv2
import time
from components.core.depth_estimation import DepthEstimator
# Add the current directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# Import page functions from pages module
from pages.project_segmentation_mask_on_pointcloud import project_segmentation_mask_on_pointcloud_page
from pages.kitti_groundtruth import kitti_groundtruth_page
from pages.depth_estimation import depth_estimation_page
from components.core.utils import load_dataset_sample

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
if 'sam_masks' not in st.session_state:
    st.session_state.sam_masks = None  # List of masks, one per bbox/instance
if 'sam_integration' not in st.session_state:
    st.session_state.sam_integration = None
if 'sam_model_type' not in st.session_state:
    st.session_state.sam_model_type = 'sam2_t'

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
    sample_index_str = st.sidebar.text_input(
        "Sample Index",
        value="0",
        key="sample_index",
        help=f"0-{max_sample} for {dataset.upper()}"
    )
    # Handle empty or invalid input
    if sample_index_str.strip() == "":
        sample_index = 0
    else:
        sample_index = int(sample_index_str)
        # Clamp to valid range
        sample_index = max(0, min(sample_index, max_sample))

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

    # SAM segmentation settings
    st.sidebar.markdown("### SAM Segmentation")
    sam_model_type = st.sidebar.selectbox(
        "SAM Model",
        options=['sam2_t', 'sam3'],
        index=0,
        key="sam_model_type",
        help="SAM2: Uses bounding boxes as prompts. SAM3: Uses text prompts for class-based segmentation."
    )
    
    # Initialize SAM integration if not already done, if None, or if model type changed
    # Use a separate key to track the initialized model type (can't modify widget key directly)
    initialized_model_type = st.session_state.get('sam_initialized_model_type')
    needs_init = (
        'sam_integration' not in st.session_state or 
        st.session_state.sam_integration is None or
        initialized_model_type != sam_model_type
    )
    
    if needs_init:
        try:
            from components.core.sam_integration import SAMIntegration
            st.session_state.sam_integration = SAMIntegration(model_type=sam_model_type)
            st.session_state.sam_initialized_model_type = sam_model_type
            print(f"SAM integration initialized successfully with model type: {sam_model_type}")
        except Exception as e:
            error_msg = str(e)
            st.sidebar.warning(f"SAM initialization failed: {error_msg}")
            print(f"SAM initialization error: {error_msg}")
            import traceback
            print(traceback.format_exc())
            st.session_state.sam_integration = None
    
    # Depth estimation settings (sparse depth only in this branch)
    st.sidebar.markdown("### Depth Estimation (Sparse Depth Only)")
    # Pose estimation parameters - always enabled, prefer l_shape_fitting
    st.sidebar.markdown("### Pose Estimation")
    pose_estimation_method = st.sidebar.selectbox(
        "Pose Estimation Method",
        options=['l_shape', 'pca'],
        index=0,
        key="pose_estimation_method",
        help="L-Shape: Robust to partial views (preferred). PCA: Fast, works well for dense point clouds."
    )
    # Always use pose estimation
    use_pose_estimation = True
    st.session_state.use_pose_estimation_checkbox = True
    
    # Cuboid fitting parameters
    st.sidebar.markdown("### Cuboid Fitting")
    if 'cuboid_fitting' not in st.session_state.params:
        st.session_state.params['cuboid_fitting'] = {
            'w_distance': 1.0,
            'w_geometric': 0.5,
            'w_outlier': 2.0,
            'step_center_search': 0.2,
            'max_step_center': 10,
            'd_theta': 0.05
        }
    
    with st.sidebar.expander("Cuboid Fitting Parameters", expanded=False):
        st.session_state.params['cuboid_fitting']['w_distance'] = st.slider(
            "Weight: Distance to Faces",
            min_value=0.0, max_value=5.0,
            value=st.session_state.params['cuboid_fitting']['w_distance'],
            step=0.1, key="cuboid_w_dist",
            help="Weight for squared distance from points to visible cuboid faces")
        
        st.session_state.params['cuboid_fitting']['w_geometric'] = st.slider(
            "Weight: Geometric Consistency",
            min_value=0.0, max_value=5.0,
            value=st.session_state.params['cuboid_fitting']['w_geometric'],
            step=0.1, key="cuboid_w_geo",
            help="Weight for geometric consistency (surface normal alignment)")
        
        st.session_state.params['cuboid_fitting']['w_outlier'] = st.slider(
            "Weight: Outlier Penalty",
            min_value=0.0, max_value=10.0,
            value=st.session_state.params['cuboid_fitting']['w_outlier'],
            step=0.1, key="cuboid_w_out",
            help="Weight for penalty on points outside the cuboid")
        
        st.session_state.params['cuboid_fitting']['step_center_search'] = st.slider(
            "Center Search Step Size",
            min_value=0.05, max_value=1.0,
            value=st.session_state.params['cuboid_fitting']['step_center_search'],
            step=0.05, key="cuboid_step_center",
            help="Step size for center search along the ray (meters)")
        
        st.session_state.params['cuboid_fitting']['max_step_center'] = st.slider(
            "Max Center Search Steps",
            min_value=1, max_value=20,
            value=st.session_state.params['cuboid_fitting']['max_step_center'],
            step=1, key="cuboid_max_steps",
            help="Maximum number of steps for center search")
        
        st.session_state.params['cuboid_fitting']['d_theta'] = st.slider(
            "Yaw Search Step (radians)",
            min_value=0.01, max_value=0.2,
            value=st.session_state.params['cuboid_fitting']['d_theta'],
            step=0.01, key="cuboid_d_theta",
            help="Angular step size for yaw search (smaller = more precise but slower)")
    
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
        
    reconstruct_full = st.sidebar.button("🔧 Generate SAM Masks", key="reconstruct_points")
    reconstruct_per_bbox = False
    
    if reconstruct_full:
        per_bbox_mode = False
        sample_data = st.session_state.get("sample_data")
        if not sample_data:
            st.sidebar.warning("Load a sample first to reconstruct points.")
        else:
            # Load image
            img = cv2.imread(sample_data['image_path'])
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Generate SAM masks if SAM integration is available and we have KITTI data
            sam_masks = None
            if st.session_state.sam_integration is not None:
                ground_truth_boxes = sample_data.get('ground_truth_boxes', [])
                if ground_truth_boxes:
                    sam_integration = st.session_state.sam_integration

                    if st.session_state.sam_model_type.startswith('sam2'):
                        masks = []
                        for gt_box in ground_truth_boxes:
                            bbox_2d = gt_box.get('bbox_2d')
                            if bbox_2d is not None:
                                bbox_list = [
                                    bbox_2d['left'],
                                    bbox_2d['top'],
                                    bbox_2d['right'],
                                    bbox_2d['bottom'],
                                ]
                                mask = sam_integration.get_mask_from_bbox(img_rgb, bbox_list)
                                masks.append(mask)
                        sam_masks = masks
                        print(f"Generated {len(masks)} masks using SAM2 from bounding boxes")

                    elif st.session_state.sam_model_type == 'sam3':
                        class_names = list(
                            set([box.get('category', 'unknown') for box in ground_truth_boxes])
                        )
                        class_names = [
                            c for c in class_names if c != 'unknown' and c != 'DontCare'
                        ]

                        if class_names:
                            segment_results = sam_integration.segment_by_classes(
                                img_rgb, class_names
                            )
                            all_masks = segment_results['masks']

                            bboxes_list = []
                            for gt_box in ground_truth_boxes:
                                bbox_2d = gt_box.get('bbox_2d')
                                if bbox_2d is not None:
                                    bboxes_list.append(
                                        [
                                            bbox_2d['left'],
                                            bbox_2d['top'],
                                            bbox_2d['right'],
                                            bbox_2d['bottom'],
                                        ]
                                    )

                            if bboxes_list:
                                matches = sam_integration.match_instances_to_bboxes(
                                    all_masks, bboxes_list, iou_threshold=0.3
                                )

                                masks = [None] * len(bboxes_list)
                                for mask_idx, bbox_idx in matches.items():
                                    masks[bbox_idx] = all_masks[mask_idx]

                                sam_masks = [m for m in masks if m is not None]
                                print(
                                    f"Generated {len(all_masks)} masks using SAM3, "
                                    f"matched {len(sam_masks)} to bounding boxes"
                                )
                            else:
                                sam_masks = all_masks
                                print(
                                    f"Generated {len(all_masks)} masks using SAM3 (no bbox matching)"
                                )

            st.session_state.sam_masks = sam_masks
            
            # Get boundaries from masks and remove boundary points from reprojected points
            if sam_masks is not None and len(sam_masks) > 0:
                # Get combined boundary mask from all masks
                boundary_mask = sam_integration.get_object_boundaries(sam_masks)
                
                # Remove boundary points from colored_sparse_points
                if (st.session_state.get('colored_sparse_points') is not None and 
                    len(st.session_state.get('colored_sparse_points', [])) > 0):
                    
                    colored_points = st.session_state.colored_sparse_points
                    colored_colors = st.session_state.colored_sparse_colors
                    
                    # Project points to 2D to check if they're at boundaries
                    from components.core.pointcloud_projection import Projection
                    projection = Projection(
                        camera_intrinsic=sample_data['camera_intrinsic'],
                        camera_extrinsic=sample_data.get('camera_extrinsic', np.eye(4)),
                        camera_to_lidar_transform=sample_data['camera_to_lidar_transform'],
                        point_cloud=colored_points
                    )
                    
                    pixels, valid_mask = projection.point_to_pixel(colored_points)
                    h, w = img_rgb.shape[:2]
                    
                    # Check which points are at boundaries
                    in_bounds = (
                        (pixels[:, 0] >= 0) & (pixels[:, 0] < w) &
                        (pixels[:, 1] >= 0) & (pixels[:, 1] < h)
                    )
                    valid_mask &= in_bounds
                    
                    # Filter out points at boundaries
                    keep_mask = np.ones(len(colored_points), dtype=bool)
                    valid_pixels = pixels[valid_mask].astype(int)
                    valid_indices = np.where(valid_mask)[0]
                    
                    for i, (u, v) in enumerate(valid_pixels):
                        if 0 <= v < h and 0 <= u < w:
                            if boundary_mask[v, u] > 0:
                                # Point is at boundary, mark for removal
                                point_idx = valid_indices[i]
                                keep_mask[point_idx] = False
                    
                    # Filter points and colors
                    filtered_points = colored_points[keep_mask]
                    
                    # Handle colors (should be numpy array with same length as points)
                    if isinstance(colored_colors, np.ndarray) and len(colored_colors) == len(colored_points):
                        filtered_colors = colored_colors[keep_mask]
                    else:
                        # Fallback: keep original colors if format doesn't match
                        filtered_colors = colored_colors
                    
                    n_removed = np.sum(~keep_mask)
                    print(f"Removed {n_removed} boundary points from {len(colored_points)} reprojected points")
                    
                    # Update session state
                    st.session_state.colored_sparse_points = filtered_points
                    st.session_state.colored_sparse_colors = filtered_colors
            
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
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "SEGMENTATION AND PROJECTION", "DEPTH ESTIMATION AND RECONSTRUCTION", "DETECTION RESULTS", "KITTI Ground Truth", "Mask & Backprojection", "Clustering"
    ])

    with tab1:
        project_segmentation_mask_on_pointcloud_page(st.session_state.sample_data, points)
    with tab2:
        depth_estimation_page()
    with tab3:
        from pages.detection_result import detection_result_page
        detection_result_page()
    with tab4:
        kitti_groundtruth_page()
    with tab5:
        from pages.sam_segmentation import sam_segmentation_page
        sam_segmentation_page()
    with tab6:
        from pages.clustering import clustering_page
        clustering_page()

if __name__ == "__main__":
    main()