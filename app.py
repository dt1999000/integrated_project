import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import sys
import os
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd

# Add the current directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our pipeline components
from nuscenes_dataset_loader import NuScenesDatasetLoader
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

def load_dataset_sample(sample_index: int = 0, distance_threshold: float = 0.3, ransac_n: int = 3, num_iterations: int = 1000):
    """Load a sample from the nuScenes dataset"""
    try:
        # Initialize dataset loader
        dataset_loader = NuScenesDatasetLoader(dataroot='v1.0-mini')
        dataset_loader.load_dataset()
        
        # Get sample token
        sample_token = dataset_loader.nusc.sample[sample_index]['token']
        
        # Load synchronized camera and LiDAR data
        sample_data = dataset_loader.load_nuscenes_data(sample_token)
        
        if sample_data is None:
            st.error(f"Failed to load sample {sample_index}")
            return None, None, None

        # Load point cloud
        point_cloud = PointCloud(sample_data['point_cloud'])
        
        point_cloud.remove_ground_plane_ransac(distance_threshold=distance_threshold, ransac_n=ransac_n, num_iterations=num_iterations)
        #segmentation_detector = SegmentationDetector()
        return sample_data, point_cloud

    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None, None

def create_3d_scatter_plot(points: np.ndarray, labels: Optional[np.ndarray] = None,
                          title: str = "3D Point Cloud") -> go.Figure:
    """Create a 3D scatter plot using Plotly for web compatibility"""
    fig = go.Figure()

    if labels is None:
        # Single color for all points
        fig.add_trace(go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(size=2, color='lightblue'),
            name='Points'
        ))
    else:
        # Color by cluster
        unique_labels = np.unique(labels)
        colors = px.colors.qualitative.Plotly[:len(unique_labels)]

        for i, label in enumerate(unique_labels):
            if label == -1:  # Noise points
                mask = labels == label
                fig.add_trace(go.Scatter3d(
                    x=points[mask, 0],
                    y=points[mask, 1],
                    z=points[mask, 2],
                    mode='markers',
                    marker=dict(size=2, color='gray'),
                    name='Noise'
                ))
            else:
                mask = labels == label
                fig.add_trace(go.Scatter3d(
                    x=points[mask, 0],
                    y=points[mask, 1],
                    z=points[mask, 2],
                    mode='markers',
                    marker=dict(size=2, color=colors[i % len(colors)]),
                    name=f'Cluster {label}'
                ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z'),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        height=600
    )

    return fig

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
            points = point_cloud.point_cloud_plane_removed
            
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
        fig = create_3d_scatter_plot(point_cloud.point_cloud_plane_removed, labels, "DBSCAN Clustering Results")
        st.plotly_chart(fig, use_container_width=True)

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
        fig = create_3d_scatter_plot(point_cloud.point_cloud_plane_removed, labels, "OPTICS Clustering Results")
        st.plotly_chart(fig, use_container_width=True)

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
        fig = create_3d_scatter_plot(point_cloud.point_cloud_plane_removed, labels, "BIRCH Clustering Results")
        st.plotly_chart(fig, use_container_width=True)

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
        fig = create_3d_scatter_plot(point_cloud.point_cloud_plane_removed, labels, "Agglomerative Clustering Results")
        st.plotly_chart(fig, use_container_width=True)

        # Parameter summary
        st.subheader("Parameters Used")
        st.json(result['params'])

def comparison_page(point_cloud):
    """Compare different clustering algorithms side by side"""
    st.header("⚖️ Algorithm Comparison")

    # Select algorithms to compare
    with st.sidebar.expander("Comparison Settings", expanded=True):
        algorithms = st.multiselect(
            "Select Algorithms to Compare",
            options=['dbscan', 'optics', 'birch', 'agglomerative'],
            default=['dbscan', 'optics']
        )
        
        run_comparison = st.button("🚀 Run Comparison", key="run_comparison")
    
    if run_comparison:
        if not algorithms:
            st.error("Please select at least one algorithm to compare")
            return

        with st.spinner("Running comparison..."):
            start_time = time.time()
            
            # Get points
            points = point_cloud.point_cloud_plane_removed
            
            # Initialize clustering manager
            clustering_manager = ClusteringManager(points)
            
            # Run selected algorithms with default parameters
            comparison_results = {}
            
            if 'dbscan' in algorithms:
                labels = clustering_manager.run_dbscan(eps=0.5, min_samples=10)
                comparison_results['dbscan'] = {
                    'labels': labels,
                    'runtime': time.time() - start_time
                }
                start_time = time.time()
            
            if 'optics' in algorithms:
                labels = clustering_manager.run_optics(min_samples=10, max_eps=1.0)
                comparison_results['optics'] = {
                    'labels': labels,
                    'runtime': time.time() - start_time
                }
                start_time = time.time()
            
            if 'birch' in algorithms:
                labels = clustering_manager.run_birch(threshold=0.5, n_clusters=5)
                comparison_results['birch'] = {
                    'labels': labels,
                    'runtime': time.time() - start_time
                }
                start_time = time.time()
            
            if 'agglomerative' in algorithms:
                labels = clustering_manager.run_agglomerative(n_clusters=5)
                comparison_results['agglomerative'] = {
                    'labels': labels,
                    'runtime': time.time() - start_time
                }
            
            # Store results
            st.session_state.clustering_results['comparison'] = comparison_results
            st.success(f"Comparison completed in {time.time() - start_time:.2f} seconds")

    # Display comparison results if available
    if 'comparison' in st.session_state.clustering_results:
        results = st.session_state.clustering_results['comparison']

        # Create comparison table
        comparison_data = []
        for algo_name, result in results.items():
            labels = result['labels']
            unique_labels = np.unique(labels)
            n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
            n_noise = np.sum(labels == -1) if -1 in unique_labels else 0
            
            comparison_data.append({
                'Algorithm': algo_name.capitalize(),
                'Clusters': n_clusters,
                'Noise Points': n_noise,
                'Runtime (s)': f"{result['runtime']:.2f}"
            })
        
        # Display table
        df = pd.DataFrame(comparison_data)
        st.table(df)
        
        # Visualizations
        st.subheader("Visualizations")
        
        # Create subplots
        n_algos = len(results)
        fig = make_subplots(
            rows=1, cols=n_algos,
            specs=[[{'type': 'scatter3d'} for _ in range(n_algos)]],
            subplot_titles=list(results.keys())
        )
        
        # Add each algorithm's visualization
        for i, (algo_name, result) in enumerate(results.items()):
            fig.add_trace(
                go.Scatter3d(
                    x=point_cloud.point_cloud_plane_removed[:, 0],
                    y=point_cloud.point_cloud_plane_removed[:, 1],
                    z=point_cloud.point_cloud_plane_removed[:, 2],
                    mode='markers',
                    marker=dict(size=2, color=result['labels'], colorscale='Viridis'),
                    name=algo_name
                ),
                row=1, col=i+1
            )
        
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

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
    
    # Sample selection
    sample_index = st.sidebar.slider("Sample Index", min_value=0, max_value=10, value=0, step=1, key="sample_index")
    #add some slider to change the parameters of ransac interactively before loading the data
    distance_threshold = st.sidebar.slider("Distance Threshold", min_value=0.1, max_value=1.0, value=0.3, step=0.01, key="distance_threshold")
    ransac_n = st.sidebar.slider("RANSAC N", min_value=3, max_value=10, value=3, step=1, key="ransac_n")
    num_iterations = st.sidebar.slider("Number of Iterations", min_value=100, max_value=1000, value=1000, step=100, key="num_iterations")
    # Load data button
    if st.sidebar.button("🔄 Load Sample", key="load_sample"):
        with st.spinner("Loading dataset sample..."):
            sample_data, point_cloud = load_dataset_sample(sample_index, distance_threshold, ransac_n, num_iterations)
            
            if sample_data is not None:
                st.session_state.data_loaded = True
                st.session_state.point_cloud = point_cloud
                st.session_state.clustering_results = {}
                st.success(f"Sample {sample_index} loaded successfully!")
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
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 DBSCAN", "🔭 OPTICS", "🌳 BIRCH", "🔗 Agglomerative", "⚖️ Comparison"
    ])

    with tab1:
        dbscan_page(point_cloud)

    with tab2:
        optics_page(point_cloud)

    with tab3:
        birch_page(point_cloud)

    with tab4:
        agglomerative_page(point_cloud)

    with tab5:
        comparison_page(point_cloud)

if __name__ == "__main__":
    main()