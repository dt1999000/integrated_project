"""
Page module - extracted from app.py
"""
import streamlit as st
import numpy as np
import pandas as pd
import time

from visualization_helper import create_3d_scatter_plot
from frustum_manager import FrustumManager
from clustering_manager import ClusteringManager

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

    # Run clustering automatically for selected algorithm
    if st.session_state.params['pipeline']['clustering_algorithm'] == 'dbscan':
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

                # Get points - combine ground-removed LiDAR with reconstructed points if available
                points = st.session_state.point_cloud.point_cloud_plane_removed
                if st.session_state.get('reconstructed_points') is not None:
                    reconstructed = st.session_state.reconstructed_points
                    points = np.vstack([points, reconstructed])
                    print(f"Using combined point cloud: {len(points):,} points ({len(st.session_state.point_cloud.point_cloud_plane_removed):,} LiDAR + {len(reconstructed):,} reconstructed)")

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

                # Get pose estimation settings - always enabled, prefer l_shape
                use_pose_estimation = True  # Always use pose estimation
                pose_estimation_method = st.session_state.get('pose_estimation_method', 'l_shape')
                
                # Get template dimensions (only used for PCA, L-shape returns its own dimensions)
                from clustering_manager import KITTI_CUBOID_TEMPLATES
                template_dims = KITTI_CUBOID_TEMPLATES if pose_estimation_method == 'pca' else None

                # Run per-frustum clustering with overlap validation
                depth_map = st.session_state.get('depth_map')
                cuboids, per_frustum_results = fm.cluster_in_frustums(
                    points, frustums,
                    min_cluster_size=min_samples,
                    min_samples=min_samples,
                    algorithm='dbscan',
                    validate_overlap=st.session_state.validate_overlap,
                    overlap_threshold=st.session_state.overlap_threshold,
                    use_templates=st.session_state.use_templates and not use_pose_estimation,
                    clustering_params=clustering_params,
                    ground_plane_model=st.session_state.get('ground_plane_model'),
                    use_pose_estimation=use_pose_estimation,
                    pose_estimation_method=pose_estimation_method,
                    template_dims=template_dims,
                    depth_map=depth_map
                )
                bbox_results = FrustumManager.results_to_bbox_summary(per_frustum_results)

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
                fig = create_3d_scatter_plot(
                    points=cluster_points,
                    labels=cluster_labels,
                    cuboids=st.session_state.cuboids,
                    title="Frustum-Based DBSCAN Results (Colored by Cluster)"
                )
            else:
                fig = create_3d_scatter_plot(
                    points=point_cloud,
                    cuboids=st.session_state.cuboids,
                    title="Frustum-Based DBSCAN Results"
                )
        else:
            fig = create_3d_scatter_plot(
                points=point_cloud,
                labels=result['labels'],
                cuboids=st.session_state.cuboids,
                title="DBSCAN Clustering Results"
            )
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

