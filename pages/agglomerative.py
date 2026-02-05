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

    # Run clustering automatically for selected algorithm
    if st.session_state.params['pipeline']['clustering_algorithm'] == 'agglomerative':
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

                # Get points - combine ground-removed LiDAR with reconstructed points if available
                points = st.session_state.point_cloud.point_cloud_plane_removed
                if st.session_state.get('reconstructed_points') is not None:
                    reconstructed = st.session_state.reconstructed_points
                    points = np.vstack([points, reconstructed])
                    print(f"Using combined point cloud: {len(points):,} points ({len(st.session_state.point_cloud.point_cloud_plane_removed):,} LiDAR + {len(reconstructed):,} reconstructed)")

                # Build clustering params from UI sliders
                clustering_params = {
                    'agglomerative': {
                        'n_clusters': n_clusters,
                        'linkage': linkage
                    }
                }

                # Get pose estimation settings - always enabled, prefer l_shape
                use_pose_estimation = True  # Always use pose estimation
                pose_estimation_method = st.session_state.get('pose_estimation_method', 'l_shape')
                
                # Get template dimensions (only used for PCA, L-shape returns its own dimensions)
                from clustering_manager import KITTI_CUBOID_TEMPLATES
                template_dims = KITTI_CUBOID_TEMPLATES if pose_estimation_method == 'pca' else None

                # Run per-frustum clustering with Agglomerative
                depth_map = st.session_state.get('depth_map')
                cuboids, per_frustum_results = fm.cluster_in_frustums(
                    points, frustums,
                    min_cluster_size=10,
                    min_samples=5,
                    algorithm='agglomerative',
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
                fig = create_3d_scatter_plot(
                    points=cluster_points,
                    labels=cluster_labels,
                    cuboids=st.session_state.cuboids,
                    title="Frustum-Based Agglomerative Results (Colored by Cluster)"
                )
            else:
                fig = create_3d_scatter_plot(
                    points=point_cloud,
                    cuboids=st.session_state.cuboids,
                    title="Frustum-Based Agglomerative Results"
                )
        else:
            fig = create_3d_scatter_plot(
                points=point_cloud,
                labels=result['labels'],
                cuboids=st.session_state.cuboids,
                title="Agglomerative Clustering Results"
            )
        st.plotly_chart(fig, width='stretch', key='agglomerative_clustering_chart')

        # Parameter summary
        st.subheader("Parameters Used")
        st.json(result['params'])

