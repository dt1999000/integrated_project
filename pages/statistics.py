"""
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
                min_samples = st.number_input("Min Samples", 2, 50, 5, key="stats_dbscan_ms")
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


