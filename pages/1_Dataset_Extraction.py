"""
Dataset Extraction Page
Loads and extracts samples from different dataset formats (KITTI, nuScenes, sim).
"""
import streamlit as st
import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Dict, List, Tuple

from components.dataset_loaders.utils import detect_dataset_type, load_dataset_sample
from components.dataset_loaders.dataset_loader import LinkedDataHandler
from components.utils.visualization_helper import create_3d_scatter_plot
from components.core.pointcloud_projection import PointCloud
from components.core.filter import (
    filter_kitti_images,
    filter_nuscenes_images,
    filter_sim_images,
)


def main():
    """Main extraction page function"""
    st.set_page_config(
        page_title="Dataset Extraction",
        page_icon="📂",
        layout="wide"
    )
    
    st.header("📂 Dataset Extraction")
    st.markdown("""
    Load samples from different dataset formats:
    - **KITTI**: Standard KITTI dataset structure
    - **nuScenes**: nuScenes dataset format
    - **sim**: Custom format (uses LinkedDataHandler)
    """)
    
    # Initialize session state for sample
    if 'sample' not in st.session_state:
        st.session_state.sample = None
    
    # Dataset path input
    st.subheader("Dataset Selection")
    dataset_path = st.text_input(
        "Dataset Path",
        value="",
        help="Enter the root directory path of your dataset"
    )

    # Output directory for saving processed samples (images + LiDAR)
    if "output_root_dir" not in st.session_state:
        st.session_state.output_root_dir = ""
    st.subheader("Output Directory")
    st.session_state.output_root_dir = st.text_input(
        "Output folder for saved samples",
        value=st.session_state.output_root_dir,
        help="Root folder where processed samples will be saved. "
             "Subfolders 'images' and 'lidar' will be created automatically."
    )
    
    # Auto-detect dataset type
    dataset_type = None
    if dataset_path:
        dataset_path_obj = Path(dataset_path)
        if dataset_path_obj.exists():
            # Persist dataset root so other pages (e.g. 4_Export) can reuse it
            st.session_state.dataset_path = dataset_path
            dataset_type = detect_dataset_type(dataset_path)
            if dataset_type:
                st.success(f"✅ Detected dataset type: **{dataset_type.upper()}**")
            else:
                st.warning("⚠️ Could not determine dataset type. Please check the folder structure.")
                st.info("""
                **Expected structures:**
                - **KITTI**: `training/` or `testing/` with `image_2/`, `velodyne/`, `calib/`
                - **nuScenes**: `samples/`, `sweeps/`, `v1.0-*/` folders
                - **sim**: `dataset.json` file in root directory
                """)
        else:
            st.error(f"❌ Path does not exist: {dataset_path}")
    
    # Sample selection based on dataset type
    if dataset_type:
        st.subheader("Sample Selection")
        
        if dataset_type == "kitti":
            # KITTI: Use numeric indices
            try:
                # Try to determine number of samples
                training_dir = Path(dataset_path) / "training"
                testing_dir = Path(dataset_path) / "testing"
                
                split_dir = training_dir if training_dir.exists() else testing_dir
                if split_dir.exists():
                    image_dir = split_dir / "image_2"
                    if image_dir.exists():
                        image_files = sorted([f for f in image_dir.iterdir() if f.suffix == '.png'])
                        num_samples = len(image_files)
                        
                        sample_index = st.number_input(
                            "Sample Index",
                            min_value=0,
                            max_value=max(0, num_samples - 1),
                            value=0,
                            help=f"Select sample index (0 to {num_samples - 1})"
                        )
                        
                        if st.button("🔄 Load Sample", key="load_kitti_sample"):
                            with st.spinner(f"Loading KITTI sample {sample_index}..."):
                                sample_meta_data, image, point_cloud = load_dataset_sample(
                                    dataset_path=dataset_path,
                                    sample_index=int(sample_index),
                                    dataset_type=dataset_type,
                                    filter_forward_only=True
                                )
                                
                                if sample_meta_data and image is not None and point_cloud is not None:
                                    st.session_state.sample = {
                                        'sample_meta_data': sample_meta_data,
                                        'image': image,
                                        'point_cloud': point_cloud
                                    }
                                    st.success(f"✅ Sample {sample_index} loaded successfully!")
                                    st.rerun()
                                else:
                                    st.error("❌ Failed to load sample")

                        # KITTI: random batch filtering + send to detection
                        if 'kitti_filter_params' not in st.session_state:
                            st.session_state.kitti_filter_params = {
                                'blur_gate': 120,
                                'hash_thresh': 6,
                                'motion_thresh': 5,
                                'min_contrast': 0.10,
                                'min_bright': 30,
                                'max_bright': 235,
                                'enable_blur': True,
                                'enable_dedup': True,
                                'enable_motion': False,
                                'enable_brightness': True,
                                'enable_contrast': True
                            }
                        if 'kitti_filtered_batch' not in st.session_state:
                            st.session_state.kitti_filtered_batch = None

                        st.markdown("---")
                        st.subheader("🖼️ Image Filtering (KITTI Dataset)")
                        st.markdown("""
                        Sample a random batch of KITTI images, filter them by quality, and send the filtered batch to the detection page.
                        """)

                        # Filter configuration
                        with st.expander("⚙️ Filter Settings (KITTI)", expanded=False):
                            col1, col2 = st.columns(2)

                            with col1:
                                st.session_state.kitti_filter_params['enable_blur'] = st.checkbox(
                                    "Enable Blur Filter",
                                    value=st.session_state.kitti_filter_params['enable_blur'],
                                    help="Remove blurry images using Laplacian variance",
                                    key="kitti_enable_blur"
                                )
                                if st.session_state.kitti_filter_params['enable_blur']:
                                    st.session_state.kitti_filter_params['blur_gate'] = st.slider(
                                        "Blur Gate (Laplacian Variance)",
                                        0, 500,
                                        st.session_state.kitti_filter_params['blur_gate'],
                                        help="Minimum Laplacian variance (higher = sharper)",
                                        key="kitti_blur_gate"
                                    )

                                st.session_state.kitti_filter_params['enable_dedup'] = st.checkbox(
                                    "Enable Deduplication",
                                    value=st.session_state.kitti_filter_params['enable_dedup'],
                                    help="Remove visually similar images",
                                    key="kitti_enable_dedup"
                                )
                                if st.session_state.kitti_filter_params['enable_dedup']:
                                    st.session_state.kitti_filter_params['hash_thresh'] = st.slider(
                                        "Deduplication Threshold (Hamming)",
                                        0, 16,
                                        st.session_state.kitti_filter_params['hash_thresh'],
                                        help="Maximum Hamming distance for duplicates",
                                        key="kitti_hash_thresh"
                                    )

                                st.session_state.kitti_filter_params['enable_motion'] = st.checkbox(
                                    "Enable Motion Filter",
                                    value=st.session_state.kitti_filter_params['enable_motion'],
                                    help="Skip static frames (sequential indices)",
                                    key="kitti_enable_motion"
                                )
                                if st.session_state.kitti_filter_params['enable_motion']:
                                    st.session_state.kitti_filter_params['motion_thresh'] = st.slider(
                                        "Motion Threshold",
                                        0, 20,
                                        st.session_state.kitti_filter_params['motion_thresh'],
                                        help="Minimum motion score between frames",
                                        key="kitti_motion_thresh"
                                    )

                            with col2:
                                st.session_state.kitti_filter_params['enable_brightness'] = st.checkbox(
                                    "Enable Brightness Filter",
                                    value=st.session_state.kitti_filter_params['enable_brightness'],
                                    help="Remove over/under-exposed images",
                                    key="kitti_enable_brightness"
                                )
                                if st.session_state.kitti_filter_params['enable_brightness']:
                                    st.session_state.kitti_filter_params['min_bright'] = st.slider(
                                        "Min Brightness",
                                        0, 255,
                                        st.session_state.kitti_filter_params['min_bright'],
                                        key="kitti_min_bright"
                                    )
                                    st.session_state.kitti_filter_params['max_bright'] = st.slider(
                                        "Max Brightness",
                                        0, 255,
                                        st.session_state.kitti_filter_params['max_bright'],
                                        key="kitti_max_bright"
                                    )

                                st.session_state.kitti_filter_params['enable_contrast'] = st.checkbox(
                                    "Enable Contrast Filter",
                                    value=st.session_state.kitti_filter_params['enable_contrast'],
                                    help="Remove low-contrast images",
                                    key="kitti_enable_contrast"
                                )
                                if st.session_state.kitti_filter_params['enable_contrast']:
                                    st.session_state.kitti_filter_params['min_contrast'] = st.slider(
                                        "Min Contrast",
                                        0.0, 0.5,
                                        st.session_state.kitti_filter_params['min_contrast'],
                                        step=0.01,
                                        key="kitti_min_contrast"
                                    )

                        # Random batch parameters
                        col_a, col_b = st.columns(2)
                        with col_a:
                            kitti_batch_size = st.number_input(
                                "Batch size (random KITTI samples)",
                                min_value=1,
                                max_value=num_samples,
                                value=min(64, num_samples),
                                step=1,
                                key="kitti_batch_size"
                            )
                        with col_b:
                            kitti_seed = st.number_input(
                                "Random seed",
                                min_value=0,
                                max_value=1_000_000,
                                value=42,
                                step=1,
                                key="kitti_batch_seed"
                            )

                        # Filter random batch button
                        if st.button("🔍 Filter Random Batch (KITTI)", type="primary", key="filter_kitti_batch"):
                            with st.spinner("Sampling and filtering KITTI images..."):
                                rng = np.random.default_rng(int(kitti_seed))
                                all_indices = np.arange(num_samples)
                                if kitti_batch_size < num_samples:
                                    sampled_indices = rng.choice(all_indices, size=int(kitti_batch_size), replace=False)
                                else:
                                    sampled_indices = all_indices

                                filtered_batch = filter_kitti_images(
                                    dataset_path=dataset_path,
                                    indices=sorted(int(i) for i in sampled_indices),
                                    filter_params=st.session_state.kitti_filter_params
                                )
                                st.session_state.kitti_filtered_batch = filtered_batch
                            st.success(f"✅ Filtered {len(st.session_state.kitti_filtered_batch)} samples from {num_samples} total images")
                            st.rerun()

                        # Display filtered batch summary and allow sending to detection
                        if st.session_state.kitti_filtered_batch:
                            filtered_batch = st.session_state.kitti_filtered_batch
                            st.markdown("---")
                            st.subheader("📋 Filtered Sample Batch (KITTI)")
                            st.info(f"Found {len(filtered_batch)} KITTI samples that passed all filters")

                            if st.button("📚 Load all filtered samples for detection", key="load_all_kitti_for_detection"):
                                batch_samples = []
                                for item in filtered_batch:
                                    batch_samples.append({
                                        "dataset_type": "kitti",
                                        "dataset_path": dataset_path,
                                        "sample_index": item["sample_index"],
                                        "image_path": item.get("image_path", ""),
                                    })
                                st.session_state.batch_samples = batch_samples
                                st.session_state.process_all_samples = True
                                st.success(f"✅ Prepared {len(batch_samples)} KITTI samples. Go to **2_Detection** and click **Process entire batch**.")
                                st.rerun()
            except Exception as e:
                st.error(f"Error: {str(e)}")
        
        elif dataset_type == "nuscenes":
            # nuScenes: Use sample tokens
            st.info("nuScenes dataset loading - Enter sample token or use batch filtering below.")
            sample_token = st.text_input(
                "Sample Token",
                value="",
                help="Enter the nuScenes sample token"
            )
            
            if st.button("🔄 Load Sample", key="load_nuscenes_sample"):
                if sample_token:
                    with st.spinner(f"Loading nuScenes sample {sample_token}..."):
                        sample_meta_data, image, point_cloud = load_dataset_sample(
                            dataset_path=dataset_path,
                            sample_index=sample_token,
                            dataset_type=dataset_type
                        )
                        
                        if sample_meta_data and image is not None and point_cloud is not None:
                            st.session_state.sample = {
                                'sample_meta_data': sample_meta_data,
                                'image': image,
                                'point_cloud': point_cloud
                            }
                            st.success(f"✅ Sample {sample_token} loaded successfully!")
                            st.rerun()
                        else:
                            st.error("❌ Failed to load sample")
                else:
                    st.warning("Please enter a sample token")

            # nuScenes: random batch filtering + send to detection
            from components.dataset_loaders.nuscenes_dataset_loader import NuScenesDatasetLoader  # local import to avoid heavy dependency if unused

            if 'nuscenes_filter_params' not in st.session_state:
                st.session_state.nuscenes_filter_params = {
                    'blur_gate': 120,
                    'hash_thresh': 6,
                    'motion_thresh': 5,
                    'min_contrast': 0.10,
                    'min_bright': 30,
                    'max_bright': 235,
                    'enable_blur': True,
                    'enable_dedup': True,
                    'enable_motion': False,
                    'enable_brightness': True,
                    'enable_contrast': True
                }
            if 'nuscenes_filtered_batch' not in st.session_state:
                st.session_state.nuscenes_filtered_batch = None

            st.markdown("---")
            st.subheader("🖼️ Image Filtering (nuScenes Dataset)")
            st.markdown("""
            Sample a random batch of nuScenes images (CAM_FRONT), filter them by quality, and send the filtered batch to the detection page.
            """)

            # Filter configuration
            with st.expander("⚙️ Filter Settings (nuScenes)", expanded=False):
                col1, col2 = st.columns(2)

                with col1:
                    st.session_state.nuscenes_filter_params['enable_blur'] = st.checkbox(
                        "Enable Blur Filter",
                        value=st.session_state.nuscenes_filter_params['enable_blur'],
                        help="Remove blurry images using Laplacian variance",
                        key="nuscenes_enable_blur"
                    )
                    if st.session_state.nuscenes_filter_params['enable_blur']:
                        st.session_state.nuscenes_filter_params['blur_gate'] = st.slider(
                            "Blur Gate (Laplacian Variance)",
                            0, 500,
                            st.session_state.nuscenes_filter_params['blur_gate'],
                            help="Minimum Laplacian variance (higher = sharper)",
                            key="nuscenes_blur_gate"
                        )

                    st.session_state.nuscenes_filter_params['enable_dedup'] = st.checkbox(
                        "Enable Deduplication",
                        value=st.session_state.nuscenes_filter_params['enable_dedup'],
                        help="Remove visually similar images",
                        key="nuscenes_enable_dedup"
                    )
                    if st.session_state.nuscenes_filter_params['enable_dedup']:
                        st.session_state.nuscenes_filter_params['hash_thresh'] = st.slider(
                            "Deduplication Threshold (Hamming)",
                            0, 16,
                            st.session_state.nuscenes_filter_params['hash_thresh'],
                            help="Maximum Hamming distance for duplicates",
                            key="nuscenes_hash_thresh"
                        )

                    st.session_state.nuscenes_filter_params['enable_motion'] = st.checkbox(
                        "Enable Motion Filter",
                        value=st.session_state.nuscenes_filter_params['enable_motion'],
                        help="Skip static frames (sequential samples)",
                        key="nuscenes_enable_motion"
                    )
                    if st.session_state.nuscenes_filter_params['enable_motion']:
                        st.session_state.nuscenes_filter_params['motion_thresh'] = st.slider(
                            "Motion Threshold",
                            0, 20,
                            st.session_state.nuscenes_filter_params['motion_thresh'],
                            help="Minimum motion score between frames",
                            key="nuscenes_motion_thresh"
                        )

                with col2:
                    st.session_state.nuscenes_filter_params['enable_brightness'] = st.checkbox(
                        "Enable Brightness Filter",
                        value=st.session_state.nuscenes_filter_params['enable_brightness'],
                        help="Remove over/under-exposed images",
                        key="nuscenes_enable_brightness"
                    )
                    if st.session_state.nuscenes_filter_params['enable_brightness']:
                        st.session_state.nuscenes_filter_params['min_bright'] = st.slider(
                            "Min Brightness",
                            0, 255,
                            st.session_state.nuscenes_filter_params['min_bright'],
                            key="nuscenes_min_bright"
                        )
                        st.session_state.nuscenes_filter_params['max_bright'] = st.slider(
                            "Max Brightness",
                            0, 255,
                            st.session_state.nuscenes_filter_params['max_bright'],
                            key="nuscenes_max_bright"
                        )

                    st.session_state.nuscenes_filter_params['enable_contrast'] = st.checkbox(
                        "Enable Contrast Filter",
                        value=st.session_state.nuscenes_filter_params['enable_contrast'],
                        help="Remove low-contrast images",
                        key="nuscenes_enable_contrast"
                    )
                    if st.session_state.nuscenes_filter_params['enable_contrast']:
                        st.session_state.nuscenes_filter_params['min_contrast'] = st.slider(
                            "Min Contrast",
                            0.0, 0.5,
                            st.session_state.nuscenes_filter_params['min_contrast'],
                            step=0.01,
                            key="nuscenes_min_contrast"
                        )

            # Random batch parameters
            col_n1, col_n2 = st.columns(2)
            with col_n1:
                nuscenes_batch_size = st.number_input(
                    "Batch size (random nuScenes samples)",
                    min_value=1,
                    value=64,
                    step=1,
                    key="nuscenes_batch_size"
                )
            with col_n2:
                nuscenes_seed = st.number_input(
                    "Random seed (nuScenes)",
                    min_value=0,
                    max_value=1_000_000,
                    value=42,
                    step=1,
                    key="nuscenes_batch_seed"
                )

            # Filter random batch button
            if st.button("🔍 Filter Random Batch (nuScenes)", type="primary", key="filter_nuscenes_batch"):
                try:
                    # Determine nuScenes version from directory structure
                    version = None
                    for d in Path(dataset_path).iterdir():
                        if d.is_dir() and d.name.startswith("v1.0-"):
                            version = d.name
                            break
                    if version is None:
                        st.error("❌ Could not determine nuScenes version (expected a 'v1.0-*' directory).")
                    else:
                        with st.spinner("Sampling and filtering nuScenes images..."):
                            loader = NuScenesDatasetLoader(dataroot=str(dataset_path), version=version, verbose=False)
                            loader.load_dataset()
                            camera_samples = loader.get_camera_samples(camera_channel="CAM_FRONT")
                            total_samples = len(camera_samples)
                            if total_samples == 0:
                                st.error("❌ No nuScenes camera samples found (CAM_FRONT).")
                            else:
                                bs = min(int(nuscenes_batch_size), total_samples)
                                rng = np.random.default_rng(int(nuscenes_seed))
                                indices = rng.choice(np.arange(total_samples), size=bs, replace=False)
                                selected = [camera_samples[int(i)] for i in indices]

                                filtered_batch = filter_nuscenes_images(
                                    samples=selected,
                                    filter_params=st.session_state.nuscenes_filter_params
                                )
                                st.session_state.nuscenes_filtered_batch = filtered_batch
                                st.success(f"✅ Filtered {len(filtered_batch)} samples from {total_samples} total nuScenes samples")
                                st.rerun()
                except Exception as e:
                    st.error(f"Error during nuScenes batch filtering: {str(e)}")

            # Display filtered batch summary and allow sending to detection
            if st.session_state.nuscenes_filtered_batch:
                filtered_batch = st.session_state.nuscenes_filtered_batch
                st.markdown("---")
                st.subheader("📋 Filtered Sample Batch (nuScenes)")
                st.info(f"Found {len(filtered_batch)} nuScenes samples that passed all filters")

                if st.button("📚 Load all filtered samples for detection", key="load_all_nuscenes_for_detection"):
                    batch_samples = []
                    for item in filtered_batch:
                        batch_samples.append({
                            "dataset_type": "nuscenes",
                            "dataset_path": dataset_path,
                            "sample_index": item["sample_token"],
                            "image_path": item.get("image_path", ""),
                        })
                    st.session_state.batch_samples = batch_samples
                    st.session_state.process_all_samples = True
                    st.success(f"✅ Prepared {len(batch_samples)} nuScenes samples. Go to **2_Detection** and click **Process entire batch**.")
                    st.rerun()
        
        elif dataset_type == "sim":
            # sim/LinkedDataHandler: Filter images and create batch selection
            try:
                handler = LinkedDataHandler(root_dir=dataset_path, load_dataset=True)
                subsets = handler.list_subsets()
                
                if subsets:
                    selected_subset = st.selectbox(
                        "Select Subset",
                        options=subsets,
                        help="Choose a subset from the dataset",
                        key="sim_subset_selector"
                    )
                    
                    if selected_subset:
                        subset = handler.subsets[selected_subset]
                        links = subset['links']
                        
                        # Initialize session state for filtered batch
                        if 'sim_filtered_batch' not in st.session_state:
                            st.session_state.sim_filtered_batch = None
                        if 'sim_filter_params' not in st.session_state:
                            st.session_state.sim_filter_params = {
                                'blur_gate': 120,
                                'hash_thresh': 6,
                                'motion_thresh': 5,
                                'min_contrast': 0.10,
                                'min_bright': 30,
                                'max_bright': 235,
                                'enable_blur': True,
                                'enable_dedup': True,
                                'enable_motion': False,
                                'enable_brightness': True,
                                'enable_contrast': True
                            }
                        
                        st.markdown("---")
                        st.subheader("🖼️ Image Filtering (Sim Dataset)")
                        st.markdown("""
                        Filter images from the dataset using quality metrics.
                        Only images that pass all enabled filters will be available for selection.
                        """)
                        
                        # Filter configuration
                        with st.expander("⚙️ Filter Settings", expanded=True):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.session_state.sim_filter_params['enable_blur'] = st.checkbox(
                                    "Enable Blur Filter", 
                                    value=st.session_state.sim_filter_params['enable_blur'],
                                    help="Remove blurry images using Laplacian variance"
                                )
                                if st.session_state.sim_filter_params['enable_blur']:
                                    st.session_state.sim_filter_params['blur_gate'] = st.slider(
                                        "Blur Gate (Laplacian Variance)", 
                                        0, 500, 
                                        st.session_state.sim_filter_params['blur_gate'],
                                        help="Minimum Laplacian variance (higher = sharper)"
                                    )
                                
                                st.session_state.sim_filter_params['enable_dedup'] = st.checkbox(
                                    "Enable Deduplication", 
                                    value=st.session_state.sim_filter_params['enable_dedup'],
                                    help="Remove visually similar images"
                                )
                                if st.session_state.sim_filter_params['enable_dedup']:
                                    st.session_state.sim_filter_params['hash_thresh'] = st.slider(
                                        "Deduplication Threshold (Hamming)", 
                                        0, 16, 
                                        st.session_state.sim_filter_params['hash_thresh'],
                                        help="Maximum Hamming distance for duplicates"
                                    )
                                
                                st.session_state.sim_filter_params['enable_motion'] = st.checkbox(
                                    "Enable Motion Filter", 
                                    value=st.session_state.sim_filter_params['enable_motion'],
                                    help="Skip static frames (requires sequential processing)"
                                )
                                if st.session_state.sim_filter_params['enable_motion']:
                                    st.session_state.sim_filter_params['motion_thresh'] = st.slider(
                                        "Motion Threshold", 
                                        0, 20, 
                                        st.session_state.sim_filter_params['motion_thresh'],
                                        help="Minimum motion score between frames"
                                    )
                            
                            with col2:
                                st.session_state.sim_filter_params['enable_brightness'] = st.checkbox(
                                    "Enable Brightness Filter", 
                                    value=st.session_state.sim_filter_params['enable_brightness'],
                                    help="Remove over/under-exposed images"
                                )
                                if st.session_state.sim_filter_params['enable_brightness']:
                                    st.session_state.sim_filter_params['min_bright'] = st.slider(
                                        "Min Brightness", 
                                        0, 255, 
                                        st.session_state.sim_filter_params['min_bright']
                                    )
                                    st.session_state.sim_filter_params['max_bright'] = st.slider(
                                        "Max Brightness", 
                                        0, 255, 
                                        st.session_state.sim_filter_params['max_bright']
                                    )
                                
                                st.session_state.sim_filter_params['enable_contrast'] = st.checkbox(
                                    "Enable Contrast Filter", 
                                    value=st.session_state.sim_filter_params['enable_contrast'],
                                    help="Remove low-contrast images"
                                )
                                if st.session_state.sim_filter_params['enable_contrast']:
                                    st.session_state.sim_filter_params['min_contrast'] = st.slider(
                                        "Min Contrast", 
                                        0.0, 0.5, 
                                        st.session_state.sim_filter_params['min_contrast'],
                                        step=0.01
                                    )
                        
                        # Filter images button
                        if st.button("🔍 Filter Images", type="primary", key="filter_sim_images"):
                            with st.spinner("Filtering images from dataset..."):
                                print(f"dataset_path: {dataset_path}")
                                filtered_batch = filter_sim_images(
                                    handler=handler,
                                    subset_name=selected_subset,
                                    links=links,
                                    dataset_path=dataset_path,
                                    filter_params=st.session_state.sim_filter_params,
                                )
                                st.session_state.sim_filtered_batch = filtered_batch
                                st.success(f"✅ Filtered {len(filtered_batch)} samples from {len(links)} total links")
                                st.rerun()
                        
                        # Display filtered batch and allow selection
                        if st.session_state.sim_filtered_batch is not None:
                            filtered_batch = st.session_state.sim_filtered_batch
                            
                            st.markdown("---")
                            st.subheader("📋 Filtered Sample Batch")
                            st.info(f"Found {len(filtered_batch)} samples that passed all filters")
                            
                            if len(filtered_batch) > 0:
                                # Display batch as grid with thumbnails
                                st.markdown("**Select a sample from the filtered batch:**")
                                
                                # Create selection interface
                                num_cols = 3
                                cols = st.columns(num_cols)
                                
                                selected_sample_idx = None
                                
                                for idx, sample_info in enumerate(filtered_batch):
                                    col_idx = idx % num_cols
                                    with cols[col_idx]:
                                        # Display thumbnail
                                        image = sample_info['image']
                                        link_token = sample_info['link_token']
                                        metrics = sample_info['metrics']
                                        
                                        # Resize for thumbnail
                                        h, w = image.shape[:2]
                                        max_size = 200
                                        if w > max_size or h > max_size:
                                            scale = max_size / max(w, h)
                                            new_w, new_h = int(w * scale), int(h * scale)
                                            thumbnail = cv2.resize(image, (new_w, new_h))
                                        else:
                                            thumbnail = image
                                        
                                        st.image(thumbnail, use_container_width=True)
                                        
                                        # Display metrics
                                        st.caption(f"**Token:** {link_token[:12]}...")
                                        st.caption(f"Blur: {metrics.get('blur', 0):.1f}")
                                        st.caption(f"Contrast: {metrics.get('contrast', 0):.3f}")
                                        st.caption(f"Brightness: {metrics.get('brightness', 0):.1f}")
                                        
                                        # Selection button
                                        if st.button(f"Select", key=f"select_sample_{idx}"):
                                            selected_sample_idx = idx
                                
                                # Handle selection
                                if selected_sample_idx is not None:
                                    selected_sample = filtered_batch[selected_sample_idx]
                                    
                                    # Load the full sample data
                                    with st.spinner(f"Loading sample {selected_sample['link_token']}..."):
                                        sample_meta_data, image, point_cloud = load_dataset_sample(
                                            dataset_path=dataset_path,
                                            sample_index=selected_sample['link_token'],
                                            dataset_type=dataset_type
                                        )
                                        
                                        if sample_meta_data and image is not None and point_cloud is not None:
                                            st.session_state.sample = {
                                                'sample_meta_data': sample_meta_data,
                                                'image': image,
                                                'point_cloud': point_cloud
                                            }
                                            st.success(f"✅ Sample {selected_sample['link_token']} loaded successfully!")
                                            st.rerun()
                                        else:
                                            st.error("❌ Failed to load selected sample")
                                # Load entire batch for detection (process all on 2_Detection)
                                if st.button("📚 Load all filtered samples for detection", key="load_all_sim_for_detection"):
                                    batch_samples = []
                                    for item in filtered_batch:
                                        batch_samples.append({
                                            "dataset_type": "sim",
                                            "dataset_path": dataset_path,
                                            "sample_index": item["link_token"],
                                            "image_path": item.get("image_path", ""),
                                        })
                                    st.session_state.batch_samples = batch_samples
                                    st.session_state.process_all_samples = True
                                    st.success(f"✅ Prepared {len(batch_samples)} samples. Go to **2_Detection** and click **Process entire batch**.")
                                    st.rerun()
                            else:
                                st.warning("⚠️ No samples passed the filters. Try adjusting filter settings.")
                        else:
                            st.info("👆 Click 'Filter Images' to process the dataset and create a filtered batch")
                else:
                    st.warning("No subsets found in dataset")
            except Exception as e:
                st.error(f"Error loading LinkedDataHandler: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
    # Display loaded sample
    if st.session_state.sample is not None:
        st.subheader("📊 Loaded Sample Preview")
        
        sample = st.session_state.sample
        sample_meta_data = sample['sample_meta_data']
        image = sample['image']
        point_cloud = sample['point_cloud']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Image Preview**")
            st.image(image, use_container_width=True)
            st.caption(f"Image shape: {image.shape}")
        
        with col2:
            st.markdown("**Point Cloud Info**")
            st.metric("Number of Points", f"{len(point_cloud):,}")
            if len(point_cloud) > 0:
                st.caption(f"X range: [{point_cloud[:, 0].min():.2f}, {point_cloud[:, 0].max():.2f}]")
                st.caption(f"Y range: [{point_cloud[:, 1].min():.2f}, {point_cloud[:, 1].max():.2f}]")
                st.caption(f"Z range: [{point_cloud[:, 2].min():.2f}, {point_cloud[:, 2].max():.2f}]")
        
        # Point cloud visualization with ground removal
        st.markdown("---")
        st.subheader("📊 Point Cloud Visualization")
        
        if len(point_cloud) > 0:
            try:
                # Apply ground removal
                with st.spinner("Removing ground plane..."):
                    point_cloud_obj = PointCloud(point_cloud)
                    
                    # Get ground-removed points
                    point_cloud_ground_removed = point_cloud_obj.original_point_cloud
                    
                    # Display statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Original Points", f"{len(point_cloud):,}")
                    with col2:
                        st.metric("After Ground Removal", f"{len(point_cloud_ground_removed):,}")
                    with col3:
                        reduction_pct = (1 - len(point_cloud_ground_removed) / len(point_cloud)) * 100
                        st.metric("Reduction", f"{reduction_pct:.1f}%")
                    
                    if len(point_cloud_ground_removed) > 0:
                        # Visualize ground-removed point cloud
                        fig = create_3d_scatter_plot(
                            points=point_cloud_ground_removed,
                            labels=None,
                            mask_points=None,
                            cuboids=None,
                            rays=None,
                            points_in_frustums=None,
                            reconstructed_points=None,
                            show_lidar=True,
                            show_reconstructed=False,
                            color_by_depth=False,
                            title="Point Cloud (After Ground Removal)"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        st.success(f"✅ Point cloud visualized successfully! {len(point_cloud_ground_removed):,} points remaining after ground removal.")
                    else:
                        st.warning("⚠️ Point cloud is empty after ground removal. Try adjusting the parameters.")
                        
            except Exception as e:
                st.error(f"❌ Error processing point cloud: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
        else:
            st.warning("⚠️ Point cloud is empty")
        
        # Sample metadata
        with st.expander("Sample Metadata", expanded=False):
            st.json(sample_meta_data)
        
        st.success("✅ Sample is ready! Navigate to **2_Detection.py** to run the detection pipeline.")


def _filter_kitti_images(*args, **kwargs):
    """Backward-compat wrapper (deprecated). Use components.core.filter.filter_kitti_images instead."""
    return filter_kitti_images(*args, **kwargs)


def _filter_nuscenes_images(*args, **kwargs):
    """Backward-compat wrapper (deprecated). Use components.core.filter.filter_nuscenes_images instead."""
    return filter_nuscenes_images(*args, **kwargs)


def _filter_sim_images(*args, **kwargs):
    """Backward-compat wrapper (deprecated). Use components.core.filter.filter_sim_images instead."""
    return filter_sim_images(*args, **kwargs)


if __name__ == "__main__":
    main()

