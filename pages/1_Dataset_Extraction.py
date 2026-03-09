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
from components.dataset_loaders.nuscenes_dataset_loader import NuScenesDatasetLoader
from components.dataset_loaders.rosbag_extractor import (
    get_image_topics as get_ros_image_topics,
    get_pointcloud_topics as get_ros_pointcloud_topics,
    get_camera_info_topics as get_ros_camera_info_topics,
    get_tf_topics as get_ros_tf_topics,
    suggest_topics_from_metadata,
    filter_rosbag_frames,
    extract_bag_to_folder,
    sanitize_bag_name,
)
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
    # Batch mode is False by default; only "Load all ... for detection" sets it to True.
    if 'process_all_samples' not in st.session_state:
        st.session_state.process_all_samples = False

    # Dataset path input
    st.subheader("Dataset Selection")
    dataset_path = st.text_input(
        "Dataset Path",
        value="",
        help="Enter the root directory path of your dataset"
    )
    if dataset_path:
        dataset_path = str(Path(dataset_path).expanduser().resolve(strict=False))

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
                                    # Ensure single-sample mode on Detection page
                                    st.session_state.process_all_samples = False
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

                        # Quick presets for indoor / outdoor scenes
                        col_kitti_preset1, col_kitti_preset2 = st.columns(2)
                        with col_kitti_preset1:
                            if st.button("🌳 Outdoor Scenes", key="kitti_outdoor_preset"):
                                st.session_state.kitti_filter_params.update(
                                    {
                                        "motion_thresh": 8,
                                        "blur_gate": 140,
                                        "hash_thresh": 8,
                                        "min_bright": 30,
                                        "max_bright": 245,
                                        "min_contrast": 0.12,
                                        "enable_blur": True,
                                        "enable_dedup": True,
                                        "enable_motion": True,
                                        "enable_brightness": True,
                                        "enable_contrast": True,
                                    }
                                )
                                st.rerun()
                        with col_kitti_preset2:
                            if st.button("🏠 Indoor Scenes", key="kitti_indoor_preset"):
                                st.session_state.kitti_filter_params.update(
                                    {
                                        "motion_thresh": 4,
                                        "blur_gate": 110,
                                        "hash_thresh": 6,
                                        "min_bright": 40,
                                        "max_bright": 235,
                                        "min_contrast": 0.10,
                                        "enable_blur": True,
                                        "enable_dedup": True,
                                        "enable_motion": True,
                                        "enable_brightness": True,
                                        "enable_contrast": True,
                                    }
                                )
                                st.rerun()

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
                                # Standardized batch sample descriptor:
                                # {
                                #   "dataset_type": str,
                                #   "dataset_path": str,
                                #   "sample_index": Union[int, str],
                                #   "image_path": str,
                                #   "point_cloud_path": str,
                                # }
                                batch_samples = []
                                for item in filtered_batch:
                                    batch_samples.append({
                                        "dataset_type": "kitti",
                                        "dataset_path": dataset_path,
                                        "sample_index": item["sample_index"],
                                        "image_path": item.get("image_path", ""),
                                        # For KITTI we resolve LiDAR internally in loaders; keep path optional here.
                                        "point_cloud_path": "",
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
                            # Ensure single-sample mode on Detection page
                            st.session_state.process_all_samples = False
                            st.success(f"✅ Sample {sample_token} loaded successfully!")
                            st.rerun()
                        else:
                            st.error("❌ Failed to load sample")
                else:
                    st.warning("Please enter a sample token")

            # nuScenes: random batch filtering + send to detection

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

            # Quick presets for indoor / outdoor scenes
            col_nusc_preset1, col_nusc_preset2 = st.columns(2)
            with col_nusc_preset1:
                if st.button("🌳 Outdoor Scenes", key="nuscenes_outdoor_preset"):
                    st.session_state.nuscenes_filter_params.update(
                        {
                            "motion_thresh": 8,
                            "blur_gate": 140,
                            "hash_thresh": 8,
                            "min_bright": 30,
                            "max_bright": 245,
                            "min_contrast": 0.12,
                            "enable_blur": True,
                            "enable_dedup": True,
                            "enable_motion": True,
                            "enable_brightness": True,
                            "enable_contrast": True,
                        }
                    )
                    st.rerun()
            with col_nusc_preset2:
                if st.button("🏠 Indoor Scenes", key="nuscenes_indoor_preset"):
                    st.session_state.nuscenes_filter_params.update(
                        {
                            "motion_thresh": 4,
                            "blur_gate": 110,
                            "hash_thresh": 6,
                            "min_bright": 40,
                            "max_bright": 235,
                            "min_contrast": 0.10,
                            "enable_blur": True,
                            "enable_dedup": True,
                            "enable_motion": True,
                            "enable_brightness": True,
                            "enable_contrast": True,
                        }
                    )
                    st.rerun()

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
                            # nuScenes LiDAR path is resolved by its dataset loader.
                            "point_cloud_path": "",
                        })
                    st.session_state.batch_samples = batch_samples
                    st.session_state.process_all_samples = True
                    st.success(f"✅ Prepared {len(batch_samples)} nuScenes samples. Go to **2_Detection** and click **Process entire batch**.")
                    st.rerun()
        
        elif dataset_type == "rosbag":
            # ROS bag: configure topics, filter frames, then extract and load batch
            st.info("ROS bag detected. Configure topics and filters, then process the bag into samples.")

            bag_path_obj = Path(dataset_path)

            # Topic discovery (best-effort)
            suggestions = suggest_topics_from_metadata(bag_path_obj)

            image_topics: List[str] = []
            pointcloud_topics: List[str] = []
            camera_info_topics: List[str] = []
            tf_topics: List[str] = []

            try:
                image_topics = get_ros_image_topics(bag_path_obj)
                pointcloud_topics = get_ros_pointcloud_topics(bag_path_obj)
                camera_info_topics = get_ros_camera_info_topics(bag_path_obj)
                tf_topics = get_ros_tf_topics(bag_path_obj)
            except Exception as e:
                st.warning(f"Could not inspect ROS bag topics automatically: {e}")

            st.subheader("ROS Bag Topics")
            col_t1, col_t2 = st.columns(2)

            with col_t1:
                # Image topic
                if image_topics:
                    default_image = suggestions.get("image_topic")
                    idx = 0
                    if default_image and default_image in image_topics:
                        idx = image_topics.index(default_image)
                    image_topic = st.selectbox(
                        "Image topic",
                        image_topics,
                        index=idx,
                        help="Topic containing camera images",
                        key="rosbag_image_topic",
                    )
                else:
                    image_topic = st.text_input(
                        "Image topic",
                        value=suggestions.get("image_topic") or "",
                        help="Enter the image topic manually (e.g. /sync/flir/compressed)",
                        key="rosbag_image_topic_text",
                    )

                # CameraInfo topic
                if camera_info_topics:
                    default_ci = suggestions.get("camera_info_topic")
                    idx_ci = 0
                    if default_ci and default_ci in camera_info_topics:
                        idx_ci = camera_info_topics.index(default_ci)
                    camera_info_topic = st.selectbox(
                        "CameraInfo topic (for intrinsics)",
                        camera_info_topics,
                        index=idx_ci,
                        help="Topic containing sensor_msgs/CameraInfo",
                        key="rosbag_camera_info_topic",
                    )
                else:
                    camera_info_topic = st.text_input(
                        "CameraInfo topic (optional)",
                        value=suggestions.get("camera_info_topic") or "",
                        help="Leave empty to skip intrinsics from CameraInfo",
                        key="rosbag_camera_info_topic_text",
                    ) or None

            with col_t2:
                # PointCloud2 topic
                if pointcloud_topics:
                    default_pc = suggestions.get("pointcloud_topic")
                    idx_pc = 0
                    if default_pc and default_pc in pointcloud_topics:
                        idx_pc = pointcloud_topics.index(default_pc)
                    pc_choice = st.selectbox(
                        "PointCloud2 topic (optional)",
                        ["(none)"] + pointcloud_topics,
                        index=(idx_pc + 1) if default_pc and default_pc in pointcloud_topics else 0,
                        help="LiDAR PointCloud2 topic (set to '(none)' for image-only)",
                        key="rosbag_pc_topic",
                    )
                    pointcloud_topic = None if pc_choice == "(none)" else pc_choice
                else:
                    pointcloud_topic = st.text_input(
                        "PointCloud2 topic (optional)",
                        value=suggestions.get("pointcloud_topic") or "",
                        help="Leave empty if no LiDAR is present in the bag",
                        key="rosbag_pc_topic_text",
                    ) or None

                # TF topic
                if tf_topics:
                    default_tf = suggestions.get("tf_topic") or "/tf_static"
                    idx_tf = tf_topics.index(default_tf) if default_tf in tf_topics else 0
                    tf_topic = st.selectbox(
                        "TF topic for extrinsics",
                        tf_topics,
                        index=idx_tf,
                        help="Topic containing TFMessage with static transforms",
                        key="rosbag_tf_topic",
                    )
                else:
                    tf_topic = st.text_input(
                        "TF topic (optional)",
                        value=suggestions.get("tf_topic") or "/tf_static",
                        help="Usually /tf_static or /tf",
                        key="rosbag_tf_topic_text",
                    ) or None

            # Initialize ROS bag filter params (reuse KITTI defaults)
            if "rosbag_filter_params" not in st.session_state:
                st.session_state.rosbag_filter_params = {
                    "blur_gate": 120,
                    "hash_thresh": 6,
                    "motion_thresh": 5,
                    "min_contrast": 0.10,
                    "min_bright": 30,
                    "max_bright": 235,
                    "enable_blur": True,
                    "enable_dedup": True,
                    "enable_motion": False,
                    "enable_brightness": True,
                    "enable_contrast": True,
                }
            if "rosbag_filtered_frames" not in st.session_state:
                st.session_state.rosbag_filtered_frames = None

            st.markdown("---")
            st.subheader("🖼️ Frame Filtering (ROS bag)")
            st.markdown(
                "Apply the same quality filters as for KITTI/nuScenes, but directly on the ROS bag frames."
            )

            with st.expander("⚙️ Filter Settings (ROS bag)", expanded=False):
                col_f1, col_f2 = st.columns(2)

                with col_f1:
                    rp = st.session_state.rosbag_filter_params
                    rp["enable_blur"] = st.checkbox(
                        "Enable Blur Filter",
                        value=rp["enable_blur"],
                        help="Remove blurry images using Laplacian variance",
                        key="rosbag_enable_blur",
                    )
                    if rp["enable_blur"]:
                        rp["blur_gate"] = st.slider(
                            "Blur Gate (Laplacian Variance)",
                            0,
                            500,
                            rp["blur_gate"],
                            help="Minimum Laplacian variance (higher = sharper)",
                            key="rosbag_blur_gate",
                        )

                    rp["enable_dedup"] = st.checkbox(
                        "Enable Deduplication",
                        value=rp["enable_dedup"],
                        help="Remove visually similar images",
                        key="rosbag_enable_dedup",
                    )
                    if rp["enable_dedup"]:
                        rp["hash_thresh"] = st.slider(
                            "Deduplication Threshold (Hamming)",
                            0,
                            16,
                            rp["hash_thresh"],
                            help="Maximum Hamming distance for duplicates",
                            key="rosbag_hash_thresh",
                        )

                    rp["enable_motion"] = st.checkbox(
                        "Enable Motion Filter",
                        value=rp["enable_motion"],
                        help="Skip static frames (sequential messages)",
                        key="rosbag_enable_motion",
                    )
                    if rp["enable_motion"]:
                        rp["motion_thresh"] = st.slider(
                            "Motion Threshold",
                            0,
                            20,
                            rp["motion_thresh"],
                            help="Minimum motion score between frames",
                            key="rosbag_motion_thresh",
                        )

                with col_f2:
                    rp = st.session_state.rosbag_filter_params
                    rp["enable_brightness"] = st.checkbox(
                        "Enable Brightness Filter",
                        value=rp["enable_brightness"],
                        help="Remove over/under-exposed images",
                        key="rosbag_enable_brightness",
                    )
                    if rp["enable_brightness"]:
                        rp["min_bright"] = st.slider(
                            "Min Brightness",
                            0,
                            255,
                            rp["min_bright"],
                            key="rosbag_min_bright",
                        )
                        rp["max_bright"] = st.slider(
                            "Max Brightness",
                            0,
                            255,
                            rp["max_bright"],
                            key="rosbag_max_bright",
                        )

                    rp["enable_contrast"] = st.checkbox(
                        "Enable Contrast Filter",
                        value=rp["enable_contrast"],
                        help="Remove low-contrast images",
                        key="rosbag_enable_contrast",
                    )
                    if rp["enable_contrast"]:
                        rp["min_contrast"] = st.slider(
                            "Min Contrast",
                            0.0,
                            0.5,
                            rp["min_contrast"],
                            step=0.01,
                            key="rosbag_min_contrast",
                        )

                    rosbag_max_frames = st.number_input(
                        "Max frames to consider from bag (0 = all)",
                        min_value=0,
                        value=0,
                        step=1,
                        key="rosbag_max_frames",
                    )

            # Button to process bag (filter frames)
            if st.button("🎬 Process bag (filter frames)", type="primary", key="rosbag_process_bag"):
                if not image_topic:
                    st.error("Please specify an image topic for the bag.")
                else:
                    with st.spinner("Filtering ROS bag frames..."):
                        try:
                            max_frames_val = (
                                int(st.session_state.rosbag_max_frames)
                                if st.session_state.rosbag_max_frames > 0
                                else None
                            )
                        except Exception:
                            max_frames_val = None

                        try:
                            accepted = filter_rosbag_frames(
                                bag_path=bag_path_obj,
                                image_topic=image_topic,
                                filter_params=st.session_state.rosbag_filter_params,
                                max_frames=max_frames_val,
                                progress_callback=None,
                            )
                            st.session_state.rosbag_filtered_frames = accepted
                            st.success(f"✅ {len(accepted)} frames passed all filters.")
                        except Exception as e:
                            st.error(f"ROS bag filtering failed: {e}")

            filtered_frames = st.session_state.get("rosbag_filtered_frames") or []
            if filtered_frames:
                st.markdown("---")
                st.subheader("📋 Filtered Frame Summary (ROS bag)")
                st.info(
                    f"Found {len(filtered_frames)} frames that passed all filters. "
                    "You can now extract them to disk and load them as a batch for processing."
                )

                if st.button(
                    "💾 Save data batch and load batch for processing",
                    type="primary",
                    key="rosbag_save_and_load_batch",
                ):
                    with st.spinner("Extracting filtered frames from ROS bag..."):
                        bag_name = sanitize_bag_name(bag_path_obj.name)
                        out_dir = st.session_state.output_root_dir + "/" + bag_name + "_extracted"

                        timestamps = [f["timestamp_ns"] for f in filtered_frames]
                        try:
                            frames, stats = extract_bag_to_folder(
                                bag_path=bag_path_obj,
                                out_dir=out_dir,
                                image_topic=image_topic,
                                pointcloud_topic=pointcloud_topic,
                                accepted_timestamps_ns=timestamps,
                                max_frames=None,
                                progress_callback=None,
                                camera_info_topic=camera_info_topic,
                                tf_topics=[tf_topic] if tf_topic else None,
                            )
                        except Exception as e:
                            st.error(f"Failed to extract filtered frames: {e}")
                            frames = []
                    if frames:
                        batch_samples = []
                        for f in frames:
                            batch_samples.append(
                                {
                                    "dataset_type": "rosbag",
                                    "dataset_path": str(out_dir),
                                    "sample_index": f["frame_index"],
                                    "image_path": f.get("image_path", ""),
                                    # For ROS bags we often know the exact LiDAR file path.
                                    "point_cloud_path": f.get("point_cloud_path", ""),
                                }
                            )
                        st.session_state.batch_samples = batch_samples
                        st.session_state.process_all_samples = True
                        # Indicate that raw samples have already been saved for batch
                        st.session_state.batch_samples_saved = True
                        st.success(
                            f"✅ Prepared {len(batch_samples)} ROS bag samples. "
                            "Go to **2_Detection** and click **Process entire batch**."
                        )
                        st.info(f"Extracted data directory: `{out_dir}`")
                        st.rerun()
            else:
                st.info("Click **Process bag (filter frames)** to create a filtered frame list.")

            # After a rosbag extraction: show saved images and let user pick one sample to load (like KITTI/nuScenes)
            if st.session_state.get("batch_samples_saved") and st.session_state.get("batch_samples"):
                _batch = st.session_state.batch_samples
                if _batch and all(s.get("dataset_type") == "rosbag" for s in _batch):
                    st.markdown("---")
                    st.subheader("📷 Extracted samples")
                    st.info(
                        f"**{len(_batch)}** samples were saved. "
                        "Choose one below to load for the **2_Detection** tab and check image–LiDAR fusion."
                    )
                    # Thumbnail grid (first 24 in 4 columns)
                    n_show = min(24, len(_batch))
                    cols = st.columns(4)
                    for i in range(n_show):
                        with cols[i % 4]:
                            img_path = _batch[i].get("image_path") or ""
                            if img_path and Path(img_path).exists():
                                img = cv2.imread(img_path)
                                if img is not None:
                                    st.image(
                                        cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                                        caption=f"Frame {_batch[i]['sample_index']}",
                                    )
                    if len(_batch) > n_show:
                        st.caption(f"Showing first {n_show} of {len(_batch)}. Use the selector below to load any sample.")
                    # Persist selected index so it survives reruns when Load button is clicked
                    if "rosbag_selected_sample_idx" not in st.session_state:
                        st.session_state.rosbag_selected_sample_idx = 0
                    selected_idx = st.selectbox(
                        "Choose sample to load for Detection tab",
                        options=list(range(len(_batch))),
                        format_func=lambda i, b=_batch: f"Frame {b[i]['sample_index']}",
                        key="rosbag_extracted_sample_selector",
                    )
                    st.session_state.rosbag_selected_sample_idx = selected_idx
                    load_idx = min(st.session_state.rosbag_selected_sample_idx, len(_batch) - 1) if _batch else 0
                    if st.button("🔄 Load this sample for Detection", key="rosbag_load_one_sample"):
                        with st.spinner("Loading sample..."):
                            meta, image, pc = load_dataset_sample(
                                dataset_path=_batch[load_idx]["dataset_path"],
                                sample_index=_batch[load_idx]["sample_index"],
                                dataset_type="rosbag",
                                filter_forward_only=False,
                            )
                        if meta is not None and image is not None and pc is not None:
                            st.session_state.sample = {
                                "sample_meta_data": meta,
                                "image": image,
                                "point_cloud": pc,
                            }
                            # Ensure single-sample mode on Detection page
                            st.session_state.process_all_samples = False
                            st.success("✅ Sample loaded. Go to **2_Detection** to run the pipeline and check fusion.")
                            st.rerun()
                        else:
                            st.error("Failed to load sample. Check that image and point cloud files exist in the extracted folder.")
                    # Show calibration matrices from extracted calib.npz for debugging fusion
                    calib_path = Path(_batch[0]["dataset_path"]) / "calib.npz"
                    if calib_path.exists():
                        with st.expander("📐 Calibration (calib.npz) — inspect for fusion issues"):
                            try:
                                calib = np.load(str(calib_path), allow_pickle=True)
                                camera_intrinsic = calib.get("camera_intrinsic")
                                camera_to_lidar = calib.get("camera_to_lidar")
                                camera_frame = calib.get("camera_frame", None)
                                lidar_frame = calib.get("lidar_frame", None)
                                if camera_frame is not None and hasattr(camera_frame, "item"):
                                    camera_frame = camera_frame.item()
                                if lidar_frame is not None and hasattr(lidar_frame, "item"):
                                    lidar_frame = lidar_frame.item()
                                st.markdown("**Frames**")
                                st.text(f"camera_frame: {camera_frame}\nlidar_frame: {lidar_frame}")
                                st.markdown("**Camera intrinsic (3×3)** — maps camera 3D to image (u,v)")
                                if camera_intrinsic is not None:
                                    st.dataframe(np.asarray(camera_intrinsic).round(4))
                                else:
                                    st.warning("Missing")
                                st.markdown("**Camera → LiDAR transform (4×4)** — transforms points from camera to LiDAR frame")
                                if camera_to_lidar is not None:
                                    st.dataframe(np.asarray(camera_to_lidar).round(4))
                                else:
                                    st.warning("Missing")
                            except Exception as e:
                                st.error(f"Could not load calib.npz: {e}")
                    else:
                        with st.expander("📐 Calibration (calib.npz)"):
                            st.warning(f"No calib.npz found at {calib_path}. Fusion may use identity matrices.")

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

                        # Quick presets for indoor / outdoor scenes
                        col_sim_preset1, col_sim_preset2 = st.columns(2)
                        with col_sim_preset1:
                            if st.button("🌳 Outdoor Scenes", key="sim_outdoor_preset"):
                                st.session_state.sim_filter_params.update(
                                    {
                                        "motion_thresh": 8,
                                        "blur_gate": 140,
                                        "hash_thresh": 8,
                                        "min_bright": 30,
                                        "max_bright": 245,
                                        "min_contrast": 0.12,
                                        "enable_blur": True,
                                        "enable_dedup": True,
                                        "enable_motion": True,
                                        "enable_brightness": True,
                                        "enable_contrast": True,
                                    }
                                )
                                st.rerun()
                        with col_sim_preset2:
                            if st.button("🏠 Indoor Scenes", key="sim_indoor_preset"):
                                st.session_state.sim_filter_params.update(
                                    {
                                        "motion_thresh": 4,
                                        "blur_gate": 110,
                                        "hash_thresh": 6,
                                        "min_bright": 40,
                                        "max_bright": 235,
                                        "min_contrast": 0.10,
                                        "enable_blur": True,
                                        "enable_dedup": True,
                                        "enable_motion": True,
                                        "enable_brightness": True,
                                        "enable_contrast": True,
                                    }
                                )
                                st.rerun()
                        
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
                                # Persist selected index so it survives reruns (button click triggers new run)
                                sim_batch_key = f"sim_selected_idx_{selected_subset}"
                                if sim_batch_key not in st.session_state:
                                    st.session_state[sim_batch_key] = None

                                # Create selection interface
                                num_cols = 3
                                cols = st.columns(num_cols)

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

                                        st.image(thumbnail)

                                        # Display metrics
                                        st.caption(f"**Token:** {link_token[:12]}...")
                                        st.caption(f"Blur: {metrics.get('blur', 0):.1f}")
                                        st.caption(f"Contrast: {metrics.get('contrast', 0):.3f}")
                                        st.caption(f"Brightness: {metrics.get('brightness', 0):.1f}")

                                        # Selection button: persist index in session state
                                        if st.button(f"Select", key=f"select_sample_{idx}"):
                                            st.session_state[sim_batch_key] = idx
                                            st.rerun()

                                selected_sample_idx = st.session_state[sim_batch_key]
                                if selected_sample_idx is not None and selected_sample_idx >= len(filtered_batch):
                                    st.session_state[sim_batch_key] = None
                                    selected_sample_idx = None

                                # Handle selection
                                if selected_sample_idx is not None:
                                    if st.button('Load selected sample for detection', key='load_selected_sample_for_detection'):
                                        st.session_state.process_all_samples = False
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
                                            # Sim LiDAR path is resolved in the sim dataset loader.
                                            "point_cloud_path": "",
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

