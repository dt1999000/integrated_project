"""
Dataset Extraction Page
Loads and extracts samples from different dataset formats (KITTI, nuScenes, sim).
"""
import streamlit as st
import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any

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
    get_tf_frames as get_ros_tf_frames,
    compute_rosbag_calibration,
)
from components.utils.visualization_helper import create_3d_scatter_plot
from components.core.pointcloud_projection import PointCloud
from components.core.filter import (
    filter_kitti_images,
    filter_nuscenes_images,
    filter_sim_images,
    sample_rosbag_frames_every_nth,
)

DEFAULT_FILTER_PARAMS = {
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

OUTDOOR_SCENE_PRESET = {
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

INDOOR_SCENE_PRESET = {
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


def _every_nth_indices(total_count: int, stride: int, start: int = 0) -> np.ndarray:
    """
    Utility: return indices [start, start+stride, ...] < total_count as int array.
    """
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    if total_count <= 0:
        return np.array([], dtype=int)
    start = max(0, min(start, total_count - 1))
    return np.arange(start, total_count, stride, dtype=int)


def _sim_rgb_path_for_link(dataset_path: str, subset_name: str, link: Dict) -> Optional[Path]:
    """Resolve on-disk path for a sim link's RGB sample (same rules as filter_sim_images)."""
    rgb_sample = link.get("samples", {}).get("rgb", {})
    if not rgb_sample or "filename" not in rgb_sample:
        return None
    filename = rgb_sample["filename"]
    filename = filename.lstrip("/").lstrip("\\")
    if len(filename) > 1 and filename[1] == ":":
        parts = filename.split("\\", 2)
        filename = parts[2] if len(parts) > 2 else parts[-1]
    subset_path = Path(dataset_path) / subset_name
    return subset_path / "samples" / filename


def ensure_filter_state(prefix: str) -> None:
    """Ensure st.session_state[f'{prefix}_filter_params'] exists with defaults."""
    key = f"{prefix}_filter_params"
    if key not in st.session_state:
        st.session_state[key] = DEFAULT_FILTER_PARAMS.copy()


def render_filter_controls(prefix: str, title: str, expanded: bool = False) -> Dict[str, Any]:
    """
    Render common blur/dedup/motion/brightness/contrast controls for image filtering.

    Updates and returns st.session_state[f'{prefix}_filter_params'].
    """
    state_key = f"{prefix}_filter_params"
    ensure_filter_state(prefix)
    params = st.session_state[state_key]

    with st.expander(f"⚙️ Filter Settings ({title})", expanded=expanded):
        col1, col2 = st.columns(2)

        with col1:
            params["enable_blur"] = st.checkbox(
                "Enable Blur Filter",
                value=params["enable_blur"],
                help="Remove blurry images using Laplacian variance",
                key=f"{prefix}_enable_blur",
            )
            if params["enable_blur"]:
                params["blur_gate"] = st.slider(
                    "Blur Gate (Laplacian Variance)",
                    0,
                    500,
                    params["blur_gate"],
                    help="Minimum Laplacian variance (higher = sharper)",
                    key=f"{prefix}_blur_gate",
                )

            params["enable_dedup"] = st.checkbox(
                "Enable Deduplication",
                value=params["enable_dedup"],
                help="Remove visually similar images",
                key=f"{prefix}_enable_dedup",
            )
            if params["enable_dedup"]:
                params["hash_thresh"] = st.slider(
                    "Deduplication Threshold (Hamming)",
                    0,
                    16,
                    params["hash_thresh"],
                    help="Maximum Hamming distance for duplicates",
                    key=f"{prefix}_hash_thresh",
                )

            params["enable_motion"] = st.checkbox(
                "Enable Motion Filter",
                value=params["enable_motion"],
                help="Skip static frames",
                key=f"{prefix}_enable_motion",
            )
            if params["enable_motion"]:
                params["motion_thresh"] = st.slider(
                    "Motion Threshold",
                    0,
                    20,
                    params["motion_thresh"],
                    help="Minimum motion score between frames",
                    key=f"{prefix}_motion_thresh",
                )

        with col2:
            params["enable_brightness"] = st.checkbox(
                "Enable Brightness Filter",
                value=params["enable_brightness"],
                help="Remove over/under-exposed images",
                key=f"{prefix}_enable_brightness",
            )
            if params["enable_brightness"]:
                params["min_bright"] = st.slider(
                    "Min Brightness",
                    0,
                    255,
                    params["min_bright"],
                    key=f"{prefix}_min_bright",
                )
                params["max_bright"] = st.slider(
                    "Max Brightness",
                    0,
                    255,
                    params["max_bright"],
                    key=f"{prefix}_max_bright",
                )

            params["enable_contrast"] = st.checkbox(
                "Enable Contrast Filter",
                value=params["enable_contrast"],
                help="Remove low-contrast images",
                key=f"{prefix}_enable_contrast",
            )
            if params["enable_contrast"]:
                params["min_contrast"] = st.slider(
                    "Min Contrast",
                    0.0,
                    0.5,
                    params["min_contrast"],
                    step=0.01,
                    key=f"{prefix}_min_contrast",
                )

    st.session_state[state_key] = params
    return params


def render_scene_preset_buttons(prefix: str) -> None:
    """Quick outdoor/indoor presets for image filter params (shared across dataset types)."""
    ensure_filter_state(prefix)
    state_key = f"{prefix}_filter_params"
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🌳 Outdoor Scenes", key=f"{prefix}_outdoor_preset"):
            st.session_state[state_key].update(OUTDOOR_SCENE_PRESET)
            st.rerun()
    with col_b:
        if st.button("🏠 Indoor Scenes", key=f"{prefix}_indoor_preset"):
            st.session_state[state_key].update(INDOOR_SCENE_PRESET)
            st.rerun()


def _resolve_current_calibration(dataset_flag: Optional[str]) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[Any]]:
    """
    Resolve calibration for the current session based on a dataset flag.
    
    Returns a tuple of (calib_dict_or_None, dataset_type, sample_index).
    calib_dict has keys: camera_intrinsic, camera_to_lidar, camera_frame, lidar_frame.
    """
    calib: Optional[Dict[str, Any]] = None
    dataset_type: Optional[str] = dataset_flag
    sample_index: Optional[Any] = None

    # ROS bag: always prefer the live calibration computed from TF/frame selection.
    # This is recomputed whenever frames change in ROS mode.
    if dataset_type == "rosbag":
        rosbag_calib = st.session_state.get("rosbag_calibration")
        if rosbag_calib:
            camera_intrinsic = rosbag_calib.get("camera_intrinsic")
            camera_to_lidar = rosbag_calib.get("camera_to_lidar")
            camera_frame = rosbag_calib.get("camera_frame")
            lidar_frame = rosbag_calib.get("lidar_frame")
            calib = {
                "camera_intrinsic": camera_intrinsic,
                "camera_to_lidar": camera_to_lidar,
                "camera_frame": camera_frame,
                "lidar_frame": lidar_frame,
            }
            return calib, dataset_type, sample_index

    # Other datasets (KITTI, nuScenes, sim, extracted rosbag samples):
    # prefer calibration coming from the currently loaded sample's metadata.
    if "sample" in st.session_state and st.session_state.sample is not None:
        meta = st.session_state.sample.get("sample_meta_data", {})
        meta_dataset_type = meta.get("dataset_type")
        if dataset_type is None:
            dataset_type = meta_dataset_type
        if meta_dataset_type == dataset_type or dataset_type is None:
            camera_intrinsic = meta.get("camera_intrinsic")
            camera_to_lidar = meta.get("camera_to_lidar_transform")
            sample_index = meta.get("sample_index")
            camera_frame = meta.get("camera_frame")
            lidar_frame = meta.get("lidar_frame")
            if camera_intrinsic is not None or camera_to_lidar is not None:
                calib = {
                    "camera_intrinsic": camera_intrinsic,
                    "camera_to_lidar": camera_to_lidar,
                    "camera_frame": camera_frame,
                    "lidar_frame": lidar_frame,
                }

    return calib, dataset_type, sample_index

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
    # Ensure global params dict exists so we can store bag frequency for tracking.
    if 'params' not in st.session_state:
        st.session_state.params = {}

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
            # Persist the detected dataset type as a flag for other pages/helpers
            st.session_state.current_dataset_type = dataset_type
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
        _filtered_scope = (dataset_type, str(dataset_path))
        if st.session_state.get("_filtered_batch_scope") != _filtered_scope:
            st.session_state.filtered_batch = None
            st.session_state._filtered_batch_scope = _filtered_scope
        elif "filtered_batch" not in st.session_state:
            st.session_state.filtered_batch = None
        
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
                                    kitti_gt = sample_meta_data.get("ground_truth_boxes", [])
                                    st.session_state.sample = {
                                        'sample_meta_data': sample_meta_data,
                                        'image': image,
                                        'point_cloud': point_cloud
                                    }
                                    st.session_state.ground_truth_annotations = kitti_gt
                                    st.session_state.ground_truth_2d_boxes = [
                                        box for box in kitti_gt if box.get("bbox_2d") is not None
                                    ]
                                    # Ensure single-sample mode on Detection page
                                    st.session_state.process_all_samples = False
                                    st.success(f"✅ Sample {sample_index} loaded successfully!")
                                    st.rerun()
                                else:
                                    st.error("❌ Failed to load sample")

                        # KITTI: random batch filtering + send to detection
                        ensure_filter_state("kitti")
                        st.markdown("---")
                        st.subheader("🖼️ Image Filtering (KITTI Dataset)")
                        st.markdown("""
                        Sample a random batch of KITTI images, filter them by quality, and send the filtered batch to the detection page.
                        """)

                        render_scene_preset_buttons("kitti")

                        # Filter configuration (shared UI)
                        render_filter_controls(prefix="kitti", title="KITTI")

                        # Random batch parameters
                        col_a, col_b = st.columns(2)
                        with col_a:
                            kitti_stride = st.number_input(
                                "Take every n-th frame (KITTI)",
                                min_value=1,
                                max_value=max(1, num_samples),
                                value=min(10, max(1, num_samples)),
                                step=1,
                                key="kitti_stride_every_nth",
                            )
                        with col_b:
                            kitti_start = st.number_input(
                                "Start index",
                                min_value=0,
                                max_value=max(0, num_samples - 1),
                                value=0,
                                step=1,
                                key="kitti_every_nth_start",
                            )

                        if st.button("📚 Prepare KITTI batch (every n-th frame)", type="primary", key="filter_kitti_batch"):
                            with st.spinner("Preparing KITTI batch..."):
                                step = int(kitti_stride)
                                start_idx = int(kitti_start)
                                indices_every_nth = _every_nth_indices(num_samples, step, start_idx)
                                filtered_batch = []
                                for idx in indices_every_nth:
                                    if 0 <= int(idx) < len(image_files):
                                        filtered_batch.append(
                                            {
                                                "sample_index": int(idx),
                                                "image_path": str(image_files[int(idx)]),
                                            }
                                        )
                                st.session_state.filtered_batch = filtered_batch
                            st.success(
                                f"✅ Selected {len(st.session_state.filtered_batch)} KITTI frames "
                                f"using start={start_idx}, step={step}."
                            )
                            st.rerun()

                        # Display filtered batch summary and allow sending to detection
                        if st.session_state.filtered_batch:
                            filtered_batch = st.session_state.filtered_batch
                            st.markdown("---")
                            st.subheader("📋 Subsampled Sample Batch (KITTI)")
                            st.info(f"Prepared {len(filtered_batch)} KITTI samples (every n-th frame)")

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

            # nuScenes: simple subsampling (every n-th sample) + send to detection

            ensure_filter_state("nuscenes")
            st.markdown("---")
            st.subheader("🖼️ Image Filtering (nuScenes Dataset)")
            st.markdown("""
            Sample a random batch of nuScenes images (CAM_FRONT), filter them by quality, and send the filtered batch to the detection page.
            """)

            render_scene_preset_buttons("nuscenes")

            # Filter configuration (shared UI)
            render_filter_controls(prefix="nuscenes", title="nuScenes")

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
                                st.session_state.filtered_batch = filtered_batch
                                st.success(f"✅ Filtered {len(filtered_batch)} samples from {total_samples} total nuScenes samples")
                                st.rerun()
                except Exception as e:
                    st.error(f"Error during nuScenes batch filtering: {str(e)}")

            # Display filtered batch summary and allow sending to detection
            if st.session_state.filtered_batch:
                filtered_batch = st.session_state.filtered_batch
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

            # Sidebar controls to choose calibration frames (once TF topic is known)
            if tf_topic:
                tf_frames = get_ros_tf_frames(bag_path_obj, [tf_topic])
                if tf_frames:
                    # Initialize defaults in session_state so choices persist
                    if "rosbag_camera_frame_override" not in st.session_state:
                        st.session_state.rosbag_camera_frame_override = tf_frames[0]
                    if "rosbag_lidar_frame_override" not in st.session_state:
                        st.session_state.rosbag_lidar_frame_override = tf_frames[min(1, len(tf_frames) - 1)]

                    with st.sidebar:
                        st.markdown("### ROS calibration frames")
                        st.session_state.rosbag_camera_frame_override = st.selectbox(
                            "Camera frame for calibration",
                            tf_frames,
                            index=tf_frames.index(st.session_state.rosbag_camera_frame_override)
                            if st.session_state.rosbag_camera_frame_override in tf_frames
                            else 0,
                            help="TF frame to treat as camera for camera→LiDAR transform",
                            key="rosbag_camera_frame_override_select",
                        )
                        st.session_state.rosbag_lidar_frame_override = st.selectbox(
                            "LiDAR frame for calibration",
                            tf_frames,
                            index=tf_frames.index(st.session_state.rosbag_lidar_frame_override)
                            if st.session_state.rosbag_lidar_frame_override in tf_frames
                            else min(1, len(tf_frames) - 1),
                            help="TF frame to treat as LiDAR for camera→LiDAR transform",
                            key="rosbag_lidar_frame_override_select",
                        )

                        # Recompute calibration immediately when frame selection changes
                        calib = compute_rosbag_calibration(
                            bag_path=bag_path_obj,
                            image_topic=image_topic,
                            pointcloud_topic=pointcloud_topic,
                            camera_info_topic=camera_info_topic,
                            tf_topics=[tf_topic] if tf_topic else None,
                            camera_frame_override=st.session_state.rosbag_camera_frame_override,
                            lidar_frame_override=st.session_state.rosbag_lidar_frame_override,
                        )
                        st.session_state.rosbag_calibration = {
                            "camera_intrinsic": calib.camera_intrinsic,
                            "camera_to_lidar": calib.camera_to_lidar_transform,
                            "camera_frame": calib.camera_frame,
                            "lidar_frame": calib.lidar_frame,
                        }

            ensure_filter_state("rosbag")
            if "rosbag_filtered_frames" not in st.session_state:
                st.session_state.rosbag_filtered_frames = None

            st.markdown("---")
            st.subheader("🖼️ Frame Filtering (ROS bag)")
            st.markdown(
                "Apply the same quality filters as for KITTI/nuScenes, but directly on the ROS bag frames."
            )

            render_scene_preset_buttons("rosbag")

            # Filter configuration (shared UI) plus ROS bag–specific limit
            render_filter_controls(prefix="rosbag", title="ROS bag")
            rosbag_max_frames = st.number_input(
                "Max frames to consider from bag (0 = all)",
                min_value=0,
                value=0,
                step=1,
                key="rosbag_max_frames",
            )

            # Simple subsampling option: take every n-th ROS bag frame without applying quality filters
            st.markdown("**Alternatively, create a simple batch by taking every n-th ROS bag frame (no quality filters).**")
            col_stride1, col_stride2 = st.columns(2)
            with col_stride1:
                rosbag_stride = st.number_input(
                    "Take every n-th frame (ROS bag)",
                    min_value=1,
                    value=10,
                    step=1,
                    key="rosbag_stride_every_nth",
                )
            with col_stride2:
                rosbag_simple_max = st.number_input(
                    "Max sampled frames (0 = unlimited)",
                    min_value=0,
                    value=0,
                    step=1,
                    key="rosbag_simple_max_frames",
                )

            if st.button(
                "📚 Sample ROS bag every n-th frame (skip quality filters)",
                key="rosbag_sample_every_nth",
            ):
                with st.spinner("Sampling ROS bag frames (every n-th)..."):
                    stride = int(rosbag_stride)
                    max_simple = int(rosbag_simple_max) if rosbag_simple_max > 0 else None
                    sampled_frames = sample_rosbag_frames_every_nth(
                        bag_path=bag_path_obj,
                        image_topic=image_topic,
                        stride=stride,
                        max_frames=max_simple,
                    )
                    st.session_state.rosbag_filtered_frames = sampled_frames
                if sampled_frames:
                    st.success(
                        f"✅ Selected {len(sampled_frames)} frames by taking every {stride}-th message."
                    )
                    st.rerun()
                else:
                    st.warning("No frames were sampled. Check the image topic and stride.")

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
                                camera_frame_override=st.session_state.get("rosbag_camera_frame_override"),
                                lidar_frame_override=st.session_state.get("rosbag_lidar_frame_override"),
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
                        print(f'batch_samples_saved: {st.session_state.batch_samples_saved}, batch_samples: {batch_samples}')
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
                    # Optional: load all extracted ROS bag samples as a batch for the Detection page
                    if st.button("📚 Load all extracted ROS bag samples for detection", key="load_all_rosbag_for_detection"):
                        st.session_state.batch_samples = _batch
                        st.session_state.process_all_samples = True
                        st.success(
                            f"✅ Prepared {len(_batch)} ROS bag samples. "
                            "Go to **2_Detection** and click **Process entire batch**."
                        )
                        st.rerun()
                    # Calibration is now tracked live in st.session_state.rosbag_calibration

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
                        if st.session_state.get("_sim_batch_subset") != selected_subset:
                            st.session_state.filtered_batch = None
                            st.session_state._sim_batch_subset = selected_subset

                        subset = handler.subsets[selected_subset]
                        links = subset['links']
                        
                        ensure_filter_state("sim")

                        st.markdown("---")
                        st.subheader("🖼️ Image Filtering (Sim Dataset)")
                        st.markdown("""
                        Filter images from the dataset using quality metrics.
                        Only images that pass all enabled filters will be available for selection.
                        """)

                        # Simple subsampling option: take every n-th sim link without quality filtering
                        st.markdown("**Alternatively, create a simple batch by taking every n-th sim sample (no quality filters).**")
                        sim_stride = st.number_input(
                            "Take every n-th sample (Sim)",
                            min_value=1,
                            max_value=max(1, len(links)),
                            value=min(10, max(1, len(links))),
                            step=1,
                            key=f"sim_stride_every_nth_{selected_subset}",
                        )
                        if st.button(
                            "📚 Prepare every n-th sim batch",
                            key=f"prepare_every_nth_sim_batch_{selected_subset}",
                        ):
                            with st.spinner("Preparing every n-th sim batch..."):
                                step = int(sim_stride)
                                indices_every_nth = _every_nth_indices(len(links), step)
                                subsample_batch: List[Dict] = []
                                for idx in indices_every_nth:
                                    link = links[int(idx)]
                                    image_path = _sim_rgb_path_for_link(dataset_path, selected_subset, link)
                                    if image_path is None or not image_path.exists():
                                        continue
                                    image_bgr = cv2.imread(str(image_path))
                                    if image_bgr is None:
                                        continue
                                    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                                    subsample_batch.append(
                                        {
                                            "link_token": link["token"],
                                            "link": link,
                                            "image": image_rgb,
                                            "image_path": str(image_path),
                                            "metrics": {
                                                "blur": 0.0,
                                                "contrast": 0.0,
                                                "brightness": 0.0,
                                            },
                                            "subsample_only": True,
                                        }
                                    )
                                st.session_state.filtered_batch = subsample_batch
                                st.session_state._sim_batch_is_subsample = True
                                st.success(
                                    f"✅ Prepared {len(subsample_batch)} sim samples by taking every {step}-th link "
                                    "(no quality filters). Use the batch below to load one sample or the full batch."
                                )
                                st.rerun()

                        render_scene_preset_buttons("sim")

                        # Filter configuration (shared UI)
                        render_filter_controls(prefix="sim", title="Sim Dataset", expanded=True)
                        
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
                                st.session_state.filtered_batch = filtered_batch
                                st.session_state._sim_batch_is_subsample = False
                                st.success(f"✅ Filtered {len(filtered_batch)} samples from {len(links)} total links")
                                st.rerun()
                        
                        # Display filtered batch and allow selection
                        if st.session_state.filtered_batch is not None:
                            filtered_batch = st.session_state.filtered_batch
                            
                            st.markdown("---")
                            st.subheader("📋 Filtered Sample Batch")
                            batch_is_subsample = st.session_state.get("_sim_batch_is_subsample") is True
                            if batch_is_subsample:
                                st.info(
                                    f"Prepared {len(filtered_batch)} samples (every n-th link, no quality filters)"
                                )
                            else:
                                st.info(f"Found {len(filtered_batch)} samples that passed all filters")
                            
                            if len(filtered_batch) > 0:
                                st.markdown(
                                    "**Load a sample for detection:** use the button under a thumbnail, "
                                    "or load the full batch below for **2_Detection → Process entire batch**."
                                )

                                num_cols = 3
                                cols = st.columns(num_cols)

                                for idx, sample_info in enumerate(filtered_batch):
                                    col_idx = idx % num_cols
                                    with cols[col_idx]:
                                        image = sample_info['image']
                                        link_token = sample_info['link_token']
                                        metrics = sample_info['metrics']

                                        h, w = image.shape[:2]
                                        max_size = 200
                                        if w > max_size or h > max_size:
                                            scale = max_size / max(w, h)
                                            new_w, new_h = int(w * scale), int(h * scale)
                                            thumbnail = cv2.resize(image, (new_w, new_h))
                                        else:
                                            thumbnail = image

                                        st.image(thumbnail)

                                        st.caption(f"**Token:** {link_token[:12]}...")
                                        if sample_info.get("subsample_only"):
                                            st.caption("Quality metrics: not computed (subsample)")
                                        else:
                                            st.caption(f"Blur: {metrics.get('blur', 0):.1f}")
                                            st.caption(f"Contrast: {metrics.get('contrast', 0):.3f}")
                                            st.caption(f"Brightness: {metrics.get('brightness', 0):.1f}")

                                        if st.button(
                                            "Load for detection",
                                            key=f"load_sim_sample_{selected_subset}_{idx}",
                                        ):
                                            st.session_state.process_all_samples = False
                                            with st.spinner(f"Loading sample {link_token}..."):
                                                sample_meta_data, image, point_cloud = load_dataset_sample(
                                                    dataset_path=dataset_path,
                                                    sample_index=sample_info['link_token'],
                                                    dataset_type=dataset_type,
                                                )

                                                if sample_meta_data and image is not None and point_cloud is not None:
                                                    st.session_state.sample = {
                                                        'sample_meta_data': sample_meta_data,
                                                        'image': image,
                                                        'point_cloud': point_cloud
                                                    }
                                                    st.success(f"✅ Sample {link_token} loaded successfully!")
                                                    st.rerun()
                                                else:
                                                    st.error("❌ Failed to load sample")

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
                            st.info(
                                "👆 Use **Prepare every n-th sim batch** or **Filter Images** to build a batch, "
                                "then load one sample or the full batch for detection."
                            )
                else:
                    st.warning("No subsets found in dataset")
            except Exception as e:
                st.error(f"Error loading LinkedDataHandler: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
    # Global bag frequency control for batch processing / tracking
    if st.session_state.get("process_all_samples") or st.session_state.get("batch_samples"):
        if "bag_freq_hz" not in st.session_state.params:
            st.session_state.params["bag_freq_hz"] = 45.0
        with st.sidebar.expander("⏱️ Bag Frequency (tracking)", expanded=False):
            st.session_state.params["bag_freq_hz"] = st.number_input(
                "Bag Frequency (Hz)",
                min_value=1.0,
                max_value=200.0,
                value=float(st.session_state.params["bag_freq_hz"]),
                step=1.0,
                help="Sampling frequency of the sequence. Used for motion-based tracking (default 45 Hz).",
                key="bag_freq_hz_input",
            )

    # Persist currently loaded sample GT for downstream pages (Evaluation/Export).
    if st.session_state.get("sample") is not None:
        current_meta = st.session_state.sample.get("sample_meta_data", {})
        gt_boxes = current_meta.get("ground_truth_boxes", [])
        st.session_state.ground_truth_annotations = gt_boxes
        st.session_state.ground_truth_2d_boxes = [
            box for box in gt_boxes if box.get("bbox_2d") is not None
        ]
    elif "ground_truth_annotations" not in st.session_state:
        st.session_state.ground_truth_annotations = []
        st.session_state.ground_truth_2d_boxes = []

    # Global calibration summary, resolved from the current session/sample
    st.markdown("---")
    with st.expander("📐 Current calibration (from loaded sample)"):
        calib, dataset_type, sample_index = _resolve_current_calibration(
            st.session_state.get("current_dataset_type")
        )

        if calib:
            camera_intrinsic = calib.get("camera_intrinsic")
            camera_to_lidar = calib.get("camera_to_lidar")
            camera_frame = calib.get("camera_frame")
            lidar_frame = calib.get("lidar_frame")

            if dataset_type is not None:
                st.markdown(f"**Dataset:** {str(dataset_type).upper()}  |  **Sample:** {sample_index}")

            if camera_frame is not None or lidar_frame is not None:
                st.markdown("**Frames**")
                st.text(f"camera_frame: {camera_frame}\nlidar_frame: {lidar_frame}")

            st.markdown("**Camera intrinsic (3×3)** — maps camera 3D to image (u,v)")
            if camera_intrinsic is not None:
                st.dataframe(np.asarray(camera_intrinsic).round(4))
            else:
                st.warning("Missing camera_intrinsic")

            st.markdown("**Camera → LiDAR transform (4×4)** — transforms points from camera to LiDAR frame")
            if camera_to_lidar is not None:
                st.dataframe(np.asarray(camera_to_lidar).round(4))
            else:
                st.warning("Missing camera_to_lidar")
        else:
            st.info(
                "No calibration data available yet. "
                "Load a sample on this page (KITTI / nuScenes / sim / ROS bag) to populate calibration "
                "used by the Projection object in **2_Detection**."
            )


if __name__ == "__main__":
    main()

