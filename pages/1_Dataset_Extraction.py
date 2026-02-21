"""
Dataset Extraction Page
Loads and extracts samples from different dataset formats (KITTI, nuScenes, sim).
"""
import streamlit as st
import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from PIL import Image
import imagehash

from components.dataset_loaders.utils import detect_dataset_type, load_dataset_sample
from components.dataset_loaders.dataset_loader import LinkedDataHandler

# Import image quality functions from 1_Extraction.py
def variance_of_laplacian(img_gray):
    """Calculate blur metric using Laplacian variance."""
    lap = cv2.Laplacian(img_gray, cv2.CV_64F)
    return float(lap.var())

def contrast_score(img_gray):
    """Calculate normalized RMS contrast."""
    return float(np.std(img_gray)) / 255.0

def brightness(img_gray):
    """Calculate average brightness."""
    return float(img_gray.mean())

def hamming(a, b):
    """Calculate Hamming distance between two perceptual hashes."""
    return int(a - b)

def calculate_motion_score(frame1_gray, frame2_gray):
    """Calculate motion score between consecutive frames."""
    if frame1_gray is None or frame2_gray is None:
        return 0
    diff = cv2.absdiff(frame1_gray, frame2_gray)
    return np.mean(diff)


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
    
    # Auto-detect dataset type
    dataset_type = None
    if dataset_path:
        dataset_path_obj = Path(dataset_path)
        if dataset_path_obj.exists():
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
            except Exception as e:
                st.error(f"Error: {str(e)}")
        
        elif dataset_type == "nuscenes":
            # nuScenes: Use sample tokens
            st.info("nuScenes dataset loading - Enter sample token")
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
                                filtered_batch = _filter_sim_images(
                                    handler, selected_subset, links, dataset_path,
                                    st.session_state.sim_filter_params
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
        
        # Sample metadata
        with st.expander("Sample Metadata", expanded=False):
            st.json(sample_meta_data)
        
        st.success("✅ Sample is ready! Navigate to **2_Detection.py** to run the detection pipeline.")


def _filter_sim_images(
    handler: LinkedDataHandler,
    subset_name: str,
    links: List[Dict],
    dataset_path: str,
    filter_params: Dict
) -> List[Dict]:
    """
    Filter images from sim dataset using quality metrics.
    
    Args:
        handler: LinkedDataHandler instance
        subset_name: Name of the subset
        links: List of link dictionaries
        dataset_path: Root dataset path
        filter_params: Dictionary with filter parameters
    
    Returns:
        List of filtered sample dictionaries with image, link_token, and metrics
    """
    filtered_samples = []
    seen_hashes = []
    prev_frame_gray = None
    
    root_path = Path(dataset_path)
    subset_path = root_path / subset_name
    # Process each link
    for link_idx, link in enumerate(links):
        try:
            rgb_sample = link.get('samples', {}).get('rgb', {})
            if not rgb_sample or 'filename' not in rgb_sample:
                continue
            
            # Get filename and normalize it (handle absolute paths)
            filename = rgb_sample['filename']
            
            # Normalize filename: remove leading slashes and handle absolute paths
            # Filename format might be: /rgb/filename.jpg or C:\rgb\filename.jpg
            # We want: rgb/filename.jpg (relative to samples folder)
            filename = filename.lstrip('/').lstrip('\\')
            
            # If it's an absolute Windows path (starts with drive letter), extract relative part
            if len(filename) > 1 and filename[1] == ':':
                # Windows absolute path like C:\rgb\file.jpg
                # Extract everything after the first backslash after the drive
                parts = filename.split('\\', 2)
                if len(parts) > 2:
                    filename = parts[2]  # Take everything after C:\rgb\
                else:
                    # Just drive and filename, take filename
                    filename = parts[-1]
            
            # Construct image path: dataset_path / subset_name / samples / filename
            image_path = subset_path / "samples" / filename
            if not image_path.exists():
                continue
            
            image_bgr = cv2.imread(str(image_path))
            if image_bgr is None:
                continue
            
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            
            # Calculate metrics
            blur_score = variance_of_laplacian(image_gray)
            contrast_val = contrast_score(image_gray)
            brightness_val = brightness(image_gray)
            
            # Apply filters
            passed = True
            
            # Blur filter
            if filter_params['enable_blur']:
                if blur_score < filter_params['blur_gate']:
                    passed = False
            
            # Motion filter (if enabled and we have previous frame)
            if filter_params['enable_motion'] and prev_frame_gray is not None:
                motion_score = calculate_motion_score(prev_frame_gray, image_gray)
                if motion_score < filter_params['motion_thresh']:
                    passed = False
            
            # Deduplication
            if filter_params['enable_dedup'] and passed:
                try:
                    ph = imagehash.dhash(Image.fromarray(image_gray))
                    if any(hamming(ph, old) <= filter_params['hash_thresh'] for old in seen_hashes):
                        passed = False
                    else:
                        seen_hashes.append(ph)
                except Exception:
                    pass  # Skip hash errors
            
            # Brightness filter
            if filter_params['enable_brightness'] and passed:
                if not (filter_params['min_bright'] <= brightness_val <= filter_params['max_bright']):
                    passed = False
            
            # Contrast filter
            if filter_params['enable_contrast'] and passed:
                if contrast_val < filter_params['min_contrast']:
                    passed = False
            
            # If passed all filters, add to batch
            if passed:
                filtered_samples.append({
                    'link_token': link['token'],
                    'link': link,
                    'image': image_rgb,
                    'image_path': str(image_path),
                    'metrics': {
                        'blur': blur_score,
                        'contrast': contrast_val,
                        'brightness': brightness_val
                    }
                })
            
            # Update previous frame for motion detection
            if filter_params['enable_motion']:
                prev_frame_gray = image_gray
            
        except Exception as e:
            # Skip links that fail to process
            continue
    
    return filtered_samples


if __name__ == "__main__":
    main()

