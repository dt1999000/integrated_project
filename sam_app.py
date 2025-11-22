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
import cv2

# Add the current directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our pipeline components
from nuscenes_dataset_loader import NuScenesDatasetLoader
from pointcloud_projection import PointCloud, Projection2DTo3D
from clustering_manager import ClusteringManager
from sam_integration import SAMModelManager, BoundingBoxToSAM

# Configure Streamlit page
st.set_page_config(
    page_title="3D Object Detection with SAM",
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
if 'sam_manager' not in st.session_state:
    st.session_state.sam_manager = None

def load_dataset_sample(sample_index: int = 0):
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

        # Load image
        image = cv2.imread(sample_data['image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Initialize point cloud
        point_cloud = PointCloud()
        point_cloud.set_points(sample_data['point_cloud'])

        # Remove ground plane and ego vehicle
        point_cloud.remove_ground_plane_ransac()
        point_cloud.remove_ego_vehicle_points()

        # Initialize projection
        projection = Projection2DTo3D(
            camera_intrinsic=sample_data['camera_intrinsic'],
            camera_extrinsic=sample_data['camera_extrinsic'],
            camera_to_lidar_transform=sample_data['camera_to_lidar_transform'],
            point_cloud=sample_data['point_cloud'],
            image=image
        )

        return sample_data, point_cloud, projection, image

    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None, None, None

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

def model_selection_page():
    """Model selection and configuration page"""
    st.header("🤖 Model Selection")
    
    # Model selection
    with st.sidebar.expander("Model Configuration", expanded=True):
        model_type = st.selectbox(
            "Select SAM Model",
            options=["sam2_t", "sam2_b", "sam2_l", "sam_b", "mobile_sam"],
            index=0,
            help="Select the SAM model to use for segmentation"
        )
        
        # Model info
        if st.button("🔄 Load Model", key="load_model"):
            with st.spinner(f"Loading {model_type} model..."):
                try:
                    sam_manager = SAMModelManager(model_type)
                    st.session_state.sam_manager = sam_manager
                    st.success(f"{model_type} model loaded successfully!")
                except Exception as e:
                    st.error(f"Error loading model: {str(e)}")
    
    # Display model info if loaded
    if st.session_state.sam_manager is not None:
        st.success("Model loaded successfully!")
        
        # Model capabilities
        with st.expander("Model Capabilities", expanded=True):
            if st.session_state.sam_manager.model_type.startswith("sam2"):
                st.write("**SAM2 Dynamic Interactive Predictor**")
                st.write("- Supports bounding box prompts")
                st.write("- Supports point prompts")
                st.write("- Memory updates for consistent object tracking")
            elif st.session_state.sam_manager.model_type == "sam_b":
                st.write("**SAM-B Model**")
                st.write("- Standard SAM model")
                st.write("- Good general-purpose segmentation")
            elif st.session_state.sam_manager.model_type == "mobile_sam":
                st.write("**MobileSAM (FastSAM)**")
                st.write("- Optimized for mobile/edge devices")
                st.write("- Faster inference with slightly lower accuracy")

def sam_segmentation_page():
    """SAM segmentation page"""
    st.header("✂️ SAM Segmentation")
    
    # Check if model is loaded
    if st.session_state.sam_manager is None:
        st.error("Please load a model first in the Model Selection page")
        return
    
    # Check if data is loaded
    if st.session_state.point_cloud is None:
        st.error("Please load a dataset sample first")
        return
    
    # Get data
    point_cloud = st.session_state.point_cloud
    projection = point_cloud.projection if hasattr(point_cloud, 'projection') else None
    image = st.session_state.get('image', None)
    
    if image is None:
        st.error("No image available")
        return
    
    # Segmentation options
    with st.sidebar.expander("Segmentation Options", expanded=True):
        seg_method = st.selectbox(
            "Segmentation Method",
            options=["Bounding Boxes", "Point Prompts", "Automatic"],
            index=0,
            help="Method to use for segmentation"
        )
        
        if seg_method == "Bounding Boxes":
            st.write("Use bounding boxes from nuScenes annotations as prompts")
            
            # Load bounding boxes
            if st.button("📦 Load Bounding Boxes", key="load_bboxes"):
                with st.spinner("Loading bounding boxes..."):
                    try:
                        from bounding_boxes import BoundingBoxes
                        from nuscenes_dataset_loader import NuScenesDatasetLoader
                        
                        dataset_loader = NuScenesDatasetLoader(dataroot='v1.0-mini')
                        dataset_loader.load_dataset()
                        
                        sample_token = dataset_loader.nusc.sample[0]['token']
                        bbox_extractor = BoundingBoxes(nusc=dataset_loader.nusc)
                        bboxes = bbox_extractor.get_boxes_for_sample(sample_token, "CAM_FRONT")
                        
                        # Convert to format [x1, y1, x2, y2]
                        bbox_list = []
                        for bbox in bboxes:
                            bbox_list.append(bbox.bbox_2d)
                        
                        st.session_state.bboxes = bbox_list
                        st.success(f"Loaded {len(bbox_list)} bounding boxes")
                    except Exception as e:
                        st.error(f"Error loading bounding boxes: {str(e)}")
            
            # Display bounding boxes if available
            if 'bboxes' in st.session_state:
                st.write(f"**Loaded {len(st.session_state.bboxes)} bounding boxes**")
                
                # Draw bounding boxes on image
                fig, ax = plt.subplots(1, 1, figsize=(10, 6))
                ax.imshow(image)
                
                for bbox in st.session_state.bboxes:
                    x1, y1, x2, y2 = bbox
                    width = x2 - x1
                    height = y2 - y1
                    rect = plt.Rectangle((x1, y1), width, height, 
                                       linewidth=1, edgecolor='r', facecolor='none')
                    ax.add_patch(rect)
                
                ax.set_title("Bounding Boxes")
                ax.axis('off')
                st.pyplot(fig)
        
        elif seg_method == "Point Prompts":
            st.write("Click on the image to add point prompts")
            
            # Display image with click event
            st.write("Click on the image below to add point prompts:")
            clicked_point = st.image(image, use_column_width=True, 
                                 caption="Click to add point prompts")
            
            # Store clicked points
            if 'clicked_points' not in st.session_state:
                st.session_state.clicked_points = []
            
            if clicked_point is not None:
                # Get click coordinates (this is a simplified example)
                # In a real implementation, you'd need to handle click events properly
                st.session_state.clicked_points.append(clicked_point)
                st.write(f"Added point: {clicked_point}")
            
            # Display current points
            if st.session_state.clicked_points:
                st.write(f"**Current points:** {st.session_state.clicked_points}")
                
                # Draw points on image
                fig, ax = plt.subplots(1, 1, figsize=(10, 6))
                ax.imshow(image)
                
                for point in st.session_state.clicked_points:
                    ax.plot(point[0], point[1], 'ro', markersize=5)
                
                ax.set_title("Point Prompts")
                ax.axis('off')
                st.pyplot(fig)
            
            # Clear points button
            if st.button("🗑️ Clear Points"):
                st.session_state.clicked_points = []
                st.rerun()
        
        else:  # Automatic
            st.write("Automatic segmentation without prompts")
    
    # Run segmentation button
    if st.sidebar.button("🚀 Run Segmentation", key="run_segmentation"):
        with st.spinner("Running SAM segmentation..."):
            try:
                sam_manager = st.session_state.sam_manager
                
                if seg_method == "Bounding Boxes" and 'bboxes' in st.session_state:
                    # Use bounding boxes as prompts
                    mask = sam_manager.predict_from_bboxes(
                        image, st.session_state.bboxes
                    )
                elif seg_method == "Point Prompts" and 'clicked_points' in st.session_state:
                    # Use point prompts
                    mask = sam_manager.predict_from_points(
                        image, st.session_state.clicked_points
                    )
                else:  # Automatic
                    # Use automatic segmentation
                    results = sam_manager.predict(image)
                    mask = sam_manager.get_segmentation_masks(results)
                
                # Store mask
                st.session_state.segmentation_mask = mask
                st.success("Segmentation completed!")
                
                # Display mask
                fig, ax = plt.subplots(1, 2, figsize=(15, 6))
                
                # Original image
                ax[0].imshow(image)
                ax[0].set_title("Original Image")
                ax[0].axis('off')
                
                # Segmentation mask
                ax[1].imshow(mask, cmap='jet')
                ax[1].set_title("Segmentation Mask")
                ax[1].axis('off')
                
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Error running segmentation: {str(e)}")
    
    # 3D projection button
    if 'segmentation_mask' in st.session_state and st.sidebar.button("🚀 Project to 3D", key="project_to_3d"):
        with st.spinner("Projecting to 3D..."):
            try:
                # Project mask to 3D
                from segmentation_detection import SegmentationToPointCloud
                seg_to_3d = SegmentationToPointCloud(projection)
                mask_points = seg_to_3d.project_all_masks(st.session_state.segmentation_mask)
                
                # Add to point cloud
                point_cloud.add_segmentation_projected_points(mask_points)
                
                # Cluster based on masks
                clusters = point_cloud.cluster_with_sam_masks(
                    st.session_state.sam_manager, image, 
                    bboxes=st.session_state.get('bboxes', None)
                )
                
                # Store results
                st.session_state.clustering_results['sam'] = {
                    'mask_points': mask_points,
                    'clusters': clusters
                }
                
                st.success("Projection and clustering completed!")
                
            except Exception as e:
                st.error(f"Error projecting to 3D: {str(e)}")
    
    # Display results if available
    if 'sam' in st.session_state.clustering_results:
        result = st.session_state.clustering_results['sam']
        
        # Metrics
        clusters = result['clusters']
        n_clusters = len(clusters)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Clusters", n_clusters)
        with col2:
            st.metric("Masks", len(result['mask_points']))
        with col3:
            st.metric("Points", sum(len(points) for points in result['mask_points'].values()))
        
        # 3D Visualization
        st.subheader("3D Visualization")
        fig = create_3d_scatter_plot(point_cloud.get_points(), None, "Point Cloud with Projected Points")
        st.plotly_chart(fig, use_container_width=True)
        
        # Cluster visualization
        if clusters:
            st.subheader("Cluster Visualization")
            fig = create_3d_scatter_plot(point_cloud.get_points(), 
                                       np.concatenate(clusters), 
                                       "SAM-based Clustering")
            st.plotly_chart(fig, use_container_width=True)

def main():
    """Main application function"""
    # Header
    st.markdown('<h1 class="main-header">3D Object Detection with SAM</h1>', 
                unsafe_allow_html=True)
    
    # Introduction
    st.markdown("""
    This application provides an interface for using SAM (Segment Anything Model) with 3D point cloud data
    from autonomous driving datasets.

    ### Features:
    - **Multiple SAM Models**: SAM2-t, SAM2-b, SAM2-l, SAM-B, MobileSAM
    - **Flexible Prompts**: Bounding boxes, point prompts, or automatic segmentation
    - **3D Projection**: Project 2D segmentations to 3D point clouds
    - **Clustering**: Cluster projected points based on segmentation masks

    ### Getting Started:
    1. Select and load a SAM model in the Model Selection page
    2. Load a dataset sample
    3. Choose a segmentation method and run segmentation
    4. Project to 3D and cluster the points
    5. Analyze the results through visualizations
    """)
    
    # Navigation tabs
    tab1, tab2 = st.tabs(["🤖 Model Selection", "✂️ SAM Segmentation"])
    
    with tab1:
        model_selection_page()
    
    with tab2:
        sam_segmentation_page()

if __name__ == "__main__":
    main()
