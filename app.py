"""
3D Object Detection Pipeline - Main Application
Streamlit entry point for the unified detection pipeline.
"""
import streamlit as st

# Configure Streamlit page
st.set_page_config(
    page_title="3D Object Detection Pipeline",
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
    .page-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
    .page-card h3 {
        margin-top: 0;
        color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)


def main():
    """Main application entry point"""
    # Header
    st.markdown('<h1 class="main-header">🎯 3D Object Detection Pipeline</h1>', 
                unsafe_allow_html=True)
    
    # Introduction
    st.markdown("""
    Welcome to the **3D Object Detection Pipeline** application!
    
    This application provides a streamlined workflow for 3D object detection from autonomous driving datasets.
    Navigate through the pages using the sidebar menu to access different stages of the pipeline.
    """)
    
    # Pipeline Overview
    st.markdown("---")
    st.subheader("📋 Pipeline Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="page-card">
            <h3>📂 1. Dataset Extraction</h3>
            <p>Load and extract samples from different dataset formats:</p>
            <ul>
                <li><strong>KITTI</strong>: Standard KITTI dataset structure</li>
                <li><strong>nuScenes</strong>: nuScenes dataset format</li>
                <li><strong>sim</strong>: Custom format with LinkedDataHandler</li>
            </ul>
            <p><em>Features:</em> Multi-format support, image quality filtering (for sim datasets), sample preview</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="page-card">
            <h3>🎯 2. Detection Pipeline</h3>
            <p>Run the complete 3D object detection pipeline:</p>
            <ul>
                <li><strong>Step 1:</strong> Ground plane removal</li>
                <li><strong>Step 2:</strong> Sparse depth backprojection</li>
                <li><strong>Step 3:</strong> SAM segmentation</li>
                <li><strong>Step 4:</strong> Clustering (DBSCAN)</li>
                <li><strong>Step 5:</strong> Detection & pose estimation</li>
            </ul>
            <p><em>Features:</em> Full pipeline execution, step-by-step execution, progress tracking</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="page-card">
            <h3>📊 3. Evaluation</h3>
            <p>Evaluate detection results against ground truth:</p>
            <ul>
                <li>3D IoU computation</li>
                <li>2D IoU computation</li>
                <li>Comparison visualizations</li>
                <li>Matching statistics</li>
            </ul>
            <p><em>Features:</em> Comprehensive metrics, visual comparisons, detailed reports</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="page-card">
            <h3>💾 4. Export</h3>
            <p>Export detection results to various formats:</p>
            <ul>
                <li><strong>JSON</strong>: Custom format with metadata</li>
                <li><strong>KITTI</strong>: Standard KITTI format</li>
                <li><strong>COCO</strong>: Coming soon</li>
            </ul>
            <p><em>Features:</em> Multiple export formats, metadata inclusion, download support</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick Start Guide
    st.markdown("---")
    st.subheader("🚀 Quick Start Guide")
    
    st.markdown("""
    1. **Start with Dataset Extraction** (📂 1_Dataset_Extraction)
       - Select your dataset path
       - Choose a sample to load
       - For sim datasets: Use image filtering to create a quality-filtered batch
       - Preview and load your sample
    
    2. **Run Detection Pipeline** (🎯 2_Detection)
       - Use "Run Full Pipeline" to execute all steps at once
       - Or run each step individually to see intermediate results
       - Adjust parameters in the sidebar
       - View visualizations for each step
    
    3. **Evaluate Results** (📊 3_Evaluation)
       - Compare detections with ground truth
       - Review IoU metrics
       - Analyze detection quality
    
    4. **Export Results** (💾 4_Export)
       - Choose export format
       - Download results for downstream use
    """)
    
    # Session State Info
    st.markdown("---")
    st.subheader("ℹ️ Application Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **Session State:**
        - Sample data persists across pages
        - Pipeline state is maintained
        - Filter parameters are saved
        """)
    
    with col2:
        st.info("""
        **Navigation:**
        - Use the sidebar menu to switch between pages
        - Pages are automatically discovered by Streamlit
        - Data flows seamlessly between pages
        """)
    
    # Status Check
    if 'sample' in st.session_state and st.session_state.sample is not None:
        st.success("✅ Sample loaded and ready for processing!")
        sample_meta = st.session_state.sample.get('sample_meta_data', {})
        st.caption(f"Dataset: {sample_meta.get('dataset_type', 'unknown').upper()} | "
                  f"Sample: {sample_meta.get('sample_index', 'N/A')}")
    else:
        st.info("👈 Navigate to **1_Dataset_Extraction** to load a sample first.")


if __name__ == "__main__":
    main()
