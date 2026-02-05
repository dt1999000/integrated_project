# Repository Summary - 3D Object Detection Pipeline

## Architecture Overview (Mermaid Diagram)

```mermaid
graph TB
    subgraph "Entry Point"
        APP[app.py<br/>Streamlit Main Application]
    end
    
    subgraph "Dataset Loaders"
        KITTI[kitti_dataset_loader.py<br/>KITTI Dataset]
        NUSCENES[nuscenes_dataset_loader.py<br/>NuScenes Dataset]
    end
    
    subgraph "2D Detection & Segmentation"
        SEG[segmentation_detection.py<br/>SAM2 Segmentation]
        BB[bounding_boxes.py<br/>2D BBox Management]
    end
    
    subgraph "Depth Estimation"
        DEPTH[depth_estimation.py<br/>DepthEstimator]
        MARIGOLD[marigold_dc.py<br/>Marigold-DC Pipeline]
    end
    
    subgraph "Point Cloud Processing"
        PROJ[pointcloud_projection.py<br/>2D-to-3D Projection<br/>Frustum Filtering]
        FRUSTUM[frustum_manager.py<br/>FrustumManager<br/>Frustum Creation]
    end
    
    subgraph "Clustering"
        CLUSTER_MGR[clustering_manager.py<br/>ClusteringManager<br/>Multi-Algorithm Support]
        HDBSCAN[pages/hdbscan.py]
        DBSCAN[pages/dbscan.py]
        OPTICS[pages/optics.py]
        BIRCH[pages/birch.py]
        AGGLOM[pages/agglomerative.py]
    end
    
    subgraph "3D Object Detection"
        POSE[pose_estimation.py<br/>L-Shape & PCA]
        EVAL[evaluation.py<br/>Metrics & Evaluation]
    end
    
    subgraph "Visualization & UI"
        VIZ[visualization_helper.py<br/>3D Visualization]
        STATS[pages/statistics.py]
        KITTI_GT[pages/kitti_groundtruth.py]
        DEPTH_PAGE[pages/depth_estimation.py]
        PROJ_PAGE[pages/project_segmentation_mask_on_pointcloud.py]
    end
    
    %% Data Flow
    APP --> KITTI
    APP --> NUSCENES
    APP --> SEG
    APP --> DEPTH
    APP --> PROJ
    APP --> CLUSTER_MGR
    APP --> VIZ
    
    KITTI --> PROJ
    NUSCENES --> PROJ
    
    SEG --> BB
    BB --> FRUSTUM
    
    DEPTH --> MARIGOLD
    MARIGOLD --> PROJ
    
    PROJ --> FRUSTUM
    FRUSTUM --> CLUSTER_MGR
    
    CLUSTER_MGR --> HDBSCAN
    CLUSTER_MGR --> DBSCAN
    CLUSTER_MGR --> OPTICS
    CLUSTER_MGR --> BIRCH
    CLUSTER_MGR --> AGGLOM
    
    CLUSTER_MGR --> POSE
    POSE --> EVAL
    
    APP --> STATS
    APP --> KITTI_GT
    APP --> DEPTH_PAGE
    APP --> PROJ_PAGE
    
    EVAL --> VIZ
    CLUSTER_MGR --> VIZ
    
    style APP fill:#e1f5ff
    style CLUSTER_MGR fill:#fff4e1
    style PROJ fill:#e8f5e9
    style DEPTH fill:#f3e5f5
    style FRUSTUM fill:#e8f5e9
```

## Pipeline Flow

```mermaid
flowchart LR
    A[Load Dataset<br/>KITTI/NuScenes] --> B[Extract 2D BBoxes<br/>YOLO/SAM2]
    B --> C[Create Frustums<br/>2D-to-3D Projection]
    C --> D[Load LiDAR Point Cloud]
    D --> E[Ground Plane Removal<br/>RANSAC]
    E --> F[Depth Estimation<br/>Marigold-DC]
    F --> G[3D Point Reconstruction]
    G --> H[Merge with LiDAR]
    H --> I[Filter Points in Frustums]
    I --> J[Clustering<br/>HDBSCAN/DBSCAN/etc]
    J --> K[Pose Estimation<br/>L-Shape/PCA]
    K --> L[3D Cuboid Generation]
    L --> M[Evaluation & Metrics]
    M --> N[Visualization]
    
    style A fill:#e3f2fd
    style J fill:#fff3e0
    style M fill:#f1f8e9
    style N fill:#fce4ec
```

## Module Dependencies

```mermaid
graph TD
    subgraph "Core Modules"
        A[app.py]
        B[pointcloud_projection.py]
        C[frustum_manager.py]
        D[clustering_manager.py]
        E[depth_estimation.py]
    end
    
    subgraph "Supporting Modules"
        F[segmentation_detection.py]
        G[pose_estimation.py]
        H[visualization_helper.py]
        I[evaluation.py]
    end
    
    subgraph "Data Modules"
        J[kitti_dataset_loader.py]
        K[nuscenes_dataset_loader.py]
        L[bounding_boxes.py]
    end
    
    A --> B
    A --> C
    A --> D
    A --> E
    A --> F
    A --> J
    A --> K
    
    C --> B
    C --> D
    F --> L
    E --> M[marigold_dc.py]
    
    D --> G
    D --> H
    D --> I
    
    style A fill:#ffebee
    style D fill:#e8f5e9
    style C fill:#e1f5fe
```

## Key Features

### 1. **Multi-Dataset Support**
   - KITTI dataset loader
   - NuScenes dataset loader
   - Unified data format interface

### 2. **2D Detection & Segmentation**
   - SAM2 (Segment Anything Model 2) integration
   - 2D bounding box extraction
   - Segmentation mask generation

### 3. **Depth Estimation**
   - Marigold-DC for depth completion
   - Sparse depth guidance from LiDAR
   - 3D point reconstruction from depth maps

### 4. **Point Cloud Processing**
   - 2D-to-3D frustum projection
   - Ground plane removal (RANSAC)
   - Frustum-based point filtering

### 5. **Clustering Algorithms**
   - HDBSCAN (Hierarchical DBSCAN)
   - DBSCAN
   - OPTICS
   - BIRCH
   - Agglomerative Clustering

### 6. **3D Object Detection**
   - Pose estimation (L-Shape fitting, PCA)
   - 3D cuboid generation
   - Template-based cuboid sizing (KITTI statistics)

### 7. **Evaluation & Visualization**
   - Comprehensive metrics (IoU, precision, recall)
   - 3D interactive visualization
   - Statistics and comparison tools
   - Ground truth comparison

## Technology Stack

- **Framework**: Streamlit (Web UI)
- **ML Models**: SAM2, Marigold-DC, YOLO
- **Clustering**: scikit-learn (HDBSCAN, DBSCAN, OPTICS, BIRCH, Agglomerative)
- **3D Processing**: Open3D, NumPy
- **Computer Vision**: OpenCV, PIL
- **Visualization**: Plotly, Matplotlib

## File Structure

```
Project/
├── app.py                          # Main Streamlit application
├── clustering_manager.py           # Multi-algorithm clustering manager
├── pointcloud_projection.py        # 2D-to-3D projection & frustum operations
├── frustum_manager.py              # Frustum creation & management
├── depth_estimation.py             # Depth estimation & 3D reconstruction
├── marigold_dc.py                  # Marigold-DC depth completion
├── segmentation_detection.py       # SAM2 segmentation integration
├── pose_estimation.py              # Object pose estimation (L-Shape, PCA)
├── evaluation.py                   # Metrics & evaluation functions
├── visualization_helper.py         # 3D visualization utilities
├── kitti_dataset_loader.py         # KITTI dataset loader
├── nuscenes_dataset_loader.py      # NuScenes dataset loader
├── bounding_boxes.py               # 2D bounding box management
├── pages/                          # Streamlit page modules
│   ├── hdbscan.py
│   ├── dbscan.py
│   ├── optics.py
│   ├── birch.py
│   ├── agglomerative.py
│   ├── statistics.py
│   ├── depth_estimation.py
│   ├── kitti_groundtruth.py
│   └── project_segmentation_mask_on_pointcloud.py
└── dataset/                         # Dataset storage
    ├── kitti/
    └── nuscenes/
```

