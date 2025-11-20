# 🚀 CVAT Integration for 3D Point Cloud Clustering

Complete workflow to export clustering results and import them into CVAT for annotation and quality control.

## 📋 Overview

This system provides:
- **Automatic CVAT export** from clustering results
- **Multiple format support** (COCO 3D, YOLO 3D, CVAT XML, JSON, PLY)
- **Docker-based CVAT setup** for easy deployment
- **One-click import** of clustering annotations
- **Real dataset integration** with nuScenes

## 🎯 Quick Start

### 1. Run Clustering with CVAT Export
```bash
# Quick mode with CVAT export
python example_clustering_vibed.py --quick --export-cvat --no-visualization

# Or with specific algorithms
python example_clustering_vibed.py --algorithms dbscan birch --export-cvat
```

### 2. Start CVAT Container
```bash
# On Windows
setup_cvat.bat

# On Linux/Mac
bash setup_cvat.sh
```

### 3. Import Annotations
1. Open **http://localhost:3000**
2. Login with **admin/admin**
3. Create new task → "PointCloudClusters"
4. Upload `cvat_exports/clusters_coco3d.json` as COCO format

## 📁 Export Files

After clustering, you'll find these files in `cvat_exports/`:

| File | Format | Description | Best For |
|------|--------|-------------|----------|
| `clusters_coco3d.json` | COCO 3D | **Rich metadata + 3D boxes** | **CVAT Import** |
| `clusters_yolo3d.txt` | YOLO 3D | Training format | Model Training |
| `clusters_cvat.xml` | CVAT XML | Native format | CVAT Backup |
| `clusters_simple.json` | Simple JSON | Human-readable | Analysis |
| `clusters_colored.ply` | PLY | Point cloud + colors | Visualization |

## 🐳 CVAT Container Setup

### Prerequisites
- Docker Desktop installed and running
- At least 4GB RAM available
- 10GB disk space

### Docker Compose Configuration
```yaml
# docker-compose.yml
services:
  cvat_server:
    image: cvat/server:latest
    ports:
      - "8080:8080"
    volumes:
      - ./cvat_exports:/data/exports:ro  # Mount our exports

  cvat_ui:
    image: cvat/ui:latest
    ports:
      - "3000:80"
    depends_on:
      - cvat_server
```

## 📊 Export Data Structure

### COCO 3D Format (Recommended)
```json
{
  "annotations": [
    {
      "id": 1,
      "category_id": 1,
      "bbox3d": {
        "location": [x, y, z],        # Cluster centroid
        "dimension": [w, h, d],       # Bounding box size
        "rotation": [0, 0, 0]        # No rotation (axis-aligned)
      },
      "num_points": 52,              # Points in cluster
      "density": 1.23,               # Point density
      "points": [[x1,y1,z1], ...]   # All cluster points
    }
  ],
  "categories": [
    {"id": 1, "name": "cluster_0", "supercategory": "point_cloud_cluster"}
  ]
}
```

### YOLO 3D Format
```
# class_id center_x center_y center_z width height depth rotation_x rotation_y rotation_z
0 -5.579899 0.245231 -0.953457 0.177972 1.639992 0.541208 0 0 0
1 0.001334 -0.126385 -0.188773 1.262628 2.932315 0.919253 0 0 0
```

## 🎨 Clustering Results

### Recent Example (nuScenes Dataset)
- **Algorithm**: BIRCH (threshold=0.7, branching_factor=20)
- **Points processed**: 19,441 LiDAR points
- **Clusters found**: 1,278 clusters
- **Processing time**: 284.15 seconds
- **Noise ratio**: 0.0% (all points clustered)

### Cluster Statistics
- **Average points per cluster**: ~15 points
- **Cluster density**: 0.5-5.0 points/m³
- **Bounding box sizes**: 0.1-3.0 meters
- **Height distribution**: Ground level to ~3m

## 🔧 Advanced Usage

### Custom Export Options
```bash
# Custom output directory
python example_clustering_vibed.py --export-cvat --cvat-output-dir my_annotations

# Custom filename prefix
python example_clustering_vibed.py --export-cvat --cvat-prefix nuscenes_sample

# Skip PLY export (smaller files)
python example_clustering_vibed.py --export-cvat --no-ply-export
```

### Integration with Existing CVAT Setup
```bash
# Add export files to existing CVAT volume
docker cp cvat_exports/ cvat_server:/data/annotations/

# Import via CLI
docker exec cvat_server python3 manage.py import_annotations \
  --task_id 1 --format coco --file /data/annotations/clusters_coco3d.json
```

## 📈 Quality Control in CVAT

### Review Clusters
1. **Visual Inspection**: Check cluster boundaries in 3D view
2. **Merge Clusters**: Combine adjacent clusters that should be single objects
3. **Split Clusters**: Divide large clusters into separate objects
4. **Add Classes**: Assign semantic labels (car, pedestrian, etc.)

### Common Issues & Solutions
- **Over-segmentation**: Too many small clusters → Increase eps or decrease min_samples
- **Under-segmentation**: Few large clusters → Decrease eps or increase min_samples
- **Noise points**: Unassigned points → Adjust density parameters

## 🚢 Export Workflow

### Complete Pipeline
```
1. nuScenes Dataset
   ↓
2. 2D Bounding Box Extraction
   ↓
3. 2D→3D Projection
   ↓
4. Ground Plane Removal
   ↓
5. Multi-Algorithm Clustering
   ↓
6. Best Algorithm Selection (BIRCH)
   ↓
7. CVAT Export (5 formats)
   ↓
8. CVAT Import & Quality Control
   ↓
9. Export for Training (YOLO/COCO)
```

## 📚 API Integration

### REST API Usage
```python
import requests

# Create CVAT task
task_data = {
    "name": "PointCloudClusters",
    "labels": [{"name": "cluster"}]
}
response = requests.post(
    "http://localhost:8080/api/v1/tasks",
    json=task_data,
    headers={"Authorization": "Token YOUR_TOKEN"}
)

# Upload annotations
with open("cvat_exports/clusters_coco3d.json", "rb") as f:
    files = {"annotation_file": f}
    response = requests.post(
        f"http://localhost:8080/api/v1/tasks/{task_id}/annotations",
        files=files,
        data={"format": "coco"},
        headers={"Authorization": "Token YOUR_TOKEN"}
    )
```

## 🎯 Next Steps

### Training Data Preparation
1. **Export annotated data** from CVAT
2. **Split into train/val/test** sets
3. **Convert to training format** (YOLO, PyTorch, etc.)
4. **Train 3D object detection** models

### Model Evaluation
1. **Import model predictions** to CVAT
2. **Compare with ground truth**
3. **Calculate metrics** (IoU, precision, recall)
4. **Iterate and improve** clustering parameters

## 🛠️ Troubleshooting

### CVAT Import Issues
- **Format mismatch**: Ensure COCO 3D format compatibility
- **Coordinate system**: Check LiDAR vs camera coordinates
- **Memory limits**: Large point clouds may need chunking

### Docker Issues
- **Port conflicts**: Change ports in docker-compose.yml
- **Memory limits**: Increase Docker Desktop memory allocation
- **Network issues**: Check firewall settings

### Clustering Issues
- **Too few clusters**: Adjust algorithm parameters
- **Too many clusters**: Increase density thresholds
- **Performance issues**: Reduce point cloud density or sample rate

## 📞 Support

- **CVAT Documentation**: https://opencv.github.io/cvat/
- **Docker Hub**: https://hub.docker.com/u/cvat
- **Issues**: Report clustering or export issues in project repository

---

**Ready to annotate!** 🎉 Your clustering results are now ready for import into CVAT.