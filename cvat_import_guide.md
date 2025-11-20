# CVAT Import Guide for 3D Point Cloud Clustering Results

This guide shows how to import your clustering results into CVAT using Docker containers.

## 🐳 Docker Setup

### 1. Pull CVAT Docker Image
```bash
docker pull cvat/server:latest
docker pull cvat/ui:latest
docker pull cvat/lts:latest
```

### 2. Start CVAT Container
```bash
docker-compose -f docker-compose.yml up -d
```

## 📁 Our Export Files

Your clustering results are exported to `cvat_exports/` directory:

### File Formats:
- `clusters_coco3d.json` - **Recommended for CVAT import** (COCO 3D format)
- `clusters_yolo3d.txt` - YOLO 3D format
- `clusters_cvat.xml` - CVAT native XML format
- `clusters_simple.json` - Human-readable format
- `clusters_colored.ply` - Point cloud visualization

## 🚀 Import Methods

### Method 1: CVAT Web Interface (Recommended)

1. **Access CVAT Web UI**
   - Open browser: `http://localhost:8080`
   - Login with admin/admin

2. **Create New Task**
   - Click "Create task"
   - Task name: "Point Cloud Clustering"
   - Labels: "cluster" (or let CVAT auto-detect)

3. **Import Annotations**
   - Go to your task
   - Click "Actions" → "Upload annotations"
   - Select `clusters_coco3d.json`
   - Choose "COCO 1.0" format
   - Click "Upload"

### Method 2: CVAT CLI

```bash
# Create task
docker exec -it cvat_server bash -c "python3 manage.py create_task --name PointCloudClusters --labels cluster"

# Import annotations
docker exec -it cvat_server bash -c "python3 manage.py import_annotations --task_id 1 --format coco --file /app/data/clusters_coco3d.json"
```

### Method 3: REST API

```bash
# Upload task data first, then import annotations
curl -X POST "http://localhost:8080/api/v1/tasks/1/annotations" \
  -H "Authorization: Token YOUR_API_TOKEN" \
  -F "format=coco" \
  -F "annotation_file=@cvat_exports/clusters_coco3d.json"
```

## 📊 Export File Analysis

Our export contains:
- **1278 clusters** detected by BIRCH algorithm
- **3D bounding boxes** with position and dimensions
- **Point cloud metadata** with algorithm parameters
- **Cluster statistics** (density, volume, point count)

### Sample COCO 3D Format:
```json
{
  "annotations": [
    {
      "id": 1,
      "category_id": 1,
      "bbox3d": {
        "location": [x, y, z],
        "dimension": [width, height, depth],
        "rotation": [0, 0, 0]
      },
      "num_points": 52,
      "density": 1.23
    }
  ]
}
```

## 🎯 Next Steps in CVAT

1. **Review Annotations**
   - Navigate through clusters in 3D view
   - Verify cluster boundaries
   - Merge or split clusters if needed

2. **Add Ground Truth**
   - Use cluster suggestions as starting points
   - Refine with manual annotations
   - Add class labels (car, pedestrian, etc.)

3. **Export Training Data**
   - Export in YOLO or COCO format
   - Use for 3D object detection training
   - Generate point cloud datasets

## 🔧 Troubleshooting

**Import Issues:**
- Ensure COCO format matches CVAT requirements
- Check 3D coordinate system alignment
- Verify cluster count matches expectations

**Performance:**
- Large point clouds may need chunking
- Consider sample rate for annotation density
- Use CVAT's filtering options

## 📝 Notes

- Our export uses **LiDAR coordinate system** (origin at sensor)
- Clusters are **axis-aligned bounding boxes** (no rotation)
- **Noise points** are labeled as -1 and excluded from main clusters
- **Density information** is included for each cluster

## 🚀 Quick Start Script

```bash
#!/bin/bash
# Quick CVAT setup and import

echo "🚀 Starting CVAT with our clustering results..."

# Start CVAT
docker-compose up -d

echo "⏳ Waiting for CVAT to start..."
sleep 30

# Create task and import annotations
docker exec -it cvat_server bash -c "
python3 manage.py create_task --name PointCloudClusters --labels cluster
python3 manage.py import_annotations --task_id 1 --format coco --file /data/clusters_coco3d.json
"

echo "✅ Import completed! Visit http://localhost:8080"
```

Save this as `setup_cvat.sh` and run it to quickly set up CVAT with your clustering results.