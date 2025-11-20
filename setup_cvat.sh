#!/bin/bash

echo "🚀 CVAT Setup for 3D Point Cloud Clustering Import"
echo "================================================="

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

echo "✅ Docker is running"

# Check if our export files exist
if [ ! -f "cvat_exports/clusters_coco3d.json" ]; then
    echo "❌ CVAT export files not found. Please run clustering with --export-cvat first."
    echo "💡 Run: python example_clustering_vibed.py --export-cvat"
    exit 1
fi

echo "✅ Found CVAT export files"
echo "   - $(ls -1 cvat_exports/ | wc -l) files ready for import"

# Pull Docker images
echo ""
echo "📦 Pulling CVAT Docker images..."
docker pull cvat/server:latest
docker pull cvat/ui:latest

# Start containers
echo ""
echo "🐳 Starting CVAT containers..."
docker-compose up -d

# Wait for CVAT to start
echo ""
echo "⏳ Waiting for CVAT to initialize (this may take 2-3 minutes)..."
sleep 60

# Check if CVAT is ready
for i in {1..30}; do
    if curl -s http://localhost:8080/api/v1/server/health > /dev/null 2>&1; then
        echo "✅ CVAT is ready!"
        break
    fi
    echo "   Waiting... ($i/30)"
    sleep 5
done

# Create admin user (if it doesn't exist)
echo ""
echo "👤 Creating admin user..."
docker exec -it cvat_server bash -c "
python3 manage.py createsuperuser \
  --username admin \
  --email admin@cvat.org \
  --noinput || echo 'Admin user already exists'
"

# Set admin password
docker exec -it cvat_server bash -c "
python3 manage.py shell -c \"from django.contrib.auth.models import User; u=User.objects.get(username='admin'); u.set_password('admin'); u.save()\"
"

echo ""
echo "🎯 CVAT Setup Complete!"
echo "======================"
echo ""
echo "📱 Access CVAT Web Interface:"
echo "   URL: http://localhost:3000"
echo "   Username: admin"
echo "   Password: admin"
echo ""
echo "📊 API Documentation:"
echo "   URL: http://localhost:8080/api/v1/docs"
echo ""
echo "📁 Import Instructions:"
echo "   1. Open http://localhost:3000"
echo "   2. Login with admin/admin"
echo "   3. Click 'Create task'"
echo "   4. Task name: 'PointCloudClusters'"
echo "   5. Click 'Actions' → 'Upload annotations'"
echo "   6. Select 'coco3d.json' format"
echo "   7. Upload cvat_exports/clusters_coco3d.json"
echo ""
echo "🎉 Your clustering results are ready to import!"

# Show export summary
echo ""
echo "📈 Your Export Summary:"
echo "===================="
echo "📁 Export files: $(ls -1 cvat_exports/ | wc -l)"
echo "🎯 Clusters found: $(grep -c '"id"' cvat_exports/clusters_coco3d.json)"
echo "📊 Total file size: $(du -sh cvat_exports/ | cut -f1)"

echo ""
echo "💡 Quick Import Command:"
echo "curl -X POST 'http://localhost:8080/api/v1/tasks' \\"
echo "  -H 'Authorization: Token YOUR_TOKEN' \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{\"name\": \"PointCloudClusters\", \"labels\": [{\"name\": \"cluster\"}]}'"