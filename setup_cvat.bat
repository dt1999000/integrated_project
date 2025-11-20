@echo off
echo 🚀 CVAT Setup for 3D Point Cloud Clustering Import
echo =================================================

REM Check if Docker is running
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker is not running. Please start Docker Desktop first.
    pause
    exit /b 1
)

echo ✅ Docker is running

REM Check if our export files exist
if not exist "cvat_exports\clusters_coco3d.json" (
    echo ❌ CVAT export files not found. Please run clustering with --export-cvat first.
    echo 💡 Run: python example_clustering_vibed.py --export-cvat
    pause
    exit /b 1
)

echo ✅ Found CVAT export files
dir cvat_exports\ | find "File(s)"

REM Pull Docker images
echo.
echo 📦 Pulling CVAT Docker images...
docker pull cvat/server:latest
docker pull cvat/ui:latest

REM Start containers
echo.
echo 🐳 Starting CVAT containers...
docker-compose up -d

REM Wait for CVAT to start
echo.
echo ⏳ Waiting for CVAT to initialize (this may take 2-3 minutes)...
timeout /t 60 /nobreak >nul

REM Check if CVAT is ready
for /l %%i in (1,1,30) do (
    curl -s http://localhost:8080/api/v1/server/health >nul 2>&1
    if !errorlevel! equ 0 (
        echo ✅ CVAT is ready!
        goto :cvat_ready
    )
    echo    Waiting... (%%i/30)
    timeout /t 5 /nobreak >nul
)

:cvat_ready
REM Create admin user (if it doesn't exist)
echo.
echo 👤 Creating admin user...
docker exec cvat_server bash -c "python3 manage.py createsuperuser --username admin --email admin@cvat.org --noinput || echo 'Admin user already exists'"

REM Set admin password
docker exec cvat_server bash -c "python3 manage.py shell -c \"from django.contrib.auth.models import User; u=User.objects.get(username='admin'); u.set_password('admin'); u.save()\""

echo.
echo 🎯 CVAT Setup Complete!
echo ======================
echo.
echo 📱 Access CVAT Web Interface:
echo    URL: http://localhost:3000
echo    Username: admin
echo    Password: admin
echo.
echo 📊 API Documentation:
echo    URL: http://localhost:8080/api/v1/docs
echo.
echo 📁 Import Instructions:
echo    1. Open http://localhost:3000
echo    2. Login with admin/admin
echo    3. Click 'Create task'
echo    4. Task name: 'PointCloudClusters'
echo    5. Click 'Actions' ^> 'Upload annotations'
echo    6. Select 'coco3d.json' format
echo    7. Upload cvat_exports\clusters_coco3d.json
echo.
echo 🎉 Your clustering results are ready to import!
echo.
echo 📈 Your Export Summary:
echo ====================
dir cvat_exports\
echo.
echo 💡 Press any key to open CVAT in your browser...
pause >nul
start http://localhost:3000