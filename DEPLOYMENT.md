# Deployment Guide - 3D Object Detection Pipeline

Deploy your Streamlit app to a **public URL** for free using one of these platforms.

---

## Quick Comparison

| Platform | Free Tier | Setup Time | Best For |
|----------|-----------|------------|----------|
| **Railway** | $5 credit/month | 5 min | Easiest Docker deployment |
| **Render** | 750 hrs/month | 5 min | Good free tier |
| **Streamlit Cloud** | Free | 3 min | Pure Streamlit apps |
| **Hugging Face Spaces** | Free | 5 min | ML/AI projects |

---

## Option 1: Railway (Recommended)

Railway provides the easiest Docker deployment with a generous free tier.

### Steps:

1. **Push your code to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
   git push -u origin main
   ```

2. **Deploy on Railway**
   - Go to [railway.app](https://railway.app)
   - Sign in with GitHub
   - Click "New Project" → "Deploy from GitHub repo"
   - Select your repository
   - Railway will auto-detect the Dockerfile and deploy

3. **Get your public URL**
   - Go to Settings → Networking
   - Click "Generate Domain"
   - Your app is live at `https://your-app.up.railway.app`

### Dataset Note:
For Railway, you'll need to either:
- Include sample datasets in the repo (small samples only)
- Use cloud storage (S3, GCS) for datasets
- Or configure a persistent volume

---

## Option 2: Render

Render offers a free tier with 750 hours/month.

### Steps:

1. **Push your code to GitHub** (if not already done)

2. **Deploy on Render**
   - Go to [render.com](https://render.com)
   - Sign in with GitHub
   - Click "New +" → "Web Service"
   - Connect your repository
   - Select "Docker" as the environment
   - Click "Create Web Service"

3. **Configure (if needed)**
   - Render will use the `render.yaml` config automatically
   - Or manually set:
     - Build Command: (leave blank for Docker)
     - Start Command: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`

4. **Your public URL**: `https://your-app.onrender.com`

---

## Option 3: Streamlit Community Cloud

Best for pure Streamlit apps (may have issues with Open3D).

### Steps:

1. **Push to GitHub** with this structure:
   ```
   your-repo/
   ├── app.py
   ├── requirements.txt
   ├── .streamlit/
   │   └── config.toml
   └── ... other files
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repo, branch, and `app.py`
   - Click "Deploy"

3. **Your public URL**: `https://your-app.streamlit.app`

### Limitations:
- 1GB memory limit (may not support large point clouds)
- Some C++ dependencies (Open3D) may have issues

---

## Option 4: Hugging Face Spaces

Great for ML projects with GPU support available.

### Steps:

1. **Create a Space**
   - Go to [huggingface.co/spaces](https://huggingface.co/spaces)
   - Click "Create new Space"
   - Select "Streamlit" as the SDK
   - Choose a name and visibility

2. **Upload your code**
   ```bash
   git clone https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE
   cd YOUR_SPACE
   # Copy your files here
   git add .
   git commit -m "Add app"
   git push
   ```

3. **Your public URL**: `https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE`

---

## Dataset Handling for Cloud Deployment

Since datasets (NuScenes, KITTI) are large, here are your options:

### Option A: Sample Data (Recommended for Demo)
Include small sample datasets in your repo:
```
dataset/
├── nuscenes/
│   └── v1.0-mini/  # ~1GB - OK for demos
└── kitti/
    └── training/   # Use first 10-20 samples only
```

### Option B: Cloud Storage
Modify the loaders to fetch from S3/GCS:
```python
# In nuscenes_dataset_loader.py
import boto3
# Download on first access
```

### Option C: Git LFS (Large File Storage)
```bash
git lfs install
git lfs track "*.bin"
git lfs track "*.png"
git add .gitattributes
```

---

## Environment Variables

Set these in your deployment platform:

| Variable | Value | Description |
|----------|-------|-------------|
| `PORT` | (auto-set) | Server port |
| `STREAMLIT_SERVER_HEADLESS` | `true` | No browser popup |
| `STREAMLIT_BROWSER_GATHER_USAGE_STATS` | `false` | Disable telemetry |

---

## Troubleshooting

### "Out of Memory" Error
- Reduce point cloud size in app
- Use sampling for large datasets
- Upgrade to paid tier for more RAM

### "Module not found" Error
- Ensure all dependencies are in `requirements.txt`
- Check that system dependencies are in Dockerfile

### App not starting
- Check logs in the platform's dashboard
- Verify the PORT environment variable is used
- Make sure health check endpoint works

---

## Quick Deploy Commands

### Railway CLI
```bash
npm install -g @railway/cli
railway login
railway init
railway up
```

### Render CLI
```bash
# Use web dashboard - no CLI needed for free tier
```

### Streamlit Cloud
```bash
# No CLI - use web dashboard at share.streamlit.io
```

---

## Share Your App!

Once deployed, share your public URL:
- `https://your-app.up.railway.app`
- `https://your-app.onrender.com`
- `https://your-app.streamlit.app`
