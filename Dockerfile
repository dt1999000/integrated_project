# Dockerfile for 3D Object Detection & Clustering Pipeline
# Streamlit-based application for autonomous driving dataset visualization

FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies required for OpenCV, Open3D, and other libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libusb-1.0-0 \
    libx11-6 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY *.py ./
COPY .streamlit/ ./.streamlit/

# Create dataset directories (will be mounted as volumes or contain sample data)
RUN mkdir -p dataset/nuscenes dataset/kitti

# Expose default Streamlit port (cloud platforms may override)
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:${PORT:-8501}/_stcore/health || exit 1

# Set Streamlit configuration
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Default port (can be overridden by cloud platform)
ENV PORT=8501

# Run the Streamlit application with dynamic port
CMD streamlit run app.py --server.port=${PORT} --server.address=0.0.0.0
