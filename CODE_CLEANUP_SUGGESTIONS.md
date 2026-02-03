# Code Cleanup Suggestions for Depth Estimation Pipeline

## ✅ Completed Improvements

### 1. Camera Parameters as Object Variables
- **Before**: Camera parameters (`camera_intrinsic`, `camera_extrinsic`, `camera_to_lidar_transform`) were passed as arguments to every method
- **After**: Camera parameters are stored as instance variables in `DepthEstimator` class
- **Benefits**:
  - Cleaner method signatures
  - Reduced parameter passing overhead
  - Easier to maintain and update camera parameters
  - Supports initialization-time or runtime setting via `set_camera_params()`

### 2. Cached Projection Object
- Added `_get_projection()` method that creates and caches a `Projection` object
- Reuses projection object when camera parameters don't change
- Reduces object creation overhead

## 🔧 Further Cleanup Suggestions

### 1. **Extract Constants and Configuration**

```python
# Create a constants file: depth_config.py
class DepthConfig:
    DEFAULT_DEPTH_THRESHOLD_MIN = 0.1
    DEFAULT_DEPTH_THRESHOLD_MAX = 100.0
    DEFAULT_STRIDE = 1
    DEFAULT_PROCESSING_RESOLUTION = 768
    DEFAULT_NUM_INFERENCE_STEPS = 50
    DEFAULT_ENSEMBLE_SIZE = 1
    DEFAULT_SEED = 2024
    
    # CPU fallback values
    CPU_PROCESSING_RESOLUTION = 512
    CPU_NUM_INFERENCE_STEPS = 10
    CPU_ENSEMBLE_SIZE = 1
```

**Benefits**: Centralized configuration, easier to adjust defaults

### 2. **Create Separate Classes for Different Functionality**

Split `DepthEstimator` into focused classes:

```python
class DepthEstimationModel:
    """Handles depth estimation (Marigold/Depth Anything)"""
    def __init__(self, use_marigold=True, ...):
        # Model initialization
    
    def estimate(self, image):
        # Depth estimation logic

class DepthReconstruction:
    """Handles 3D point cloud reconstruction from depth maps"""
    def __init__(self, camera_intrinsic, camera_to_lidar_transform):
        # Camera setup
    
    def reconstruct(self, depth_map, ...):
        # Reconstruction logic

class DepthCompletion:
    """Handles depth completion using Marigold-DC"""
    def __init__(self, ...):
        # DC pipeline setup
    
    def complete(self, image, sparse_depth, ...):
        # Completion logic

class DepthEstimator:
    """Main facade class that coordinates all components"""
    def __init__(self, ...):
        self.estimation_model = DepthEstimationModel(...)
        self.reconstruction = DepthReconstruction(...)
        self.completion = DepthCompletion(...)
```

**Benefits**: Single Responsibility Principle, easier testing, better modularity

### 3. **Add Type Hints Throughout**

```python
from typing import Optional, Tuple, Dict
import numpy.typing as npt

def reconstruct_points_from_depth(
    self, 
    depth_map: npt.NDArray[np.float32],
    depth_threshold_min: float = 0.1,
    depth_threshold_max: float = 100.0,
    stride: int = 1
) -> npt.NDArray[np.float32]:
    ...
```

**Benefits**: Better IDE support, catch type errors early, self-documenting code

### 4. **Extract Validation Logic**

```python
def _validate_camera_params(self):
    """Validate that camera parameters are set."""
    if self.camera_intrinsic is None or self.camera_to_lidar_transform is None:
        raise ValueError(
            "Camera parameters not set. "
            "Initialize with camera params or call set_camera_params() first."
        )

def _validate_depth_map(self, depth_map: np.ndarray):
    """Validate depth map format."""
    if depth_map.ndim != 2:
        raise ValueError(f"Depth map must be 2D, got {depth_map.ndim}D")
    if depth_map.shape[0] == 0 or depth_map.shape[1] == 0:
        raise ValueError("Depth map cannot be empty")
```

**Benefits**: Reusable validation, consistent error messages

### 5. **Use Dataclasses for Return Values**

```python
from dataclasses import dataclass

@dataclass
class DepthEstimationResult:
    depth_map: np.ndarray
    points_lidar: np.ndarray
    metadata: Dict[str, any] = None

def estimate_depth_and_reconstruct(...) -> DepthEstimationResult:
    ...
    return DepthEstimationResult(
        depth_map=depth_map,
        points_lidar=points_lidar,
        metadata={'stride': stride, 'thresholds': (min, max)}
    )
```

**Benefits**: Type-safe return values, easier to extend with metadata

### 6. **Add Logging Instead of Print Statements**

```python
import logging

logger = logging.getLogger(__name__)

# Instead of print()
logger.info(f"Reconstructing point cloud: {len(u)} pixels")
logger.debug(f"Valid depth pixels: {len(depths_valid)}")
logger.warning(f"CUDA not found: Reducing resolution")
```

**Benefits**: Configurable log levels, better for production, can redirect to files

### 7. **Extract Magic Numbers**

```python
# In create_sparse_depth_map
MIN_DEPTH_FOR_PIXEL = 0  # Minimum depth value to consider valid
DEPTH_COMPARISON_THRESHOLD = 0  # When multiple points map to same pixel

# In reconstruct_points_from_depth  
HOMOGENEOUS_COORDINATE = 1.0  # For homogeneous coordinates [u, v, 1]
```

**Benefits**: Self-documenting code, easier to adjust thresholds

### 8. **Add Context Managers for Resource Management**

```python
@contextmanager
def _temporary_projection(self, point_cloud):
    """Context manager for temporary projection objects."""
    projection = self._get_projection(point_cloud)
    try:
        yield projection
    finally:
        # Cleanup if needed
        pass
```

**Benefits**: Guaranteed cleanup, better resource management

### 9. **Create Utility Functions for Common Operations**

```python
def _back_project_pixel_to_3d(
    self, 
    u: np.ndarray, 
    v: np.ndarray, 
    depths: np.ndarray
) -> np.ndarray:
    """Back-project pixels to 3D camera coordinates."""
    K_inv = np.linalg.inv(self.camera_intrinsic)
    pixels_homogeneous = np.stack([u, v, np.ones_like(u)], axis=0)
    points_normalized = K_inv @ pixels_homogeneous
    return (points_normalized * depths).T

def _transform_to_lidar(self, points_camera: np.ndarray) -> np.ndarray:
    """Transform points from camera to LiDAR coordinates."""
    points_homo = np.hstack([points_camera, np.ones((len(points_camera), 1))])
    return (self.camera_to_lidar_transform @ points_homo.T).T[:, :3]
```

**Benefits**: Reusable code, easier to test individual operations

### 10. **Add Progress Callbacks**

```python
from typing import Callable, Optional

def complete_depth(
    self, 
    image: np.ndarray, 
    sparse_depth: np.ndarray,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    ...
) -> np.ndarray:
    """Complete depth with optional progress callback."""
    total_steps = num_inference_steps * ensemble_size
    for step in range(num_inference_steps):
        if progress_callback:
            progress_callback(step, total_steps)
        # ... processing
```

**Benefits**: Better UX for long-running operations, can integrate with UI progress bars

### 11. **Error Handling Improvements**

```python
class DepthEstimationError(Exception):
    """Base exception for depth estimation errors."""
    pass

class CameraNotConfiguredError(DepthEstimationError):
    """Raised when camera parameters are not set."""
    pass

class InvalidDepthMapError(DepthEstimationError):
    """Raised when depth map is invalid."""
    pass

# Usage
if self.camera_intrinsic is None:
    raise CameraNotConfiguredError(
        "Camera parameters not set. Call set_camera_params() first."
    )
```

**Benefits**: More specific error handling, easier debugging

### 12. **Add Unit Tests**

```python
# tests/test_depth_estimation.py
import pytest
import numpy as np
from depth_estimation import DepthEstimator

def test_camera_params_validation():
    estimator = DepthEstimator()
    with pytest.raises(ValueError):
        estimator.reconstruct_points_from_depth(np.zeros((100, 100)))

def test_sparse_depth_map_creation():
    estimator = DepthEstimator(
        camera_intrinsic=np.eye(3),
        camera_to_lidar_transform=np.eye(4)
    )
    point_cloud = np.array([[1, 0, 5], [0, 1, 10]])
    sparse = estimator.create_sparse_depth_map(point_cloud, (100, 100))
    assert sparse.shape == (100, 100)
```

**Benefits**: Catch regressions, document expected behavior

### 13. **Documentation Improvements**

```python
def reconstruct_points_from_depth(
    self, 
    depth_map: np.ndarray, 
    depth_threshold_min: float = 0.1,
    depth_threshold_max: float = 100.0,
    stride: int = 1
) -> np.ndarray:
    """
    Reconstruct 3D point cloud from metric depth map.
    
    This method back-projects depth values to 3D camera coordinates
    and transforms them to LiDAR coordinate system.
    
    Args:
        depth_map: Metric depth map (H, W) in meters. Must be 2D array.
        depth_threshold_min: Minimum valid depth value in meters. 
                           Points with depth < threshold are filtered out.
        depth_threshold_max: Maximum valid depth value in meters.
                           Points with depth > threshold are filtered out.
        stride: Sampling stride for point cloud. 
               - 1: Use all pixels
               - 2: Use every other pixel (50% reduction)
               - N: Use every Nth pixel
    
    Returns:
        points_lidar: Nx3 array of 3D points in LiDAR coordinate system.
                     Points are ordered by pixel position (row-major).
    
    Raises:
        ValueError: If camera parameters are not set.
        ValueError: If depth_map is not 2D.
    
    Example:
        >>> estimator = DepthEstimator(
        ...     camera_intrinsic=K,
        ...     camera_to_lidar_transform=T
        ... )
        >>> depth_map = np.load('depth.npy')
        >>> points = estimator.reconstruct_points_from_depth(
        ...     depth_map, 
        ...     stride=2  # Subsample for faster processing
        ... )
        >>> print(f"Reconstructed {len(points)} points")
    """
```

**Benefits**: Better API documentation, examples help users

### 14. **Performance Optimizations**

```python
# Use vectorized operations where possible
# Instead of:
for (u, v), depth in zip(valid_pixels, depths):
    if sparse_depth[v, u] == 0 or depth < sparse_depth[v, u]:
        sparse_depth[v, u] = depth

# Use:
# Group by pixel coordinates and take minimum depth
from scipy.sparse import coo_matrix
coo = coo_matrix((depths, (valid_pixels[:, 1], valid_pixels[:, 0])), 
                 shape=sparse_depth.shape)
sparse_depth = np.minimum.reduceat(coo.toarray(), ...)  # More efficient
```

**Benefits**: Faster execution, especially for large point clouds

### 15. **Configuration Management**

```python
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class DepthEstimatorConfig:
    use_marigold: bool = True
    use_full_precision: bool = False
    use_tiny_vae: bool = False
    camera_intrinsic: Optional[np.ndarray] = None
    camera_extrinsic: Optional[np.ndarray] = None
    camera_to_lidar_transform: Optional[np.ndarray] = None
    default_depth_threshold_min: float = 0.1
    default_depth_threshold_max: float = 100.0
    default_stride: int = 1

class DepthEstimator:
    def __init__(self, config: DepthEstimatorConfig = None):
        if config is None:
            config = DepthEstimatorConfig()
        self.config = config
        # ... initialization
```

**Benefits**: Easier to serialize/deserialize configs, better defaults management

## Summary

The refactoring has already improved code cleanliness significantly by:
1. ✅ Removing repetitive parameter passing
2. ✅ Caching projection objects
3. ✅ Centralizing camera parameter management

The suggested improvements would further enhance:
- **Maintainability**: Better structure, documentation, error handling
- **Testability**: Separated concerns, unit tests
- **Performance**: Optimized operations, better resource management
- **Usability**: Better error messages, progress callbacks, type hints

Consider implementing these improvements incrementally, prioritizing based on your project's needs.


