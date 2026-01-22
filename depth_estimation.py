from transformers import pipeline
import torch
import numpy as np
from PIL import Image

try:
    from depth_anything_v2.dpt import DepthAnythingV2
    DEPTH_ANYTHING_AVAILABLE = True
except ImportError:
    print("Depth Anything not available")
    DEPTH_ANYTHING_AVAILABLE = False

if DEPTH_ANYTHING_AVAILABLE:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DepthAnythingV2(device=device)
    print("Depth Anything model loaded successfully")
else:
    print("Depth Anything model not available locally")

class DepthEstimator:
    def __init__(self):
        if DEPTH_ANYTHING_AVAILABLE:
            self.model = DepthAnythingV2(device=device)
        else:
            self.pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")

    def get_depth_map(self, image):
        if DEPTH_ANYTHING_AVAILABLE:
            return self.model(image)
        else:
            image = Image.fromarray(image)
            depth_image = self.pipe(image)["depth"]
            depth = np.array(depth_image)
            print(depth)
            return depth_image

