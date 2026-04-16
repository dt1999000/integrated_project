"""
SUNRGBD dataset loader.

This loader indexes SUNRGBD scenes and provides a normalized sample output
compatible with the rest of the pipeline:
  - RGB image stream: HxWx3 RGB numpy array
  - Point cloud stream: Nx3 numpy array reconstructed from depth
  - Metadata with camera intrinsics and optional ground-truth boxes
"""

from pathlib import Path
from typing import Dict, List, Optional
import json

import cv2
import numpy as np

from components.core.pointcloud_projection import PointCloud, load_sunrgbd_intrinsics


class SUNRGBDDatasetLoader:
    """
    Loader for SUNRGBD directory trees.

    Expected per-scene layout:
      <scene_root>/
        image/<name>.jpg
        depth_bfx/<name>.png
        intrinsics.txt
        annotation2D3D/index.json   (optional)
    """

    def __init__(self, dataroot: str):
        self.dataroot = Path(dataroot)
        self.samples: List[Dict] = []

    def load_dataset(self) -> List[Dict]:
        if not self.dataroot.exists():
            raise FileNotFoundError(f"SUNRGBD root not found: {self.dataroot}")

        self.samples = []
        intrinsics_files = sorted(self.dataroot.glob("**/intrinsics.txt"))
        for intr_file in intrinsics_files:
            scene_root = intr_file.parent
            image_dir = scene_root / "image"
            depth_dir = scene_root / "depth_bfx"
            if not image_dir.exists() or not depth_dir.exists():
                continue

            image_files = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".png"}])
            depth_files = sorted([p for p in depth_dir.iterdir() if p.suffix.lower() == ".png"])
            if len(image_files) == 0 or len(depth_files) == 0:
                continue

            image_path = image_files[0]
            depth_path = depth_files[0]
            annotation_path = scene_root / "annotation2D3D" / "index.json"
            if not annotation_path.exists():
                annotation_path = None

            self.samples.append(
                {
                    "scene_root": str(scene_root),
                    "scene_id": scene_root.name,
                    "image_path": str(image_path),
                    "depth_path": str(depth_path),
                    "intrinsics_path": str(intr_file),
                    "annotation_path": str(annotation_path) if annotation_path else None,
                }
            )

        return self.samples

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _load_ground_truth_boxes(annotation_path: Optional[str], image_shape: tuple) -> List[Dict]:
        if annotation_path is None:
            return []

        with open(annotation_path, "r", encoding="utf-8") as f:
            ann = json.load(f)

        objects = ann.get("objects", [])
        frames = ann.get("frames", [])
        frame_polygons = []
        if len(frames) > 0:
            frame_polygons = frames[0].get("polygon", [])

        h, w = image_shape[:2]
        gt_boxes: List[Dict] = []

        for poly2d in frame_polygons:
            obj_idx = poly2d.get("object")
            if obj_idx is None or obj_idx < 0 or obj_idx >= len(objects):
                continue

            obj = objects[obj_idx]
            name = obj.get("name", "Unknown")
            base_name = name.split(":")[0] if ":" in name else name
            poly3d_list = obj.get("polygon", [])
            if len(poly3d_list) == 0:
                continue

            poly3d = poly3d_list[0]
            xs = np.asarray(poly3d.get("X", []), dtype=np.float64)
            zs = np.asarray(poly3d.get("Z", []), dtype=np.float64)
            ymin = float(poly3d.get("Ymin", 0.0))
            ymax = float(poly3d.get("Ymax", 0.0))
            if xs.size == 0 or zs.size == 0:
                continue

            # SUNRGBD stores "up/down" limits in Ymin/Ymax; normalize to min/max.
            y_low = min(ymin, ymax)
            y_high = max(ymin, ymax)

            x2d = np.asarray(poly2d.get("x", []), dtype=np.float64)
            y2d = np.asarray(poly2d.get("y", []), dtype=np.float64)
            bbox_2d = None
            if x2d.size > 0 and y2d.size > 0:
                left = int(np.clip(np.min(x2d), 0, w - 1))
                right = int(np.clip(np.max(x2d), 0, w - 1))
                top = int(np.clip(np.min(y2d), 0, h - 1))
                bottom = int(np.clip(np.max(y2d), 0, h - 1))
                if right > left and bottom > top:
                    bbox_2d = {"left": left, "top": top, "right": right, "bottom": bottom}

            gt_boxes.append(
                {
                    "category": base_name if base_name else "Unknown",
                    "class": base_name if base_name else "Unknown",
                    "bbox_2d": bbox_2d,
                    "min_x": float(np.min(xs)),
                    "max_x": float(np.max(xs)),
                    "min_y": float(y_low),
                    "max_y": float(y_high),
                    "min_z": float(np.min(zs)),
                    "max_z": float(np.max(zs)),
                }
            )

        return gt_boxes

    @staticmethod
    def _sunrgbd_camera_to_pipeline_lidar_transform() -> np.ndarray:
        """
        Return a rigid transform that maps SUNRGBD camera coordinates
        (x-right, y-down, z-front) to the pipeline LiDAR-like frame
        (x-front, y-left, z-up).

        Mapping:
          x_l =  z_c
          y_l = -x_c
          z_l = -y_c
        """
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = np.array(
            [
                [0.0, 0.0, 1.0],
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
            ],
            dtype=np.float64,
        )
        return transform

    @staticmethod
    def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points.reshape(-1, 3)
        pts = np.asarray(points, dtype=np.float64)
        ones = np.ones((pts.shape[0], 1), dtype=np.float64)
        pts_h = np.hstack([pts[:, :3], ones])
        pts_t = (transform @ pts_h.T).T[:, :3]
        return pts_t.astype(np.float32)

    @staticmethod
    def _transform_gt_boxes_to_pipeline_lidar(gt_boxes: List[Dict], transform: np.ndarray) -> List[Dict]:
        """
        Convert axis-aligned SUNRGBD camera-frame min/max boxes into
        axis-aligned bounds in the pipeline LiDAR-like frame.
        """
        transformed_boxes: List[Dict] = []
        for box in gt_boxes:
            min_x, max_x = float(box["min_x"]), float(box["max_x"])
            min_y, max_y = float(box["min_y"]), float(box["max_y"])
            min_z, max_z = float(box["min_z"]), float(box["max_z"])

            corners_cam = np.array(
                [
                    [min_x, min_y, min_z],
                    [max_x, min_y, min_z],
                    [max_x, max_y, min_z],
                    [min_x, max_y, min_z],
                    [min_x, min_y, max_z],
                    [max_x, min_y, max_z],
                    [max_x, max_y, max_z],
                    [min_x, max_y, max_z],
                ],
                dtype=np.float64,
            )
            corners_lidar = SUNRGBDDatasetLoader._transform_points(corners_cam, transform)

            transformed_box = dict(box)
            transformed_box["min_x"] = float(np.min(corners_lidar[:, 0]))
            transformed_box["max_x"] = float(np.max(corners_lidar[:, 0]))
            transformed_box["min_y"] = float(np.min(corners_lidar[:, 1]))
            transformed_box["max_y"] = float(np.max(corners_lidar[:, 1]))
            transformed_box["min_z"] = float(np.min(corners_lidar[:, 2]))
            transformed_box["max_z"] = float(np.max(corners_lidar[:, 2]))
            transformed_boxes.append(transformed_box)
        return transformed_boxes

    def load_sunrgbd_data(
        self,
        sample_index: int,
        depth_scale: float = 10000.0,
        depth_trunc: float = 10.0,
        stride: int = 1,
        keep_fraction: float = 0.8,
    ) -> Dict:
        if len(self.samples) == 0:
            self.load_dataset()

        if sample_index < 0 or sample_index >= len(self.samples):
            raise IndexError(f"SUNRGBD sample_index out of range: {sample_index} (n={len(self.samples)})")

        sample = self.samples[sample_index]
        image_bgr = cv2.imread(sample["image_path"])
        if image_bgr is None:
            raise FileNotFoundError(f"Could not read image: {sample['image_path']}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        depth_img = cv2.imread(sample["depth_path"], cv2.IMREAD_UNCHANGED)
        if depth_img is None:
            raise FileNotFoundError(f"Could not read depth: {sample['depth_path']}")

        camera_intrinsic = load_sunrgbd_intrinsics(sample["intrinsics_path"])
        point_cloud_obj = PointCloud.from_rgbd_sunrgbd(
            rgb_image=image_rgb,
            depth_image=depth_img,
            camera_intrinsic=camera_intrinsic,
            depth_scale=depth_scale,
            depth_trunc=depth_trunc,
            stride=stride,
            keep_fraction=keep_fraction,
        )

        gt_boxes_cam = self._load_ground_truth_boxes(sample.get("annotation_path"), image_rgb.shape)
        camera_to_lidar_transform = self._sunrgbd_camera_to_pipeline_lidar_transform()
        point_cloud_lidar = self._transform_points(
            point_cloud_obj.original_point_cloud,
            camera_to_lidar_transform,
        )
        gt_boxes = self._transform_gt_boxes_to_pipeline_lidar(
            gt_boxes_cam,
            camera_to_lidar_transform,
        )

        return {
            "sample_index": sample_index,
            "scene_id": sample["scene_id"],
            "image_path": sample["image_path"],
            "depth_path": sample["depth_path"],
            "intrinsics_path": sample["intrinsics_path"],
            "point_cloud": point_cloud_lidar,
            "point_cloud_colors": getattr(point_cloud_obj, "colors", None),
            "camera_intrinsic": camera_intrinsic,
            "camera_extrinsic": np.eye(4, dtype=np.float64),
            "camera_to_lidar_transform": camera_to_lidar_transform,
            "ground_truth_boxes": gt_boxes,
            "dataset_type": "sunrgbd",
            "image_shape": image_rgb.shape[:2],
            "depth_scale": depth_scale,
            "depth_trunc": depth_trunc,
            "stride": stride,
            "keep_fraction": keep_fraction,
        }
