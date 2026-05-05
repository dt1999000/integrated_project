"""
SUNRGBD dataset loader.

This loader indexes SUNRGBD scenes and provides a normalized sample output
compatible with the rest of the pipeline:
  - RGB image stream: HxWx3 RGB numpy array
  - Point cloud stream: Nx3 numpy array loaded from depth .mat
  - Metadata with camera intrinsics and optional ground-truth boxes
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re

import cv2
import numpy as np
from scipy.io import loadmat

from components.core.pointcloud_projection import load_sunrgbd_calibration

SUNRGBD_KEEP_FRACTION_SESSION_KEY = "sunrgbd_keep_fraction"
SUNRGBD_KEEP_FRACTION_DEFAULT = 0.8

# Official SUNRGBD trainval validation range (1-based scene index; paper-comparable).
# After numeric sorting of scene IDs, this is samples[0]..samples[5049] (0-based).
SUNRGBD_VAL_SPLIT_1BASED_START = 1
SUNRGBD_VAL_SPLIT_1BASED_END = 5050


def sunrgbd_val_split_indices_zero_based(num_samples: int) -> List[int]:
    """
    0-based indices for the standard validation split (1-based scenes 1..5050).

    Assumes ``load_dataset`` orders ``sample_ids`` with numeric sort so index
    :math:`k` matches scene order used in common benchmarks.
    """
    if num_samples <= 0:
        return []
    start_z = SUNRGBD_VAL_SPLIT_1BASED_START - 1
    end_excl = min(num_samples, SUNRGBD_VAL_SPLIT_1BASED_END)
    if start_z >= end_excl:
        return []
    return list(range(start_z, end_excl))


def sunrgbd_keep_fraction_for_load() -> float:
    """
    Return the fraction of SUNRGBD depth points to keep (0–1], from Streamlit session.

    Safe when the key is missing (e.g. user never opened Dataset Extraction): uses
    ``SUNRGBD_KEEP_FRACTION_DEFAULT``. Invalid stored values fall back to the default.
    """
    import streamlit as st

    v = st.session_state.get(
        SUNRGBD_KEEP_FRACTION_SESSION_KEY, SUNRGBD_KEEP_FRACTION_DEFAULT
    )
    if v is None:
        return float(SUNRGBD_KEEP_FRACTION_DEFAULT)
    x = float(v)
    if x <= 0.0 or x > 1.0:
        return float(SUNRGBD_KEEP_FRACTION_DEFAULT)
    return x


class SUNRGBDDatasetLoader:
    """
    Loader for SUNRGBD directory trees.

    Expected SUNRGBD trainval layout:
      <dataroot>/sunrgbd_trainval/
        calib/<id>.txt
        depth/<id>.mat
        image/<id>.jpg
        label_v1/<id>.txt   (preferred; matches upright-depth .mat point clouds)
        label/<id>.txt      (fallback, version 2)
    """

    def __init__(self, dataroot: str):
        self.dataroot = Path(dataroot)
        self.samples: List[Dict] = []

    def load_dataset(self) -> List[Dict]:
        if not self.dataroot.exists():
            raise FileNotFoundError(f"SUNRGBD root not found: {self.dataroot}")

        self.samples = []
        trainval_root = self.dataroot / "sunrgbd_trainval"
        data_root = trainval_root if trainval_root.exists() else self.dataroot

        calib_dir = data_root / "calib"
        depth_dir = data_root / "depth"
        image_dir = data_root / "image"
        label_dir = data_root / "label"
        label_v1_dir = data_root / "label_v1"

        if not calib_dir.exists() or not depth_dir.exists() or not image_dir.exists():
            return self.samples

        image_map: Dict[str, Path] = {}
        for p in sorted(image_dir.iterdir()):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                image_map[p.stem] = p

        depth_map: Dict[str, Path] = {}
        for p in sorted(depth_dir.iterdir()):
            if p.suffix.lower() == ".mat":
                depth_map[p.stem] = p

        calib_map: Dict[str, Path] = {}
        for p in sorted(calib_dir.iterdir()):
            if p.suffix.lower() == ".txt":
                calib_map[p.stem] = p

        raw_ids = set(image_map.keys()) & set(depth_map.keys()) & set(calib_map.keys())
        # Numeric order matches standard train/val index conventions (1..N), not lexicographic.
        sample_ids = sorted(
            raw_ids, key=lambda s: (0, int(s)) if str(s).isdigit() else (1, str(s))
        )
        for sample_id in sample_ids:
            annotation_v2_path = label_dir / f"{sample_id}.txt"
            annotation_v1_path = label_v1_dir / f"{sample_id}.txt"
            if annotation_v1_path.exists():
                annotation_path = annotation_v1_path
            elif annotation_v2_path.exists():
                annotation_path = annotation_v2_path
            else:
                annotation_path = None
            self.samples.append(
                {
                    "scene_root": str(data_root),
                    "scene_id": sample_id,
                    "image_path": str(image_map[sample_id]),
                    "depth_path": str(depth_map[sample_id]),
                    "intrinsics_path": str(calib_map[sample_id]),
                    "annotation_path": str(annotation_path) if annotation_path is not None else None,
                }
            )

        return self.samples

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _load_ground_truth_boxes(annotation_path: Optional[str], image_shape: tuple) -> List[Dict]:
        if annotation_path is None:
            return []

        h, w = image_shape[:2]
        gt_boxes: List[Dict] = []
        with open(annotation_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        float_pattern = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")
        for line in lines:
            tokens = line.split()
            if len(tokens) < 2:
                continue

            label = tokens[0]
            numeric_values = [float(tok) for tok in tokens[1:] if float_pattern.match(tok)]
            if len(numeric_values) < 6:
                continue

            bbox_2d = None
            if len(numeric_values) >= 4:
                # SUNRGBD raw labels store 2D boxes as (x, y, w, h), not (x1, y1, x2, y2).
                x, y, bw, bh = numeric_values[0], numeric_values[1], numeric_values[2], numeric_values[3]
                left = int(np.clip(round(x), 0, w - 1))
                top = int(np.clip(round(y), 0, h - 1))
                right = int(np.clip(round(x + bw), 0, w - 1))
                bottom = int(np.clip(round(y + bh), 0, h - 1))
                if right > left and bottom > top:
                    bbox_2d = {"left": left, "top": top, "right": right, "bottom": bottom}

            # Raw SUNRGBD text labels:
            # <class> x y w h cx cy cz sx sy sz ox oy
            # 3D is in upright-depth frame: X right, Y forward, Z up.
            if len(numeric_values) >= 10:
                cx, cy, cz = numeric_values[4], numeric_values[5], numeric_values[6]
                sx, sy, sz = numeric_values[7], numeric_values[8], numeric_values[9]

                # In SUNRGBD labels, coeffs are half-sizes (not full box lengths).
                half_x = float(abs(sx))
                half_y = float(abs(sy))
                half_z = float(abs(sz))

                # Orientation is a 2D heading vector on the ground plane.
                # For upright-depth, use yaw around +Z.
                if len(numeric_values) >= 12:
                    ox, oy = float(numeric_values[10]), float(numeric_values[11])
                    # SUNRGBD orientation vector is defined in upright-depth XY, but our
                    # cuboid local axis convention is rotated by +90° relative to that
                    # heading. Apply the offset so boxes align with visible objects.
                    yaw = float(np.arctan2(oy, ox) - (0.5 * np.pi))
                else:
                    ox, oy = 1.0, 0.0
                    yaw = 0.0

                local_corners = np.array(
                    [
                        [-half_x, -half_y, -half_z],
                        [half_x, -half_y, -half_z],
                        [half_x, half_y, -half_z],
                        [-half_x, half_y, -half_z],
                        [-half_x, -half_y, half_z],
                        [half_x, -half_y, half_z],
                        [half_x, half_y, half_z],
                        [-half_x, half_y, half_z],
                    ],
                    dtype=np.float64,
                )
                cos_yaw = float(np.cos(yaw))
                sin_yaw = float(np.sin(yaw))
                rot_z = np.array(
                    [
                        [cos_yaw, -sin_yaw, 0.0],
                        [sin_yaw, cos_yaw, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                    dtype=np.float64,
                )
                center = np.array([cx, cy, cz], dtype=np.float64)
                corners = (local_corners @ rot_z.T) + center

                min_x = float(np.min(corners[:, 0]))
                max_x = float(np.max(corners[:, 0]))
                min_y = float(np.min(corners[:, 1]))
                max_y = float(np.max(corners[:, 1]))
                min_z = float(np.min(corners[:, 2]))
                max_z = float(np.max(corners[:, 2]))
            else:
                min_x, min_y, min_z, max_x, max_y, max_z = numeric_values[-6:]
                ox, oy = 1.0, 0.0
                yaw = 0.0
                corners = None

            gt_boxes.append(
                {
                    "category": label if label else "Unknown",
                    "class": label if label else "Unknown",
                    "bbox_2d": bbox_2d,
                    "center": [float(cx), float(cy), float(cz)] if len(numeric_values) >= 10 else None,
                    "size": [
                        float(2.0 * half_x),
                        float(2.0 * half_y),
                        float(2.0 * half_z),
                    ]
                    if len(numeric_values) >= 10
                    else None,
                    "orientation": [float(ox), float(oy)],
                    "yaw": float(yaw),
                    "corners": corners.astype(np.float32) if corners is not None else None,
                    "min_x": float(min(min_x, max_x)),
                    "max_x": float(max(min_x, max_x)),
                    "min_y": float(min(min_y, max_y)),
                    "max_y": float(max(min_y, max_y)),
                    "min_z": float(min(min_z, max_z)),
                    "max_z": float(max(min_z, max_z)),
                }
            )

        return gt_boxes

    @staticmethod
    def _load_point_cloud_from_mat(depth_mat_path: str) -> Dict[str, Optional[np.ndarray]]:
        mat_data = loadmat(depth_mat_path)
        candidate_arrays: List[np.ndarray] = []
        for key, value in mat_data.items():
            if key.startswith("__"):
                continue
            if isinstance(value, np.ndarray) and value.ndim == 2:
                candidate_arrays.append(value)

        if len(candidate_arrays) == 0:
            return {"points": np.zeros((0, 3), dtype=np.float32), "colors": None}

        best = max(candidate_arrays, key=lambda arr: int(arr.shape[0] * arr.shape[1]))
        arr = np.asarray(best, dtype=np.float32)
        if arr.shape[0] in {3, 6} and arr.shape[1] > arr.shape[0]:
            arr = arr.T

        if arr.shape[1] < 3:
            return {"points": np.zeros((0, 3), dtype=np.float32), "colors": None}

        points = arr[:, :3].astype(np.float32)
        colors = None
        if arr.shape[1] >= 6:
            rgb = arr[:, 3:6]
            if np.max(rgb) <= 1.0:
                rgb = np.clip(rgb * 255.0, 0.0, 255.0)
            colors = rgb.astype(np.uint8)

        return {"points": points, "colors": colors}

    @staticmethod
    def _subsample_point_cloud(
        points: np.ndarray,
        colors: Optional[np.ndarray],
        stride: int,
        keep_fraction: float,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Reduce point count from SUNRGBD ``depth/*.mat`` clouds.

        ``stride`` keeps every n-th row (same ordering as in the .mat array).
        ``keep_fraction`` randomly retains that fraction of the remaining points
        (matches the intent of the extraction-page slider / RGBD path).
        """
        if points.size == 0:
            return points, colors

        pts = points
        clr = colors
        step = max(1, int(stride))
        if step > 1:
            idx_stride = np.arange(0, len(pts), step, dtype=int)
            pts = pts[idx_stride]
            if clr is not None:
                clr = clr[idx_stride]

        n = len(pts)
        frac = float(keep_fraction)
        if frac >= 1.0 or n == 0:
            return pts, clr

        frac = max(0.0, min(1.0, frac))
        if frac <= 0.0:
            empty = pts[:0]
            return empty, clr[:0] if clr is not None else None

        target_n = int(float(n) * frac)
        if target_n <= 0:
            target_n = 1
        if target_n >= n:
            return pts, clr

        rng = np.random.default_rng()
        keep_idx = rng.choice(n, size=target_n, replace=False)
        pts_out = pts[keep_idx]
        clr_out = clr[keep_idx] if clr is not None else None
        return pts_out, clr_out

    @staticmethod
    def _sunrgbd_camera_to_depth_transform(rtilt: np.ndarray) -> np.ndarray:
        """
        Build camera->upright-depth transform from SUNRGBD ``Rtilt``.

        The SUNRGBD point cloud stored in ``depth/*.mat`` follows an upright-depth
        convention used by SUNRGBD preprocessing:
          - X right
          - Y forward
          - Z up

        SUNRGBD depth points are stored in an upright-depth frame while
        image projection expects camera optical coordinates. In practice, the
        mapping from upright-depth -> camera combines:

          1) ``Rtilt`` alignment
          2) axis remap depth->camera:
             x_c =  x_d
             y_c = -z_d
             z_c =  y_d

        Since the projection class consumes ``camera_to_lidar_transform``
        (camera -> depth-like frame), we return the inverse of that composed
        depth->camera map.
        """
        # depth(upright) -> camera optical axis conversion
        depth_to_camera_axes = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )

        # depth -> camera
        depth_to_camera = depth_to_camera_axes @ np.asarray(rtilt, dtype=np.float64)

        # camera -> depth
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = np.linalg.inv(depth_to_camera)
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

        camera_intrinsic, camera_rtilt = load_sunrgbd_calibration(sample["intrinsics_path"])
        point_cloud_data = self._load_point_cloud_from_mat(sample["depth_path"])
        point_cloud_depth = point_cloud_data["points"]
        point_cloud_colors = point_cloud_data["colors"]

        point_cloud_depth, point_cloud_colors = self._subsample_point_cloud(
            point_cloud_depth,
            point_cloud_colors,
            stride=int(stride),
            keep_fraction=float(keep_fraction),
        )

        gt_boxes = self._load_ground_truth_boxes(sample.get("annotation_path"), image_rgb.shape)
        camera_to_lidar_transform = self._sunrgbd_camera_to_depth_transform(camera_rtilt)

        return {
            "sample_index": sample_index,
            "scene_id": sample["scene_id"],
            "scene_root": sample.get("scene_root"),
            "image_path": sample["image_path"],
            "depth_path": sample["depth_path"],
            "intrinsics_path": sample["intrinsics_path"],
            "annotation_path": sample.get("annotation_path"),
            "point_cloud": point_cloud_depth,
            "point_cloud_colors": point_cloud_colors,
            "camera_intrinsic": camera_intrinsic,
            "camera_rtilt": camera_rtilt,
            "camera_extrinsic": np.eye(4, dtype=np.float64),
            "camera_to_lidar_transform": camera_to_lidar_transform,
            "camera_frame": "camera_optical",
            "lidar_frame": "sunrgbd_upright_depth",
            "ground_truth_boxes": gt_boxes,
            "dataset_type": "sunrgbd",
            "image_shape": image_rgb.shape[:2],
            "depth_scale": depth_scale,
            "depth_trunc": depth_trunc,
            "stride": stride,
            "keep_fraction": keep_fraction,
        }
