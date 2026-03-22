"""
ROS bag utilities for dataset integration.

This module provides:
- Topic discovery helpers (image, point cloud)
- Frame-level quality filtering using components.core.filter
- Extraction of filtered frames to a KITTI-like folder (image_2/, velodyne/)
- Optional calibration export (camera intrinsics + camera→LiDAR transform)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml
import re
from scipy.spatial.transform import Rotation as R
import open3d as o3d

from components.core.filter import image_passes_quality_filters

# Optional: rosbags (ROS1/ROS2)
HAS_ROSBAGS = False
AnyReader = None  # type: ignore[assignment]
Ros2Reader = None  # type: ignore[assignment]
message_to_cvimage = None  # type: ignore[assignment]

try:
    from rosbags.highlevel import AnyReader as _AnyReader

    AnyReader = _AnyReader  # type: ignore[assignment]
    HAS_ROSBAGS = True
except ImportError:  # pragma: no cover - rosbags is optional
    AnyReader = None  # type: ignore[assignment]

try:
    from rosbags.rosbag2 import Reader as _Ros2Reader  # type: ignore[import]

    Ros2Reader = _Ros2Reader  # type: ignore[assignment]
except ImportError:
    Ros2Reader = None  # type: ignore[assignment]

try:
    from rosbags.image import message_to_cvimage as _message_to_cvimage  # type: ignore[import]

    message_to_cvimage = _message_to_cvimage  # type: ignore[assignment]
except ImportError:
    message_to_cvimage = None  # type: ignore[assignment]


# -----------------------------------------------------------------------------
# Bag discovery and topic listing
# -----------------------------------------------------------------------------


def open_reader(bag_path: Path):
    """Open ROS bag (file or folder). Returns context manager."""
    if not HAS_ROSBAGS:
        raise RuntimeError(
            "rosbags package is required for bag extraction. "
            "Install with: pip install rosbags"
        )
    bag_path = Path(bag_path)
    if not bag_path.exists():
        raise FileNotFoundError(f"Bag not found: {bag_path}")
    try:
        return AnyReader([bag_path])
    except Exception:
        if Ros2Reader is not None:
            try:
                return Ros2Reader(str(bag_path))  # type: ignore[operator]
            except Exception as e:  # pragma: no cover - only on failure
                raise RuntimeError(f"Failed to open bag {bag_path}: {e}") from e
        # If we get here, AnyReader failed and Ros2Reader is unavailable
        raise RuntimeError(f"Failed to open bag {bag_path}: unsupported format or rosbags installation is incomplete")


def get_image_topics(bag_path: Path) -> List[str]:
    """Return sorted list of image topic names."""
    topics = set()
    try:
        with open_reader(bag_path) as reader:
            for conn in reader.connections:
                if "Image" in conn.msgtype or "CompressedImage" in conn.msgtype:
                    topics.add(conn.topic)
    except Exception:
        pass
    return sorted(topics)


def get_pointcloud_topics(bag_path: Path) -> List[str]:
    """Return sorted list of PointCloud2 topic names."""
    topics = set()
    try:
        with open_reader(bag_path) as reader:
            for conn in reader.connections:
                if "PointCloud2" in conn.msgtype:
                    topics.add(conn.topic)
    except Exception:
        pass
    return sorted(topics)


def get_camera_info_topics(bag_path: Path) -> List[str]:
    """Return sorted list of CameraInfo topics."""
    topics = set()
    try:
        with open_reader(bag_path) as reader:
            for conn in reader.connections:
                if "sensor_msgs/msg/CameraInfo" in conn.msgtype or "CameraInfo" in conn.msgtype:
                    topics.add(conn.topic)
    except Exception:
        pass
    return sorted(topics)


def get_tf_topics(bag_path: Path) -> List[str]:
    """Return sorted list of TF/TF_STATIC topics."""
    topics = set()
    try:
        with open_reader(bag_path) as reader:
            for conn in reader.connections:
                if "tf2_msgs/msg/TFMessage" in conn.msgtype or conn.topic in ("/tf", "/tf_static"):
                    topics.add(conn.topic)
    except Exception:
        pass
    return sorted(topics)


def suggest_topics_from_metadata(bag_root: Path) -> Dict[str, Optional[str]]:
    """
    Suggest image, point cloud, camera_info and tf topics from rosbag2 metadata.yaml.

    Args:
        bag_root: Folder that contains metadata.yaml (ROS2 bag) or its parent.

    Returns:
        Dict with optional keys:
            image_topic, pointcloud_topic, camera_info_topic, tf_topic
    """
    bag_root = Path(bag_root)
    meta_path = None
    if bag_root.is_file():
        # e.g. path/to/rosbag2_..._0.mcap.zstd → look in parent
        candidate = bag_root.parent / "metadata.yaml"
        if candidate.exists():
            meta_path = candidate
    else:
        candidate = bag_root / "metadata.yaml"
        if candidate.exists():
            meta_path = candidate

    if meta_path is None or not meta_path.exists():
        return {
            "image_topic": None,
            "pointcloud_topic": None,
            "camera_info_topic": None,
            "tf_topic": None,
        }

    try:
        with meta_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception:
        return {
            "image_topic": None,
            "pointcloud_topic": None,
            "camera_info_topic": None,
            "tf_topic": None,
        }

    info = data.get("rosbag2_bagfile_information", {})
    topics_info = info.get("topics_with_message_count", []) or []

    image_candidates: List[str] = []
    pc_candidates: List[str] = []
    cam_info_candidates: List[str] = []
    tf_candidates: List[str] = []

    for entry in topics_info:
        meta = entry.get("topic_metadata", {})
        name = meta.get("name", "")
        typ = meta.get("type", "")

        if "CompressedImage" in typ or typ.endswith("sensor_msgs/msg/Image"):
            image_candidates.append(name)
        elif "PointCloud2" in typ:
            pc_candidates.append(name)
        elif "CameraInfo" in typ:
            cam_info_candidates.append(name)
        elif "TFMessage" in typ and name in ("/tf", "/tf_static"):
            tf_candidates.append(name)

    # Keep this simple: just pick the first available topic in each category,
    # without any additional preference ordering.
    image_topic = image_candidates[0] if image_candidates else None
    pointcloud_topic = pc_candidates[0] if pc_candidates else None
    camera_info_topic = cam_info_candidates[0] if cam_info_candidates else None
    tf_topic = tf_candidates[0] if tf_candidates else None

    return {
        "image_topic": image_topic,
        "pointcloud_topic": pointcloud_topic,
        "camera_info_topic": camera_info_topic,
        "tf_topic": tf_topic,
    }


# -----------------------------------------------------------------------------
# Calibration helpers (CameraInfo + TF tree)
# -----------------------------------------------------------------------------


@dataclass
class RosbagCalibration:
    camera_intrinsic: Optional[np.ndarray]
    camera_to_lidar_transform: Optional[np.ndarray]
    camera_frame: Optional[str]
    lidar_frame: Optional[str]


def _transform_to_matrix(transform) -> np.ndarray:
    """Convert geometry_msgs/Transform to 4x4 matrix."""
    t = transform.translation
    q = transform.rotation
    trans = np.array([t.x, t.y, t.z], dtype=np.float64)
    quat = np.array([q.x, q.y, q.z, q.w], dtype=np.float64)
    Rm = R.from_quat(quat).as_matrix()
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rm
    T[:3, 3] = trans
    return T


def _build_tf_adjacency(reader, tf_topics: Iterable[str]) -> Dict[str, Dict[str, np.ndarray]]:
    """Build adjacency dict: T[src][dst] maps coords from src frame to dst frame."""
    adj: Dict[str, Dict[str, np.ndarray]] = {}
    tf_conns = [c for c in reader.connections if c.topic in tf_topics]
    if not tf_conns:
        return adj

    for conn, _ts, raw in reader.messages(connections=tf_conns):
        msg = reader.deserialize(raw, conn.msgtype)
        transforms = getattr(msg, "transforms", []) or []
        for t in transforms:
            parent = t.header.frame_id or ""
            child = t.child_frame_id or ""
            if not parent or not child:
                continue
            print(f'parent={parent}, child={child}')
            T_parent_child = _transform_to_matrix(t.transform)
            # Store transforms so that adj[src][dst] always maps coordinates
            # from src frame to dst frame.
            adj.setdefault(parent, {})[child] = np.linalg.inv(T_parent_child)
            adj.setdefault(child, {})[parent] = T_parent_child
    return adj


def get_tf_frames(bag_path: Path, tf_topics: Optional[Iterable[str]] = None) -> List[str]:
    """
    Inspect TF topics in a ROS bag and return a sorted list of frame names.
    """
    if tf_topics is None:
        tf_topics = ("/tf_static", "/tf")

    if not HAS_ROSBAGS:
        return []

    bag_path = Path(bag_path)
    with open_reader(bag_path) as reader:
        adj = _build_tf_adjacency(reader, tf_topics)
    return sorted(adj.keys())


def _find_transform(
    adj: Dict[str, Dict[str, np.ndarray]], source_frame: str, target_frame: str
) -> Optional[np.ndarray]:
    """
    Find homogeneous transform from source_frame to target_frame using BFS over TF tree.
    Returns 4x4 matrix T such that X_target = T @ X_source.
    """
    from collections import deque
    print(f'source_frame={source_frame}, target_frame={target_frame}')
    if source_frame == target_frame:
        return np.eye(4, dtype=np.float64)

    if source_frame not in adj or target_frame not in adj:
        return None

    visited = {source_frame}
    queue = deque([(source_frame, np.eye(4, dtype=np.float64))])

    while queue:
        frame, T_source_current = queue.popleft()
        if frame == target_frame:
            return T_source_current

        for nbr, T_current_nbr in adj.get(frame, {}).items():
            if nbr in visited:
                continue
            visited.add(nbr)
            T_source_nbr = T_current_nbr @ T_source_current
            queue.append((nbr, T_source_nbr))

    return None


def compute_rosbag_calibration(
    bag_path: Path,
    image_topic: str,
    pointcloud_topic: Optional[str],
    camera_info_topic: Optional[str],
    tf_topics: Optional[Iterable[str]] = None,
    camera_frame_override: Optional[str] = None,
    lidar_frame_override: Optional[str] = None,
) -> RosbagCalibration:
    """
    Compute camera intrinsics and camera→LiDAR transform from ROS bag.

    Uses:
    - sensor_msgs/CameraInfo on camera_info_topic for intrinsics
    - tf2_msgs/TFMessage on /tf_static and /tf for extrinsics
    """
    if tf_topics is None:
        tf_topics = ("/tf_static", "/tf")

    bag_path = Path(bag_path)
    camera_intrinsic: Optional[np.ndarray] = None
    # Respect explicit overrides; only infer frames when override is None.
    camera_frame: Optional[str] = camera_frame_override
    lidar_frame: Optional[str] = lidar_frame_override

    if not HAS_ROSBAGS:
        return RosbagCalibration(None, None, None, None)

    # First pass: CameraInfo + frame_ids
    with open_reader(bag_path) as reader:
        # CameraInfo → intrinsic. Only derive camera_frame from header when
        # caller did not provide an explicit override.
        if camera_info_topic and camera_intrinsic is None:
            cam_conns = [c for c in reader.connections if c.topic == camera_info_topic]
            for conn, _ts, raw in reader.messages(connections=cam_conns):
                msg = reader.deserialize(raw, conn.msgtype)
                K = getattr(msg, "k", None)
                if K is None:
                    K = getattr(msg, "K", None)
                if K is not None and len(K) == 9:
                    camera_intrinsic = np.array(K, dtype=np.float64).reshape(3, 3)
                    header = getattr(msg, "header", None)
                    if camera_frame is None and header is not None:
                        camera_frame = getattr(header, "frame_id", "")
                    break

        # If no CameraInfo-derived frame override, derive camera_frame from image topic
        if camera_frame is None:
            img_conns = [c for c in reader.connections if c.topic == image_topic]
            for conn, _ts, raw in reader.messages(connections=img_conns):
                msg = reader.deserialize(raw, conn.msgtype)
                header = getattr(msg, "header", None)
                if header is not None and getattr(header, "frame_id", ""):
                    camera_frame = header.frame_id
                    break

        # LiDAR frame from PointCloud2 (if not overridden)
        if pointcloud_topic and lidar_frame is None:
            pc_conns = [c for c in reader.connections if c.topic == pointcloud_topic]
            for conn, _ts, raw in reader.messages(connections=pc_conns):
                msg = reader.deserialize(raw, conn.msgtype)
                header = getattr(msg, "header", None)
                if header is not None and getattr(header, "frame_id", ""):
                    lidar_frame = header.frame_id
                    break
    print(f'camera_frame={camera_frame}, lidar_frame={lidar_frame}')
    if camera_frame is None or lidar_frame is None:
        return RosbagCalibration(camera_intrinsic, None, camera_frame, lidar_frame)

    # Second pass: TF tree for extrinsics
    with open_reader(bag_path) as reader:
        adj = _build_tf_adjacency(reader, tf_topics)
    print(f'adj={adj}')
    if not adj:
        return RosbagCalibration(camera_intrinsic, None, camera_frame, lidar_frame)

    T_cam_lidar = _find_transform(adj, camera_frame, lidar_frame)
    print(f'T_cam_lidar={T_cam_lidar}')
    return RosbagCalibration(camera_intrinsic, T_cam_lidar, camera_frame, lidar_frame)

# -----------------------------------------------------------------------------
# Filtering: inspect bag and select good frames (no disk writes)
# -----------------------------------------------------------------------------


def filter_rosbag_frames(
    bag_path: Path,
    image_topic: str,
    filter_params: Dict[str, Any],
    max_frames: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> List[Dict[str, Any]]:
    """
    Filter frames from a ROS bag image topic using components.core.filter.

    Returns a list of accepted frames:
        [
          {
            "frame_index": int,          # sequential index (0..N-1 in accepted order)
            "timestamp_ns": int,         # bag timestamp
            "metrics": {...},            # blur/contrast/brightness/motion
          },
          ...
        ]
    """
    if not HAS_ROSBAGS:
        raise RuntimeError("rosbags is required. Install with: pip install rosbags")

    bag_path = Path(bag_path)
    accepted: List[Dict[str, Any]] = []
    seen_hashes: List[Any] = []
    prev_gray: Optional[np.ndarray] = None

    with open_reader(bag_path) as reader:
        img_conns = [c for c in reader.connections if c.topic == image_topic]
        if not img_conns:
            return []

        total_msgs = sum(c.msgcount for c in img_conns)
        processed = 0

        for conn, ts, raw in reader.messages(connections=img_conns):
            if max_frames is not None and len(accepted) >= max_frames:
                break

            processed += 1
            if progress_callback and total_msgs:
                progress_callback(processed, total_msgs, "Filtering ROS bag frames...")

            if message_to_cvimage is None:
                raise RuntimeError(
                    "rosbags.image.message_to_cvimage is required for ROS image decoding. "
                    "Ensure the 'rosbags[image]' extras and OpenCV are installed."
                )

            try:
                msg = reader.deserialize(raw, conn.msgtype)
                frame_bgr = message_to_cvimage(msg, "bgr8")  # type: ignore[call-arg]
                if frame_bgr is None:
                    continue
            except Exception:
                continue

            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            passed, metrics = image_passes_quality_filters(
                gray, prev_gray, seen_hashes, filter_params
            )
            if filter_params.get("enable_motion", False):
                prev_gray = gray

            if not passed:
                continue

            accepted.append(
                {
                    "frame_index": len(accepted),
                    "timestamp_ns": int(ts),
                    "metrics": metrics,
                }
            )

    return accepted


# -----------------------------------------------------------------------------
# Main extraction to KITTI-like folder (image_2, velodyne)
# -----------------------------------------------------------------------------


def extract_bag_to_folder(
    bag_path: Path,
    out_dir: Path,
    image_topic: str,
    pointcloud_topic: Optional[str] = None,
    accepted_timestamps_ns: Optional[Sequence[int]] = None,
    max_frames: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    camera_info_topic: Optional[str] = None,
    tf_topics: Optional[Iterable[str]] = None,
    camera_frame_override: Optional[str] = None,
    lidar_frame_override: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Extract images (and optionally point clouds) from a ROS bag.
    
    If accepted_timestamps_ns is provided, ONLY those timestamps are exported and
    no internal quality filtering is applied. This is intended to be used after
    a prior filtering pass (e.g. via filter_rosbag_frames).
    
    The output layout is:
        out_dir/
          image_2/    000000.png, 000001.png, ...
          velodyne/   000000.pcd, 000001.pcd, ...   (PCD: XYZ)
          calib.npz   camera_intrinsic, camera_to_lidar, camera_frame, lidar_frame
    
    Returns:
        frames: list of {
            "frame_index": int,
            "image_path": str,
            "pointcloud_path": Optional[str],
            "timestamp_ns": int,
        }
        stats: dict with total_images, exported_images
    """
    if not HAS_ROSBAGS:
        raise RuntimeError("rosbags is required. Install with: pip install rosbags")

    print(f'image_topic={image_topic}, pointcloud_topic={pointcloud_topic}')
    bag_path = Path(bag_path)
    out_dir = Path(out_dir)
    out_images = out_dir / "image_2"
    out_velodyne = out_dir / "velodyne"
    out_images.mkdir(parents=True, exist_ok=True)
    out_velodyne.mkdir(parents=True, exist_ok=True)

    # Calibration (best-effort)
    calib = compute_rosbag_calibration(
        bag_path=bag_path,
        image_topic=image_topic,
        pointcloud_topic=pointcloud_topic,
        camera_info_topic=camera_info_topic,
        tf_topics=tf_topics,
        camera_frame_override=camera_frame_override,
        lidar_frame_override=lidar_frame_override,
    )
    calib_path = out_dir / "calib.npz"
    try:
        np.savez(
            calib_path,
            camera_intrinsic=calib.camera_intrinsic
            if calib.camera_intrinsic is not None
            else np.eye(3, dtype=np.float64),
            camera_to_lidar=calib.camera_to_lidar_transform
            if calib.camera_to_lidar_transform is not None
            else np.eye(4, dtype=np.float64),
            camera_frame=calib.camera_frame if calib.camera_frame is not None else "",
            lidar_frame=calib.lidar_frame if calib.lidar_frame is not None else "",
        )
    except Exception:
        # Calibration export is best-effort; continue even if it fails
        pass

    # If caller passed accepted timestamps, use them as a lookup set
    accepted_ts_set: Optional[set[int]] = None
    if accepted_timestamps_ns is not None:
        accepted_ts_set = {int(ts) for ts in accepted_timestamps_ns}
        print(
            "[rosbag_extractor] accepted_timestamps_ns provided: count=",
            len(accepted_ts_set),
        )
        if accepted_ts_set:
            # Show a few sample timestamps for debugging
            sample_ts = list(sorted(accepted_ts_set))[:5]
            print(
                "[rosbag_extractor] accepted_timestamps_ns sample=",
                sample_ts,
            )

    stats = {
        "total_images": 0,
        "exported_images": 0,
    }

    frames: List[Dict[str, Any]] = []

    # First pass: write images
    with open_reader(bag_path) as reader:
        img_conns = [c for c in reader.connections if c.topic == image_topic]
        if not img_conns:
            return [], stats

        total_msgs = sum(c.msgcount for c in img_conns)
        processed = 0
        frame_index = 0
        debug_ts_checks = 0

        for conn, ts, raw in reader.messages(connections=img_conns):
            stats["total_images"] += 1
            processed += 1
            if progress_callback and total_msgs:
                progress_callback(processed, total_msgs, "Exporting images...")

            if max_frames is not None and stats["exported_images"] >= max_frames:
                break

            if debug_ts_checks < 5:
                in_set = (
                    accepted_ts_set is not None and int(ts) in accepted_ts_set
                )
                print(
                    "[rosbag_extractor] image msg ts=",
                    int(ts),
                    "in accepted_ts_set=",
                    in_set,
                )
                debug_ts_checks += 1

            if accepted_ts_set is not None and int(ts) not in accepted_ts_set:
                continue

            if message_to_cvimage is None:
                raise RuntimeError(
                    "rosbags.image.message_to_cvimage is required for ROS image decoding. "
                    "Ensure the 'rosbags[image]' extras and OpenCV are installed."
                )

            try:
                msg = reader.deserialize(raw, conn.msgtype)
                frame_bgr = message_to_cvimage(msg, "bgr8")  # type: ignore[call-arg]
                if frame_bgr is None:
                    continue
            except Exception:
                continue

            img_name = f"{frame_index:06d}.png"
            img_path = out_images / img_name
            cv2.imwrite(str(img_path), frame_bgr)

            frames.append(
                {
                    "frame_index": frame_index,
                    "image_path": str(img_path),
                    "pointcloud_path": None,
                    "timestamp_ns": int(ts),
                }
            )

            stats["exported_images"] += 1
            frame_index += 1

    if not frames or not pointcloud_topic:
        return frames, stats

    # Second pass: export point clouds and time-sync to frames
    ts_list = np.array([f["timestamp_ns"] for f in frames])
    frame_indices = [f["frame_index"] for f in frames]
    assigned: Dict[int, Tuple[int, np.ndarray]] = {}

    with open_reader(bag_path) as reader:
        pc_conns = [c for c in reader.connections if c.topic == pointcloud_topic]
        if not pc_conns:
            return frames, stats

        total_pc = sum(c.msgcount for c in pc_conns)
        pc_processed = 0

        for conn, ts, raw in reader.messages(connections=pc_conns):
            if progress_callback and total_pc:
                pc_processed += 1
                if pc_processed % 50 == 0:
                    progress_callback(
                        pc_processed, total_pc, "Syncing point clouds..."
                    )
            try:
                msg = reader.deserialize(raw, conn.msgtype)
                xyz = _pointcloud2_to_xyz(msg)
                if xyz is None or len(xyz) == 0:
                    continue
            except Exception:
                continue

            ts_ns = int(ts)
            idx = int(np.argmin(np.abs(ts_list - ts_ns)))
            fi = frame_indices[idx]
            if fi not in assigned or abs(ts_list[idx] - ts_ns) < abs(
                ts_list[idx] - assigned[fi][0]
            ):
                assigned[fi] = (ts_ns, xyz)

    for fi, (_, xyz) in assigned.items():
        # Save each point cloud as a PCD file (XYZ only).
        # This format is widely supported (e.g., Open3D, PCL, CVAT).
        pc_path = out_velodyne / f"{fi:06d}.pcd"
        try:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float32))
            # Use ASCII for maximum interoperability; change to write_ascii=False
            # if binary PCD is preferred.
            o3d.io.write_point_cloud(str(pc_path), pcd, write_ascii=True)
        except Exception as e:
            print(f"[rosbag_extractor] Failed to write PCD file {pc_path}: {e}")
            continue

        for f in frames:
            if f["frame_index"] == fi:
                f["pointcloud_path"] = str(pc_path)
                break

    return frames, stats


def _pointcloud2_to_xyz(msg: Any) -> Optional[np.ndarray]:
    """Convert PointCloud2 message to Nx3 float32 (x,y,z). Returns None on failure."""
    data = getattr(msg, "data", None)
    if data is None:
        data = getattr(msg, "point_cloud_data", None)
    if data is None:
        return None

    if isinstance(data, (bytes, bytearray)):
        raw = np.frombuffer(data, dtype=np.uint8)
    else:
        raw = np.array(data, dtype=np.uint8)

    point_step = getattr(msg, "point_step", 12)

    # Assume x,y,z at 0,4,8 (float32) – standard sensor_msgs/PointCloud2
    n = len(raw) // point_step
    if n == 0:
        print("[rosbag_extractor] _pointcloud2_to_xyz: n == 0 (no points)")
        return None

    xyzw = np.frombuffer(raw[: n * point_step].tobytes(), dtype=np.float32)

    # Reshape so we can take columns 0,1,2 (x,y,z)
    num_fields = point_step // 4
    if num_fields < 3:
        print(
            "[rosbag_extractor] _pointcloud2_to_xyz: "
            f"num_fields={num_fields} < 3; cannot extract x,y,z"
        )
        return None

    pts = xyzw.reshape(-1, num_fields)[:, :3].astype(np.float32)
    return pts
    

def sanitize_bag_name(name: str) -> str:
    """Sanitize bag file/folder name for use as directory name, without extension."""
    # Strip file extension(s) first (e.g. foo.mcap -> foo)
    base = Path(name).stem
    name = re.sub(r"[^\w\-.]", "_", base)
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "rosbag"
