"""
CVAT Export Utilities for Point Cloud Clustering Results

This module provides functionality to export point cloud clustering results
in formats compatible with CVAT (Computer Vision Annotation Tool).

Supported export formats:
- COCO 3D Point Cloud format
- YOLO format for 3D bounding boxes
- CVAT XML format
- Simple JSON format for cluster annotations

Author: Claude Code
"""

import json
import numpy as np
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict


class CVATExporter:
    """
    Export point cloud clustering results in CVAT-compatible formats.
    """

    def __init__(self, points: np.ndarray, labels: np.ndarray,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize the CVAT exporter.

        Args:
            points: Nx3 array of 3D points
            labels: Array of cluster labels for each point
            metadata: Optional metadata about the point cloud
        """
        self.points = points
        self.labels = labels
        self.metadata = metadata or {}

        # Analyze clusters
        self.clusters = self._analyze_clusters()

    def _analyze_clusters(self) -> Dict[int, Dict[str, Any]]:
        """
        Analyze clusters to extract bounding box and centroid information.

        Returns:
            Dictionary mapping cluster IDs to cluster information
        """
        clusters = {}
        unique_labels = np.unique(self.labels[self.labels != -1])  # Exclude noise

        for cluster_id in unique_labels:
            mask = self.labels == cluster_id
            cluster_points = self.points[mask]

            if len(cluster_points) == 0:
                continue

            # Calculate bounding box (axis-aligned)
            min_coords = np.min(cluster_points, axis=0)
            max_coords = np.max(cluster_points, axis=0)

            # Calculate centroid
            centroid = np.mean(cluster_points, axis=0)

            # Calculate dimensions
            dimensions = max_coords - min_coords

            # Calculate cluster statistics
            clusters[cluster_id] = {
                'id': int(cluster_id),
                'num_points': len(cluster_points),
                'centroid': centroid.tolist(),
                'min_coords': min_coords.tolist(),
                'max_coords': max_coords.tolist(),
                'dimensions': dimensions.tolist(),
                'points': cluster_points.tolist(),
                'volume': np.prod(dimensions) if np.all(dimensions > 0) else 0.0,
                'density': len(cluster_points) / np.prod(dimensions) if np.all(dimensions > 0) else 0.0
            }

        return clusters

    def export_coco_3d(self, output_path: str,
                      image_filename: Optional[str] = None) -> None:
        """
        Export clustering results in COCO 3D format.

        Args:
            output_path: Path to save the COCO JSON file
            image_filename: Optional reference image filename
        """
        # COCO 3D format structure
        coco_data = {
            "info": {
                "description": "3D Point Cloud Clustering Annotations",
                "version": "1.0",
                "year": datetime.now().year,
                "contributor": "Claude Code - Point Cloud Clustering",
                "date_created": datetime.now().isoformat()
            },
            "licenses": [
                {
                    "id": 1,
                    "name": "Unknown License",
                    "url": ""
                }
            ],
            "images": [],
            "annotations": [],
            "categories": []
        }

        # Add image info (if provided)
        if image_filename:
            coco_data["images"].append({
                "id": 1,
                "width": 1920,  # Default camera resolution
                "height": 1080,
                "file_name": image_filename,
                "license": 1,
                "date_captured": datetime.now().isoformat()
            })

        # Add categories (one per cluster)
        categories = set()
        for cluster_id in self.clusters.keys():
            category_name = f"cluster_{cluster_id}"
            if category_name not in categories:
                coco_data["categories"].append({
                    "id": int(cluster_id + 1),  # COCO categories start from 1
                    "name": category_name,
                    "supercategory": "point_cloud_cluster"
                })
                categories.add(category_name)

        # Add annotations for each cluster
        annotation_id = 1
        for cluster_id, cluster_info in self.clusters.items():
            # Create 3D bounding box annotation
            annotation = {
                "id": annotation_id,
                "image_id": 1 if image_filename else 0,
                "category_id": int(cluster_id + 1),
                "segmentation": [],
                "area": float(cluster_info['volume']),
                "iscrowd": 0,
                "bbox3d": {
                    "location": cluster_info['centroid'],
                    "dimension": cluster_info['dimensions'],
                    "rotation": [0, 0, 0],  # No rotation for axis-aligned boxes
                    "alpha": 0.0  # Observation angle
                },
                "num_points": cluster_info['num_points'],
                "density": float(cluster_info['density']),
                "points": cluster_info['points']  # Include all points in the cluster
            }
            coco_data["annotations"].append(annotation)
            annotation_id += 1

        # Save to file
        with open(output_path, 'w') as f:
            json.dump(coco_data, f, indent=2)

        print(f"Exported {len(self.clusters)} clusters to COCO 3D format: {output_path}")

    def export_yolo_3d(self, output_path: str) -> None:
        """
        Export clustering results in YOLO 3D format.

        Args:
            output_path: Path to save the YOLO TXT file
        """
        # YOLO 3D format: class_id center_x center_y center_z width height depth rotation_x rotation_y rotation_z
        yolo_lines = []

        for cluster_id, cluster_info in self.clusters.items():
            # Use cluster_id as class_id
            class_id = cluster_id

            # Extract position and dimensions
            cx, cy, cz = cluster_info['centroid']
            dx, dy, dz = cluster_info['dimensions']

            # YOLO format (no rotation for simplicity)
            yolo_line = f"{class_id} {cx:.6f} {cy:.6f} {cz:.6f} {dx:.6f} {dy:.6f} {dz:.6f} 0 0 0"
            yolo_lines.append(yolo_line)

        # Save to file
        with open(output_path, 'w') as f:
            f.write('\n'.join(yolo_lines))

        print(f"Exported {len(self.clusters)} clusters to YOLO 3D format: {output_path}")

    def export_cvat_xml(self, output_path: str,
                       image_filename: Optional[str] = None) -> None:
        """
        Export clustering results in CVAT XML format.

        Args:
            output_path: Path to save the CVAT XML file
            image_filename: Optional reference image filename
        """
        # Create XML structure
        root = ET.Element("annotations")

        # Add metadata
        meta = ET.SubElement(root, "meta")
        job = ET.SubElement(meta, "job")
        ET.SubElement(job, "id").text = "0"
        ET.SubElement(job, "name").text = "Point Cloud Clustering"
        ET.SubElement(job, "size").text = str(len(self.clusters))

        # Add original image info (if provided)
        if image_filename:
            original = ET.SubElement(root, "original")
            ET.SubElement(original, "image").text = image_filename

        # Add cluster annotations
        annotation_id = 1
        for cluster_id, cluster_info in self.clusters.items():
            box = ET.SubElement(root, "box")
            ET.SubElement(box, "id").text = str(annotation_id)
            ET.SubElement(box, "label").text = f"cluster_{cluster_id}"
            ET.SubElement(box, "occluded").text = "0"

            # For 3D data, we'll store bounding box info in attributes
            attributes = ET.SubElement(box, "attribute", name="3d_info")
            attributes.text = json.dumps({
                "centroid": cluster_info['centroid'],
                "dimensions": cluster_info['dimensions'],
                "min_coords": cluster_info['min_coords'],
                "max_coords": cluster_info['max_coords'],
                "num_points": cluster_info['num_points'],
                "volume": cluster_info['volume'],
                "density": cluster_info['density']
            })

            annotation_id += 1

        # Format and save XML
        self._prettify_xml(root)
        tree = ET.ElementTree(root)
        tree.write(output_path, encoding='utf-8', xml_declaration=True)

        print(f"Exported {len(self.clusters)} clusters to CVAT XML format: {output_path}")

    def export_json_simple(self, output_path: str) -> None:
        """
        Export clustering results in simple JSON format.

        Args:
            output_path: Path to save the JSON file
        """
        # Convert numpy types to native Python types for JSON serialization
        clusters_json = {}
        for cluster_id, cluster_info in self.clusters.items():
            clusters_json[str(cluster_id)] = {
                'id': int(cluster_info['id']),
                'num_points': int(cluster_info['num_points']),
                'centroid': [float(x) for x in cluster_info['centroid']],
                'min_coords': [float(x) for x in cluster_info['min_coords']],
                'max_coords': [float(x) for x in cluster_info['max_coords']],
                'dimensions': [float(x) for x in cluster_info['dimensions']],
                'volume': float(cluster_info['volume']),
                'density': float(cluster_info['density']),
                'points': [[float(x) for x in point] for point in cluster_info['points']]
            }

        export_data = {
            "metadata": {
                "num_points": int(len(self.points)),
                "num_clusters": int(len(self.clusters)),
                "num_noise_points": int(np.sum(self.labels == -1)),
                "timestamp": datetime.now().isoformat(),
                **self.metadata
            },
            "clusters": clusters_json,
            "point_cloud": {
                "points": [[float(x) for x in point] for point in self.points.tolist()],
                "labels": [int(x) for x in self.labels.tolist()]
            }
        }

        # Save to file
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)

        print(f"Exported {len(self.clusters)} clusters to simple JSON format: {output_path}")

    def export_ply_with_clusters(self, output_path: str) -> None:
        """
        Export point cloud as PLY file with cluster colors.

        Args:
            output_path: Path to save the PLY file
        """
        # Generate colors for each cluster
        colors = self._generate_cluster_colors()

        # Create PLY header
        ply_content = [
            "ply",
            "format ascii 1.0",
            f"element vertex {len(self.points)}",
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "property uchar alpha",
            "property int cluster_id",
            "end_header"
        ]

        # Add vertex data
        for i, (point, label) in enumerate(zip(self.points, self.labels)):
            x, y, z = point

            if label == -1:  # Noise points
                r, g, b = 128, 128, 128  # Gray
            else:
                r, g, b = colors[label]

            alpha = 255
            ply_content.append(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b} {alpha} {label}")

        # Save to file
        with open(output_path, 'w') as f:
            f.write('\n'.join(ply_content))

        print(f"Exported point cloud with cluster colors to PLY format: {output_path}")

    def export_all_formats(self, output_dir: str,
                          filename_prefix: str = "pointcloud_clusters",
                          image_filename: Optional[str] = None) -> None:
        """
        Export clustering results in all supported CVAT-compatible formats.

        Args:
            output_dir: Directory to save all export files
            filename_prefix: Prefix for output filenames
            image_filename: Optional reference image filename
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        print(f"Exporting clustering results in all CVAT formats...")

        # Export in all formats
        self.export_coco_3d(
            str(output_path / f"{filename_prefix}_coco3d.json"),
            image_filename
        )

        self.export_yolo_3d(
            str(output_path / f"{filename_prefix}_yolo3d.txt")
        )

        self.export_cvat_xml(
            str(output_path / f"{filename_prefix}_cvat.xml"),
            image_filename
        )

        self.export_json_simple(
            str(output_path / f"{filename_prefix}_simple.json")
        )

        self.export_ply_with_clusters(
            str(output_path / f"{filename_prefix}_colored.ply")
        )

        print(f"\nAll export files saved to: {output_path.absolute()}")
        print("Files created:")
        print(f"  • {filename_prefix}_coco3d.json - COCO 3D format (best for CVAT)")
        print(f"  • {filename_prefix}_yolo3d.txt - YOLO 3D format")
        print(f"  • {filename_prefix}_cvat.xml - CVAT XML format")
        print(f"  • {filename_prefix}_simple.json - Simple JSON format")
        print(f"  • {filename_prefix}_colored.ply - PLY point cloud with colors")

    def _generate_cluster_colors(self) -> Dict[int, Tuple[int, int, int]]:
        """
        Generate distinct colors for each cluster.

        Returns:
            Dictionary mapping cluster IDs to RGB colors
        """
        colors = {}
        unique_labels = np.unique(self.labels[self.labels != -1])

        # Generate colors using a colormap
        import matplotlib.pyplot as plt
        cmap = plt.cm.get_cmap('tab20')

        for i, label in enumerate(unique_labels):
            color = cmap(i / len(unique_labels))
            colors[label] = tuple(int(c * 255) for c in color[:3])

        return colors

    def _prettify_xml(self, elem: ET.Element) -> None:
        """
        Pretty-print XML formatting - simplified approach.

        Args:
            elem: XML element to format
        """
        # Skip XML prettification to avoid complexity - ElementTree handles basic formatting
        pass


def export_clustering_results_to_cvat(points: np.ndarray, labels: np.ndarray,
                                    output_dir: str = "cvat_exports",
                                    filename_prefix: str = "clusters",
                                    metadata: Optional[Dict[str, Any]] = None,
                                    image_filename: Optional[str] = None) -> None:
    """
    Convenience function to export clustering results to CVAT-compatible formats.

    Args:
        points: Nx3 array of 3D points
        labels: Array of cluster labels
        output_dir: Directory to save export files
        filename_prefix: Prefix for output filenames
        metadata: Optional metadata about the clustering
        image_filename: Optional reference image filename
    """
    exporter = CVATExporter(points, labels, metadata)
    exporter.export_all_formats(output_dir, filename_prefix, image_filename)


if __name__ == "__main__":
    # Example usage
    print("CVAT Export Example")

    # Create sample data
    np.random.seed(42)
    points = np.random.randn(300, 3)
    labels = np.random.randint(-1, 5, 300)  # -1 for noise, 0-4 for clusters

    # Export to CVAT formats
    export_clustering_results_to_cvat(
        points=points,
        labels=labels,
        output_dir="cvat_example_export",
        filename_prefix="sample_clusters",
        metadata={"algorithm": "DBSCAN", "eps": 0.5, "min_samples": 10},
        image_filename="sample_camera.jpg"
    )