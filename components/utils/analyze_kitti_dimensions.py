"""
KITTI Ground Truth Cuboid Dimension Analysis

This script iterates through the KITTI dataset, extracts ground truth cuboid
measurements for each object class, and plots the distribution of dimensions.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, List
from ..dataset_loaders.kitti_dataset_loader import KITTIDatasetLoader


def collect_cuboid_dimensions(dataroot: str = "dataset/kitti") -> Dict[str, Dict[str, List[float]]]:
    """
    Iterate through KITTI dataset and collect cuboid dimensions for each class.

    Args:
        dataroot: Path to KITTI dataset root

    Returns:
        Dictionary mapping class names to dimension lists:
        {
            'Car': {'length': [...], 'width': [...], 'height': [...]},
            'Pedestrian': {...},
            ...
        }
    """
    # Initialize loader
    loader = KITTIDatasetLoader(dataroot=dataroot, split="training", verbose=True)
    loader.load_dataset()

    # Collect dimensions per class
    dimensions = defaultdict(lambda: {'length': [], 'width': [], 'height': []})

    print(f"\nProcessing {loader.num_samples} samples...")

    for idx in range(loader.num_samples):
        if idx % 100 == 0:
            print(f"  Processing sample {idx}/{loader.num_samples}...")

        try:
            sample_data = loader.load_kitti_data(idx)
            gt_boxes = sample_data.get('ground_truth_boxes', [])

            for box in gt_boxes:
                category = box.get('category', 'Unknown')

                # Compute dimensions from min/max bounds
                length = box['max_x'] - box['min_x']
                width = box['max_y'] - box['min_y']
                height = box['max_z'] - box['min_z']

                dimensions[category]['length'].append(length)
                dimensions[category]['width'].append(width)
                dimensions[category]['height'].append(height)

        except Exception as e:
            print(f"  Error processing sample {idx}: {e}")
            continue

    # Convert to regular dict and print summary
    dimensions = dict(dimensions)
    print("\n" + "=" * 60)
    print("Collection Summary")
    print("=" * 60)
    for category, dims in dimensions.items():
        count = len(dims['length'])
        print(f"  {category}: {count} objects")

    return dimensions


def compute_statistics(dimensions: Dict[str, Dict[str, List[float]]]) -> Dict[str, Dict]:
    """
    Compute statistics for each class and dimension.

    Returns:
        Dictionary with mean, std, min, max for each class/dimension
    """
    stats = {}

    for category, dims in dimensions.items():
        stats[category] = {}
        for dim_name in ['length', 'width', 'height']:
            values = np.array(dims[dim_name])
            if len(values) > 0:
                stats[category][dim_name] = {
                    'count': len(values),
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'median': float(np.median(values))
                }

    return stats


def print_statistics(stats: Dict[str, Dict]):
    """Print formatted statistics table."""
    print("\n" + "=" * 80)
    print("Ground Truth Cuboid Dimension Statistics (meters)")
    print("=" * 80)

    for category, dim_stats in sorted(stats.items()):
        count = dim_stats.get('length', {}).get('count', 0)
        print(f"\n{category} (n={count})")
        print("-" * 70)
        print(f"{'Dimension':<12} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'Median':>8}")
        print("-" * 70)

        for dim_name in ['length', 'width', 'height']:
            if dim_name in dim_stats:
                s = dim_stats[dim_name]
                print(f"{dim_name:<12} {s['mean']:>8.2f} {s['std']:>8.2f} "
                      f"{s['min']:>8.2f} {s['max']:>8.2f} {s['median']:>8.2f}")


def plot_distributions(dimensions: Dict[str, Dict[str, List[float]]], save_path: str = None):
    """
    Plot dimension distributions for each class.

    Args:
        dimensions: Dictionary of dimensions per class
        save_path: Optional path to save the figure
    """
    # Filter classes with enough samples
    min_samples = 10
    valid_classes = {k: v for k, v in dimensions.items()
                     if len(v['length']) >= min_samples}

    if not valid_classes:
        print("No classes with sufficient samples to plot.")
        return

    n_classes = len(valid_classes)
    fig, axes = plt.subplots(n_classes, 3, figsize=(15, 4 * n_classes))

    if n_classes == 1:
        axes = axes.reshape(1, -1)

    dim_names = ['length', 'width', 'height']
    colors = {'length': 'steelblue', 'width': 'darkorange', 'height': 'forestgreen'}

    for row_idx, (category, dims) in enumerate(sorted(valid_classes.items())):
        for col_idx, dim_name in enumerate(dim_names):
            ax = axes[row_idx, col_idx]
            values = np.array(dims[dim_name])

            # Plot histogram
            ax.hist(values, bins=30, color=colors[dim_name], alpha=0.7, edgecolor='black')

            # Add mean and std lines
            mean_val = np.mean(values)
            std_val = np.std(values)
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {mean_val:.2f}m')
            ax.axvline(mean_val - std_val, color='red', linestyle=':', linewidth=1)
            ax.axvline(mean_val + std_val, color='red', linestyle=':', linewidth=1)

            ax.set_xlabel(f'{dim_name.capitalize()} (m)')
            ax.set_ylabel('Count')
            ax.set_title(f'{category} - {dim_name.capitalize()} (n={len(values)})')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")

    plt.show()


def plot_class_comparison(dimensions: Dict[str, Dict[str, List[float]]], save_path: str = None):
    """
    Plot box plots comparing dimensions across classes.

    Args:
        dimensions: Dictionary of dimensions per class
        save_path: Optional path to save the figure
    """
    # Filter classes with enough samples
    min_samples = 10
    valid_classes = sorted([k for k, v in dimensions.items()
                            if len(v['length']) >= min_samples])

    if not valid_classes:
        print("No classes with sufficient samples to plot.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    dim_names = ['length', 'width', 'height']

    for ax_idx, dim_name in enumerate(dim_names):
        ax = axes[ax_idx]

        # Prepare data for box plot
        data = [dimensions[cls][dim_name] for cls in valid_classes]

        bp = ax.boxplot(data, labels=valid_classes, patch_artist=True)

        # Color the boxes
        colors = plt.cm.Set3(np.linspace(0, 1, len(valid_classes)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        ax.set_xlabel('Object Class')
        ax.set_ylabel(f'{dim_name.capitalize()} (m)')
        ax.set_title(f'{dim_name.capitalize()} Distribution by Class')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")

    plt.show()


def main():
    """Main function to run the analysis."""
    import argparse

    parser = argparse.ArgumentParser(description='Analyze KITTI ground truth cuboid dimensions')
    parser.add_argument('--dataroot', type=str, default='dataset/kitti',
                        help='Path to KITTI dataset root')
    parser.add_argument('--save-hist', type=str, default=None,
                        help='Path to save histogram figure')
    parser.add_argument('--save-box', type=str, default=None,
                        help='Path to save box plot figure')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip plotting (just print statistics)')

    args = parser.parse_args()

    # Collect dimensions
    print("=" * 60)
    print("KITTI Ground Truth Cuboid Dimension Analysis")
    print("=" * 60)

    dimensions = collect_cuboid_dimensions(args.dataroot)

    if not dimensions:
        print("No ground truth boxes found!")
        return

    # Compute and print statistics
    stats = compute_statistics(dimensions)
    print_statistics(stats)

    # Plot distributions
    if not args.no_plot:
        print("\nGenerating distribution plots...")
        plot_distributions(dimensions, save_path=args.save_hist)
        plot_class_comparison(dimensions, save_path=args.save_box)


if __name__ == "__main__":
    main()
