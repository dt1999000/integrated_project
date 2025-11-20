"""
ClusteringManager - Comprehensive clustering system for 3D point clouds with multiple algorithms
and evaluation metrics.

This module provides:
- Multiple clustering algorithms (DBSCAN, HDBSCAN, OPTICS, BIRCH, Agglomerative)
- Parameter grid search and automatic tuning
- Comprehensive evaluation metrics (standard and 3D-specific)
- Algorithm comparison and ranking
- 3D point cloud specialized metrics
"""

import numpy as np
import warnings
from typing import Dict, List, Tuple, Any, Optional
from itertools import product
import time

# Standard clustering algorithms from scikit-learn
from sklearn.cluster import DBSCAN, OPTICS, Birch, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Try to import HDBSCAN (optional dependency)
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    warnings.warn("HDBSCAN not available. Install with: pip install hdbscan")


# Default parameter grids for different clustering algorithms
DEFAULT_PARAM_GRIDS = {
    'dbscan': {
        'eps': np.linspace(0.3, 2.0, 10),
        'min_samples': range(5, 51, 5)
    },
    'optics': {
        'min_samples': range(5, 51, 5),
        'max_eps': np.linspace(1.0, 5.0, 5),
        'xi': np.linspace(0.01, 0.1, 5)
    },
    'birch': {
        'threshold': np.linspace(0.1, 1.0, 10),
        'branching_factor': [20, 50, 100],
        'n_clusters': [None] + list(range(2, 21))
    },
    'agglomerative': {
        'n_clusters': range(2, 21),
        'linkage': ['ward', 'complete', 'average', 'single']
    }
}

if HDBSCAN_AVAILABLE:
    DEFAULT_PARAM_GRIDS['hdbscan'] = {
        'min_cluster_size': range(5, 51, 5),
        'min_samples': range(1, 21, 2),
        'cluster_selection_method': ['eom', 'leaf'],
        'cluster_selection_epsilon': np.linspace(0.0, 1.0, 6)
    }


class ClusteringMetrics:
    """
    Collection of clustering evaluation metrics including 3D-specific metrics.
    """

    @staticmethod
    def silhouette_score_safe(points: np.ndarray, labels: np.ndarray) -> float:
        """Calculate silhouette score with error handling."""
        try:
            unique_labels = np.unique(labels[labels != -1])
            if len(unique_labels) < 2:
                return 0.0
            return silhouette_score(points, labels, metric='euclidean')
        except (ValueError, RuntimeError):
            return 0.0

    @staticmethod
    def calinski_harabasz_score_safe(points: np.ndarray, labels: np.ndarray) -> float:
        """Calculate Calinski-Harabasz score with error handling."""
        try:
            unique_labels = np.unique(labels[labels != -1])
            if len(unique_labels) < 2:
                return 0.0
            return calinski_harabasz_score(points, labels)
        except (ValueError, RuntimeError):
            return 0.0

    @staticmethod
    def davies_bouldin_score_safe(points: np.ndarray, labels: np.ndarray) -> float:
        """Calculate Davies-Bouldin score with error handling."""
        try:
            unique_labels = np.unique(labels[labels != -1])
            if len(unique_labels) < 2:
                return float('inf')
            return davies_bouldin_score(points, labels)
        except (ValueError, RuntimeError):
            return float('inf')

    @staticmethod
    def calculate_3d_metrics(points: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """
        Calculate 3D-specific clustering metrics including density, spatial extent, and height analysis.
        """
        metrics = {}
        unique_labels = np.unique(labels[labels != -1])

        if len(unique_labels) == 0:
            return {'num_clusters': 0, 'noise_ratio': 1.0}

        # Basic cluster statistics
        metrics['num_clusters'] = len(unique_labels)
        metrics['noise_ratio'] = np.sum(labels == -1) / len(labels)

        # Cluster-wise 3D metrics
        total_volume = 0.0
        total_points = 0
        all_densities = []
        all_height_stds = []

        for cluster_id in unique_labels:
            cluster_points = points[labels == cluster_id]

            if len(cluster_points) < 2:
                continue

            # Bounding box analysis
            bbox_min = np.min(cluster_points, axis=0)
            bbox_max = np.max(cluster_points, axis=0)
            bbox_volume = np.prod(bbox_max - bbox_min)

            # Point density (points per cubic meter)
            density = len(cluster_points) / bbox_volume if bbox_volume > 0 else 0
            all_densities.append(density)

            # Height distribution analysis (Z-axis)
            heights = cluster_points[:, 2]
            height_std = np.std(heights)
            all_height_stds.append(height_std)

            total_volume += bbox_volume
            total_points += len(cluster_points)

            # Per-cluster metrics
            metrics[f'cluster_{cluster_id}_points'] = len(cluster_points)
            metrics[f'cluster_{cluster_id}_volume'] = bbox_volume
            metrics[f'cluster_{cluster_id}_density'] = density
            metrics[f'cluster_{cluster_id}_height_range'] = np.max(heights) - np.min(heights)
            metrics[f'cluster_{cluster_id}_height_std'] = height_std

        # Aggregate 3D metrics
        if all_densities:
            metrics['avg_density'] = np.mean(all_densities)
            metrics['std_density'] = np.std(all_densities)
            metrics['min_density'] = np.min(all_densities)
            metrics['max_density'] = np.max(all_densities)

        if all_height_stds:
            metrics['avg_height_std'] = np.mean(all_height_stds)
            metrics['std_height_std'] = np.std(all_height_stds)

        # Overall density
        overall_volume = total_volume if total_volume > 0 else 1.0
        metrics['overall_density'] = total_points / overall_volume

        return metrics

    @staticmethod
    def calculate_composite_score(points: np.ndarray, labels: np.ndarray) -> float:
        """
        Calculate composite clustering score combining multiple metrics.
        Higher scores indicate better clustering quality.
        """
        if len(np.unique(labels[labels != -1])) < 2:
            return 0.0

        # Standard metrics
        silhouette = ClusteringMetrics.silhouette_score_safe(points, labels)
        calinski_harabasz = ClusteringMetrics.calinski_harabasz_score_safe(points, labels)
        davies_bouldin = ClusteringMetrics.davies_bouldin_score_safe(points, labels)

        # 3D-specific metrics
        metrics_3d = ClusteringMetrics.calculate_3d_metrics(points, labels)

        # Normalize and combine metrics
        # Silhouette: [-1, 1] -> [0, 1]
        normalized_silhouette = (silhouette + 1) / 2

        # Calinski-Harabasz: scale to [0, 1] (approximate)
        normalized_ch = min(calinski_harabasz / 10000, 1.0)

        # Davies-Bouldin: invert (lower is better) -> [0, 1]
        normalized_db = 1 / (1 + davies_bouldin)

        # Noise ratio penalty (lower is better)
        noise_penalty = 1 - metrics_3d['noise_ratio']

        # Density score (prefer moderate densities)
        if 'avg_density' in metrics_3d:
            density_score = min(metrics_3d['avg_density'] / 1000, 1.0)  # Scale to reasonable range
        else:
            density_score = 0.5

        # Composite score (weighted combination)
        composite = (
            0.30 * normalized_silhouette +
            0.20 * normalized_ch +
            0.20 * normalized_db +
            0.15 * noise_penalty +
            0.15 * density_score
        )

        return composite


class ClusteringManager:
    """
    Comprehensive clustering system that manages multiple algorithms,
    parameter optimization, and evaluation metrics for 3D point clouds.
    """

    def __init__(self, point_cloud: np.ndarray):
        """
        Initialize ClusteringManager with point cloud data.

        Args:
            point_cloud: Nx3 numpy array of 3D points (x, y, z)
        """
        if not isinstance(point_cloud, np.ndarray) or point_cloud.ndim != 2 or point_cloud.shape[1] != 3:
            raise ValueError("point_cloud must be a Nx3 numpy array")

        self.point_cloud = point_cloud
        self.algorithms = {}
        self.results = {}
        self.metrics = ClusteringMetrics()

        # Register default algorithms
        self._register_default_algorithms()

    def _register_default_algorithms(self):
        """Register default clustering algorithms with their parameter grids."""

        # DBSCAN
        self.register_algorithm('dbscan', DBSCAN, DEFAULT_PARAM_GRIDS['dbscan'])

        # OPTICS
        self.register_algorithm('optics', OPTICS, DEFAULT_PARAM_GRIDS['optics'])

        # BIRCH
        self.register_algorithm('birch', Birch, DEFAULT_PARAM_GRIDS['birch'])

        # Agglomerative Clustering
        self.register_algorithm('agglomerative', AgglomerativeClustering, DEFAULT_PARAM_GRIDS['agglomerative'])

        # HDBSCAN (if available)
        if HDBSCAN_AVAILABLE:
            self.register_algorithm('hdbscan', hdbscan.HDBSCAN, DEFAULT_PARAM_GRIDS['hdbscan'])

    def register_algorithm(self, name: str, algorithm_class, param_grid: Dict[str, List]):
        """
        Register a clustering algorithm with its parameter grid.

        Args:
            name: Algorithm name identifier
            algorithm_class: Scikit-learn compatible clustering class
            param_grid: Dictionary of parameter names to list of values
        """
        self.algorithms[name] = {
            'class': algorithm_class,
            'param_grid': param_grid
        }

    def generate_param_combinations(self, param_grid: Dict[str, List]) -> List[Dict[str, Any]]:
        """
        Generate all parameter combinations from a parameter grid.

        Args:
            param_grid: Dictionary of parameter names to list of values

        Returns:
            List of parameter dictionaries
        """
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())

        combinations = []
        for combination in product(*param_values):
            param_dict = dict(zip(param_names, combination))
            combinations.append(param_dict)

        return combinations

    def fit_algorithm(self, algorithm_name: str, params: Dict[str, Any]) -> np.ndarray:
        """
        Fit a clustering algorithm with given parameters.

        Args:
            algorithm_name: Name of registered algorithm
            params: Dictionary of parameters

        Returns:
            Array of cluster labels (-1 for noise points)
        """
        if algorithm_name not in self.algorithms:
            raise ValueError(f"Algorithm '{algorithm_name}' not registered")

        algo_config = self.algorithms[algorithm_name]
        algorithm_class = algo_config['class']

        # Special handling for different algorithms
        if algorithm_name == 'hdbscan':
            # HDBSCAN uses different parameter names
            clusterer = algorithm_class(**params)
            labels = clusterer.fit_predict(self.point_cloud)
        elif algorithm_name == 'agglomerative':
            # Agglomerative clustering doesn't support predict method
            clusterer = algorithm_class(**params)
            labels = clusterer.fit_predict(self.point_cloud)
        else:
            # Standard scikit-learn interface
            clusterer = algorithm_class(**params)
            labels = clusterer.fit_predict(self.point_cloud)

        return labels

    def evaluate_clustering(self, labels: np.ndarray) -> Dict[str, float]:
        """
        Evaluate clustering results using multiple metrics.

        Args:
            labels: Array of cluster labels

        Returns:
            Dictionary of evaluation metrics
        """
        # Standard metrics
        silhouette = self.metrics.silhouette_score_safe(self.point_cloud, labels)
        calinski_harabasz = self.metrics.calinski_harabasz_score_safe(self.point_cloud, labels)
        davies_bouldin = self.metrics.davies_bouldin_score_safe(self.point_cloud, labels)

        # 3D-specific metrics
        metrics_3d = self.metrics.calculate_3d_metrics(self.point_cloud, labels)

        # Composite score
        composite_score = self.metrics.calculate_composite_score(self.point_cloud, labels)

        evaluation = {
            'silhouette_score': silhouette,
            'calinski_harabasz_score': calinski_harabasz,
            'davies_bouldin_score': davies_bouldin,
            'composite_score': composite_score,
            **metrics_3d
        }

        return evaluation

    def grid_search(self, algorithm_name: str, max_combinations: int = 50) -> Dict[str, Any]:
        """
        Perform grid search to find best parameters for an algorithm.

        Args:
            algorithm_name: Name of registered algorithm
            max_combinations: Maximum number of parameter combinations to test

        Returns:
            Dictionary with best parameters, labels, and evaluation metrics
        """
        if algorithm_name not in self.algorithms:
            raise ValueError(f"Algorithm '{algorithm_name}' not registered")

        algo_config = self.algorithms[algorithm_name]
        param_grid = algo_config['param_grid']

        # Generate parameter combinations (limit if too many)
        all_combinations = self.generate_param_combinations(param_grid)
        if len(all_combinations) > max_combinations:
            # Random sample if too many combinations
            np.random.shuffle(all_combinations)
            combinations = all_combinations[:max_combinations]
        else:
            combinations = all_combinations

        best_score = -np.inf
        best_params = None
        best_labels = None
        best_evaluation = None

        print(f"Testing {len(combinations)} parameter combinations for {algorithm_name}...")

        for i, params in enumerate(combinations):
            try:
                labels = self.fit_algorithm(algorithm_name, params)

                # Skip if all points are noise or only one cluster
                if np.all(labels == -1) or len(np.unique(labels[labels != -1])) < 2:
                    continue

                evaluation = self.evaluate_clustering(labels)

                if evaluation['composite_score'] > best_score:
                    best_score = evaluation['composite_score']
                    best_params = params
                    best_labels = labels
                    best_evaluation = evaluation

                if (i + 1) % 10 == 0:
                    print(f"  Tested {i + 1}/{len(combinations)} combinations...")

            except Exception as e:
                warnings.warn(f"Error with params {params}: {str(e)}")
                continue

        if best_params is None:
            raise RuntimeError(f"No valid clustering found for {algorithm_name}")

        return {
            'algorithm_name': algorithm_name,
            'best_params': best_params,
            'best_labels': best_labels,
            'best_evaluation': best_evaluation,
            'num_combinations_tested': len(combinations)
        }

    def run_clustering_comparison(self, algorithms: List[str] = None,
                                max_combinations_per_algorithm: int = 30) -> Dict[str, Dict[str, Any]]:
        """
        Run comparison of multiple clustering algorithms.

        Args:
            algorithms: List of algorithm names to test (None = all registered)
            max_combinations_per_algorithm: Max parameter combos per algorithm

        Returns:
            Dictionary with results for each algorithm
        """
        if algorithms is None:
            algorithms = list(self.algorithms.keys())

        print(f"\n{'='*60}")
        print("CLUSTERING ALGORITHM COMPARISON")
        print(f"{'='*60}")
        print(f"Point cloud: {self.point_cloud.shape[0]} points")
        print(f"Algorithms to test: {algorithms}")
        print(f"{'='*60}\n")

        comparison_results = {}

        for algo_name in algorithms:
            print(f"\n{'='*20} {algo_name.upper()} {'='*20}")

            try:
                start_time = time.time()
                result = self.grid_search(algo_name, max_combinations_per_algorithm)
                end_time = time.time()

                result['computation_time'] = end_time - start_time
                comparison_results[algo_name] = result

                # Print results
                print(f"\n[OK] {algo_name} completed successfully")
                print(f"  Best parameters: {result['best_params']}")
                print(f"  Composite score: {result['best_evaluation']['composite_score']:.4f}")
                print(f"  Number of clusters: {result['best_evaluation']['num_clusters']}")
                print(f"  Noise ratio: {result['best_evaluation']['noise_ratio']:.3f}")
                print(f"  Computation time: {result['computation_time']:.2f}s")

            except Exception as e:
                print(f"\n[FAIL] {algo_name} failed: {str(e)}")
                comparison_results[algo_name] = {
                    'algorithm_name': algo_name,
                    'error': str(e),
                    'computation_time': 0,
                    'best_evaluation': {'composite_score': 0.0}
                }

        # Rank algorithms by composite score
        valid_results = {k: v for k, v in comparison_results.items()
                        if 'error' not in v and 'composite_score' in v.get('best_evaluation', {})}

        if valid_results:
            sorted_results = dict(sorted(valid_results.items(),
                                       key=lambda x: x[1]['best_evaluation']['composite_score'],
                                       reverse=True))

            print(f"\n{'='*60}")
            print("ALGORITHM RANKING (by composite score)")
            print(f"{'='*60}")

            for i, (algo_name, result) in enumerate(sorted_results.items(), 1):
                eval_metrics = result['best_evaluation']
                print(f"{i}. {algo_name:15s} | Score: {eval_metrics['composite_score']:.4f} | "
                      f"Clusters: {eval_metrics['num_clusters']:2d} | "
                      f"Noise: {eval_metrics['noise_ratio']:.3f} | "
                      f"Time: {result['computation_time']:.2f}s")

        # Store full results
        self.results = comparison_results

        return comparison_results

    def get_best_algorithm(self) -> Tuple[str, Dict[str, Any]]:
        """
        Get the best performing algorithm from the comparison.

        Returns:
            Tuple of (algorithm_name, result_dict)
        """
        if not self.results:
            raise ValueError("No clustering results available. Run run_clustering_comparison() first.")

        valid_results = {k: v for k, v in self.results.items()
                        if 'error' not in v and 'composite_score' in v.get('best_evaluation', {})}

        if not valid_results:
            raise ValueError("No valid clustering results found")

        best_algo_name = max(valid_results.keys(),
                           key=lambda k: valid_results[k]['best_evaluation']['composite_score'])

        return best_algo_name, valid_results[best_algo_name]

    def convert_labels_to_clusters(self, labels: np.ndarray) -> List[np.ndarray]:
        """
        Convert cluster labels to list of point index arrays.

        Args:
            labels: Array of cluster labels

        Returns:
            List of numpy arrays, each containing point indices for a cluster
        """
        unique_labels = np.unique(labels)
        clusters = []

        for label in unique_labels:
            if label != -1:  # Skip noise points
                cluster_indices = np.where(labels == label)[0]
                clusters.append(cluster_indices)

        return clusters