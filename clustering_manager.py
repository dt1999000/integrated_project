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


class ClusteringManager:
    """
    Class to manage and run multiple clustering algorithms on 3D point cloud data.
    Provides comprehensive evaluation metrics and parameter optimization.
    """
    
    def __init__(self, points: np.ndarray):
        """
        Initialize clustering manager with point cloud data.
        
        Args:
            points: Nx3 array of 3D points
        """
        self.points = points
        self.n_points = len(points)
        self.labels = None
        self.results = {}
        
        # Precompute distances for efficiency
        print("Precomputing pairwise distances for efficiency...")
        start_time = time.time()
        from sklearn.neighbors import NearestNeighbors
        self.nn = NearestNeighbors(n_neighbors=min(5, self.n_points), algorithm='auto')
        self.nn.fit(points)
        self.distances = self.nn.kneighbors_graph(mode='distance')
        print(f"Distance precomputation completed in {time.time() - start_time:.2f} seconds")
    
    def run_dbscan(self, eps: float = 0.5, min_samples: int = 10,
                   metric: str = 'euclidean', algorithm: str = 'auto',
                   leaf_size: int = 30) -> np.ndarray:
        """
        Run DBSCAN clustering algorithm.
        
        Args:
            eps: The maximum distance between two samples for one to be considered
                as in the neighborhood of the other.
            min_samples: The number of samples in a neighborhood for a point
                to be considered as a core point.
            metric: The metric to use when calculating distance between instances.
            algorithm: Algorithm used to compute the nearest neighbors.
            leaf_size: Leaf size passed to BallTree or KDTree.
            
        Returns:
            Array of cluster labels for each point.
        """
        print(f"Running DBSCAN with eps={eps}, min_samples={min_samples}")
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric,
                       algorithm=algorithm, leaf_size=leaf_size)
        self.labels = dbscan.fit_predict(self.points)
        
        # Store results
        self.results['dbscan'] = {
            'labels': self.labels,
            'params': {
                'eps': eps,
                'min_samples': min_samples,
                'metric': metric,
                'algorithm': algorithm,
                'leaf_size': leaf_size
            }
        }
        
        return self.labels
    
    def run_optics(self, min_samples: int = 10, max_eps: float = 1.0,
                  xi: float = 0.05, min_cluster_size: int = 10,
                  metric: str = 'euclidean') -> np.ndarray:
        """
        Run OPTICS clustering algorithm.
        
        Args:
            min_samples: Number of samples in a neighborhood for a point to be
                considered as a core point.
            max_eps: Maximum distance between two samples for one to be
                considered as in the neighborhood of the other.
            xi: Determines the minimum steepness on the reachability plot.
            min_cluster_size: Minimum number of points in a cluster.
            metric: The metric to use when calculating distance between instances.
            
        Returns:
            Array of cluster labels for each point.
        """
        print(f"Running OPTICS with min_samples={min_samples}, max_eps={max_eps}")
        optics = OPTICS(min_samples=min_samples, max_eps=max_eps, xi=xi,
                       min_cluster_size=min_cluster_size, metric=metric)
        self.labels = optics.fit_predict(self.points)
        
        # Store results
        self.results['optics'] = {
            'labels': self.labels,
            'params': {
                'min_samples': min_samples,
                'max_eps': max_eps,
                'xi': xi,
                'min_cluster_size': min_cluster_size,
                'metric': metric
            }
        }
        
        return self.labels
    
    def run_birch(self, threshold: float = 0.5, branching_factor: int = 50,
                  n_clusters: int = 5) -> np.ndarray:
        """
        Run BIRCH clustering algorithm.
        
        Args:
            threshold: The radius of the subcluster obtained by merging a new sample
                and the closest subcluster.
            branching_factor: Maximum number of CF subclusters in each node.
            n_clusters: Number of clusters after clustering.
            
        Returns:
            Array of cluster labels for each point.
        """
        print(f"Running BIRCH with threshold={threshold}, branching_factor={branching_factor}")
        birch = Birch(threshold=threshold, branching_factor=branching_factor, n_clusters=n_clusters)
        self.labels = birch.fit_predict(self.points)
        
        # Store results
        self.results['birch'] = {
            'labels': self.labels,
            'params': {
                'threshold': threshold,
                'branching_factor': branching_factor,
                'n_clusters': n_clusters
            }
        }
        
        return self.labels
    
    def run_agglomerative(self, n_clusters: int = 5, linkage: str = 'ward',
                       affinity: str = 'euclidean') -> np.ndarray:
        """
        Run Agglomerative clustering algorithm.
        
        Args:
            n_clusters: Number of clusters to find.
            linkage: Linkage criterion to use.
            affinity: Metric used to compute the linkage.
            
        Returns:
            Array of cluster labels for each point.
        """
        print(f"Running Agglomerative with n_clusters={n_clusters}, linkage={linkage}, affinity={affinity}")
        agglomerative = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, affinity=affinity)
        self.labels = agglomerative.fit_predict(self.points)
        
        # Store results
        self.results['agglomerative'] = {
            'labels': self.labels,
            'params': {
                'n_clusters': n_clusters,
                'linkage': linkage,
                'affinity': affinity
            }
        }
        
        return self.labels
    
    def run_hdbscan(self, min_cluster_size: int = 5, min_samples: int = 10,
                    metric: str = 'euclidean', cluster_selection_method: str = 'eom') -> np.ndarray:
        """
        Run HDBSCAN clustering algorithm.
        
        Args:
            min_cluster_size: Minimum number of points in a cluster.
            min_samples: Number of samples in a neighborhood for a point to be
                considered as a core point.
            metric: The metric to use when calculating distance between instances.
            cluster_selection_method: Method used to select clusters.
            
        Returns:
            Array of cluster labels for each point.
        """
        if not HDBSCAN_AVAILABLE:
            raise ImportError("HDBSCAN not available. Install with: pip install hdbscan")
        
        print(f"Running HDBSCAN with min_cluster_size={min_cluster_size}, min_samples={min_samples}")
        hdbscan = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                                    metric=metric, cluster_selection_method=cluster_selection_method)
        self.labels = hdbscan.fit_predict(self.points)
        
        # Store results
        self.results['hdbscan'] = {
            'labels': self.labels,
            'params': {
                'min_cluster_size': min_cluster_size,
                'min_samples': min_samples,
                'metric': metric,
                'cluster_selection_method': cluster_selection_method
            }
        }
        
        return self.labels
    