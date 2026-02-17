import numpy as np
from typing import Dict, Optional


# Standard clustering algorithms from scikit-learn
from sklearn.cluster import DBSCAN, OPTICS, Birch, AgglomerativeClustering, HDBSCAN

# Import constants
from .constants import KITTI_CUBOID_TEMPLATES



class ClusteringManager:
    """
    Class to manage and run multiple clustering algorithms on 3D point cloud data.
    Provides comprehensive evaluation metrics and parameter optimization.
    """
    
    # Default parameters for each clustering algorithm
    DEFAULT_PARAMS = {
        'hdbscan': {
            'min_cluster_size': 5,
            'min_samples': 5,
            'metric': 'euclidean',
            'cluster_selection_method': 'eom'
        },
        'dbscan': {
            'eps': 0.5,
            'min_samples': 10,
            'metric': 'euclidean',
            'algorithm': 'auto',
            'leaf_size': 30
        },
        'optics': {
            'min_samples': 10,
            'max_eps': 1.0,
            'xi': 0.05,
            'min_cluster_size': 10,
            'metric': 'euclidean'
        },
        'birch': {
            'threshold': 0.5,
            'branching_factor': 50,
            'n_clusters': 5
        },
        'agglomerative': {
            'n_clusters': 5,
            'linkage': 'ward'
        }
    }

    def __init__(self, points: np.ndarray, params: Optional[Dict[str, Dict]] = None):
        """
        Initialize clustering manager with point cloud data.

        Args:
            points: Nx3 array of 3D points
            params: Optional dictionary of algorithm parameters. If provided, will
                   override default parameters. Structure:
                   {
                       'hdbscan': {'min_cluster_size': 15, ...},
                       'dbscan': {'eps': 0.5, ...},
                       ...
                   }
        """
        self.points = points
        self.n_points = len(points)
        self.labels = None
        self.results = {}

        # Initialize params with defaults, then override with provided params
        self.params = {}
        for algo, default_params in self.DEFAULT_PARAMS.items():
            self.params[algo] = default_params.copy()

        # Override with user-provided params
        if params:
            self.update_params(params)

        # Precompute distances for efficiency (only if enough points)
        if self.n_points > 1:
            from sklearn.neighbors import NearestNeighbors
            self.nn = NearestNeighbors(n_neighbors=min(5, self.n_points), algorithm='auto')
            self.nn.fit(points)
            self.distances = self.nn.kneighbors_graph(mode='distance')

    def update_params(self, params: Dict[str, Dict]):
        """
        Update algorithm parameters.

        Args:
            params: Dictionary of algorithm parameters to update.
                   Can update individual algorithms or specific parameters.
                   Example: {'hdbscan': {'min_cluster_size': 20}}
        """
        for algo, algo_params in params.items():
            if algo in self.params:
                self.params[algo].update(algo_params)
            else:
                self.params[algo] = algo_params

    def get_params(self, algorithm: str) -> Dict:
        """
        Get parameters for a specific algorithm.

        Args:
            algorithm: Algorithm name ('hdbscan', 'dbscan', etc.)

        Returns:
            Dictionary of parameters for the algorithm
        """
        return self.params.get(algorithm, {})

    # Valid parameters for each algorithm (used to filter override params)
    VALID_PARAMS = {
        'hdbscan': {'min_cluster_size', 'min_samples', 'metric', 'cluster_selection_method'},
        'dbscan': {'eps', 'min_samples', 'metric', 'algorithm', 'leaf_size'},
        'optics': {'min_samples', 'max_eps', 'xi', 'min_cluster_size', 'metric'},
        'birch': {'threshold', 'branching_factor', 'n_clusters'},
        'agglomerative': {'n_clusters', 'linkage'}
    }

    def run_clustering(self, algorithm: str = 'hdbscan', **override_params) -> np.ndarray:
        """
        Run clustering using stored parameters with optional overrides.

        Args:
            algorithm: Algorithm to use ('hdbscan', 'dbscan', 'optics', 'birch', 'agglomerative')
            **override_params: Parameters to override for this run only

        Returns:
            Array of cluster labels for each point
        """
        # Get base params and apply overrides (filter to valid params for this algorithm)
        params = self.params.get(algorithm, {}).copy()
        valid_keys = self.VALID_PARAMS.get(algorithm, set())
        filtered_overrides = {k: v for k, v in override_params.items() if k in valid_keys}
        params.update(filtered_overrides)

        if algorithm == 'hdbscan':
            return self.run_hdbscan(**params)
        elif algorithm == 'dbscan':
            return self.run_dbscan(**params)
        elif algorithm == 'optics':
            return self.run_optics(**params)
        elif algorithm == 'birch':
            return self.run_birch(**params)
        elif algorithm == 'agglomerative':
            return self.run_agglomerative(**params)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    
    def run_dbscan(self, eps: float = 0.5, min_samples: int = 5,
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
    
    def run_optics(self, min_samples: int = 5, max_eps: float = 1.0,
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
    
    def run_agglomerative(self, n_clusters: int = 5, linkage: str = 'ward') -> np.ndarray:
        """
        Run Agglomerative clustering algorithm.
        
        Args:
            n_clusters: Number of clusters to find.
            linkage: Linkage criterion to use.
            
        Returns:
            Array of cluster labels for each point.
        """
        print(f"Running Agglomerative with n_clusters={n_clusters}, linkage={linkage}")
        agglomerative = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        self.labels = agglomerative.fit_predict(self.points)
        
        # Store results
        self.results['agglomerative'] = {
            'labels': self.labels,
            'params': {
                'n_clusters': n_clusters,
                'linkage': linkage
            }
        }
        
        return self.labels
    
    def run_hdbscan(self, min_cluster_size: int = 5, min_samples: int = 5,
                    metric: str = 'euclidean', cluster_selection_method: str = 'eom') -> np.ndarray:
        """
        Run HDBSCAN clustering algorithm (using sklearn implementation).

        Args:
            min_cluster_size: Minimum number of points in a cluster.
            min_samples: Number of samples in a neighborhood for a point to be
                considered as a core point.
            metric: The metric to use when calculating distance between instances.
            cluster_selection_method: Method used to select clusters.

        Returns:
            Array of cluster labels for each point.
        """
        print(f"Running HDBSCAN with min_cluster_size={min_cluster_size}, min_samples={min_samples}")
        clusterer = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                           metric=metric, cluster_selection_method=cluster_selection_method)
        self.labels = clusterer.fit_predict(self.points)
        
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
    
    def generate_cuboids_from_clusters(self, labels: np.ndarray, 
                                     cuboid_size: float = 1.0) -> Dict[str, Dict]:
        """
        Generate cuboids from cluster labels for interactive visualization.
        
        Args:
            labels: Cluster labels for each point
            cuboid_size: Size of each cuboid in meters
            
        Returns:
            Dictionary mapping cluster_id to cuboid data
        """
        unique_labels = np.unique(labels)
        cuboids = []
        
        for cluster_id in unique_labels:
            if cluster_id == -1:  # Skip noise points
                continue
            
            # Get points in this cluster
            cluster_points = self.points[labels == cluster_id]
            if len(cluster_points) < 4:  # Need at least 4 points for a cuboid
                continue
            
            # Calculate bounding box
            min_x = np.min(cluster_points[:,0])
            max_x = np.max(cluster_points[:,0])
            min_y = np.min(cluster_points[:,1])
            max_y = np.max(cluster_points[:,1])
            min_z = np.min(cluster_points[:,2])
            max_z = np.max(cluster_points[:,2])
            
            cuboids.append({
                'min_x': min_x,
                'min_y': min_y,
                'min_z': min_z,
                'max_x': max_x,
                'max_y': max_y,
                'max_z': max_z,
                'label': cluster_id
            })
        
        return cuboids

    def generate_cuboid_from_template(
        self,
        cluster_points: np.ndarray,
        category: str,
        cluster_label: int = -1,
        ground_z: Optional[float] = None
    ) -> Optional[Dict]:
        """
        Generate a cuboid using a class-specific template fitted to cluster points.

        The cuboid is positioned so that:
        - X: Starts at min_x of cluster (front face), extends by template length
        - Y: Centered on the cluster's Y centroid
        - Z: Starts at ground_z (or min_z of cluster if not provided), extends by template height

        Args:
            cluster_points: Nx3 array of points in the cluster
            category: Object category (e.g., 'Car', 'Pedestrian')
            cluster_label: Cluster label for reference
            ground_z: Optional ground plane z value at this location. If provided,
                      uses this as the base z instead of cluster min_z.

        Returns:
            Cuboid dict with min/max bounds, or None if insufficient points
        """
        if len(cluster_points) < 4:
            return None

        # Get template for this category (fallback to 'Unknown')
        template = KITTI_CUBOID_TEMPLATES.get(category, KITTI_CUBOID_TEMPLATES['Unknown'])
        length = template['length']
        width = template['width']
        height = template['height']

        # Compute anchor points from cluster
        min_x = np.min(cluster_points[:, 0])  # Front face of object
        center_y = np.mean(cluster_points[:, 1])  # Y centroid

        # Use ground_z if provided, otherwise fall back to cluster min_z
        if ground_z is not None:
            base_z = ground_z
        else:
            base_z = np.min(cluster_points[:, 2])  # Ground level from cluster

        # Position cuboid:
        # - X: starts at min_x, extends backward by length
        # - Y: centered on center_y
        # - Z: starts at base_z (ground), extends upward by height
        cuboid = {
            'min_x': min_x,
            'max_x': min_x + length,
            'min_y': center_y - width / 2,
            'max_y': center_y + width / 2,
            'min_z': base_z,
            'max_z': base_z + height,
            'label': cluster_label,
            'template_category': category,
            'template_dims': template
        }

        return cuboid

    def generate_cuboid_from_pose_estimation(
        self,
        cluster_points: np.ndarray,
        category: str,
        cluster_label: int = -1,
        pose_estimation_method: str = 'pca',
        ground_plane_model: Optional[np.ndarray] = None,
        template_dims: Optional[Dict[str, float]] = None
    ) -> Optional[Dict]:
        """
        Generate a KITTI-format cuboid using pose estimation (PCA or L-shape fitting).

        Args:
            cluster_points: Nx3 array of points in the cluster
            category: Object category (e.g., 'Car', 'Pedestrian')
            cluster_label: Cluster label for reference
            pose_estimation_method: 'pca' or 'l_shape' - method for pose estimation
            ground_plane_model: Optional [a, b, c, d] plane equation from RANSAC.
                              Used to compute ground z for height calculation.
            template_dims: Optional dict with 'length', 'width', 'height' from templates.
                          Only used for PCA method (L-shape returns its own dimensions).

        Returns:
            KITTI-format cuboid dict with 'center', 'yaw', 'length', 'width', 'height',
            'corners', and min/max bounds, or None if insufficient points
        """
        if len(cluster_points) < 4:
            return None

        from .pose_estimation import estimate_pose_pca, estimate_pose_l_shape, cuboid_from_pose

        # Compute ground_z at cluster centroid using plane model
        ground_z = None
        if ground_plane_model is not None:
            a, b, c, d = ground_plane_model
            if abs(c) > 1e-6:
                center_x = np.mean(cluster_points[:, 0])
                center_y = np.mean(cluster_points[:, 1])
                ground_z = -(a * center_x + b * center_y + d) / c

        # Estimate pose
        if pose_estimation_method == 'pca':
            pose_result = estimate_pose_pca(cluster_points)
            # PCA doesn't return dimensions, so we need templates
            if template_dims is None:
                template_dims = KITTI_CUBOID_TEMPLATES.get(category, KITTI_CUBOID_TEMPLATES['Unknown'])
        elif pose_estimation_method == 'l_shape':
            pose_result = estimate_pose_l_shape(cluster_points, ground_plane_model=ground_plane_model)
            # L-shape returns dimensions, so don't use templates
            template_dims = None
        else:
            raise ValueError(f"Unknown pose estimation method: {pose_estimation_method}")

        # Create KITTI-format cuboid from pose
        pose_cuboid = cuboid_from_pose(
            pose_result,
            category=category,
            template_dims=template_dims,
            ground_z=ground_z
        )
        pose_cuboid['label'] = cluster_label

        return pose_cuboid

    