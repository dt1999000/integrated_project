import numpy as np
from typing import Dict, Optional, Tuple, Set


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
        'adaptive_dbscan': {
            'base_eps': 0.35,
            'min_samples': 10,
            'eps_growth_rate': 1.0,
            'reference_distance': 15.0,
            'min_scale': 0.7,
            'max_scale': 4.0,
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
        # sklearn requires n_neighbors < n_samples_fit.
        if self.n_points > 1:
            from sklearn.neighbors import NearestNeighbors
            n_neighbors = min(5, self.n_points - 1)
            self.nn = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto')
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
        'adaptive_dbscan': {
            'base_eps', 'min_samples', 'eps_growth_rate', 'reference_distance',
            'min_scale', 'max_scale', 'algorithm', 'leaf_size'
        },
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
        elif algorithm == 'adaptive_dbscan':
            return self.run_adaptive_dbscan(**params)
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

    def run_adaptive_dbscan(
        self,
        base_eps: float = 0.35,
        min_samples: int = 5,
        eps_growth_rate: float = 1.0,
        reference_distance: float = 15.0,
        min_scale: float = 0.7,
        max_scale: float = 4.0,
        algorithm: str = 'auto',
        leaf_size: int = 30,
    ) -> np.ndarray:
        """
        Run distance-adaptive DBSCAN by isotropic coordinate scaling before clustering.

        For each point i, let r_i be 3D distance from the sensor origin (same frame as
        ``points``). Scale is centered on the **median** r of this point set:

            s_i = clip(1 + eps_growth_rate * (r_i - median(r)) / reference_distance,
                       min_scale, max_scale)

        Transformed coordinates: p'_i = p_i / s_i (x, y, and z).

        DBSCAN uses fixed ``eps=base_eps`` in transformed space. Physically, neighbor
        tolerance grows roughly with range for points farther than the median and
        shrinks for closer points. Using 3D range and scaling z avoids the old XY-only
        warp, where vertical gaps stayed in meters while horizontal gaps were shrunk,
        which blocked merging tall/sparse structure along z.

        **Why growth rate used to feel inert:** if all points shared nearly the same
        XY radius from the sensor, s_i was almost constant, so geometry only rescales
        uniformly and cluster connectivity barely changes. Median centering spreads
        s_i within each mask (roof vs bumper, etc.), so ``eps_growth_rate`` actually
        changes which pairs fall within ``base_eps``.
        """
        print(
            f"Running adaptive DBSCAN with base_eps={base_eps}, "
            f"min_samples={min_samples}, eps_growth_rate={eps_growth_rate}"
        )

        points = np.asarray(self.points, dtype=np.float64)
        range_3d = np.linalg.norm(points[:, :3], axis=1)
        r_med = float(np.median(range_3d))
        safe_ref = max(float(reference_distance), 1e-6)
        g = float(eps_growth_rate)
        scale = 1.0 + g * (range_3d - r_med) / safe_ref
        scale = np.clip(scale, float(min_scale), float(max_scale))

        transformed_points = points / scale[:, np.newaxis]

        clusterer = DBSCAN(
            eps=base_eps,
            min_samples=min_samples,
            metric='euclidean',
            algorithm=algorithm,
            leaf_size=leaf_size
        )
        self.labels = clusterer.fit_predict(transformed_points)

        self.results['adaptive_dbscan'] = {
            'labels': self.labels,
            'params': {
                'base_eps': base_eps,
                'min_samples': min_samples,
                'eps_growth_rate': eps_growth_rate,
                'reference_distance': reference_distance,
                'min_scale': min_scale,
                'max_scale': max_scale,
                'algorithm': algorithm,
                'leaf_size': leaf_size,
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

        # Build dimensions tuple for L-shape when template_dims is provided (caller passes from session_state)
        dimensions_tuple = None
        if template_dims is not None:
            dimensions_tuple = (
                float(template_dims.get('length', 4.0)),
                float(template_dims.get('width', 1.8)),
                float(template_dims.get('height', 1.6)),
            )

        # Estimate pose
        if pose_estimation_method == 'pca':
            pose_result = estimate_pose_pca(cluster_points)
            # PCA doesn't return dimensions, so we need templates
            if template_dims is None:
                template_dims = KITTI_CUBOID_TEMPLATES.get(category, KITTI_CUBOID_TEMPLATES['Unknown'])
        elif pose_estimation_method == 'l_shape':
            pose_result = estimate_pose_l_shape(
                cluster_points,
                category=category,
                ground_plane_model=ground_plane_model,
                dimensions=dimensions_tuple,
            )
            # L-shape returns dimensions; pass template_dims to cuboid_from_pose for consistency
            template_dims = template_dims  # keep for cuboid_from_pose if needed
        else:
            raise ValueError(f"Unknown pose estimation method: {pose_estimation_method}")

        # Create KITTI-format cuboid from pose
        pose_cuboid = cuboid_from_pose(
            pose_result,
            category=category,
            template_dims=template_dims,
            dimensions=dimensions_tuple,
            ground_z=ground_z
        )
        pose_cuboid['label'] = cluster_label

        return pose_cuboid


def scene_is_indoor_from_point_cloud(
    points: np.ndarray,
    max_horizontal_span_m: float = 52.0,
    min_points_per_m3: float = 1.0,
    min_points: int = 400,
    max_aabb_volume_m3: float = 22000.0,
    max_vertical_span_m: float = 14.0,
    near_xy_radius_m: float = 14.0,
    min_near_xy_fraction: float = 0.16,
    density_relax_factor: float = 0.42,
) -> bool:
    """
    Infer indoor vs outdoor from the raw LiDAR scan using AABB bounds, mean density,
    ceiling height, and near-field occupancy (combined rules).

    Hard outdoor vetoes (automotive / open scene):
        - max(dx, dy) > max_horizontal_span_m
        - AABB volume > max_aabb_volume_m3
        - fewer than min_points

    Indoor if any passes (after vetoes):
        1. Mean density N/V >= min_points_per_m3
        2. Low ceiling: dz <= max_vertical_span_m and N/V >= min_points_per_m3 * density_relax_factor
        3. Near field: fraction of points with ||(x,y)|| < near_xy_radius_m is at least
           min_near_xy_fraction (typical for room / corridor LiDAR with many wall returns)

    Args:
        points: (N, 3+) array of points in meters (sensor frame; origin near sensor).
        max_horizontal_span_m: Above this, classify as outdoor.
        min_points_per_m3: Primary density threshold (points per cubic meter in AABB).
        min_points: Below this count, default to outdoor.
        max_aabb_volume_m3: Above this, classify as outdoor.
        max_vertical_span_m: dz below this allows a relaxed density check (indoor ceiling).
        near_xy_radius_m: Radius in the xy plane for near-field fraction.
        min_near_xy_fraction: Minimum fraction of points inside that cylinder (0–1).
        density_relax_factor: Multiplier on min_points_per_m3 for the low-ceiling path.

    Returns:
        True if classified as indoor, False for outdoor or uncertain.
    """
    if points is None or len(points) < min_points:
        return False
    p = np.asarray(points[:, :3], dtype=np.float64)
    mn = np.min(p, axis=0)
    mx = np.max(p, axis=0)
    ext = mx - mn
    horizontal = float(np.maximum(ext[0], ext[1]))
    vert = float(ext[2])
    vol = float(ext[0] * ext[1] * ext[2])
    vol = max(vol, 1e-3)
    density = float(len(p) / vol)

    if vol > max_aabb_volume_m3:
        return False
    if horizontal > max_horizontal_span_m:
        return False

    if density >= min_points_per_m3:
        return True

    relaxed = min_points_per_m3 * float(density_relax_factor)
    if vert <= max_vertical_span_m and density >= relaxed:
        return True

    d_xy = np.linalg.norm(p[:, :2], axis=1)
    near_frac = float(np.mean(d_xy < float(near_xy_radius_m)))
    if near_frac >= float(min_near_xy_fraction):
        return True

    return False


def filter_clusters_by_max_volume(
    points: np.ndarray,
    labels: np.ndarray,
    template_volume: float, 
    volume_factor: float
) -> np.ndarray:
    """
    Filter out clusters whose 3D bounding box volume is significantly larger than a target.

    Args:
        points: Nx3 array of points (all clusters together)
        labels: N array of cluster labels for each point (e.g., from DBSCAN)
        template_volume: template volume in m^3. Clusters whose
                    volume exceeds this * volume_factor are removed.
        volume_factor: factor to scale volume to make the algorithm robust

    Returns:
        Boolean mask of shape (N,) where True means the point is kept.
    """
    if points.size == 0:
        return np.zeros(0, dtype=bool)
    max_volume = template_volume*volume_factor
    keep_mask = np.zeros(len(points), dtype=bool)
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels >= 0]

    for cluster_id in unique_labels:
        cluster_mask = labels == cluster_id
        cluster_points = points[cluster_mask]
        if len(cluster_points) == 0:
            continue

        min_xyz = np.min(cluster_points, axis=0)
        max_xyz = np.max(cluster_points, axis=0)
        extents = np.maximum(max_xyz - min_xyz, 0.0)
        volume = float(extents[0] * extents[1] * extents[2])

        if volume <= max_volume:
            keep_mask[cluster_mask] = True
    
    return keep_mask


def compute_sparse_iou_2d(
    mask: np.ndarray,
    cluster_points_3d: np.ndarray,
    projection,
    image_shape: Tuple[int, int],
) -> float:
    """
    Compute 2D IoU between a cluster's reprojected pixels and the full mask.

    We reproject the cluster's 3D points to 2D, build the set of occupied pixels,
    and compare it directly against all mask pixels (mask > 0) in image space.

    IoU = |cluster_pixels ∩ mask_pixels| / |cluster_pixels ∪ mask_pixels|

    Args:
        mask: HxW binary mask (e.g. SAM segment)
        cluster_points_3d: Nx3 array of 3D points in the cluster (LiDAR coords)
        projection: Projection instance with point_to_pixel(points_3d) -> (pixels, valid_mask)
        image_shape: (height, width)

    Returns:
        IoU in [0, 1], or 0.0 if union is empty
    """
    h, w = image_shape
    if len(cluster_points_3d) == 0:
        return 0.0

    pixels, valid_mask = projection.point_to_pixel(cluster_points_3d)
    valid_pixels = pixels[valid_mask]

    in_bounds = (
        (valid_pixels[:, 0] >= 0) & (valid_pixels[:, 0] < w) &
        (valid_pixels[:, 1] >= 0) & (valid_pixels[:, 1] < h)
    )
    valid_pixels = valid_pixels[in_bounds]
    if len(valid_pixels) == 0:
        return 0.0

    cluster_pixels: Set[Tuple[int, int]] = set()
    for u, v in valid_pixels.astype(int):
        cluster_pixels.add((int(u), int(v)))

    v_idx, u_idx = np.where(mask > 0)
    mask_pixels: Set[Tuple[int, int]] = set(zip(u_idx.tolist(), v_idx.tolist()))

    if len(mask_pixels) == 0 or len(cluster_pixels) == 0:
        return 0.0

    intersection = cluster_pixels & mask_pixels
    union = cluster_pixels | mask_pixels
    if len(union) == 0:
        return 0.0

    return len(intersection) / len(union)


def select_best_cluster_points(
    mask_points: np.ndarray,
    mask: np.ndarray,
    projection,
    image_shape: Tuple[int, int],
    cluster_labels: Optional[np.ndarray] = None,
    clustering_algorithm: str = 'dbscan',
    dbscan_eps: float = 0.5,
    dbscan_min_samples: int = 5,
    adaptive_dbscan_base_eps: float = 0.35,
    adaptive_dbscan_eps_growth_rate: float = 1.0,
    adaptive_dbscan_reference_distance: float = 15.0,
    adaptive_dbscan_min_scale: float = 0.7,
    adaptive_dbscan_max_scale: float = 4.0,
    hdbscan_min_cluster_size: int = 5,
    hdbscan_min_samples: int = 5,
    sparse_depth_map: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """
    Select the cluster with highest 2D IoU to the mask.

    Reprojects each cluster onto the 2D image and computes IoU with the full mask.
    IoU is normalized so that the IoU of all mask_points (reprojected) vs the mask
    equals 1.0; per-cluster IoUs are then relative to that (in [0, 1]). The cluster
    with the highest normalized IoU is returned. If mask is None, falls back to
    selecting the cluster closest to the mask center.

    Args:
        mask_points: Nx3 array of 3D points assigned to a mask
        mask: Binary mask (H, W) as numpy array
        projection: Projection object with point_to_pixel method
        image_shape: (height, width) tuple
        cluster_labels: Optional precomputed labels aligned with mask_points.
        clustering_algorithm: Clustering algorithm to run when cluster_labels is None.
        dbscan_eps: DBSCAN eps parameter
        dbscan_min_samples: DBSCAN min_samples parameter
        sparse_depth_map: Unused; kept for API compatibility.

    Returns:
        Points of the best cluster, or None if no valid cluster found
    """
    if cluster_labels is None and len(mask_points) < dbscan_min_samples:
        return None

    if cluster_labels is None:
        clustering_manager = ClusteringManager(mask_points)
        if clustering_algorithm == 'hdbscan':
            cluster_labels = clustering_manager.run_hdbscan(
                min_cluster_size=hdbscan_min_cluster_size,
                min_samples=hdbscan_min_samples
            )
        elif clustering_algorithm == 'adaptive_dbscan':
            cluster_labels = clustering_manager.run_adaptive_dbscan(
                base_eps=adaptive_dbscan_base_eps,
                min_samples=dbscan_min_samples,
                eps_growth_rate=adaptive_dbscan_eps_growth_rate,
                reference_distance=adaptive_dbscan_reference_distance,
                min_scale=adaptive_dbscan_min_scale,
                max_scale=adaptive_dbscan_max_scale,
            )
        else:
            cluster_labels = clustering_manager.run_dbscan(
                eps=dbscan_eps, min_samples=dbscan_min_samples
            )

    unique_labels = np.unique(cluster_labels)
    unique_labels = unique_labels[unique_labels >= 0]
    if len(unique_labels) == 0:
        return None

    # IoU is normalized so that the IoU of all mask_points (reprojected) vs the mask equals 1.0.
    if mask is not None:
        iou_all_mask_points = compute_sparse_iou_2d(
            mask=mask,
            cluster_points_3d=mask_points,
            projection=projection,
            image_shape=image_shape,
        )
        ref_iou = iou_all_mask_points if iou_all_mask_points > 0 else 1.0

        best_cluster_id = -1
        best_iou = -1.0
        for cluster_id in unique_labels:
            cluster_points = mask_points[cluster_labels == cluster_id]
            if len(cluster_points) < 5:
                continue
            raw_iou = compute_sparse_iou_2d(
                mask=mask,
                cluster_points_3d=cluster_points,
                projection=projection,
                image_shape=image_shape,
            )
            normalized_iou = min(1.0, raw_iou / ref_iou)
            print(f'iou for cluster id {cluster_id} is {normalized_iou}')
            if normalized_iou > best_iou:
                best_iou = normalized_iou
                best_cluster_id = cluster_id
        if best_cluster_id == -1:
            return None
        return mask_points[cluster_labels == best_cluster_id]

    return None

    