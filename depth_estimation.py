import torch
import numpy as np
from PIL import Image
import os
import logging
import warnings
import diffusers

if not torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # Ensure we're using CPU build
    assert not torch.cuda.is_available(), "CUDA should not be available"
    
try:
    from diffusers import DDIMScheduler, MarigoldDepthPipeline
    MARIGOLD_AVAILABLE = True
    diffusers.utils.logging.disable_progress_bar()
except (ImportError, RuntimeError) as e:
    print(f"Marigold not available: {e}")
    MARIGOLD_AVAILABLE = False
    # Create dummy classes to avoid NameError
    MarigoldDepthPipeline = None
    DDIMScheduler = None

# Only define MarigoldDepthCompletionPipeline if Marigold is available
if MARIGOLD_AVAILABLE:
    class MarigoldDepthCompletionPipeline(MarigoldDepthPipeline):
        """
        Pipeline for Marigold Depth Completion.
        Extends the MarigoldDepthPipeline to include depth completion functionality.
        """
        def __call__(
            self, image: Image.Image, sparse_depth: np.ndarray, num_inference_steps: int = 50,
            ensemble_size: int = 1, processing_resolution: int = 768, seed: int = 2024,
        ) -> np.ndarray:

            """
            Args:
                image (PIL.Image.Image): Input image of shape [H, W] with 3 channels.
                sparse_depth (np.ndarray): Sparse depth guidance of shape [H, W].
                num_inference_steps (int, optional): Number of denoising steps. Defaults to 50.
                ensemble_size (int, optional): Number of predictions to be ensembled. Defaults to 1.
                processing_resolution (int, optional): Resolution for processing. Defaults to 768.
                seed (int, optional): Random seed. Defaults to 2024.

            Returns:
                np.ndarray: Dense depth prediction of shape [H, W].

            """
            # Resolving variables
            device = self._execution_device
            generator = torch.Generator(device=device).manual_seed(seed)

            # Check inputs
            if not isinstance(num_inference_steps, int) or num_inference_steps < 1:
                raise ValueError("Invalid num_inference_steps")
            if type(sparse_depth) is not np.ndarray or sparse_depth.ndim != 2:
                raise ValueError("Sparse depth should be a 2D numpy ndarray with zeros at missing positions")
            if ensemble_size < 1:
                raise ValueError("Ensemble size must be at least 1")

            # Prepare empty text conditioning
            with torch.no_grad():
                if self.empty_text_embedding is None:
                    text_inputs = self.tokenizer("", padding="do_not_pad",
                        max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")
                    text_input_ids = text_inputs.input_ids.to(device)
                    self.empty_text_embedding = self.text_encoder(text_input_ids)[0]  # [1,2,1024]

            # Preprocess input images
            image, padding, original_resolution = self.image_processor.preprocess(
                image, processing_resolution=processing_resolution, device=device, dtype=self.dtype
            )  # [N,3,PPH,PPW]

            # Check sparse depth dimensions
            if sparse_depth.shape != original_resolution:
                raise ValueError(
                    f"Sparse depth dimensions ({sparse_depth.shape}) must match that of the image ({image.shape[-2:]})"
                )

            # Encode input image into latent space
            with torch.no_grad():
                image_latent, pred_latent = self.prepare_latents(image, None, generator, ensemble_size, 1)  # [N*E,4,h,w], [N*E,4,h,w]
            del image

            # Preprocess sparse depth
            sparse_depth = torch.from_numpy(sparse_depth)[None, None].float().to(device)
            sparse_mask = sparse_depth > 0
            logging.debug(f"Using {sparse_mask.int().sum().item()} guidance points")

            def affine_to_metric(depth: torch.Tensor) -> torch.Tensor:
                # Convert affine invariant depth predictions to metric depth predictions using the parametrized scale and shift. See Equation 2 of the paper.
                return (scale**2) * sparse_range * depth + (shift**2) * sparse_lower

            def latent_to_metric(latent: torch.Tensor) -> torch.Tensor:
                # Decode latent to affine invariant depth predictions and subsequently to metric depth predictions.
                affine_invariant_prediction = self.decode_prediction(latent)  # [E,1,PPH,PPW]
                prediction = affine_to_metric(affine_invariant_prediction)
                prediction = self.image_processor.unpad_image(prediction, padding)  # [E,1,PH,PW]
                prediction = self.image_processor.resize_antialias(
                    prediction, original_resolution, "bilinear", is_aa=False
                )  # [1,1,H,W]
                return prediction

            def loss_l1l2(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
                # Compute L1 and L2 loss between input and target.
                out_l1 = torch.nn.functional.l1_loss(input, target)
                out_l2 = torch.nn.functional.mse_loss(input, target)
                out = out_l1 + out_l2
                return out

            self.scheduler.set_timesteps(num_inference_steps, device=device)

            ensemble_predictions = []
            for ensemble_idx in self.progress_bar(range(ensemble_size), desc="Processing ensemble members...", leave=False):

                current_image_latent = image_latent[ensemble_idx:ensemble_idx+1]  # [1,4,h,w]
                current_pred_latent = pred_latent[ensemble_idx:ensemble_idx+1]   # [1,4,h,w]

                # Set up optimization targets and compute the range and lower bound of the sparse depth
                scale, shift = torch.nn.Parameter(torch.ones(1, device=device)), torch.nn.Parameter(torch.ones(1, device=device))
                current_pred_latent = torch.nn.Parameter(current_pred_latent)
                sparse_range = (sparse_depth[sparse_mask].max() - sparse_depth[sparse_mask].min()).item() # (cmax − cmin)
                sparse_lower = (sparse_depth[sparse_mask].min()).item() # cmin

                # Set up optimizer
                optimizer = torch.optim.Adam([ {"params": [scale, shift], "lr": 0.005},
                                               {"params": [current_pred_latent] , "lr": 0.05 }])

                # Denoising loop
                for _, t in enumerate(
                    self.progress_bar(self.scheduler.timesteps, desc=f"Marigold-DC steps ({str(device)})...", leave=False)
                ):
                    optimizer.zero_grad()

                    # Forward pass through the U-Net
                    batch_latent = torch.cat([current_image_latent, current_pred_latent], dim=1)  # [1,8,h,w]
                    noise = self.unet(
                        batch_latent, t, encoder_hidden_states=self.empty_text_embedding, return_dict=False
                    )[0]  # [1,4,h,w]

                    # Compute pred_epsilon to later rescale the depth latent gradient
                    with torch.no_grad():
                        alpha_prod_t = self.scheduler.alphas_cumprod[t]
                        beta_prod_t = 1 - alpha_prod_t
                        pred_epsilon = (alpha_prod_t**0.5) * noise + (beta_prod_t**0.5) * current_pred_latent

                    step_output = self.scheduler.step(noise, t, current_pred_latent, generator=generator)

                    # Preview the final output depth with Tweedie's formula (See Equation 1 of the paper)
                    pred_original_sample = step_output.pred_original_sample

                    # Decode to metric space, compute loss with guidance and backpropagate
                    current_metric_estimate = latent_to_metric(pred_original_sample)
                    loss = loss_l1l2(current_metric_estimate[sparse_mask], sparse_depth[sparse_mask])
                    loss.backward()

                    # Scale gradients up
                    with torch.no_grad():
                        pred_epsilon_norm = torch.linalg.norm(pred_epsilon).item()
                        depth_latent_grad_norm = torch.linalg.norm(current_pred_latent.grad).item()
                        scaling_factor = pred_epsilon_norm / max(depth_latent_grad_norm, 1e-8)
                        current_pred_latent.grad *= scaling_factor

                    # Execute the update step through guidance backprop
                    optimizer.step()

                    # Execute update of the latent with regular denoising diffusion step
                    with torch.no_grad():
                        current_pred_latent.data = self.scheduler.step(noise, t, current_pred_latent, generator=generator).prev_sample

                    del pred_original_sample, current_metric_estimate, step_output, pred_epsilon, noise
                    torch.cuda.empty_cache()

                # Decode prediction from latent into pixel space for current ensemble member
                with torch.no_grad():
                    current_prediction = latent_to_metric(current_pred_latent.detach())
                    ensemble_predictions.append(current_prediction)

            del image_latent

            # Ensemble the predictions
            if ensemble_size > 1:
                # Take per-pixel median
                ensemble_tensor = torch.cat(ensemble_predictions, dim=0)  # [E,1,H,W]
                prediction = ensemble_tensor.median(dim=0, keepdim=True).values  # [1,1,H,W]
            else:
                prediction = ensemble_predictions[0]

            # return Numpy array
            prediction = self.image_processor.pt_to_numpy(prediction)  # [N,H,W,1]
            self.maybe_free_model_hooks()

            return prediction.squeeze()
else:
    # Create a dummy class to avoid NameError when Marigold is not available
    MarigoldDepthCompletionPipeline = None


class DepthEstimator:
    def __init__(self, use_marigold: bool = True, use_full_precision: bool = False, use_tiny_vae: bool = False,
                 camera_intrinsic: np.ndarray = None, camera_extrinsic: np.ndarray = None,
                 camera_to_lidar_transform: np.ndarray = None):
        """
        Initialize DepthEstimator with optional camera parameters.
        
        Args:
            use_marigold: Whether to use Marigold for depth estimation
            use_full_precision: Use float32 instead of float16/bf16
            use_tiny_vae: Use lightweight VAE for depth completion
            camera_intrinsic: 3x3 camera intrinsic matrix K (optional, can be set later)
            camera_extrinsic: 4x4 camera extrinsic matrix (optional, can be set later)
            camera_to_lidar_transform: 4x4 transformation matrix from camera to LiDAR (optional, can be set later)
        """
        self.use_marigold = use_marigold
        self.pipe = None
        self.dc_pipe = None  # Depth completion pipeline
        self.use_full_precision = use_full_precision
        self.use_tiny_vae = use_tiny_vae
        
        # Camera parameters (can be set during initialization or later)
        self.camera_intrinsic = camera_intrinsic
        self.camera_extrinsic = camera_extrinsic if camera_extrinsic is not None else np.eye(4)
        self.camera_to_lidar_transform = camera_to_lidar_transform
        
        # Cached projection object (created when camera params are set)
        self._projection = None
        
        if use_marigold and MARIGOLD_AVAILABLE:
            print("Initializing Marigold depth estimation model...")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            torch_dtype = torch.float32 if use_full_precision else (torch.float16 if device.type == "cuda" else torch.float32)
            self.pipe = MarigoldDepthPipeline.from_pretrained("prs-eth/marigold-v1-0",
                torch_dtype=torch_dtype).to(device)
            print("Marigold model loaded successfully")
            
            # Initialize depth completion pipeline
            print("Initializing Marigold-DC pipeline...")
            self.dc_pipe = MarigoldDepthCompletionPipeline.from_pretrained(
                "prs-eth/marigold-depth-v1-0", prediction_type="depth"
            ).to(device, dtype=torch_dtype)
            self.dc_pipe.scheduler = DDIMScheduler.from_config(
                self.dc_pipe.scheduler.config, timestep_spacing="trailing"
            )
            
            if use_tiny_vae:
                print("Using lightweight VAE for depth completion")
                del self.dc_pipe.vae
                self.dc_pipe.vae = diffusers.AutoencoderTiny.from_pretrained("madebyollin/taesd").to(device, dtype=torch_dtype)
            
            print("Marigold-DC pipeline initialized successfully")
        else:
            if not MARIGOLD_AVAILABLE:
                raise RuntimeError("Marigold is not available. Please install diffusers: pip install diffusers")
            else:
                raise RuntimeError("Marigold initialization failed. Please check your installation.")
    
    def set_camera_params(self, camera_intrinsic: np.ndarray, 
                         camera_extrinsic: np.ndarray = None,
                         camera_to_lidar_transform: np.ndarray = None):
        """
        Set camera parameters and create/recreate projection object.
        
        Args:
            camera_intrinsic: 3x3 camera intrinsic matrix K
            camera_extrinsic: 4x4 camera extrinsic matrix (optional, defaults to identity)
            camera_to_lidar_transform: 4x4 transformation matrix from camera to LiDAR
        """
        self.camera_intrinsic = camera_intrinsic
        if camera_extrinsic is not None:
            self.camera_extrinsic = camera_extrinsic
        if camera_to_lidar_transform is not None:
            self.camera_to_lidar_transform = camera_to_lidar_transform
        
        # Invalidate cached projection object
        self._projection = None
    
    def _get_projection(self, point_cloud: np.ndarray = None):
        """
        Get or create projection object. Requires camera parameters to be set.
        
        Args:
            point_cloud: Optional point cloud for projection initialization
            
        Returns:
            Projection object
        """
        if self.camera_intrinsic is None or self.camera_to_lidar_transform is None:
            raise ValueError("Camera parameters not set. Call set_camera_params() first.")
        
        # Create projection if not cached or if point cloud changed
        if self._projection is None or (point_cloud is not None and 
                                       not np.array_equal(self._projection.point_cloud, point_cloud)):
            from pointcloud_projection import Projection
            
            # Use dummy point cloud if none provided (will be updated when needed)
            dummy_pc = point_cloud if point_cloud is not None else np.array([[0, 0, 0]])
            
            self._projection = Projection(
                camera_intrinsic=self.camera_intrinsic,
                camera_extrinsic=self.camera_extrinsic,
                camera_to_lidar_transform=self.camera_to_lidar_transform,
                point_cloud=dummy_pc
            )
        
        return self._projection

    def get_depth_map(self, image):
        """
        Get metric depth map using Marigold model.
        
        Args:
            image: Input image as numpy array (H, W, 3)
            
        Returns:
            depth_map: Metric depth map as numpy array (H, W)
        """
        return self.get_depth_map_marigold(image)

    def get_depth_map_marigold(self, image):   
        """
        Get metric depth map using Marigold model.
        
        Args:
            image: Input image as numpy array (H, W, 3)
            
        Returns:
            depth_map: Metric depth map as numpy array (H, W)
        """
        if not MARIGOLD_AVAILABLE:
            raise RuntimeError("Marigold is not available. Please install diffusers: pip install diffusers")
        
        if self.pipe is None:
            raise RuntimeError("Marigold pipeline not initialized. Please initialize DepthEstimator with use_marigold=True")
        
        # Convert image to PIL if needed
        if isinstance(image, np.ndarray):
            image_pil = Image.fromarray(image.astype(np.uint8))
        else:
            image_pil = image
        
        output = self.pipe(image_pil)
        depth = output.prediction
        depth_np = np.array(depth)
        
        # Ensure depth map is 2D (H, W)
        if depth_np.ndim > 2:
            depth_np = depth_np.squeeze()
        if depth_np.ndim != 2:
            raise ValueError(f"Expected 2D depth map from Marigold, got shape {depth_np.shape}")
        depth_np = depth_np.reshape(depth_np.shape[1], depth_np.shape[2])
        print(f"Depth map shape: {depth_np.shape}, min: {depth_np.min():.2f}, max: {depth_np.max():.2f}")
        return depth_np

    def reconstruct_points_from_depth(self, depth_map: np.ndarray, 
                                     depth_threshold_min: float = 0.1,
                                     depth_threshold_max: float = 100.0,
                                     stride: int = 1) -> np.ndarray:
        """
        Reconstruct 3D point cloud from metric depth map and project to LiDAR coordinates.
        Uses camera parameters stored in the object.
        
        Args:
            depth_map: Metric depth map (H, W) in meters
            depth_threshold_min: Minimum valid depth value (meters)
            depth_threshold_max: Maximum valid depth value (meters)
            stride: Sampling stride for point cloud (1 = all pixels, 2 = every other pixel, etc.)
            
        Returns:
            points_lidar: Nx3 array of 3D points in LiDAR coordinate system
        """
        if self.camera_intrinsic is None or self.camera_to_lidar_transform is None:
            raise ValueError("Camera parameters not set. Initialize with camera params or call set_camera_params() first.")
        
        # Ensure depth_map is 2D (H, W)
        depth_map = np.asarray(depth_map)
        original_shape = depth_map.shape
        print(f"DEBUG: reconstruct_points_from_depth received depth_map with shape: {original_shape}, ndim: {depth_map.ndim}")
        
        # Handle different possible shapes
        if depth_map.ndim == 1:
            raise ValueError(f"Depth map is 1D with shape {depth_map.shape}, expected 2D (H, W)")
        elif depth_map.ndim == 2:
            # Already 2D, use as is
            pass
        elif depth_map.ndim == 3:
            # Could be (1, H, W), (H, W, 1), or (C, H, W)
            if depth_map.shape[0] == 1:
                depth_map = depth_map[0, :, :]  # (1, H, W) -> (H, W)
            elif depth_map.shape[2] == 1:
                depth_map = depth_map[:, :, 0]  # (H, W, 1) -> (H, W)
            else:
                # (C, H, W) - take first channel
                depth_map = depth_map[0, :, :]
        elif depth_map.ndim == 4:
            # Could be (1, 1, H, W) or (B, C, H, W)
            depth_map = depth_map.squeeze()
            if depth_map.ndim != 2:
                # If still not 2D, take first slice
                depth_map = depth_map.reshape(-1, depth_map.shape[-1])[0].reshape(depth_map.shape[-2:])
        else:
            raise ValueError(f"Unexpected depth map shape: {depth_map.shape}, expected 2D (H, W)")
        
        # Final validation
        if depth_map.ndim != 2:
            raise ValueError(f"After processing, depth map still has {depth_map.ndim} dimensions with shape {depth_map.shape} (original shape: {original_shape})")
        
        print(f"DEBUG: After shape normalization, depth_map shape: {depth_map.shape}")
        height, width = depth_map.shape
        
        # Create pixel coordinate grid
        u_coords, v_coords = np.meshgrid(
            np.arange(0, width, stride),
            np.arange(0, height, stride)
        )
        
        # Flatten coordinates
        u = u_coords.flatten()
        v = v_coords.flatten()
        
        # Get corresponding depth values
        depths = depth_map[v, u]
        
        # Filter valid depths
        valid_mask = (depths >= depth_threshold_min) & (depths <= depth_threshold_max)
        u_valid = u[valid_mask]
        v_valid = v[valid_mask]
        depths_valid = depths[valid_mask]
        
        print(f"Reconstructing point cloud:")
        print(f"  Total pixels: {len(u)}")
        print(f"  Valid depth pixels: {len(depths_valid)} ({100*len(depths_valid)/len(u):.1f}%)")
        
        # Back-project to 3D camera coordinates
        # For each pixel (u, v) with depth d:
        # [X, Y, Z]^T = d * K^-1 @ [u, v, 1]^T
        
        K_inv = np.linalg.inv(self.camera_intrinsic)
        
        # Create homogeneous pixel coordinates
        pixels_homogeneous = np.stack([u_valid, v_valid, np.ones_like(u_valid)], axis=0)  # (3, N)
        
        # Back-project to normalized camera coordinates
        points_normalized = K_inv @ pixels_homogeneous  # (3, N)
        
        # Scale by depth to get 3D points in camera coordinates
        points_camera = points_normalized * depths_valid  # (3, N)
        points_camera = points_camera.T  # (N, 3)
        
        # Convert to homogeneous coordinates for transformation
        points_camera_homo = np.hstack([points_camera, np.ones((len(points_camera), 1))])  # (N, 4)
        
        # Transform to LiDAR coordinates
        points_lidar = (self.camera_to_lidar_transform @ points_camera_homo.T).T[:, :3]  # (N, 3)
        
        print(f"  Reconstructed {len(points_lidar)} points in LiDAR coordinates")
        print(f"  X range: [{points_lidar[:, 0].min():.2f}, {points_lidar[:, 0].max():.2f}]")
        print(f"  Y range: [{points_lidar[:, 1].min():.2f}, {points_lidar[:, 1].max():.2f}]")
        print(f"  Z range: [{points_lidar[:, 2].min():.2f}, {points_lidar[:, 2].max():.2f}]")
        
        return points_lidar

    def estimate_depth_and_reconstruct(self, image: np.ndarray,
                                      use_marigold: bool = None,
                                      depth_threshold_min: float = 0.1,
                                      depth_threshold_max: float = 100.0,
                                      stride: int = 1,
                                      lidar_point_cloud: np.ndarray = None,
                                      use_sparse_depth_prior: bool = True,
                                      num_inference_steps: int = 50,
                                      ensemble_size: int = 1,
                                      processing_resolution: int = 768,
                                      seed: int = 2024):
        """
        Complete pipeline: estimate depth from image and reconstruct 3D point cloud in LiDAR coordinates.
        Optionally uses sparse depth from LiDAR as a prior for better accuracy (Marigold-DC).
        Uses camera parameters stored in the object.
        
        Args:
            image: Input image as numpy array (H, W, 3)
            use_marigold: Whether to use Marigold (deprecated, always uses Marigold now). 
                         If None, uses self.use_marigold
            depth_threshold_min: Minimum valid depth value (meters)
            depth_threshold_max: Maximum valid depth value (meters)
            stride: Sampling stride for point cloud (1 = all pixels, 2 = every other pixel, etc.)
            lidar_point_cloud: Optional Nx3 array of LiDAR points to use as sparse depth prior.
                              If provided and use_sparse_depth_prior=True, uses Marigold-DC for better accuracy.
            use_sparse_depth_prior: If True and lidar_point_cloud is provided, uses Marigold-DC 
                                   with sparse depth guidance instead of regular Marigold.
            num_inference_steps: Number of denoising steps for Marigold-DC (if using sparse prior)
            ensemble_size: Number of predictions to ensemble for Marigold-DC (if using sparse prior)
            processing_resolution: Resolution for Marigold-DC processing (if using sparse prior)
            seed: Random seed for Marigold-DC (if using sparse prior)
            
        Returns:
            Dictionary containing:
                - 'depth_map': Depth map as numpy array (H, W)
                - 'points_lidar': Nx3 array of 3D points in LiDAR coordinates
                - 'sparse_depth': Sparse depth map if lidar_point_cloud was provided (H, W)
                - 'used_sparse_prior': Boolean indicating if sparse depth prior was used
        """
        if self.camera_intrinsic is None or self.camera_to_lidar_transform is None:
            raise ValueError("Camera parameters not set. Initialize with camera params or call set_camera_params() first.")
        
        if use_marigold is None:
            use_marigold = self.use_marigold
        
        print("\n" + "="*60)
        print("DEPTH ESTIMATION AND 3D RECONSTRUCTION PIPELINE")
        print("="*60)
        
        # Determine if we should use sparse depth prior
        use_prior = use_sparse_depth_prior and lidar_point_cloud is not None and self.dc_pipe is not None
        
        if use_prior:
            print("\n[Step 1/3] Creating sparse depth map from LiDAR points...")
            h, w = image.shape[:2]
            sparse_depth = self.create_sparse_depth_map(
                point_cloud=lidar_point_cloud,
                image_shape=(h, w)
            )
            
            # Check if we have enough sparse points to be useful
            n_sparse_points = np.sum(sparse_depth > 0)
            coverage = 100 * n_sparse_points / (h * w)
            
            if coverage < 0.1:  # Less than 0.1% coverage
                print(f"Warning: Sparse depth coverage is very low ({coverage:.2f}%). Falling back to regular Marigold.")
                use_prior = False
            else:
                print(f"\n[Step 2/3] Estimating depth with sparse depth prior (Marigold-DC)...")
                print(f"  Using {n_sparse_points} sparse depth points ({coverage:.2f}% coverage) as guidance")
                
                # Use Marigold-DC with sparse depth as prior
                depth_map = self.complete_depth(
                    image=image,
                    sparse_depth=sparse_depth,
                    num_inference_steps=num_inference_steps,
                    ensemble_size=ensemble_size,
                    processing_resolution=processing_resolution,
                    seed=seed
                )
        else:
            if lidar_point_cloud is not None and not use_sparse_depth_prior:
                print("\n[Step 1/2] Estimating depth map (without sparse depth prior)...")
            elif lidar_point_cloud is None:
                print("\n[Step 1/2] Estimating depth map (no LiDAR points provided)...")
            else:
                print("\n[Step 1/2] Estimating depth map...")
            
            # Use regular Marigold depth estimation
            depth_map = self.get_depth_map_marigold(image)
            sparse_depth = None
            
        # Step 2/3: Reconstruct 3D points
        step_num = "3" if use_prior else "2"
        print(f"\n[Step {step_num}/{step_num}] Reconstructing 3D point cloud...")
        points_lidar = self.reconstruct_points_from_depth(
            depth_map=depth_map,
            depth_threshold_min=depth_threshold_min,
            depth_threshold_max=depth_threshold_max,
            stride=stride
        )
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETE")
        if use_prior and sparse_depth is not None:
            n_sparse_points = np.sum(sparse_depth > 0)
            coverage = 100 * n_sparse_points / (h * w)
            print(f"Used sparse depth prior: {n_sparse_points} LiDAR points ({coverage:.2f}% coverage)")
        print("="*60 + "\n")
        
        result = {
            'depth_map': depth_map,
            'points_lidar': points_lidar,
            'used_sparse_prior': use_prior
        }
        
        if sparse_depth is not None:
            result['sparse_depth'] = sparse_depth
        
        return result
    
    def create_sparse_depth_map(self, point_cloud: np.ndarray,
                                image_shape: tuple) -> np.ndarray:
        """
        Create sparse depth map by back-projecting 3D LiDAR points onto 2D image.
        Uses camera parameters stored in the object.
        
        Args:
            point_cloud: Nx3 array of 3D points in LiDAR coordinates
            image_shape: (height, width) of the image
            
        Returns:
            sparse_depth: HxW numpy array with depth values at projected pixel locations, zeros elsewhere
        """
        if self.camera_intrinsic is None or self.camera_to_lidar_transform is None:
            raise ValueError("Camera parameters not set. Initialize with camera params or call set_camera_params() first.")
        
        # Get or create projection object
        projection = self._get_projection(point_cloud)
        
        # Update projection point cloud if needed
        if not np.array_equal(projection.point_cloud, point_cloud):
            projection.point_cloud = point_cloud
        
        # Project 3D points to 2D pixels
        pixels, valid_mask = projection.point_to_pixel(point_cloud)
        
        # Filter points that are within image bounds
        h, w = image_shape
        in_bounds = (
            (pixels[:, 0] >= 0) & (pixels[:, 0] < w) &
            (pixels[:, 1] >= 0) & (pixels[:, 1] < h)
        )
        valid_mask &= in_bounds
        
        # Initialize sparse depth map
        sparse_depth = np.zeros((h, w), dtype=np.float32)
        
        if np.any(valid_mask):
            # Get valid pixels and corresponding points
            valid_pixels = pixels[valid_mask].astype(int)
            valid_points = point_cloud[valid_mask]
            
            # Transform points to camera coordinates to get depth (z-coordinate)
            lidar_to_camera = np.linalg.inv(self.camera_to_lidar_transform)
            points_homo = np.hstack([valid_points, np.ones((len(valid_points), 1))])
            points_cam = (lidar_to_camera @ points_homo.T).T[:, :3]
            
            # Depth is the z-coordinate in camera space
            depths = points_cam[:, 2]
            
            # Only keep positive depths (in front of camera)
            positive_depth_mask = depths > 0
            valid_pixels = valid_pixels[positive_depth_mask]
            depths = depths[positive_depth_mask]
            
            # Fill sparse depth map (handle multiple points mapping to same pixel by taking closest)
            for (u, v), depth in zip(valid_pixels, depths):
                if sparse_depth[v, u] == 0 or depth < sparse_depth[v, u]:
                    sparse_depth[v, u] = depth
        
        n_points = np.sum(sparse_depth > 0)
        print(f"Created sparse depth map: {n_points} valid depth points out of {len(point_cloud)} total points")
        print(f"  Coverage: {100*n_points/(h*w):.2f}% of pixels")
        if n_points > 0:
            print(f"  Depth range: [{sparse_depth[sparse_depth>0].min():.2f}, {sparse_depth[sparse_depth>0].max():.2f}]m")
        
        return sparse_depth
    
    def complete_depth(self, image: np.ndarray, sparse_depth: np.ndarray,
                      num_inference_steps: int = 50, ensemble_size: int = 1,
                      processing_resolution: int = 768, seed: int = 2024) -> np.ndarray:
        """
        Complete sparse depth map using Marigold-DC.
        
        Args:
            image: Input image as numpy array (H, W, 3) in RGB format
            sparse_depth: Sparse depth map (H, W) with zeros at missing positions
            num_inference_steps: Number of denoising steps
            ensemble_size: Number of predictions to ensemble
            processing_resolution: Resolution for processing
            seed: Random seed
            
        Returns:
            dense_depth: Completed dense depth map (H, W)
        """
        if not MARIGOLD_AVAILABLE or self.dc_pipe is None:
            raise RuntimeError("Marigold-DC pipeline not available. Please ensure Marigold is installed and initialized.")
        
        # Convert image to PIL
        if isinstance(image, np.ndarray):
            image_pil = Image.fromarray(image.astype(np.uint8))
        else:
            image_pil = image
        
        # Adjust parameters for CPU if needed
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            processing_resolution_non_cuda = 512
            num_inference_steps_non_cuda = 10
            ensemble_size_non_cuda = 1
            
            if processing_resolution > processing_resolution_non_cuda:
                print(f"CUDA not found: Reducing processing_resolution to {processing_resolution_non_cuda}")
                processing_resolution = processing_resolution_non_cuda
            if num_inference_steps > num_inference_steps_non_cuda:
                print(f"CUDA not found: Reducing num_inference_steps to {num_inference_steps_non_cuda}")
                num_inference_steps = num_inference_steps_non_cuda
            if ensemble_size > ensemble_size_non_cuda:
                print(f"CUDA not found: Reducing ensemble_size to {ensemble_size_non_cuda}")
                ensemble_size = ensemble_size_non_cuda
        
        # Run depth completion
        print(f"\nRunning Marigold-DC depth completion...")
        print(f"  Steps: {num_inference_steps}, Ensemble: {ensemble_size}, Resolution: {processing_resolution}")
        
        dense_depth = self.dc_pipe(
            image=image_pil,
            sparse_depth=sparse_depth,
            num_inference_steps=num_inference_steps,
            ensemble_size=ensemble_size,
            processing_resolution=processing_resolution,
            seed=seed
        )
        
        print(f"Depth completion completed!")
        print(f"  Output shape: {dense_depth.shape}")
        print(f"  Depth range: [{dense_depth.min():.2f}, {dense_depth.max():.2f}]m")
        
        return dense_depth
