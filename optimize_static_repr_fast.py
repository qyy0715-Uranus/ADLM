import dataclasses
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import kornia
import torch
import torch.nn as nn
import numpy as np
import faiss
import faiss.contrib.torch_utils  # CRITICAL: Enables PyTorch interoperability
from medmnist import INFO

from vis_utils import visualize_2d_gaussians, save_image, save_progress_figure


@dataclasses.dataclass
class TrainingConfig:
    max_epochs: int = 500                                     # Number of optimization epochs
    k_neighborhood: int = 15                                  # Number of nearby Gaussians to consider (larger K -> slower)
    knn_update_rate: int = 10                                 # Number of steps after which the KNN will be updated
    reg_weight_scale: float = 0.1                             # Regularization weight for Gaussian scales being too small/big
    reg_weight_pos: float = 0.0                               # Regularization weight for Gaussian position leaving image bounds
    learning_rate_start: float = 1e-1                         # Starting learning rate for optimization
    learning_rate_end: float = 1e-5                           # Ending learning rate for optimization
    compression_factor: float = 0.1                           # Percentage of parameters used relative to number of pixels
    point_sample_prop: float = 0.9                            # Percentage of total coordinates to supervise in minibatch
    logging: bool = True                                      # If false, all prints and image logging will be skipped
    logging_vis_steps: Tuple[int, ...] = (0, 20, 100, 500)    # Epochs at which to visualize
    logging_gauss_vis_prop: float = 1.0                       # Proportion of ellipses to draw on the visualization plots


def build_rotation_matrix_2d(theta):
    '''
    Builds a batch of 2x2 rotation matrices from angles.
    theta: (K, 1) tensor of rotation angles
    '''
    K = theta.shape[0]
    theta = theta.squeeze(-1)  # (K,)

    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    R = torch.empty((K, 2, 2), device=theta.device, dtype=theta.dtype)
    R[:, 0, 0] = cos_theta
    R[:, 0, 1] = -sin_theta
    R[:, 1, 0] = sin_theta
    R[:, 1, 1] = cos_theta

    return R


def build_rotation_matrix_3d_quaternion(q):
    '''
    Builds a batch of 3x3 rotation matrices from a batch of quaternions.
    q: (K, 4) tensor (w, x, y, z)
    '''
    # Normalize quaternions to ensure they are unit quaternions
    q_norm = torch.nn.functional.normalize(q, p=2, dim=1)

    w, x, y, z = q_norm[:, 0], q_norm[:, 1], q_norm[:, 2], q_norm[:, 3]

    K = q.shape[0]
    R = torch.empty((K, 3, 3), device=q.device, dtype=q.dtype)

    # Pre-compute reused terms
    x2, y2, z2 = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    # Fill the rotation matrix
    R[:, 0, 0] = 1.0 - 2.0 * (y2 + z2)
    R[:, 0, 1] = 2.0 * (xy - wz)
    R[:, 0, 2] = 2.0 * (xz + wy)

    R[:, 1, 0] = 2.0 * (xy + wz)
    R[:, 1, 1] = 1.0 - 2.0 * (x2 + z2)
    R[:, 1, 2] = 2.0 * (yz - wx)

    R[:, 2, 0] = 2.0 * (xz - wy)
    R[:, 2, 1] = 2.0 * (yz + wx)
    R[:, 2, 2] = 1.0 - 2.0 * (x2 + y2)

    return R


class GaussianRepresentationND(nn.Module):
    def __init__(self, num_gaussians, spatial_dims, image_range=(0.0, 1.0)):
        super().__init__()
        assert len(spatial_dims) in [2, 3], "Only 2D and 3D are supported."
        self.D = len(spatial_dims)
        self.spatial_dims = torch.tensor(spatial_dims)

        # 1. Mean (mu) - Initialized randomly in [0, spatial_dims]
        self.mus = nn.Parameter(torch.rand((num_gaussians, self.D)) * (self.spatial_dims - 1))

        # 2. Scaling (inverse) - Ensures positive variance
        scaling_mean = max(1., torch.tensor(spatial_dims).amax().item() * 0.05)  # 2 voxels or 5% of image shape
        scaling_std = scaling_mean * 0.25
        noise = torch.randn(num_gaussians, self.D) * scaling_std
        min_width = 0.5  # To make sure all Gaussian stds are larger than zero
        scalings = (torch.full((num_gaussians, self.D), scaling_mean) + noise).clamp(min=min_width)
        self.scalings_inv_ = nn.Parameter(torch.log(torch.sqrt(1 / scalings)))  # log because we will be exponentiating to prevent negative values

        # 3. Rotation
        if self.D == 3:
            # Quaternions (w, x, y, z) for 3D
            self.rotations = nn.Parameter(torch.zeros(num_gaussians, 4))
            self.rotations.data[:, 0] = 1.0  # Identity
        else:
            # Single angle for 2D
            self.rotations = nn.Parameter(torch.zeros(num_gaussians, 1))

        # 4. Color (Logits)
        self.colors = nn.Parameter(torch.zeros(num_gaussians))
        self.img_min = image_range[0]
        self.img_max = image_range[1]

    @property
    def num_gaussians(self):
        return self.mus.shape[0]

    @property
    def num_params(self):
        return np.prod(self.mus.shape) + np.prod(self.scalings.shape) + np.prod(self.rotations.shape) + np.prod(self.colors.shape)

    @property
    def scalings_inv(self):
        return torch.exp(self.scalings_inv_)  # Exp to prevent it from being negative

    @property
    def scalings(self):
        return 1 / self.scalings_inv

    def initialize_from_image(self, image_tensor, lambda_init=0.3):
        """Clean, content-adaptive initialization based strictly on image gradients."""
        print(f"Initializing {self.num_gaussians} anisotropic Gaussians adaptively...")
        device = self.mus.device

        # 1. Compute gradients for structure and orientation
        grads = torch.gradient(image_tensor)
        grad_tensor = torch.stack(grads, dim=-1)
        grad_mag = torch.sqrt(torch.sum(grad_tensor ** 2, dim=-1))
        # Normalize gradient magnitude for proportional scaling
        grad_mag_norm = grad_mag / (grad_mag.max() + 1e-8)

        # 2. Probability Map
        grad_sum = grad_mag.sum()
        grad_prob = (grad_mag / grad_sum) if grad_sum > 0 else torch.zeros_like(grad_mag)
        uniform_prob = 1.0 / grad_mag.numel()

        # --- Spatial Exclusion (NMS) ON GRADIENTS ONLY ---
        pixels_per_gaussian = image_tensor.numel() / self.num_gaussians
        k_size = max(3, int(pixels_per_gaussian ** (1 / self.D)))
        if k_size % 2 == 0:
            k_size += 1
        pad = k_size // 2

        # View the raw gradients
        grad_view = grad_prob.view(1, 1, *image_tensor.shape)
        if self.D == 2:
            grad_max = torch.nn.functional.max_pool2d(grad_view, kernel_size=k_size, stride=1, padding=pad)
        else:
            grad_max = torch.nn.functional.max_pool3d(grad_view, kernel_size=k_size, stride=1, padding=pad)

        # A pixel is only a peak if it is the local max AND it actually has a gradient (ignores flat regions)
        is_peak = (grad_view == grad_max) & (grad_view > 1e-6)

        # Suppress non-peaks in the gradient map
        grad_prob_nms = torch.where(is_peak.view_as(grad_prob), grad_prob, grad_prob * 1e-4)

        # Flat regions (0 gradient) safely bypass NMS and receive exactly the uniform baseline
        P_init = (1.0 - lambda_init) * grad_prob_nms + lambda_init * uniform_prob

        # Final safety checks to prevent multinomial crashes
        P_init = P_init + 1e-8
        P_init = torch.nan_to_num(P_init, nan=1e-8, posinf=1.0, neginf=1e-8)
        P_init /= P_init.sum()

        # 3. Sample coordinates
        P_init_flat = P_init.flatten()
        # Float64 cast prevents the cumulative sum underflow bug
        sampled_indices = torch.multinomial(P_init_flat.double(), num_samples=self.num_gaussians, replacement=False)

        shape = torch.tensor(image_tensor.shape, device=device)
        grid_axes = [torch.arange(s, device=device, dtype=torch.float32) for s in shape]
        grid = torch.stack(torch.meshgrid(*grid_axes, indexing='ij'), dim=-1).view(-1, self.D)

        new_mus = grid[sampled_indices]

        # 4. Extract gradient vectors for orientation
        grad_flat = grad_tensor.view(-1, self.D)
        sampled_grads = grad_flat[sampled_indices]
        sampled_dirs = torch.nn.functional.normalize(sampled_grads, p=2, dim=-1, eps=1e-8)

        # 5. Build Properties
        with torch.no_grad():
            # --- Rotations ---
            if self.D == 2:
                theta = torch.atan2(sampled_dirs[:, 1], sampled_dirs[:, 0])
                new_rotations = theta.unsqueeze(-1)
            elif self.D == 3:
                v0, v1, v2 = sampled_dirs[:, 0], sampled_dirs[:, 1], sampled_dirs[:, 2]
                q = torch.zeros((self.num_gaussians, 4), device=device)
                q[:, 0] = 1.0 + v0
                q[:, 1] = 0.0
                q[:, 2] = -v2
                q[:, 3] = v1
                degenerate_mask = (q[:, 0] < 1e-6)
                q[degenerate_mask] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)
                new_rotations = torch.nn.functional.normalize(q, p=2, dim=-1)

            # --- Gradient-Proportional Anisotropic Scaling ---
            scaling_mean = max(1., torch.tensor(image_tensor.shape).amax().item() * 0.05)
            sampled_grad_mag = grad_mag_norm.flatten()[sampled_indices]

            new_scalings = torch.zeros((self.num_gaussians, self.D), device=device)

            # High gradient => sharper perpendicular width, longer parallel width
            perp_scale = scaling_mean / (1.0 + 8.0 * sampled_grad_mag)
            par_scale = scaling_mean * (1.0 + 7.0 * sampled_grad_mag)

            new_scalings[:, 0] = perp_scale
            if self.D == 2:
                new_scalings[:, 1] = par_scale
            else:
                new_scalings[:, 1:] = par_scale.unsqueeze(-1)

            noise = torch.randn_like(new_scalings) * (scaling_mean * 0.02)
            new_scalings = (new_scalings + noise).clamp(min=0.15)
            new_scalings_inv = torch.log(torch.sqrt(1 / new_scalings))

            # --- Color assignment ---
            img_flat = image_tensor.flatten()
            sampled_colors = img_flat[sampled_indices]

            self.img_min = image_tensor.amin().item()
            self.img_max = image_tensor.amax().item()
            colors_nrmd = sampled_colors / self.img_max
            colors_nrmd = torch.clamp(colors_nrmd, 1e-6, 1.0 - 1e-6)
            new_colors = -torch.log(1.0 / colors_nrmd - 1.0)

            # 6. Apply to Model Parameters in-place
            self.mus.data = new_mus
            self.rotations.data = new_rotations
            self.scalings_inv_.data = new_scalings_inv
            self.colors.data = new_colors

    def get_rotation_matrix(self,
                            deformation_rot: Optional[torch.Tensor] = None,
                            ):
        """Builds rotation matrices for 2D or 3D."""
        rotation = self.rotations
        if deformation_rot is not None:
            rotation += deformation_rot
        if self.D == 2:
            return build_rotation_matrix_2d(rotation)
        else:
            return build_rotation_matrix_3d_quaternion(rotation)

    def compute_sigma_inverse(self,
                              deformation_sigma: Optional[torch.Tensor] = None,
                              deformation_rot: Optional[torch.Tensor] = None,
                              ):
        R = self.get_rotation_matrix(deformation_rot)
        scaling_inv = self.scalings_inv  # 1/s
        if deformation_sigma is not None:
            scaling_inv = scaling_inv / (1 + deformation_sigma * scaling_inv)  # Equal to 1/(s+delta_s)
        S_inv = torch.diag_embed(scaling_inv ** 2)
        return torch.bmm(R, torch.bmm(S_inv, R.transpose(1, 2)))

    def forward(self,
                coords: torch.Tensor,
                top_k_idcs: torch.Tensor,
                deformation_mu: Optional[torch.Tensor] = None,
                deformation_sigma: Optional[torch.Tensor] = None,
                deformation_rot: Optional[torch.Tensor] = None,
                deformation_intens: Optional[torch.Tensor] = None, # TODO
                ):
        """
        coords: (N, D) query coordinates
        top_k_idcs: (N, K_neighbors) indices of nearest Gaussians for efficient rendering
        deformations: (num_gaussians, D) optional deformations which move Gaussian
        """
        mus = self.mus[top_k_idcs]  # (N, K_neighbors, D)
        if deformation_mu is not None:
            mus += deformation_mu
        x_minus_mu = coords.unsqueeze(1) - mus  # (N, K_neighbors, D)

        sigmas_inv = self.compute_sigma_inverse(deformation_sigma, deformation_rot)[top_k_idcs]  # (N, K_neighbors, D, D)

        # M_v = Sigma_inv @ (x - mu)
        M_v = torch.einsum('nkij,nkj->nki', sigmas_inv, x_minus_mu)

        # Distance power
        inner_term = torch.sum(x_minus_mu * M_v, dim=2)
        gaussian_weights = torch.exp(-0.5 * inner_term)
        # --- NEW: Normalized Weighted Sum ---
        # Add epsilon to prevent division by zero in empty space
        weights_nrmd = gaussian_weights / (torch.sum(gaussian_weights, dim=1, keepdim=True) + 1e-8)

        # Map color logits to [0, 1]
        colors = self.colors[top_k_idcs]
        if deformation_intens is not None:
            colors += deformation_intens
        colors = torch.sigmoid(colors) * (self.img_max - self.img_min) + self.img_min
        # Weighted sum using normalized weights
        color_pred = torch.sum(weights_nrmd * colors, dim=1)

        return color_pred


# --- Data Loading Utility ---
def load_medmnist(dataset_flag="bloodmnist", download=True, size=224, idx=0):
    info = INFO[dataset_flag]
    # DataClass = getattr(medmnist, info['python_class'])
    # dataset = DataClass(split='test', download=download, size=size)
    from medmnist import PneumoniaMNIST

    dataset = PneumoniaMNIST(split="test",
                                   download=True,
                                   size=size)
    # Get first sample
    img, _ = dataset[idx]
    img_np = np.array(img)

    # Normalize and format
    if len(img_np.shape) == 3 and img_np.shape[-1] == 3:
        # Convert RGB to Grayscale for simplicity in this demo
        img_np = np.mean(img_np, axis=-1)
    if img_np.shape[0] == 1:
        img_np = img_np[0]

    img_tensor = torch.tensor(img_np, dtype=torch.float32)
    img_tensor = (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min())

    D = len(img_tensor.shape)
    return img_tensor, D


def compute_regularization_losses(gs_model: GaussianRepresentationND,
                                  spatial_dims: torch.Tensor,
                                  lambda_pos=0.0, lambda_scale=0.0,
                                  min_scale=None, max_scale=None):
    """
    Penalizes Gaussians that drift outside the image bounds
    or shrink/grow beyond stable scaling thresholds.
    """
    # 1. Position Regularization
    if lambda_pos > 0.0:
        # Penalize any mu value that exceeds image bounds
        # relu(x - 1.0) is >0 only when x > 1.0
        out_of_bounds_pos = torch.nn.functional.relu(torch.abs(gs_model.mus - spatial_dims/2) - spatial_dims/2)
        loss_pos = out_of_bounds_pos.mean()
        loss_pos *= lambda_pos
    else:
        loss_pos = torch.tensor([0.0], dtype=gs_model.mus.dtype, device=gs_model.mus.device)

    # 2. Scale Regularization
    if min_scale is None:
        min_scale = 0.5  # Half a voxel
    if max_scale is None:
        max_scale = max(spatial_dims) / 2  # Proportion of the image
    if lambda_scale > 0.0:
        # Penalize if it gets too small (too wide) or too large (too narrow)
        too_wide = torch.nn.functional.relu(gs_model.scalings - max_scale)  # If smaller than max_scale -> negative -> no gradient
        too_narrow = torch.nn.functional.relu(min_scale - gs_model.scalings)  # If larger than min_scale -> negative -> no gradient
        loss_scale = (too_wide + too_narrow).mean()
        loss_scale *= lambda_scale
    else:
        loss_scale = torch.tensor([0.0], dtype=gs_model.mus.dtype, device=gs_model.mus.device)

    return loss_pos, loss_scale


# --- Training Loop ---
def train_gs(run_dir: Path, gs: GaussianRepresentationND, img_tensor: torch.Tensor,
             params: TrainingConfig):
    shape_tensor = torch.tensor(img_tensor.shape, device=img_tensor.device)

    # 1. Create coordinate grid
    grid_axes = [torch.arange(s, device=device) for s in img_tensor.shape]
    coords_voxel_ = torch.stack(torch.meshgrid(*grid_axes, indexing='ij'), dim=-1).view(-1, D).float()
    values_ = img_tensor.flatten()
    total_coords = coords_voxel_.shape[0]
    batch_size = int(total_coords * params.point_sample_prop)
    print(batch_size)

    # 2. Initialize Optimizers
    optimizer = torch.optim.Adam(gs.parameters(), lr=params.learning_rate_start)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params.max_epochs, eta_min=params.learning_rate_end)
    loss_fn = torch.nn.L1Loss()

    # 3. FAISS Setup (GPU)
    res = faiss.StandardGpuResources()
    cpu_index = faiss.IndexFlatL2(D)
    gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
    gpu_index.add(gs.mus.detach())
    _, top_k_idcs = gpu_index.search(coords_voxel_, params.k_neighborhood)

    t0 = time.time()
    progress_ims = []
    progress_ims_epochs = []
    progress_ims_psnrs = []
    loss, loss_rec, loss_scale, loss_pos = [torch.tensor([torch.inf])]*4
    for epoch in range(params.max_epochs):
        if params.logging:
            if epoch % 100 == 0:
                # Progress print
                pred = gs.forward(coords_voxel_, top_k_idcs).reshape(img_tensor.shape)
                psnr = kornia.metrics.psnr(img_tensor[None], pred[None], max_val=gs.img_max).mean()
                print(f"Epoch {epoch:04d} | Time: {time.time() - t0:.2f}s | LR: {optimizer.param_groups[0]['lr']:.2e} | "
                      f"PSNR: {psnr.item()} | Loss: {loss.item():.4f} | Loss recon: {loss_rec.item():.4f} | "
                      f"Loss scale: {loss_scale.item():.4f} | Loss pos: {loss_pos.item():.4f}")
            if epoch in params.logging_vis_steps:
                # Visual logging. Evaluate every coordinate.
                progress_ims_epochs.append(epoch)
                pred = gs.forward(coords_voxel_, top_k_idcs).reshape(img_tensor.shape)
                psnr = kornia.metrics.psnr(img_tensor[None], pred[None], max_val=gs.img_max).mean()
                progress_ims_psnrs.append(psnr.item())
                if gs.D == 3:
                    pred = pred[..., ::4].reshape(pred.shape[-3], pred.shape[-2], 4, 4).permute(2, 0, 3, 1).reshape(
                        4 * pred.shape[-3], 4 * pred.shape[-2])
                progress_ims.append(pred.detach().cpu())
                if gs.D == 2:
                    visualize_2d_gaussians(
                        gs_model=gs,
                        image_tensor=pred.detach(),
                        subset_ratio=params.logging_gauss_vis_prop,
                        num_std=1.5,  # Distance in number of standard devs to draw ellipses
                        run_dir=run_dir,
                        file_name=f'ellipses_{epoch}.png',
                    )

        # Compose batch from random subset of points
        rand_indices = torch.randperm(total_coords, device=device)[:batch_size]
        batch_coords = coords_voxel_[rand_indices]
        batch_values = values_[rand_indices]
        batch_top_k = top_k_idcs[rand_indices]

        # Forward, loss computation
        optimizer.zero_grad()
        preds = gs.forward(batch_coords, batch_top_k)
        loss_rec = loss_fn(preds, batch_values)
        # Regularization Losses
        loss_pos, loss_scale = compute_regularization_losses(gs, shape_tensor,
                                                             lambda_pos=params.reg_weight_pos, lambda_scale=params.reg_weight_scale)
        loss = loss_rec + loss_pos + loss_scale

        # Parameter update
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Update FAISS Index periodically to track moving Gaussians
        if (epoch + 1) % params.knn_update_rate == 0:
            gpu_index.reset()
            gpu_index.add(gs.mus.detach())
            _, top_k_idcs = gpu_index.search(coords_voxel_, params.k_neighborhood)
    # Save parameters
    print(f"Training complete. Save path: {run_dir / 'representation.pt'}")
    torch.save(gs.state_dict(), run_dir / 'representation.pt')
    epoch = params.max_epochs
    pred = gs.forward(coords_voxel_, top_k_idcs).reshape(img_tensor.shape)
    progress_ims_epochs.append(epoch)
    psnr = kornia.metrics.psnr(img_tensor[None], pred[None], max_val=gs.img_max)
    progress_ims_psnrs.append(psnr.item())

    print(f"Epoch {epoch:04d} | Time: {time.time() - t0:.2f}s | LR: {optimizer.param_groups[0]['lr']:.2e} | "
          f"PSNR: {psnr.item()} | Loss: {loss.item():.4f} | Loss recon: {loss_rec.item():.4f} | "
          f"Loss scale: {loss_scale.item():.4f} | Loss pos: {loss_pos.item():.4f}")
    if params.logging:
        gt = img_tensor
        if gs.D == 3:
            pred = pred[..., ::4].reshape(pred.shape[-3], pred.shape[-2], 4, 4).permute(2, 0, 3, 1).reshape(4*pred.shape[-3], 4*pred.shape[-2])
            gt = gt[..., ::4].reshape(gt.shape[-3], gt.shape[-2], 4, 4).permute(2, 0, 3, 1).reshape(4*gt.shape[-3], 4*gt.shape[-2])
        progress_ims.append(pred.detach().cpu())
        save_progress_figure(progress_ims, gt, progress_ims_epochs, progress_ims_psnrs, run_dir / "progress.png")
        if gs.D == 2:
            visualize_2d_gaussians(
                gs_model=gs,
                image_tensor=pred.detach(),
                subset_ratio=params.logging_gauss_vis_prop,
                num_std=1.5,  # Distance in number of standard devs to draw ellipses
                run_dir=run_dir,
                file_name=f'ellipses_{epoch}.png',
            )
    return gs, pred, coords_voxel_, top_k_idcs


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}...")
    # ----- Params ----------
    params = TrainingConfig()

    # ----- Load Data
    # Test with 2D (BloodMNIST) or 3D (OrganMNIST3D)
    dataset_flag = "pneumoniamnist"
    # dataset_flag = "synapsemnist3d"
    img, D = load_medmnist(dataset_flag, idx=3)
    assert D in {2, 3}
    # Params: position + widths + rotations + intensity
    params_per_gauss = 2+2+1+1 if D == 2 else 3+3+4+1  # If 3D, we use 4 rotation params (quartenions)
    img = img.to(device)
    num_gaussians = int(np.prod(img.shape) * params.compression_factor / params_per_gauss)

    # ----- Init Model
    gs = GaussianRepresentationND(num_gaussians, img.shape).to(device)
    gs.initialize_from_image(img)
    logging_dir = 'logging'
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
    run_id += f"_{D}D"
    run_dir = Path(logging_dir) / run_id
    run_dir.mkdir(exist_ok=True, parents=True)
    print('Run name:', run_dir)
    print(f"Image shape: {img.shape}, num pixels: {np.prod(img.shape)}, "
          f"num gaussians: {gs.num_gaussians} ({gs.num_params} params), "
          f"num neighbors: {params.k_neighborhood}")

    # ---- Optimize Representation
    gs, pred_image, coords_, final_idcs = train_gs(run_dir, gs, img, params)
