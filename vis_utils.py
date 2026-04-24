from pathlib import Path
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from PIL import Image


def save_image(arr, path):
    if arr.dtype != np.uint8:
        arr = arr - arr.min()
        if arr.max() > 0:
            arr = arr / arr.max()
        arr = (arr * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def save_progress_figure(progress_ims, gt, epochs, psnrs, save_path):
    progress_ims = [i.cpu().numpy() if isinstance(i, torch.Tensor) else i for i in progress_ims]
    gt = gt.cpu().numpy() if isinstance(gt, torch.Tensor) else gt

    n = len(progress_ims)
    fig, axes = plt.subplots(2, n + 1, figsize=(3*(n+1), 6))

    # Top row: predictions + GT
    for i, (im, ep) in enumerate(zip(progress_ims, epochs)):
        axes[0, i].imshow(im, cmap='gray', vmin=0, vmax=1)
        axes[0, i].set_title(f"Epoch {ep}")
        axes[0, i].axis('off')

    axes[0, n].imshow(gt, cmap='gray')
    axes[0, n].set_title("GT")
    axes[0, n].axis('off')

    # Bottom row: errors + PSNR
    for i, (im, psnr) in enumerate(zip(progress_ims, psnrs)):
        err = np.abs(im - gt)
        axes[1, i].imshow(err, cmap='hot', vmin=0, vmax=1)
        axes[1, i].set_xlabel(f"PSNR: {psnr:.2f}")
        # axes[1, i].axis('off')
        axes[1, i].set_xticks([])
        axes[1, i].set_yticks([])

    axes[1, n].axis('off')  # empty under GT

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def visualize_2d_gaussians(gs_model, image_tensor, subset_ratio=0.35, num_std=1.0, figsize=(10, 10), save_dir="gs_output", file_name="ellipses.png", run_dir=''):
    """
    Visualizes a subset of 2D Gaussians as ellipses overlayed on the target image.

    Args:
        gs_model: Trained GaussianSplattingND model (must be D=2).
        image_tensor: The ground truth 2D image tensor (H, W).
        subset_ratio: Float between 0 and 1. Ratio of Gaussians to draw.
        num_std: Float. Number of standard deviations for the ellipse radius.
        figsize: Tuple. Figure size for matplotlib.
    """
    assert gs_model.D == 2, "Visualization is only supported for 2D models."
    gs_model.eval()

    # 1. Get Image dimensions
    H, W = image_tensor.shape

    # 2. Extract model parameters and move to CPU numpy
    with torch.no_grad():
        mus = gs_model.mus.cpu().numpy()  # (K, 2)
        scaling_inv = gs_model.scalings_inv.cpu().numpy()  # (K, 2)
        rotations = gs_model.rotations.cpu().numpy()  # (K, 1)
        colors = torch.sigmoid(gs_model.colors).cpu().numpy()  # (K)

    K = gs_model.num_gaussians

    # 3. Randomly subsample the Gaussians
    num_to_draw = int(K * subset_ratio)
    sampled_idcs = np.random.choice(K, num_to_draw, replace=False)

    # 4. Set up the plot
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Display the ground truth image as the background
    ax.imshow(image_tensor.cpu().numpy(), cmap='gray', origin='upper')

    # 5. Draw the ellipses
    for idx in sampled_idcs:
        mu = mus[idx]
        scale_inv = scaling_inv[idx]
        rot = rotations[idx, 0]
        color_val = colors[idx]

        # --- Un-normalize Coordinates ---
        # The model mapped coords via: ((x / (S-1)) * 2 - 1) * ((S-1)/2)
        # Which algebraically simplifies to: mu = pixel_coord - (S-1)/2
        # So we reverse it: pixel_coord = mu + (S-1)/2

        # Note: indexing='ij' means mu[0] is Y, mu[1] is X
        y_center = mu[0]
        x_center = mu[1]

        # Calculate standard deviations (sigma = 1 / scaling_inv)
        # scale_inv[0] matches the Y dimension, scale_inv[1] matches the X dimension
        sigma_y = 1.0 / scale_inv[0]
        sigma_x = 1.0 / scale_inv[1]

        # Matplotlib Ellipse width/height are full diameters (2 * radius)
        # We multiply by num_std to get the desired confidence interval bound
        width = 2.0 * num_std * sigma_x
        height = 2.0 * num_std * sigma_y

        # Calculate angle. Because we mapped (y,x) to Matplotlib's (x,y) plane,
        # the rotation matrix is visually inverted. We apply a negative sign to fix the chirality.
        angle_deg = np.degrees(-rot)

        # Create an RGB color mapping from the model's grayscale color logit
        # (Using an orange/red hue with alpha so we can see the image underneath)
        ellipse_opacity = 0.8
        if len(colors.shape) == 1:
            ellipse_color = (color_val, color_val, color_val, ellipse_opacity)
        elif colors.shape[1] == 1:
            ellipse_color = (color_val[0], color_val[0], color_val[0], ellipse_opacity)
        else:
            ellipse_color = (color_val[0], color_val[1], color_val[2], ellipse_opacity)

        ellipse = Ellipse(
            xy=(x_center, y_center),
            width=width,
            height=height,
            angle=angle_deg,
            edgecolor='red',
            facecolor=ellipse_color,
            linewidth=0.5
        )
        ax.add_patch(ellipse)

    ax.set_title(f"Gaussian Splatting: {num_to_draw} / {K} Gaussians ({num_std}$\sigma$ surface)")
    ax.set_xlim(0, W - 1)
    ax.set_ylim(H - 1, 0)  # Invert Y axis to match image coordinates
    plt.axis('off')
    plt.tight_layout()
    if run_dir:
        path = Path(run_dir) / save_dir / file_name
    else:
        path = Path(save_dir) / file_name
    path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(path)
    plt.close(fig)
