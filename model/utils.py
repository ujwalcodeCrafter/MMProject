"""
Utility functions for masking, metrics, and visualization.

Provides:
    - Patch-based random masking
    - PSNR and SSIM calculation
    - Error heatmap generation
    - Tensor/image conversion helpers
"""

import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim_func
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server use
import matplotlib.pyplot as plt

# ─── Configuration ───────────────────────────────────────────────────────────
IMG_SIZE = 64        # Input image resolution (64×64)
PATCH_SIZE = 4       # Each patch is 4×4 pixels (finer masking for better reconstruction)
NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2   # 256 total patches


# ─── Masking ─────────────────────────────────────────────────────────────────

def create_patch_mask(image_size=IMG_SIZE, patch_size=PATCH_SIZE, mask_ratio=0.5):
    """
    Create a random patch-based binary mask.

    Divides the image into a grid of (image_size/patch_size)^2 patches
    and randomly zeros out `mask_ratio` fraction of them.

    Args:
        image_size  : Spatial size of the square image (pixels).
        patch_size  : Spatial size of each square patch (pixels).
        mask_ratio  : Fraction of patches to mask (0.0–1.0).

    Returns:
        mask           : Float tensor of shape (1, H, W). 1 = visible, 0 = masked.
        masked_indices : 1-D tensor with the linear indices of masked patches.
    """
    patches_per_side = image_size // patch_size
    num_patches = patches_per_side ** 2
    num_masked = int(num_patches * mask_ratio)

    # Random permutation → first `num_masked` indices are masked
    perm = torch.randperm(num_patches)
    masked_indices = perm[:num_masked]

    # Build patch-level mask and expand to pixel-level
    mask = torch.ones(num_patches)
    mask[masked_indices] = 0.0
    mask = mask.reshape(patches_per_side, patches_per_side)
    mask = mask.repeat_interleave(patch_size, dim=0)  # expand rows
    mask = mask.repeat_interleave(patch_size, dim=1)  # expand cols
    mask = mask.unsqueeze(0)  # (1, H, W)

    return mask, masked_indices


def apply_mask(image, mask):
    """
    Element-wise multiply an image by a binary mask.

    Args:
        image : Tensor (C, H, W) or (B, C, H, W).
        mask  : Tensor (1, H, W) or (B, 1, H, W).

    Returns:
        Masked image (same shape as input).
    """
    return image * mask


# ─── Metrics ─────────────────────────────────────────────────────────────────

def calculate_psnr(original, reconstructed):
    """
    Peak Signal-to-Noise Ratio between two images.

    Args:
        original      : ndarray (H, W, C) in [0, 1].
        reconstructed : ndarray (H, W, C) in [0, 1].

    Returns:
        PSNR in dB (float). Returns inf when images are identical.
    """
    mse = np.mean((original - reconstructed) ** 2)
    if mse < 1e-10:
        return float('inf')
    return 10.0 * np.log10(1.0 / mse)


def calculate_ssim(original, reconstructed):
    """
    Structural Similarity Index between two images.

    Args:
        original      : ndarray (H, W, C) in [0, 1].
        reconstructed : ndarray (H, W, C) in [0, 1].

    Returns:
        SSIM value (float in [-1, 1], higher is better).
    """
    return ssim_func(
        original,
        reconstructed,
        channel_axis=2,
        data_range=1.0,
    )


# ─── Visualization helpers ──────────────────────────────────────────────────

def generate_error_heatmap(original, reconstructed, save_path):
    """
    Compute a per-pixel absolute-error map and save it as a 'hot' heatmap.

    Args:
        original      : ndarray (H, W, C) in [0, 1].
        reconstructed : ndarray (H, W, C) in [0, 1].
        save_path     : File path for the output PNG.
    """
    error = np.mean(np.abs(original - reconstructed), axis=2)

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(error, cmap='hot', vmin=0, vmax=0.5)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=120, transparent=False,
                facecolor='#0b0d17')
    plt.close(fig)


def tensor_to_numpy(tensor):
    """
    Convert a (C, H, W) or (1, C, H, W) tensor to (H, W, C) ndarray in [0, 1].
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    return tensor.detach().cpu().permute(1, 2, 0).numpy().clip(0.0, 1.0)


def save_tensor_as_image(tensor, save_path):
    """Save a tensor as a PNG image (uses matplotlib)."""
    img = tensor_to_numpy(tensor)
    plt.imsave(save_path, img)
