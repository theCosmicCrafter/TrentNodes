"""
Shared image processing utilities for TrentNodes.

Provides GPU-accelerated image operations used across multiple nodes:
- Grayscale conversion
- Sobel edge detection
- Affine transforms
"""

import torch
import torch.nn.functional as F


# Standard grayscale weights (ITU-R BT.601)
GRAYSCALE_WEIGHTS = (0.299, 0.587, 0.114)


def to_grayscale(image: torch.Tensor) -> torch.Tensor:
    """
    Convert RGB image to grayscale using standard weights.

    Args:
        image: Tensor of shape (B, H, W, C) or (H, W, C) in [0, 1] range

    Returns:
        Grayscale tensor of shape (B, H, W) or (H, W)
    """
    if image.dim() == 3:
        # (H, W, C) -> add batch dimension
        image = image.unsqueeze(0)
        squeeze_batch = True
    else:
        squeeze_batch = False

    gray = (
        GRAYSCALE_WEIGHTS[0] * image[..., 0] +
        GRAYSCALE_WEIGHTS[1] * image[..., 1] +
        GRAYSCALE_WEIGHTS[2] * image[..., 2]
    )

    if squeeze_batch:
        gray = gray.squeeze(0)

    return gray


def get_sobel_kernels(device: torch.device) -> tuple:
    """
    Get Sobel kernels for edge detection.

    Args:
        device: torch device for the kernels

    Returns:
        Tuple of (sobel_x, sobel_y) kernels, each shape (1, 1, 3, 3)
    """
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=torch.float32, device=device
    ).view(1, 1, 3, 3)

    sobel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        dtype=torch.float32, device=device
    ).view(1, 1, 3, 3)

    return sobel_x, sobel_y


def extract_edges(image: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Extract edges using Sobel filters.

    Args:
        image: Tensor of shape (B, H, W, C) or (H, W, C) in [0, 1] range
        device: torch device for computation

    Returns:
        Edge magnitude tensor of shape (B, H, W)
    """
    if image.dim() == 3:
        image = image.unsqueeze(0)

    # Convert to grayscale
    gray = to_grayscale(image)

    # Get Sobel kernels
    sobel_x, sobel_y = get_sobel_kernels(device)

    # Apply Sobel filters
    gray_4d = gray.unsqueeze(1)
    edges_x = F.conv2d(gray_4d, sobel_x, padding=1)
    edges_y = F.conv2d(gray_4d, sobel_y, padding=1)
    edges = torch.sqrt(edges_x ** 2 + edges_y ** 2)

    return edges.squeeze(1)


def apply_affine_transform(
    image: torch.Tensor,
    scale: float,
    tx: float,
    ty: float,
    device: torch.device
) -> torch.Tensor:
    """
    Apply scale and translation transform to image.

    Args:
        image: Tensor of shape (B, H, W, C)
        scale: Scale factor (1.0 = no change)
        tx: Translation in x (pixels)
        ty: Translation in y (pixels)
        device: torch device for computation

    Returns:
        Transformed image tensor of shape (B, H, W, C)
    """
    B, H, W, C = image.shape
    inv_scale = 1.0 / scale

    # Build affine transformation matrix
    theta = torch.zeros(B, 2, 3, device=device, dtype=image.dtype)
    theta[:, 0, 0] = inv_scale
    theta[:, 1, 1] = inv_scale
    theta[:, 0, 2] = -tx / (W / 2) * inv_scale
    theta[:, 1, 2] = -ty / (H / 2) * inv_scale

    # Apply transform
    image_bchw = image.permute(0, 3, 1, 2)
    grid = F.affine_grid(theta, image_bchw.shape, align_corners=False)
    transformed = F.grid_sample(
        image_bchw, grid, mode='bilinear',
        padding_mode='border', align_corners=False
    )

    return transformed.permute(0, 2, 3, 1)


def compute_intensity_difference(
    image1: torch.Tensor,
    image2: torch.Tensor,
    scale: float = 255.0
) -> float:
    """
    Compute mean intensity difference between two images.

    Args:
        image1: First image tensor (B, H, W, C) or (H, W, C)
        image2: Second image tensor (same shape as image1)
        scale: Scale factor for output (255.0 to match 8-bit range)

    Returns:
        Mean absolute intensity difference
    """
    gray1 = to_grayscale(image1)
    gray2 = to_grayscale(image2)

    diff = torch.abs(gray1 - gray2).mean().item()
    return diff * scale
