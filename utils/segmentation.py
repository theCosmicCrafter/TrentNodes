"""
Subject segmentation and detection utilities for TrentNodes.

Provides methods for detecting subjects in images:
- BiRefNet AI-based segmentation (highest quality)
- Auto-detection using edges, center-weighting, and change detection
"""

import torch
import torch.nn.functional as F

from .image_ops import extract_edges
from .mask_ops import dilate_mask, erode_mask
from .birefnet_wrapper import (
    birefnet_segment,
    is_birefnet_available,
    clear_birefnet_cache,
)


def auto_detect_subject(
    original: torch.Tensor,
    reference: torch.Tensor,
    device: torch.device,
    edge_weight: float = 0.35,
    change_weight: float = 0.45,
    center_weight: float = 0.20
) -> torch.Tensor:
    """
    Automatically detect subject using multiple heuristics.

    Combines edge density, center-weighting, and change detection
    to create a soft mask where higher values = more likely subject.

    Args:
        original: Original image (B, H, W, C)
        reference: Reference image for change detection (B, H, W, C)
        device: torch device
        edge_weight: Weight for edge density signal
        change_weight: Weight for change detection signal
        center_weight: Weight for center bias

    Returns:
        Soft mask (B, H, W) in [0, 1] range
    """
    B, H, W, C = original.shape

    # 1. Edge density - subjects typically have more detail
    edges = extract_edges(original, device)
    edges_norm = edges / (edges.max() + 0.001)

    # 2. Center weighting - subjects are usually centered
    cy, cx = H // 2, W // 2
    y_coords = torch.arange(H, device=device, dtype=torch.float32)
    x_coords = torch.arange(W, device=device, dtype=torch.float32)
    y_dist = torch.abs(y_coords - cy) / cy
    x_dist = torch.abs(x_coords - cx) / cx
    dist_sum = y_dist.unsqueeze(1) + x_dist.unsqueeze(0)
    center_bias = 1.0 - torch.clamp(dist_sum * 0.4, 0, 0.8)
    center_bias = center_bias.unsqueeze(0).expand(B, -1, -1)

    # 3. Change detection - areas that changed most are likely subject
    diff = torch.mean(torch.abs(original - reference), dim=-1)
    diff_norm = diff / (diff.max() + 0.001)

    # Combine signals
    saliency = (
        edges_norm * edge_weight +
        diff_norm * change_weight +
        center_bias * center_weight
    )

    # Adaptive threshold
    threshold_val = saliency.mean() + 0.3 * saliency.std()
    mask = (saliency > threshold_val).float()

    # Morphological cleanup
    mask = erode_mask(mask, radius=3, device=device)
    mask = dilate_mask(mask, radius=8, device=device)

    # Smooth edges
    if mask.dim() == 3:
        mask_4d = mask.unsqueeze(1)
    else:
        mask_4d = mask.unsqueeze(0).unsqueeze(0)

    blur_kernel = torch.ones(1, 1, 5, 5, device=device) / 25.0
    mask_smooth = F.conv2d(mask_4d, blur_kernel, padding=2)

    if mask_smooth.dim() == 4:
        mask = mask_smooth.squeeze(1)
    else:
        mask = mask_smooth.squeeze(0).squeeze(0)

    return torch.clamp(mask, 0, 1)


def segment_subject(
    image: torch.Tensor,
    device: torch.device,
    method: str = "birefnet",
    reference: torch.Tensor = None
) -> torch.Tensor:
    """
    Unified subject segmentation interface.

    Args:
        image: (B, H, W, C) image tensor
        device: torch device
        method: "birefnet" or "auto"
        reference: Reference image for auto-detection (optional)

    Returns:
        Subject mask (B, H, W) tensor
    """
    if method == "birefnet":
        mask = birefnet_segment(image, device)
        if mask is not None:
            return mask
        # Fallback to auto if BiRefNet not available
        print("[TrentNodes] BiRefNet not available, using auto-detection")
        method = "auto"

    if method == "auto":
        if reference is None:
            reference = image
        return auto_detect_subject(image, reference, device)

    raise ValueError(f"Unknown segmentation method: {method}")
