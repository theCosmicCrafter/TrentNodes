"""
Mouth shape compositor node.

Composites mouth shape images onto video frames based on
the mouth sequence data from phoneme processing.
"""
from typing import Dict, Any, List, Tuple, Optional

import torch
import torch.nn.functional as F
from comfy.utils import ProgressBar


class MouthShapeCompositor:
    """
    Composite mouth shape images onto video frames.

    Takes video frames, a batch of mouth shape images (9 total: A-H + X),
    and a mouth sequence to produce lip-synced output frames.
    """

    CATEGORY = "Trent/LipSync"
    DESCRIPTION = (
        "Composite mouth shapes onto video frames for lip sync animation."
    )

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "frames": ("IMAGE", {
                    "tooltip": "Video frames batch (B, H, W, C)"
                }),
                "mouth_shapes": ("IMAGE", {
                    "tooltip": (
                        "9 mouth shape images in order A-H, X "
                        "(indices 0-8)"
                    )
                }),
                "mouth_sequence": ("MOUTH_SEQUENCE", {
                    "tooltip": "Per-frame mouth shape indices"
                }),
                "position_x": ("INT", {
                    "default": 0,
                    "min": -4096,
                    "max": 4096,
                    "step": 1,
                    "tooltip": "X position for mouth placement"
                }),
                "position_y": ("INT", {
                    "default": 0,
                    "min": -4096,
                    "max": 4096,
                    "step": 1,
                    "tooltip": "Y position for mouth placement"
                }),
            },
            "optional": {
                "mask": ("MASK", {
                    "tooltip": "Optional mask for mouth region blending"
                }),
                "blend_mode": (["alpha", "replace", "multiply", "screen"], {
                    "default": "alpha",
                    "tooltip": "How to blend mouth onto frame"
                }),
                "scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.01,
                    "tooltip": "Scale factor for mouth shapes"
                }),
                "feather": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Edge feathering in pixels"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    FUNCTION = "composite"

    def composite(
        self,
        frames: torch.Tensor,
        mouth_shapes: torch.Tensor,
        mouth_sequence: List[int],
        position_x: int,
        position_y: int,
        mask: Optional[torch.Tensor] = None,
        blend_mode: str = "alpha",
        scale: float = 1.0,
        feather: int = 0
    ) -> Tuple[torch.Tensor]:
        """
        Composite mouth shapes onto frames.

        Args:
            frames: Video frames (B, H, W, C)
            mouth_shapes: Mouth images (9, MH, MW, C or 4)
            mouth_sequence: List of mouth indices per frame
            position_x: X offset for placement
            position_y: Y offset for placement
            mask: Optional blending mask
            blend_mode: Blending method
            scale: Scale factor for mouth images
            feather: Edge feather amount

        Returns:
            Composited frames
        """
        device = frames.device
        num_frames = frames.shape[0]
        frame_h, frame_w = frames.shape[1], frames.shape[2]

        # Validate mouth shapes
        if mouth_shapes.shape[0] < 9:
            raise ValueError(
                f"Expected 9 mouth shapes (A-H + X), got {mouth_shapes.shape[0]}"
            )

        # Scale mouth shapes if needed
        if scale != 1.0:
            mouth_shapes = self._scale_images(mouth_shapes, scale)

        mouth_h, mouth_w = mouth_shapes.shape[1], mouth_shapes.shape[2]

        # Prepare output
        result = frames.clone()

        # Handle mask
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
            # Ensure mask matches frame count
            if mask.shape[0] == 1:
                mask = mask.expand(num_frames, -1, -1)

        # Extract alpha channel if present
        has_alpha = mouth_shapes.shape[-1] == 4
        if has_alpha:
            mouth_rgb = mouth_shapes[..., :3]
            mouth_alpha = mouth_shapes[..., 3:4]
        else:
            mouth_rgb = mouth_shapes
            mouth_alpha = None

        # Apply feathering to alpha/mask
        if feather > 0 and mouth_alpha is not None:
            mouth_alpha = self._feather_mask(mouth_alpha, feather)

        # Extend sequence if needed and convert to tensor
        if len(mouth_sequence) < num_frames:
            mouth_sequence = list(mouth_sequence) + [8] * (
                num_frames - len(mouth_sequence)
            )
        seq_tensor = torch.tensor(
            mouth_sequence[:num_frames], device=device, dtype=torch.long
        ).clamp(0, 8)

        # Pre-compute bounds (fixed position, so same for all frames)
        x1, y1 = position_x, position_y
        x2, y2 = x1 + mouth_w, y1 + mouth_h

        src_x1 = max(0, -x1)
        src_y1 = max(0, -y1)
        src_x2 = mouth_w - max(0, x2 - frame_w)
        src_y2 = mouth_h - max(0, y2 - frame_h)

        dst_x1 = max(0, x1)
        dst_y1 = max(0, y1)
        dst_x2 = min(frame_w, x2)
        dst_y2 = min(frame_h, y2)

        # Skip if completely out of bounds
        if dst_x1 >= dst_x2 or dst_y1 >= dst_y2:
            return (result,)

        pbar = ProgressBar(num_frames)

        # Process frames grouped by mouth shape for efficiency
        for shape_idx in range(9):
            # Find all frames using this mouth shape
            frame_mask = (seq_tensor == shape_idx)
            frame_indices = torch.where(frame_mask)[0]

            if len(frame_indices) == 0:
                continue

            # Get mouth image and alpha for this shape
            mouth_img = mouth_rgb[shape_idx]
            mouth_region = mouth_img[src_y1:src_y2, src_x1:src_x2]

            if mouth_alpha is not None:
                alpha = mouth_alpha[shape_idx]
                alpha_region = alpha[src_y1:src_y2, src_x1:src_x2]
            else:
                alpha_region = None

            # Process all frames using this shape
            for idx in frame_indices:
                i = idx.item()

                # Get alpha for blending
                if alpha_region is not None:
                    a_region = alpha_region
                elif mask is not None:
                    a_region = mask[i, dst_y1:dst_y2, dst_x1:dst_x2]
                    a_region = a_region.unsqueeze(-1)
                else:
                    a_region = torch.ones_like(mouth_region[..., :1])

                # Apply blend mode
                blended = self._blend(
                    result[i, dst_y1:dst_y2, dst_x1:dst_x2],
                    mouth_region,
                    a_region,
                    blend_mode
                )

                result[i, dst_y1:dst_y2, dst_x1:dst_x2] = blended

            pbar.update_absolute(len(frame_indices))

        return (result,)

    def _scale_images(
        self,
        images: torch.Tensor,
        scale: float
    ) -> torch.Tensor:
        """Scale a batch of images."""
        if scale == 1.0:
            return images

        # (B, H, W, C) -> (B, C, H, W) for interpolate
        images = images.permute(0, 3, 1, 2)

        new_h = int(images.shape[2] * scale)
        new_w = int(images.shape[3] * scale)

        scaled = F.interpolate(
            images,
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False
        )

        # Back to (B, H, W, C)
        return scaled.permute(0, 2, 3, 1)

    def _feather_mask(
        self,
        mask: torch.Tensor,
        radius: int
    ) -> torch.Tensor:
        """Apply gaussian blur for feathering."""
        if radius <= 0:
            return mask

        # Ensure 4D for conv2d: (B, C, H, W)
        original_shape = mask.shape
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)  # (B, 1, H, W)
        elif mask.dim() == 4 and mask.shape[-1] == 1:
            mask = mask.permute(0, 3, 1, 2)  # (B, H, W, 1) -> (B, 1, H, W)

        # Create gaussian kernel
        kernel_size = radius * 2 + 1
        sigma = radius / 3.0
        x = torch.arange(kernel_size, device=mask.device) - radius
        gauss = torch.exp(-x.pow(2) / (2 * sigma * sigma))
        gauss = gauss / gauss.sum()

        # Separable convolution
        kernel_h = gauss.view(1, 1, -1, 1)
        kernel_v = gauss.view(1, 1, 1, -1)

        padded = F.pad(mask, [radius, radius, radius, radius], mode="reflect")
        blurred = F.conv2d(padded, kernel_h.expand(1, 1, -1, 1))
        blurred = F.conv2d(blurred, kernel_v.expand(1, 1, 1, -1))

        # Restore original shape
        if len(original_shape) == 3:
            blurred = blurred.squeeze(1)
        elif len(original_shape) == 4 and original_shape[-1] == 1:
            blurred = blurred.permute(0, 2, 3, 1)

        return blurred

    def _blend(
        self,
        background: torch.Tensor,
        foreground: torch.Tensor,
        alpha: torch.Tensor,
        mode: str
    ) -> torch.Tensor:
        """Blend foreground onto background using specified mode."""
        alpha = alpha.clamp(0, 1)

        if mode == "replace":
            return foreground
        elif mode == "alpha":
            return background * (1 - alpha) + foreground * alpha
        elif mode == "multiply":
            blended = background * foreground
            return background * (1 - alpha) + blended * alpha
        elif mode == "screen":
            blended = 1 - (1 - background) * (1 - foreground)
            return background * (1 - alpha) + blended * alpha
        else:
            return background * (1 - alpha) + foreground * alpha
