"""
Tracked mouth shape compositor with background removal.

Provides nodes for:
- Removing backgrounds from mouth shape images (BiRefNet/color key)
- Compositing mouth shapes onto moving characters using tracked masks
"""
from typing import Dict, Any, List, Tuple, Optional, Union

import torch
import torch.nn.functional as F
from comfy.utils import ProgressBar

from ...utils.mask_ops import (
    get_mask_centroid,
    batch_get_centroids,
    dilate_mask,
    feather_mask,
)
from ...utils.segmentation import birefnet_segment


def color_key_mask(
    image: torch.Tensor,
    bg_color: str = "white",
    threshold: float = 0.1
) -> torch.Tensor:
    """
    Create mask by keying out a background color.

    Args:
        image: (B, H, W, C) tensor in [0, 1]
        bg_color: "white", "black", or hex color
        threshold: Color similarity threshold

    Returns:
        Mask (B, H, W) where 1 = foreground, 0 = background
    """
    B, H, W, C = image.shape

    # Define target background color
    if bg_color == "white":
        target = torch.tensor([1.0, 1.0, 1.0], device=image.device)
    elif bg_color == "black":
        target = torch.tensor([0.0, 0.0, 0.0], device=image.device)
    else:
        # Parse hex color
        try:
            hex_color = bg_color.lstrip("#")
            r = int(hex_color[0:2], 16) / 255.0
            g = int(hex_color[2:4], 16) / 255.0
            b = int(hex_color[4:6], 16) / 255.0
            target = torch.tensor([r, g, b], device=image.device)
        except (ValueError, IndexError):
            target = torch.tensor([1.0, 1.0, 1.0], device=image.device)

    # Calculate color distance
    target = target.view(1, 1, 1, 3)
    distance = torch.sqrt(((image[..., :3] - target) ** 2).sum(dim=-1))

    # Threshold to create mask (invert: background = 0)
    mask = (distance > threshold).float()

    return mask


class RemoveMouthBackground:
    """
    Remove background from mouth shape images.

    Uses BiRefNet AI segmentation or color keying to isolate
    the mouth/lips foreground from solid backgrounds.
    """

    CATEGORY = "Trent/LipSync"
    DESCRIPTION = (
        "Remove background from mouth shape images using BiRefNet "
        "or color keying."
    )

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Mouth shape images (batch of 9 or any)"
                }),
            },
            "optional": {
                "method": (["birefnet", "color_key", "auto"], {
                    "default": "birefnet",
                    "tooltip": "Background removal method"
                }),
                "quality": (["fast", "balanced", "quality"], {
                    "default": "fast",
                    "tooltip": (
                        "BiRefNet quality: fast (512px), balanced (768px), "
                        "quality (1024px)"
                    )
                }),
                "model": (["lite", "standard"], {
                    "default": "lite",
                    "tooltip": "BiRefNet model: lite (faster), standard (better)"
                }),
                "bg_color": (["white", "black"], {
                    "default": "white",
                    "tooltip": "Background color for color_key method"
                }),
                "threshold": ("FLOAT", {
                    "default": 0.15,
                    "min": 0.01,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Color key threshold (higher = more removed)"
                }),
                "expand_mask": ("INT", {
                    "default": 2,
                    "min": 0,
                    "max": 50,
                    "step": 1,
                    "tooltip": "Expand foreground mask by pixels"
                }),
                "feather": ("INT", {
                    "default": 2,
                    "min": 0,
                    "max": 50,
                    "step": 1,
                    "tooltip": "Feather mask edges"
                }),
                "output_format": (["rgba", "transparent", "premultiplied"], {
                    "default": "rgba",
                    "tooltip": (
                        "Output format: rgba (RGB + alpha), "
                        "transparent (zero BG), premultiplied (RGB * alpha)"
                    )
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("images", "masks")
    FUNCTION = "remove_background"

    def remove_background(
        self,
        images: torch.Tensor,
        method: str = "birefnet",
        quality: str = "fast",
        model: str = "lite",
        bg_color: str = "white",
        threshold: float = 0.15,
        expand_mask: int = 2,
        feather: int = 2,
        output_format: str = "rgba"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Remove background from mouth images.

        Args:
            images: Batch of mouth images (B, H, W, C)
            method: "birefnet", "color_key", or "auto"
            quality: "fast", "balanced", or "quality" (resolution)
            model: "lite" or "standard" BiRefNet variant
            bg_color: Background color for color keying
            threshold: Color key threshold
            expand_mask: Pixels to expand mask
            feather: Edge feather amount
            output_format: "rgba", "transparent", or "premultiplied"

        Returns:
            Tuple of (output images, foreground masks)
        """
        device = images.device
        B, H, W, C = images.shape

        # Map quality to resolution
        resolution_map = {"fast": 512, "balanced": 768, "quality": 1024}
        resolution = resolution_map.get(quality, 512)

        # Get masks using selected method
        if method == "birefnet" or method == "auto":
            masks = birefnet_segment(images, device, resolution, model)
            if masks is None and method == "auto":
                # Fallback to color key
                masks = color_key_mask(images, bg_color, threshold)
            elif masks is None:
                raise RuntimeError(
                    "BiRefNet not available. Install transformers or "
                    "use method='color_key'"
                )
        else:
            masks = color_key_mask(images, bg_color, threshold)

        # Post-process masks
        if expand_mask > 0:
            masks = dilate_mask(masks, expand_mask, device)
        if feather > 0:
            masks = feather_mask(masks, feather, device)

        masks = masks.clamp(0, 1)
        alpha = masks.unsqueeze(-1)
        rgb = images[..., :3]

        # Create output based on format
        if output_format == "transparent":
            # Zero out background pixels completely
            output = torch.zeros(B, H, W, 4, device=device, dtype=images.dtype)
            output[..., :3] = rgb * alpha
            output[..., 3] = masks
        elif output_format == "premultiplied":
            # RGB premultiplied by alpha
            output = torch.cat([rgb * alpha, alpha], dim=-1)
        else:
            # rgba - standard alpha composite ready
            output = torch.cat([rgb, alpha], dim=-1)

        return (output, masks)


class MouthShapeCompositorTracked:
    """
    Composite mouth shapes onto video frames with position tracking.

    Uses per-frame tracking masks to dynamically position mouth shapes
    as the character moves. Optionally removes backgrounds from mouth
    images using BiRefNet.
    """

    CATEGORY = "Trent/LipSync"
    DESCRIPTION = (
        "Composite mouth shapes onto moving characters using tracked "
        "mask positions for dynamic lip sync."
    )

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "frames": ("IMAGE", {
                    "tooltip": "Video frames batch (B, H, W, C)"
                }),
                "mouth_shapes": ("IMAGE", {
                    "tooltip": "9 mouth shape images (A-H + X)"
                }),
                "mouth_sequence": ("MOUTH_SEQUENCE", {
                    "tooltip": "Per-frame mouth shape indices"
                }),
            },
            "optional": {
                "tracking_mode": (["auto", "points", "masks"], {
                    "default": "auto",
                    "tooltip": (
                        "Tracking method: auto (use whichever is connected), "
                        "points (use point_sequence), masks (use tracking_masks)"
                    )
                }),
                "tracking_masks": ("MASK", {
                    "tooltip": "Per-frame mouth region masks from SAM3"
                }),
                "point_sequence": ("POINT_SEQUENCE", {
                    "tooltip": (
                        "Per-frame (x,y) coordinates from point tracker"
                    )
                }),
                "remove_background": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Remove background from mouth images"
                }),
                "bg_method": (["birefnet", "color_key", "auto"], {
                    "default": "birefnet",
                    "tooltip": "Background removal method"
                }),
                "bg_color": (["white", "black"], {
                    "default": "white",
                    "tooltip": "Background color for color keying"
                }),
                "scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.01,
                    "max": 5.0,
                    "step": 0.01,
                    "tooltip": "Scale mouth shapes (0.01-5.0)"
                }),
                "offset_x": ("INT", {
                    "default": 0,
                    "min": -500,
                    "max": 500,
                    "step": 1,
                    "tooltip": "X offset from centroid"
                }),
                "offset_y": ("INT", {
                    "default": 0,
                    "min": -500,
                    "max": 500,
                    "step": 1,
                    "tooltip": "Y offset from centroid"
                }),
                "output_mouth_rgba": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Output mouth shapes with alpha on transparent "
                        "background (for further compositing)"
                    )
                }),
                "output_cropped_mouths": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Output mouth shapes cropped to their bounding box "
                        "with alpha (9 shapes, untracked)"
                    )
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("frames", "mouth_rgba", "mouths_cropped")
    FUNCTION = "composite_tracked"

    def composite_tracked(
        self,
        frames: torch.Tensor,
        mouth_shapes: torch.Tensor,
        mouth_sequence: List[int],
        tracking_mode: str = "auto",
        tracking_masks: Optional[torch.Tensor] = None,
        point_sequence: Optional[List[Tuple[int, int]]] = None,
        remove_background: bool = True,
        bg_method: str = "birefnet",
        bg_color: str = "white",
        scale: float = 1.0,
        offset_x: int = 0,
        offset_y: int = 0,
        output_mouth_rgba: bool = False,
        output_cropped_mouths: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Composite mouth shapes using tracked positions.

        Args:
            frames: Video frames (B, H, W, C)
            mouth_shapes: Mouth images (9, MH, MW, C)
            mouth_sequence: Per-frame mouth indices
            tracking_mode: "auto", "points", or "masks"
            tracking_masks: Per-frame masks (B, H, W) - optional
            point_sequence: Per-frame (x,y) coordinates - optional
            remove_background: Use BiRefNet for BG removal
            bg_method: Background removal method
            bg_color: Color for color keying
            scale: Scale factor for mouths
            offset_x: X offset from centroid
            offset_y: Y offset from centroid
            output_mouth_rgba: Output per-frame mouth with alpha
            output_cropped_mouths: Output cropped mouth shapes with alpha

        Returns:
            Tuple of (composited frames, mouth RGBA frames, cropped mouths)
        """
        device = frames.device
        num_frames = frames.shape[0]
        frame_h, frame_w = frames.shape[1], frames.shape[2]

        # Validate inputs
        if mouth_shapes.shape[0] < 9:
            raise ValueError(
                f"Expected 9 mouth shapes, got {mouth_shapes.shape[0]}"
            )

        # Determine tracking mode
        if tracking_mode == "points":
            if point_sequence is None:
                raise ValueError(
                    "tracking_mode='points' but no point_sequence provided"
                )
            use_point_sequence = True
        elif tracking_mode == "masks":
            if tracking_masks is None:
                raise ValueError(
                    "tracking_mode='masks' but no tracking_masks provided"
                )
            use_point_sequence = False
        else:  # auto
            # Auto: prefer points if available, fall back to masks
            if point_sequence is not None:
                use_point_sequence = True
            elif tracking_masks is not None:
                use_point_sequence = False
            else:
                raise ValueError(
                    "Either tracking_masks or point_sequence must be provided"
                )

        if not use_point_sequence:
            # Ensure tracking masks match frame count
            if tracking_masks.dim() == 2:
                tracking_masks = tracking_masks.unsqueeze(0)
            if tracking_masks.shape[0] == 1:
                tracking_masks = tracking_masks.expand(num_frames, -1, -1)

        # Remove background from mouth shapes if requested
        # Use fast settings (512px, lite model) for speed
        if remove_background:
            if bg_method == "birefnet" or bg_method == "auto":
                fg_masks = birefnet_segment(
                    mouth_shapes, device, resolution=512, model_variant="lite"
                )
                if fg_masks is None and bg_method == "auto":
                    fg_masks = color_key_mask(mouth_shapes, bg_color, 0.15)
                elif fg_masks is None:
                    fg_masks = color_key_mask(mouth_shapes, bg_color, 0.15)
            else:
                fg_masks = color_key_mask(mouth_shapes, bg_color, 0.15)

            # Small cleanup
            fg_masks = dilate_mask(fg_masks, 1, device)
            fg_masks = feather_mask(fg_masks, 2, device)
            mouth_alphas = fg_masks.clamp(0, 1)
        else:
            # Use existing alpha or full opacity
            if mouth_shapes.shape[-1] == 4:
                mouth_alphas = mouth_shapes[..., 3]
            else:
                mouth_alphas = torch.ones(
                    mouth_shapes.shape[0],
                    mouth_shapes.shape[1],
                    mouth_shapes.shape[2],
                    device=device
                )

        # Get RGB from mouth shapes
        mouth_rgb = mouth_shapes[..., :3]

        # Scale mouth shapes if needed
        if scale != 1.0:
            mouth_rgb = self._scale_images(mouth_rgb, scale)
            mouth_alphas = self._scale_masks(mouth_alphas, scale)

        mouth_h, mouth_w = mouth_rgb.shape[1], mouth_rgb.shape[2]

        # Compute crop bounds for cropped mouth output if requested
        if output_cropped_mouths:
            crop_bounds = self._crop_mouths_to_bbox(
                mouth_rgb, mouth_alphas, device
            )
            if crop_bounds is not None:
                crop_h, crop_w = crop_bounds[0], crop_bounds[1]
                mouths_cropped = torch.zeros(
                    num_frames, crop_h, crop_w, 4,
                    device=device, dtype=frames.dtype
                )
            else:
                # No foreground found
                mouths_cropped = torch.zeros(
                    num_frames, 1, 1, 4, device=device, dtype=frames.dtype
                )
                crop_bounds = None
        else:
            # Placeholder output (single transparent pixel)
            mouths_cropped = torch.zeros(
                1, 1, 1, 4, device=device, dtype=frames.dtype
            )
            crop_bounds = None

        # Extend sequence if needed and convert to tensor
        if len(mouth_sequence) < num_frames:
            mouth_sequence = list(mouth_sequence) + [8] * (
                num_frames - len(mouth_sequence)
            )
        seq_tensor = torch.tensor(
            mouth_sequence[:num_frames], device=device, dtype=torch.long
        ).clamp(0, 8)

        # Compute placement positions from tracking data
        if use_point_sequence:
            # Direct point coordinates - extend if needed
            if len(point_sequence) < num_frames:
                last_point = point_sequence[-1] if point_sequence else (0, 0)
                point_sequence = list(point_sequence) + [last_point] * (
                    num_frames - len(point_sequence)
                )

            # Convert to tensors - points are (x, y), we need placement coords
            points_x = torch.tensor(
                [p[0] for p in point_sequence[:num_frames]],
                device=device, dtype=torch.float32
            )
            points_y = torch.tensor(
                [p[1] for p in point_sequence[:num_frames]],
                device=device, dtype=torch.float32
            )

            # Calculate placement positions (center mouth on point)
            place_x = (points_x + offset_x - mouth_w // 2).int()
            place_y = (points_y + offset_y - mouth_h // 2).int()
        else:
            # Batch compute all centroids at once (GPU accelerated)
            centroids = batch_get_centroids(tracking_masks)  # (N, 2) [cy, cx]

            # Calculate all placement positions at once
            place_x = (centroids[:, 1] + offset_x - mouth_w // 2).int()
            place_y = (centroids[:, 0] + offset_y - mouth_h // 2).int()

        # Group frames by mouth shape index for batch processing
        result = frames.clone()

        # Initialize mouth RGBA output if requested
        if output_mouth_rgba:
            mouth_rgba_out = torch.zeros(
                num_frames, frame_h, frame_w, 4,
                device=device, dtype=frames.dtype
            )
        else:
            # Placeholder output (single transparent pixel)
            mouth_rgba_out = torch.zeros(
                1, 1, 1, 4, device=device, dtype=frames.dtype
            )

        pbar = ProgressBar(num_frames)

        # Process frames grouped by mouth shape (reduces redundant operations)
        for shape_idx in range(9):
            # Find all frames using this mouth shape
            frame_mask = (seq_tensor == shape_idx)
            frame_indices = torch.where(frame_mask)[0]

            if len(frame_indices) == 0:
                continue

            mouth_img = mouth_rgb[shape_idx]
            alpha = mouth_alphas[shape_idx]

            # Process each frame in this group
            for idx in frame_indices:
                i = idx.item()
                px = place_x[i].item()
                py = place_y[i].item()

                # Calculate bounds with clipping
                src_x1 = max(0, -px)
                src_y1 = max(0, -py)
                src_x2 = mouth_w - max(0, px + mouth_w - frame_w)
                src_y2 = mouth_h - max(0, py + mouth_h - frame_h)

                dst_x1 = max(0, px)
                dst_y1 = max(0, py)
                dst_x2 = min(frame_w, px + mouth_w)
                dst_y2 = min(frame_h, py + mouth_h)

                # Skip if completely out of bounds
                if dst_x1 >= dst_x2 or dst_y1 >= dst_y2:
                    continue

                # Extract regions and composite
                alpha_region = alpha[
                    src_y1:src_y2, src_x1:src_x2
                ].unsqueeze(-1)
                mouth_region = mouth_img[src_y1:src_y2, src_x1:src_x2]

                result[i, dst_y1:dst_y2, dst_x1:dst_x2] = (
                    result[i, dst_y1:dst_y2, dst_x1:dst_x2] *
                    (1 - alpha_region) +
                    mouth_region * alpha_region
                )

                # Also write to mouth RGBA output if requested
                if output_mouth_rgba:
                    mouth_rgba_out[
                        i, dst_y1:dst_y2, dst_x1:dst_x2, :3
                    ] = mouth_region * alpha_region
                    mouth_rgba_out[
                        i, dst_y1:dst_y2, dst_x1:dst_x2, 3:4
                    ] = alpha_region

            # Write cropped mouth for all frames using this shape
            if crop_bounds is not None:
                _, _, cy1, cy2, cx1, cx2 = crop_bounds
                cropped_rgb = mouth_img[cy1:cy2, cx1:cx2]
                cropped_alpha = alpha[cy1:cy2, cx1:cx2]
                for idx in frame_indices:
                    i = idx.item()
                    mouths_cropped[i, :, :, :3] = cropped_rgb
                    mouths_cropped[i, :, :, 3] = cropped_alpha

            pbar.update_absolute(len(frame_indices))

        return (result, mouth_rgba_out, mouths_cropped)

    def _scale_images(
        self,
        images: torch.Tensor,
        scale: float
    ) -> torch.Tensor:
        """Scale batch of images."""
        if scale == 1.0:
            return images

        # (B, H, W, C) -> (B, C, H, W)
        images = images.permute(0, 3, 1, 2)

        new_h = int(images.shape[2] * scale)
        new_w = int(images.shape[3] * scale)

        scaled = F.interpolate(
            images,
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False
        )

        return scaled.permute(0, 2, 3, 1)

    def _scale_masks(
        self,
        masks: torch.Tensor,
        scale: float
    ) -> torch.Tensor:
        """Scale batch of masks."""
        if scale == 1.0:
            return masks

        # (B, H, W) -> (B, 1, H, W)
        masks = masks.unsqueeze(1)

        new_h = int(masks.shape[2] * scale)
        new_w = int(masks.shape[3] * scale)

        scaled = F.interpolate(
            masks,
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False
        )

        return scaled.squeeze(1)

    def _crop_mouths_to_bbox(
        self,
        mouth_rgb: torch.Tensor,
        mouth_alphas: torch.Tensor,
        device: torch.device
    ) -> tuple:
        """
        Get bounding box and crop dimensions for mouth shapes.

        Args:
            mouth_rgb: (9, H, W, 3) mouth RGB images
            mouth_alphas: (9, H, W) alpha masks
            device: torch device

        Returns:
            Tuple of (crop_h, crop_w, y_min, x_min) or None if no foreground
        """
        _, H, W, _ = mouth_rgb.shape

        # Find bounding box across all mouth shapes (union of all masks)
        combined_mask = (mouth_alphas > 0.01).any(dim=0)  # (H, W)

        # Find non-zero coordinates
        nonzero = torch.nonzero(combined_mask, as_tuple=True)

        if len(nonzero[0]) == 0:
            return None

        y_min = nonzero[0].min().item()
        y_max = nonzero[0].max().item() + 1
        x_min = nonzero[1].min().item()
        x_max = nonzero[1].max().item() + 1

        # Add small padding
        pad = 2
        y_min = max(0, y_min - pad)
        y_max = min(H, y_max + pad)
        x_min = max(0, x_min - pad)
        x_max = min(W, x_max + pad)

        crop_h = y_max - y_min
        crop_w = x_max - x_min

        return (crop_h, crop_w, y_min, y_max, x_min, x_max)
