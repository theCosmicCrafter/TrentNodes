"""
Batch Combine - Concatenate multiple image batches into a single batch.

A ComfyUI node for merging multiple optional image batch inputs into one
unified output batch with automatic size handling.
"""

import torch
import torch.nn.functional as F


class MultiBatchCombine:
    """
    Concatenate multiple image batches into a single output batch.

    Accepts up to 8 optional image inputs. Unconnected inputs are skipped.
    Handles dimension mismatches by resizing all batches to a target size.
    """

    SIZE_MODES = ["largest", "first", "custom"]
    RESIZE_METHODS = ["bilinear", "nearest", "bicubic", "area"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "size_mode": (cls.SIZE_MODES, {
                    "default": "largest",
                    "tooltip": "How to handle size mismatches: largest "
                               "(resize to max dimensions), first (match "
                               "first batch), custom (use target_width/height)"
                }),
                "resize_method": (cls.RESIZE_METHODS, {
                    "default": "bilinear",
                    "tooltip": "Interpolation method for resizing"
                }),
            },
            "optional": {
                "batch_1": ("IMAGE", {"tooltip": "First image batch"}),
                "batch_2": ("IMAGE", {"tooltip": "Second image batch"}),
                "batch_3": ("IMAGE", {"tooltip": "Third image batch"}),
                "batch_4": ("IMAGE", {"tooltip": "Fourth image batch"}),
                "batch_5": ("IMAGE", {"tooltip": "Fifth image batch"}),
                "batch_6": ("IMAGE", {"tooltip": "Sixth image batch"}),
                "batch_7": ("IMAGE", {"tooltip": "Seventh image batch"}),
                "batch_8": ("IMAGE", {"tooltip": "Eighth image batch"}),
                "target_width": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 8192,
                    "step": 8,
                    "tooltip": "Target width when size_mode is 'custom'"
                }),
                "target_height": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 8192,
                    "step": 8,
                    "tooltip": "Target height when size_mode is 'custom'"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("images", "total_frames")
    FUNCTION = "combine"
    CATEGORY = "Trent/Image"
    DESCRIPTION = (
        "Concatenate multiple image batches into a single batch. "
        "Unconnected inputs are skipped. Handles size mismatches via resize."
    )

    def collect_batches(
        self,
        batch_1=None,
        batch_2=None,
        batch_3=None,
        batch_4=None,
        batch_5=None,
        batch_6=None,
        batch_7=None,
        batch_8=None,
    ):
        """Collect all connected batch inputs, skipping None values."""
        batches = []
        for batch in [
            batch_1, batch_2, batch_3, batch_4,
            batch_5, batch_6, batch_7, batch_8
        ]:
            if batch is not None:
                batches.append(batch)
        return batches

    def determine_target_size(self, batches, size_mode, target_width,
                              target_height):
        """Determine target dimensions based on size_mode."""
        if size_mode == "custom":
            return (target_height, target_width)

        if size_mode == "first":
            return (batches[0].shape[1], batches[0].shape[2])

        # "largest" mode
        max_h = max(b.shape[1] for b in batches)
        max_w = max(b.shape[2] for b in batches)
        return (max_h, max_w)

    def resize_batch(self, batch, target_h, target_w, method):
        """
        Resize batch to target dimensions using GPU-accelerated interpolation.

        Args:
            batch: Image tensor (B, H, W, C) in [0, 1] range
            target_h: Target height
            target_w: Target width
            method: Interpolation method

        Returns:
            Resized tensor (B, target_h, target_w, C)
        """
        current_h, current_w = batch.shape[1], batch.shape[2]
        if current_h == target_h and current_w == target_w:
            return batch

        # BHWC -> BCHW for F.interpolate
        batch_bchw = batch.permute(0, 3, 1, 2)

        # Set align_corners based on method
        align_corners = None
        if method in ("bilinear", "bicubic"):
            align_corners = False

        resized = F.interpolate(
            batch_bchw,
            size=(target_h, target_w),
            mode=method,
            align_corners=align_corners
        )

        # BCHW -> BHWC
        return resized.permute(0, 2, 3, 1)

    def combine(
        self,
        size_mode,
        resize_method,
        batch_1=None,
        batch_2=None,
        batch_3=None,
        batch_4=None,
        batch_5=None,
        batch_6=None,
        batch_7=None,
        batch_8=None,
        target_width=512,
        target_height=512,
    ):
        """
        Combine multiple image batches into a single batch.

        Returns:
            Tuple of (combined_images, total_frame_count)
        """
        batches = self.collect_batches(
            batch_1, batch_2, batch_3, batch_4,
            batch_5, batch_6, batch_7, batch_8
        )

        if not batches:
            raise ValueError("At least one batch must be connected")

        # Single batch: passthrough
        if len(batches) == 1:
            return (batches[0], batches[0].shape[0])

        # Determine target size
        target_h, target_w = self.determine_target_size(
            batches, size_mode, target_width, target_height
        )

        # Resize all batches to target size
        resized = [
            self.resize_batch(b, target_h, target_w, resize_method)
            for b in batches
        ]

        # Concatenate along batch dimension
        combined = torch.cat(resized, dim=0)

        return (combined, combined.shape[0])


NODE_CLASS_MAPPINGS = {
    "MultiBatchCombine": MultiBatchCombine,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MultiBatchCombine": "Multi-Batch Combine",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
