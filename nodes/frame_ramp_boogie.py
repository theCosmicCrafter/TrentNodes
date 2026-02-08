"""
Frame Ramp Boogie - GPU-accelerated batch frame interpolation.

Inserts blended intermediate frames between consecutive frame pairs
in a batch, with configurable easing curves and region targeting.
Creates smooth slow-motion with actual frame blending instead of
simple duplication.
"""

import torch
from typing import Dict, Any, Tuple

from ..utils.easing import apply_easing


EASING_METHODS = [
    "linear",
    "ease_in",
    "ease_out",
    "ease_in_out",
    "smooth",
    "cubic_bezier",
]

REGIONS = ["full_batch", "start", "middle", "end"]

BEZIER_PRESETS = [
    "custom",
    "ease",
    "ease_in",
    "ease_out",
    "ease_in_out",
    "sharp",
    "gentle",
]

# CSS-standard and custom preset control points
_PRESET_VALUES = {
    "ease": (0.25, 0.1, 0.25, 1.0),
    "ease_in": (0.42, 0.0, 1.0, 1.0),
    "ease_out": (0.0, 0.0, 0.58, 1.0),
    "ease_in_out": (0.42, 0.0, 0.58, 1.0),
    "sharp": (0.75, 0.0, 0.25, 1.0),
    "gentle": (0.4, 0.0, 0.6, 1.0),
}


class FrameRampBoogie:
    """
    Insert blended intermediate frames between consecutive
    frame pairs in a batch.

    Supports multiple easing curves for controlling blend
    weight distribution, and region targeting to interpolate
    only a portion of the batch (start, middle, or end).

    Unlike BatchSlowdown (which duplicates frames), this node
    generates genuinely new frames by blending adjacent pairs.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Image batch to interpolate "
                               "(needs 2+ frames)"
                }),
                "frames_to_insert": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 32,
                    "step": 1,
                    "tooltip": "Frames to blend per pair "
                               "(1 = 2x output, 2 = 3x, "
                               "3 = 4x)"
                }),
                "easing": (EASING_METHODS, {
                    "default": "linear",
                    "tooltip": "Blend curve shape "
                               "(see preview below)"
                }),
                "target_region": (REGIONS, {
                    "default": "full_batch",
                    "tooltip": "Where to interpolate: "
                               "full_batch (all), or just "
                               "start/middle/end"
                }),
            },
            "optional": {
                "region_size": ("FLOAT", {
                    "default": 0.33,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "How much of the batch to "
                               "affect (0.33 = one-third, "
                               "0.5 = half)"
                }),
                "bezier_preset": (BEZIER_PRESETS, {
                    "default": "ease",
                    "tooltip": "Curve shape preset "
                               "(choose custom for manual "
                               "control - see preview)"
                }),
                "p1_x": ("FLOAT", {
                    "default": 0.25,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "First control handle X "
                               "(see curve preview)"
                }),
                "p1_y": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "First control handle Y "
                               "(see curve preview)"
                }),
                "p2_x": ("FLOAT", {
                    "default": 0.25,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Second control handle X "
                               "(see curve preview)"
                }),
                "p2_y": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Second control handle Y "
                               "(see curve preview)"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "STRING")
    RETURN_NAMES = ("images", "original_count", "new_count", "info")
    FUNCTION = "boogie"
    CATEGORY = "Trent/Video"
    DESCRIPTION = (
        "GPU-accelerated frame interpolation with "
        "easing curves.\n\n"
        "Creates smooth slow-motion by blending "
        "adjacent frames, not duplicating them.\n\n"
        "Easing Methods:\n"
        "- linear: Even blend (constant speed)\n"
        "- ease_in: Slow start, fast end\n"
        "- ease_out: Fast start, slow end\n"
        "- ease_in_out: Smooth S-curve\n"
        "- smooth: Hermite smoothstep\n"
        "- cubic_bezier: Custom curve with presets\n\n"
        "Region Targeting:\n"
        "- full_batch: All frame pairs\n"
        "- start/middle/end: Partial (adjustable)"
    )

    def _compute_region(
        self,
        batch_size: int,
        target_region: str,
        region_size: float,
    ) -> Tuple[int, int]:
        """
        Compute start/end pair indices for interpolation.

        Args:
            batch_size: Number of frames in the batch.
            target_region: Which region to target.
            region_size: Proportion of pairs to cover.

        Returns:
            (start_idx, end_idx) for frame pair range.
            Pair i means frames[i] and frames[i+1].
        """
        num_pairs = batch_size - 1
        if num_pairs <= 0:
            return 0, 0

        if target_region == "full_batch":
            return 0, num_pairs

        region_pairs = max(1, int(round(num_pairs * region_size)))

        if target_region == "start":
            return 0, min(region_pairs, num_pairs)
        elif target_region == "end":
            start = max(0, num_pairs - region_pairs)
            return start, num_pairs
        elif target_region == "middle":
            center = num_pairs // 2
            half = region_pairs // 2
            start = max(0, center - half)
            end = min(num_pairs, start + region_pairs)
            return start, end

        return 0, num_pairs

    def _interpolate_pair(
        self,
        frame_a: torch.Tensor,
        frame_b: torch.Tensor,
        num_frames: int,
        easing: str,
        **bezier_kwargs,
    ) -> torch.Tensor:
        """
        Generate interpolated frames between two frames.

        All N frames are computed in a single vectorized
        broadcast operation on the source device.

        Args:
            frame_a: First frame (H, W, C).
            frame_b: Second frame (H, W, C).
            num_frames: How many intermediate frames.
            easing: Easing method name.
            **bezier_kwargs: p1_x, p1_y, p2_x, p2_y.

        Returns:
            Interpolated frames (N, H, W, C).
        """
        device = frame_a.device

        # Evenly spaced t in (0, 1) exclusive of endpoints
        t = torch.linspace(
            1.0 / (num_frames + 1),
            num_frames / (num_frames + 1),
            num_frames,
            device=device,
        )

        weights = apply_easing(t, easing, **bezier_kwargs)

        # Broadcast blend: (N, 1, 1, 1) * (H, W, C)
        w = weights.view(-1, 1, 1, 1)
        return (
            frame_a.unsqueeze(0) * (1.0 - w)
            + frame_b.unsqueeze(0) * w
        )

    def _generate_info(
        self,
        original_count: int,
        new_count: int,
        frames_to_insert: int,
        easing: str,
        target_region: str,
        region_start: int,
        region_end: int,
        bezier_preset: str = "ease",
        p1_x: float = 0.25,
        p1_y: float = 0.1,
        p2_x: float = 0.25,
        p2_y: float = 1.0,
    ) -> str:
        """Generate human-readable info string."""
        total_pairs = original_count - 1
        pairs_interpolated = region_end - region_start
        inserted_total = (
            pairs_interpolated * frames_to_insert
        )
        lines = [
            "Frame Ramp Boogie Report",
            "=" * 24,
            f"Original frames: {original_count}",
            f"New frames: {new_count}",
            f"Multiplier: "
            f"{new_count / original_count:.2f}x",
            "",
            "Interpolation",
            "-" * 24,
            f"Inserted per pair: {frames_to_insert}",
            f"Easing: {easing}",
        ]

        if easing == "cubic_bezier":
            lines.append(
                f"Bezier preset: {bezier_preset}"
            )
            lines.append(
                f"P1({p1_x:.2f}, {p1_y:.2f})  "
                f"P2({p2_x:.2f}, {p2_y:.2f})"
            )

        lines.extend([
            "",
            "Region",
            "-" * 24,
            f"Target: {target_region}",
            f"Pairs interpolated: "
            f"{pairs_interpolated} of {total_pairs}",
            f"Total frames inserted: {inserted_total}",
        ])
        return "\n".join(lines)

    def boogie(
        self,
        images: torch.Tensor,
        frames_to_insert: int,
        easing: str,
        target_region: str,
        region_size: float = 0.33,
        bezier_preset: str = "ease",
        p1_x: float = 0.25,
        p1_y: float = 0.1,
        p2_x: float = 0.25,
        p2_y: float = 1.0,
    ) -> Tuple:
        """
        Main interpolation entry point.

        Args:
            images: Input batch (B, H, W, C).
            frames_to_insert: Frames to insert per pair.
            easing: Easing curve name.
            target_region: Which region to interpolate.
            region_size: Proportion of batch for region.
            bezier_preset: Named bezier preset or custom.
            p1_x: Bezier control point 1 X.
            p1_y: Bezier control point 1 Y.
            p2_x: Bezier control point 2 X.
            p2_y: Bezier control point 2 Y.

        Returns:
            (images, original_count, new_count, info)
        """
        batch_size = images.shape[0]

        if batch_size < 2:
            info = (
                "Frame Ramp Boogie: need at least 2 frames. "
                f"Got {batch_size}."
            )
            return (images, batch_size, batch_size, info)

        start_idx, end_idx = self._compute_region(
            batch_size, target_region, region_size
        )

        # Apply bezier preset if not custom
        if (
            easing == "cubic_bezier"
            and bezier_preset in _PRESET_VALUES
        ):
            p1_x, p1_y, p2_x, p2_y = _PRESET_VALUES[
                bezier_preset
            ]

        bezier_kwargs = {
            "p1_x": p1_x,
            "p1_y": p1_y,
            "p2_x": p2_x,
            "p2_y": p2_y,
        }

        chunks = []
        for i in range(batch_size - 1):
            # Always include the current original frame
            chunks.append(images[i:i + 1])

            # Insert interpolated frames for pairs in region
            if start_idx <= i < end_idx:
                interp = self._interpolate_pair(
                    images[i],
                    images[i + 1],
                    frames_to_insert,
                    easing,
                    **bezier_kwargs,
                )
                chunks.append(interp)

        # Include the final frame
        chunks.append(images[-1:])

        result = torch.cat(chunks, dim=0)
        new_count = result.shape[0]

        info = self._generate_info(
            batch_size, new_count, frames_to_insert,
            easing, target_region, start_idx, end_idx,
            bezier_preset, p1_x, p1_y, p2_x, p2_y,
        )

        return (result, batch_size, new_count, info)


NODE_CLASS_MAPPINGS = {
    "FrameRampBoogie": FrameRampBoogie,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FrameRampBoogie": "Frame Ramp Boogie",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
