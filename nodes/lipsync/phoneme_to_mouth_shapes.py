"""
Phoneme to mouth shapes node.

Converts phoneme timing data to per-frame mouth shape indices
for use with the compositor node.
"""
from typing import Dict, Any, List, Tuple

from ...utils.phoneme_utils import (
    phonemes_to_frame_sequence,
    smooth_viseme_sequence,
    get_all_mouth_shapes,
)


class PhonemeToMouthShapes:
    """
    Convert phoneme timing data to frame-by-frame mouth shape indices.

    Takes phoneme data with timing and produces a sequence of
    mouth shape indices (0-8) for each video frame.
    """

    CATEGORY = "Trent/LipSync"
    DESCRIPTION = (
        "Convert phoneme timing data to per-frame mouth shape sequence."
    )

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "phoneme_data": ("PHONEME_DATA", {
                    "tooltip": "Phoneme timing data from AudioToPhonemes"
                }),
                "duration": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 3600.0,
                    "step": 0.01,
                    "tooltip": "Audio duration in seconds"
                }),
                "fps": ("FLOAT", {
                    "default": 24.0,
                    "min": 1.0,
                    "max": 120.0,
                    "step": 0.1,
                    "tooltip": "Video frame rate"
                }),
            },
            "optional": {
                "mapping_type": (["arpabet", "ipa", "simplified"], {
                    "default": "arpabet",
                    "tooltip": "Phoneme-to-viseme mapping type"
                }),
                "hold_frames": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "tooltip": "Minimum frames to hold each mouth shape"
                }),
                "smoothing": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Apply smoothing to reduce flickering"
                }),
            },
        }

    RETURN_TYPES = ("MOUTH_SEQUENCE", "INT")
    RETURN_NAMES = ("mouth_sequence", "frame_count")
    FUNCTION = "convert_to_shapes"

    def convert_to_shapes(
        self,
        phoneme_data: List[Dict],
        duration: float,
        fps: float,
        mapping_type: str = "arpabet",
        hold_frames: int = 2,
        smoothing: bool = True
    ) -> Tuple[List[int], int]:
        """
        Convert phonemes to mouth shape sequence.

        Args:
            phoneme_data: List of phoneme timing dicts
            duration: Audio duration in seconds
            fps: Video frame rate
            mapping_type: Which phoneme mapping to use
            hold_frames: Minimum frames per mouth shape
            smoothing: Whether to apply smoothing

        Returns:
            Tuple of (mouth_shape_indices, frame_count)
        """
        # Convert phonemes to frame sequence
        sequence = phonemes_to_frame_sequence(
            phoneme_data,
            fps,
            duration,
            mapping=mapping_type
        )

        # Apply smoothing if requested
        if smoothing and hold_frames > 1:
            sequence = smooth_viseme_sequence(sequence, hold_frames)

        return (sequence, len(sequence))


class MouthShapePreview:
    """
    Preview mouth shape sequence as text for debugging.

    Shows the mouth shape letter (A-H, X) for each frame.
    """

    CATEGORY = "Trent/LipSync"
    DESCRIPTION = "Preview mouth shape sequence as text."

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "mouth_sequence": ("MOUTH_SEQUENCE", {
                    "tooltip": "Mouth shape sequence from PhonemeToMouthShapes"
                }),
            },
            "optional": {
                "max_display": ("INT", {
                    "default": 100,
                    "min": 10,
                    "max": 1000,
                    "step": 10,
                    "tooltip": "Maximum frames to display"
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("preview",)
    FUNCTION = "preview_sequence"

    def preview_sequence(
        self,
        mouth_sequence: List[int],
        max_display: int = 100
    ) -> Tuple[str]:
        """Generate text preview of mouth shape sequence."""
        shapes = get_all_mouth_shapes()

        # Convert indices to letters
        letters = [shapes.get(idx, "?") for idx in mouth_sequence]

        # Truncate if too long
        if len(letters) > max_display:
            display = letters[:max_display]
            suffix = f"... (+{len(letters) - max_display} more frames)"
        else:
            display = letters
            suffix = ""

        # Format with frame numbers every 10 frames
        lines = []
        for i in range(0, len(display), 20):
            chunk = display[i:i+20]
            frame_str = "".join(chunk)
            lines.append(f"[{i:04d}] {frame_str}")

        result = "\n".join(lines)
        if suffix:
            result += f"\n{suffix}"

        return (result,)
