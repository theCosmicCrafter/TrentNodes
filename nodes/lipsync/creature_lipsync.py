"""
All-in-one creature lip sync node.

Combines audio analysis, phoneme mapping, and frame compositing
into a single convenience node.
"""
import os
import tempfile
from typing import Dict, Any, List, Tuple, Optional

import torch
from comfy.utils import ProgressBar

from ...utils.audio_utils import (
    preprocess_audio_for_recognition,
    save_audio_to_wav,
    get_audio_duration,
    extract_audio_from_dict,
)
from ...utils.phoneme_utils import (
    phonemes_to_frame_sequence,
    smooth_viseme_sequence,
)
from .audio_to_phonemes import (
    recognize_with_vosk,
    get_model_path,
    download_vosk_model,
)
from .mouth_shape_compositor import MouthShapeCompositor


class CreatureLipSync:
    """
    All-in-one lip sync for non-human characters.

    Processes audio to extract phonemes, maps them to mouth shapes,
    and composites the results onto video frames in a single node.
    """

    CATEGORY = "Trent/LipSync"
    DESCRIPTION = (
        "Complete lip sync pipeline: audio analysis, phoneme mapping, "
        "and frame compositing for non-human character animation."
    )

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "frames": ("IMAGE", {
                    "tooltip": "Video frames to apply lip sync to"
                }),
                "audio": ("AUDIO", {
                    "tooltip": "Audio with speech to analyze"
                }),
                "mouth_shapes": ("IMAGE", {
                    "tooltip": "9 mouth shape images (A-H + X, indices 0-8)"
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
                "fps": ("FLOAT", {
                    "default": 24.0,
                    "min": 1.0,
                    "max": 120.0,
                    "step": 0.1,
                    "tooltip": "Video frame rate"
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
                "hold_frames": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "tooltip": "Minimum frames to hold each mouth shape"
                }),
                "feather": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Edge feathering in pixels"
                }),
                "model_size": (["small", "large"], {
                    "default": "small",
                    "tooltip": "Vosk model size"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MOUTH_SEQUENCE", "PHONEME_DATA")
    RETURN_NAMES = ("frames", "mouth_sequence", "phoneme_data")
    FUNCTION = "process"

    def process(
        self,
        frames: torch.Tensor,
        audio: Dict[str, Any],
        mouth_shapes: torch.Tensor,
        position_x: int,
        position_y: int,
        fps: float,
        mask: Optional[torch.Tensor] = None,
        blend_mode: str = "alpha",
        scale: float = 1.0,
        hold_frames: int = 2,
        feather: int = 0,
        model_size: str = "small"
    ) -> Tuple[torch.Tensor, List[int], List[Dict]]:
        """
        Complete lip sync processing pipeline.

        Args:
            frames: Video frames (B, H, W, C)
            audio: Audio dict with waveform and sample_rate
            mouth_shapes: 9 mouth images (A-H + X)
            position_x: X placement offset
            position_y: Y placement offset
            fps: Video frame rate
            mask: Optional blending mask
            blend_mode: Blend method
            scale: Mouth shape scale factor
            hold_frames: Minimum frames per shape
            feather: Edge feather amount
            model_size: Vosk model size

        Returns:
            Tuple of (processed_frames, mouth_sequence, phoneme_data)
        """
        num_frames = frames.shape[0]

        # Step 1: Extract and preprocess audio
        waveform, sample_rate = extract_audio_from_dict(audio)
        duration = get_audio_duration(waveform, sample_rate)

        processed_audio, target_sr = preprocess_audio_for_recognition(
            waveform, sample_rate, target_sr=16000
        )

        # Step 2: Run speech recognition
        wav_path = save_audio_to_wav(processed_audio, target_sr)

        try:
            model_name = f"en-{model_size}"
            model_path = get_model_path(model_name)

            if not os.path.exists(model_path):
                model_path = download_vosk_model(model_name)

            phonemes = recognize_with_vosk(wav_path, model_path, target_sr)

            # Add silence padding
            if phonemes:
                if phonemes[0]["start"] > 0.05:
                    phonemes.insert(0, {
                        "start": 0.0,
                        "end": phonemes[0]["start"],
                        "phoneme": "SIL"
                    })
                if phonemes[-1]["end"] < duration - 0.05:
                    phonemes.append({
                        "start": phonemes[-1]["end"],
                        "end": duration,
                        "phoneme": "SIL"
                    })
            else:
                phonemes = [{
                    "start": 0.0,
                    "end": duration,
                    "phoneme": "SIL"
                }]

        finally:
            if os.path.exists(wav_path):
                os.remove(wav_path)

        # Step 3: Convert to mouth sequence
        sequence = phonemes_to_frame_sequence(
            phonemes, fps, duration, mapping="arpabet"
        )

        if hold_frames > 1:
            sequence = smooth_viseme_sequence(sequence, hold_frames)

        # Ensure sequence matches frame count
        if len(sequence) < num_frames:
            sequence = list(sequence) + [8] * (num_frames - len(sequence))
        elif len(sequence) > num_frames:
            sequence = sequence[:num_frames]

        # Step 4: Composite mouth shapes onto frames
        compositor = MouthShapeCompositor()
        result_frames, = compositor.composite(
            frames=frames,
            mouth_shapes=mouth_shapes,
            mouth_sequence=sequence,
            position_x=position_x,
            position_y=position_y,
            mask=mask,
            blend_mode=blend_mode,
            scale=scale,
            feather=feather
        )

        return (result_frames, sequence, phonemes)


class MouthShapeLoader:
    """
    Helper node to load and organize mouth shape images.

    Ensures mouth shapes are in the correct order for lip sync.
    """

    CATEGORY = "Trent/LipSync"
    DESCRIPTION = (
        "Load and organize mouth shape images for lip sync. "
        "Expects images in A-H, X order (9 total)."
    )

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Batch of 9 mouth shape images in A-H, X order"
                }),
            },
            "optional": {
                "validate": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Validate that exactly 9 images are provided"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("mouth_shapes",)
    FUNCTION = "load"

    def load(
        self,
        images: torch.Tensor,
        validate: bool = True
    ) -> Tuple[torch.Tensor]:
        """
        Load and validate mouth shape images.

        Args:
            images: Batch of mouth images
            validate: Whether to check count

        Returns:
            Validated mouth shapes tensor
        """
        if validate and images.shape[0] != 9:
            raise ValueError(
                f"Expected 9 mouth shapes (A-H + X), got {images.shape[0]}. "
                "Order should be: A (closed), B (slight), C (open), "
                "D (wide), E (round-ish), F (UU), G (OH), H (wide open), "
                "X (idle/silence)."
            )

        # Pad with last image if fewer than 9
        if images.shape[0] < 9:
            padding = images[-1:].expand(9 - images.shape[0], -1, -1, -1)
            images = torch.cat([images, padding], dim=0)

        return (images[:9],)
