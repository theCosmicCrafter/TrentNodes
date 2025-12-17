"""
Audio preprocessing utilities for lip sync.

Handles audio format conversion, resampling, and normalization
for speech recognition models.
"""
import os
import tempfile
from typing import Tuple, Optional, Dict, Any, Union
import logging

import torch
import numpy as np

logger = logging.getLogger(__name__)


def ensure_mono(waveform: torch.Tensor) -> torch.Tensor:
    """
    Convert audio to mono by averaging channels.

    Args:
        waveform: Audio tensor, shape (samples,) or (channels, samples)

    Returns:
        Mono waveform of shape (samples,)
    """
    if waveform.dim() == 1:
        return waveform
    elif waveform.dim() == 2:
        if waveform.shape[0] > waveform.shape[1]:
            # Shape is (samples, channels), transpose first
            waveform = waveform.T
        return waveform.mean(dim=0)
    else:
        # Handle batched case (B, C, S) -> (S,) using first sample
        return waveform[0].mean(dim=0) if waveform.dim() == 3 else waveform


def resample_audio(
    waveform: torch.Tensor,
    orig_sr: int,
    target_sr: int = 16000,
    use_gpu: bool = True
) -> torch.Tensor:
    """
    Resample audio to target sample rate.

    Uses GPU-accelerated resampling via torchaudio.functional when available,
    with fallbacks to torchaudio.transforms, scipy, or linear interpolation.

    Args:
        waveform: Audio tensor of shape (samples,)
        orig_sr: Original sample rate
        target_sr: Target sample rate (default 16000 for speech models)
        use_gpu: Whether to prefer GPU resampling (default True)

    Returns:
        Resampled waveform
    """
    if orig_sr == target_sr:
        return waveform

    # Try torchaudio.functional first (most efficient, GPU-native)
    if use_gpu:
        try:
            import torchaudio.functional as AF
            # functional.resample works directly on GPU tensors
            return AF.resample(waveform, orig_sr, target_sr)
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"torchaudio.functional.resample failed: {e}")

    # Try torchaudio.transforms (creates resampler object)
    try:
        import torchaudio
        resampler = torchaudio.transforms.Resample(
            orig_freq=orig_sr,
            new_freq=target_sr
        ).to(waveform.device)
        return resampler(waveform)
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"torchaudio.transforms.Resample failed: {e}")

    # Fallback to scipy (CPU only)
    try:
        from scipy import signal
        ratio = target_sr / orig_sr
        new_length = int(len(waveform) * ratio)
        device = waveform.device
        resampled = signal.resample(
            waveform.cpu().numpy(),
            new_length
        )
        return torch.from_numpy(resampled.astype(np.float32)).to(device)
    except ImportError:
        pass

    # Last resort: GPU-accelerated linear interpolation
    logger.warning(
        "Neither torchaudio nor scipy available for resampling. "
        "Using linear interpolation."
    )
    ratio = target_sr / orig_sr
    new_length = int(len(waveform) * ratio)
    indices = torch.linspace(
        0, len(waveform) - 1, new_length,
        device=waveform.device
    )
    indices_floor = indices.long()
    indices_ceil = (indices_floor + 1).clamp(max=len(waveform) - 1)
    weights = indices - indices_floor.float()
    return (
        waveform[indices_floor] * (1 - weights) +
        waveform[indices_ceil] * weights
    )


def normalize_audio(
    waveform: torch.Tensor,
    target_db: float = -20.0
) -> torch.Tensor:
    """
    Normalize audio to target dB level.

    Args:
        waveform: Audio tensor
        target_db: Target dB level (default -20 dB)

    Returns:
        Normalized waveform
    """
    # Calculate current RMS
    rms = torch.sqrt(torch.mean(waveform ** 2) + 1e-8)

    # Convert target dB to linear
    target_rms = 10 ** (target_db / 20)

    # Scale
    if rms > 1e-8:
        waveform = waveform * (target_rms / rms)

    # Clip to prevent overflow
    waveform = torch.clamp(waveform, -1.0, 1.0)

    return waveform


def preprocess_audio_for_recognition(
    waveform: torch.Tensor,
    sample_rate: int,
    target_sr: int = 16000,
    normalize: bool = True
) -> Tuple[torch.Tensor, int]:
    """
    Full preprocessing pipeline for speech recognition.

    Args:
        waveform: Raw audio tensor
        sample_rate: Original sample rate
        target_sr: Target sample rate for model
        normalize: Whether to normalize audio

    Returns:
        Tuple of (processed_waveform, new_sample_rate)
    """
    # Convert to mono
    waveform = ensure_mono(waveform)

    # Ensure float32
    if waveform.dtype != torch.float32:
        waveform = waveform.float()

    # Normalize to [-1, 1] if needed
    max_val = waveform.abs().max()
    if max_val > 1.0:
        waveform = waveform / max_val

    # Resample
    waveform = resample_audio(waveform, sample_rate, target_sr)

    # Normalize volume
    if normalize:
        waveform = normalize_audio(waveform)

    return waveform, target_sr


def get_audio_duration(waveform: torch.Tensor, sample_rate: int) -> float:
    """
    Calculate audio duration in seconds.

    Args:
        waveform: Audio tensor
        sample_rate: Sample rate

    Returns:
        Duration in seconds
    """
    if waveform.dim() == 1:
        num_samples = waveform.shape[0]
    elif waveform.dim() == 2:
        num_samples = max(waveform.shape)
    else:
        num_samples = waveform.shape[-1]

    return num_samples / sample_rate


def save_audio_to_wav(
    waveform: torch.Tensor,
    sample_rate: int,
    file_path: Optional[str] = None
) -> str:
    """
    Save audio tensor to WAV file.

    Args:
        waveform: Audio tensor (mono, float32)
        sample_rate: Sample rate
        file_path: Output path (creates temp file if None)

    Returns:
        Path to saved WAV file
    """
    if file_path is None:
        fd, file_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

    waveform = ensure_mono(waveform)
    waveform_np = waveform.cpu().numpy()

    try:
        import soundfile as sf
        sf.write(file_path, waveform_np, sample_rate)
    except ImportError:
        try:
            from scipy.io import wavfile
            # scipy expects int16
            waveform_int = (waveform_np * 32767).astype(np.int16)
            wavfile.write(file_path, sample_rate, waveform_int)
        except ImportError:
            # Manual WAV writing as last resort
            _write_wav_manual(file_path, waveform_np, sample_rate)

    return file_path


def _write_wav_manual(
    file_path: str,
    waveform: np.ndarray,
    sample_rate: int
) -> None:
    """Write WAV file without external libraries."""
    import struct

    waveform_int = (waveform * 32767).astype(np.int16)
    num_samples = len(waveform_int)
    bytes_per_sample = 2
    num_channels = 1

    with open(file_path, "wb") as f:
        # RIFF header
        f.write(b"RIFF")
        f.write(struct.pack(
            "<I",
            36 + num_samples * bytes_per_sample
        ))
        f.write(b"WAVE")

        # fmt chunk
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))  # chunk size
        f.write(struct.pack("<H", 1))   # PCM format
        f.write(struct.pack("<H", num_channels))
        f.write(struct.pack("<I", sample_rate))
        f.write(struct.pack(
            "<I",
            sample_rate * num_channels * bytes_per_sample
        ))
        f.write(struct.pack("<H", num_channels * bytes_per_sample))
        f.write(struct.pack("<H", bytes_per_sample * 8))

        # data chunk
        f.write(b"data")
        f.write(struct.pack("<I", num_samples * bytes_per_sample))
        f.write(waveform_int.tobytes())


def extract_audio_from_dict(audio_input: Any) -> Tuple[
    torch.Tensor, int
]:
    """
    Extract waveform and sample rate from ComfyUI audio input.

    Supports:
    - Standard dict with "waveform" and "sample_rate" keys
    - VideoHelperSuite LazyAudioMap (Mapping type)
    - Raw torch.Tensor

    Args:
        audio_input: Audio data in various formats

    Returns:
        Tuple of (waveform, sample_rate)
    """
    from collections.abc import Mapping

    # Handle Mapping types (dict, LazyAudioMap, etc.)
    if isinstance(audio_input, Mapping):
        # Access via [] to trigger lazy loading for LazyAudioMap
        try:
            waveform = audio_input["waveform"]
        except KeyError:
            raise ValueError("Audio input missing 'waveform' key")

        try:
            sample_rate = audio_input["sample_rate"]
        except KeyError:
            sample_rate = 44100

        # Handle batched waveforms
        if isinstance(waveform, torch.Tensor):
            if waveform.dim() == 3:
                # (B, C, S) -> (C, S) or (S,)
                waveform = waveform[0]
            if waveform.dim() == 2 and waveform.shape[0] <= 2:
                # (C, S) with C channels
                waveform = waveform.mean(dim=0)

        return waveform, sample_rate
    elif isinstance(audio_input, torch.Tensor):
        # Raw tensor, assume 44100 Hz
        return audio_input, 44100
    else:
        # Try to access as dict-like anyway (duck typing)
        try:
            waveform = audio_input["waveform"]
            sample_rate = audio_input.get("sample_rate", 44100) \
                if hasattr(audio_input, "get") else 44100
            return waveform, sample_rate
        except (TypeError, KeyError):
            raise TypeError(
                f"Unsupported audio input type: {type(audio_input)}. "
                f"Expected dict-like with 'waveform' and 'sample_rate' keys."
            )
