"""
Model loading and caching utilities for TrentNodes.

Provides cached model loading for commonly used models:
- BiRefNet (BEN2) for subject segmentation
- SD 1.5 Inpainting model
"""

import os
import sys

import torch

import folder_paths


# BiRefNet (BEN2) model cache
_birefnet_model = None
_birefnet_device = None
_birefnet_available = None


def is_birefnet_available() -> bool:
    """
    Check if BiRefNet can be loaded (without actually loading it).

    Returns:
        True if ComfyUI-Easy-Use is installed with BiRefNet support
    """
    global _birefnet_available

    if _birefnet_available is not None:
        return _birefnet_available

    easy_use_path = os.path.join(
        folder_paths.base_path,
        "custom_nodes", "ComfyUI-Easy-Use", "py"
    )
    ben_model_path = os.path.join(
        easy_use_path, "modules", "ben", "model.py"
    )

    if os.path.exists(ben_model_path):
        _birefnet_available = True
    else:
        _birefnet_available = False
        print(
            "[TrentNodes] Note: BiRefNet requires "
            "ComfyUI-Easy-Use to be installed."
        )

    return _birefnet_available


def load_birefnet(device: torch.device):
    """
    Load BiRefNet (BEN2) model for high-quality subject segmentation.

    Model is cached for reuse across calls.

    Args:
        device: torch device to load model on

    Returns:
        BiRefNet model or None if not available
    """
    global _birefnet_model, _birefnet_device, _birefnet_available

    # Return cached model if available and on correct device
    if _birefnet_model is not None and _birefnet_device == device:
        return _birefnet_model

    try:
        # Add ComfyUI-Easy-Use to path if needed
        easy_use_path = os.path.join(
            folder_paths.base_path,
            "custom_nodes", "ComfyUI-Easy-Use", "py"
        )
        if easy_use_path not in sys.path:
            sys.path.insert(0, easy_use_path)

        from modules.ben.model import BEN_Base

        # Model path
        model_dir = os.path.join(folder_paths.models_dir, "rembg")
        model_path = os.path.join(model_dir, "BEN2_Base.pth")

        if not os.path.exists(model_path):
            from torch.hub import download_url_to_file
            os.makedirs(model_dir, exist_ok=True)
            url = (
                "https://huggingface.co/PramaLLC/BEN2/"
                "resolve/main/BEN2_Base.pth"
            )
            print("[TrentNodes] Downloading BiRefNet (~500MB)...")
            download_url_to_file(url, model_path)
            print("[TrentNodes] BiRefNet model downloaded.")

        model = BEN_Base().to(device).eval()
        model.loadcheckpoints(model_path)

        _birefnet_model = model
        _birefnet_device = device
        _birefnet_available = True
        print("[TrentNodes] BiRefNet model loaded successfully.")
        return model

    except Exception as e:
        print(f"[TrentNodes] Warning: Could not load BiRefNet: {e}")
        print("[TrentNodes] Falling back to auto-detection mode.")
        _birefnet_available = False
        return None


# SD Inpainting model cache
_inpaint_model = None
_inpaint_model_name = None


def load_sd_inpaint_model(
    checkpoint: str = "sd-v1-5-inpainting.safetensors",
    device: torch.device = None
):
    """
    Load SD 1.5 inpainting model for clean plate generation.

    Model is cached for reuse across calls.

    Args:
        checkpoint: Checkpoint filename
        device: torch device (unused, ComfyUI handles device placement)

    Returns:
        Tuple of (model, clip, vae)
    """
    global _inpaint_model, _inpaint_model_name

    import comfy.sd

    # Return cached model if available
    if _inpaint_model is not None and _inpaint_model_name == checkpoint:
        return _inpaint_model

    # Find or download checkpoint
    ckpt_path = os.path.join(
        folder_paths.models_dir, "checkpoints", checkpoint
    )

    if not os.path.exists(ckpt_path):
        alt_path = folder_paths.get_full_path("checkpoints", checkpoint)
        if alt_path and os.path.exists(alt_path):
            ckpt_path = alt_path
        else:
            print("[TrentNodes] Downloading SD 1.5 inpainting (~4GB)...")
            print("[TrentNodes] This is a one-time download.")
            from torch.hub import download_url_to_file
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            url = (
                "https://huggingface.co/webui/stable-diffusion-inpainting/"
                "resolve/main/sd-v1-5-inpainting.safetensors"
            )
            download_url_to_file(url, ckpt_path)
            print("[TrentNodes] SD inpainting model downloaded.")

    print("[TrentNodes] Loading SD inpainting model...")
    model, clip, vae, clipvision = comfy.sd.load_checkpoint_guess_config(
        ckpt_path,
        output_vae=True,
        output_clip=True,
        embedding_directory=folder_paths.get_folder_paths("embeddings")
    )

    _inpaint_model = (model, clip, vae)
    _inpaint_model_name = checkpoint
    print("[TrentNodes] SD inpainting model loaded.")

    return model, clip, vae


def clear_model_cache():
    """Clear all cached models to free memory."""
    global _birefnet_model, _birefnet_device
    global _inpaint_model, _inpaint_model_name

    _birefnet_model = None
    _birefnet_device = None
    _inpaint_model = None
    _inpaint_model_name = None

    # Force garbage collection
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("[TrentNodes] Model cache cleared.")
