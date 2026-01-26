"""
Image Caption Layout
A ComfyUI custom node for creating grid layouts of images with captions.
Accepts dynamic image/caption input pairs.
"""

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import textwrap
from typing import Dict, Any, Tuple, List


class ImageTextGrid:
    """
    Creates a grid layout of images with text captions below each.

    Dynamically adds image/text input pairs as you connect them.
    """

    BACKGROUND_COLORS = {
        "white": (255, 255, 255),
        "black": (0, 0, 0),
        "gray": (128, 128, 128),
    }

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "images_per_row": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "tooltip": "Number of images per row in the grid"
                }),
                "image_size": ("INT", {
                    "default": 256,
                    "min": 64,
                    "max": 1024,
                    "step": 8,
                    "tooltip": "Size to display each image (square)"
                }),
                "caption_height": ("INT", {
                    "default": 64,
                    "min": 0,
                    "max": 256,
                    "step": 8,
                    "tooltip": "Height reserved for caption text below each image"
                }),
                "font_size": ("INT", {
                    "default": 12,
                    "min": 8,
                    "max": 48,
                    "step": 1,
                    "tooltip": "Font size for caption text"
                }),
                "padding": ("INT", {
                    "default": 10,
                    "min": 0,
                    "max": 100,
                    "step": 2,
                    "tooltip": "Padding between images"
                }),
                "background_color": (list(cls.BACKGROUND_COLORS.keys()), {
                    "default": "white",
                    "tooltip": "Background color of the layout"
                }),
            },
            "optional": {
                # JS dynamically adds image_N and caption_N pairs
                "image_1": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("layout",)

    FUNCTION = "create_layout"
    CATEGORY = "Trent/Image"
    DESCRIPTION = "Creates a grid layout of images with captions."

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    def tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """Convert a single image tensor to PIL Image."""
        if tensor.dim() == 4:
            tensor = tensor[0]
        np_img = (tensor.cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(np_img, mode='RGB')

    def create_layout(
        self,
        images_per_row: int,
        image_size: int,
        caption_height: int,
        font_size: int,
        padding: int,
        background_color: str,
        **kwargs
    ) -> Tuple[torch.Tensor]:
        """Create the grid layout with images and captions."""

        # Collect all connected images and their captions
        image_data: List[Tuple[int, torch.Tensor, str]] = []

        for key, value in kwargs.items():
            if key.startswith("image_") and value is not None:
                try:
                    idx = int(key.split("_")[1])
                    caption_key = f"caption_{idx}"
                    caption = kwargs.get(caption_key, "") or ""
                    image_data.append((idx, value, str(caption)))
                except (ValueError, IndexError):
                    continue

        # Sort by index to maintain order
        image_data.sort(key=lambda x: x[0])

        if not image_data:
            # Return a placeholder if no images connected
            placeholder = torch.zeros((1, 256, 256, 3), dtype=torch.float32)
            return (placeholder,)

        # Get colors
        bg_color = self.BACKGROUND_COLORS.get(background_color, (255, 255, 255))
        text_color = (0, 0, 0) if background_color == "white" else (255, 255, 255)
        if background_color == "gray":
            text_color = (255, 255, 255)

        # Calculate layout dimensions
        num_images = len(image_data)
        num_rows = (num_images + images_per_row - 1) // images_per_row

        total_width = images_per_row * (image_size + padding) + padding
        total_height = num_rows * (image_size + caption_height + padding) + padding

        # Create layout canvas
        layout = Image.new('RGB', (total_width, total_height), color=bg_color)
        draw = ImageDraw.Draw(layout)

        # Try to load font
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            try:
                font = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                    font_size
                )
            except IOError:
                font = ImageFont.load_default()

        # Place each image and caption
        for i, (idx, img_tensor, caption) in enumerate(image_data):
            row = i // images_per_row
            col = i % images_per_row

            x = padding + col * (image_size + padding)
            y = padding + row * (image_size + caption_height + padding)

            # Convert tensor to PIL and resize
            pil_img = self.tensor_to_pil(img_tensor)
            pil_img = pil_img.resize((image_size, image_size), Image.LANCZOS)

            # Paste image
            layout.paste(pil_img, (x, y))

            # Draw caption if caption_height > 0
            if caption_height > 0 and caption:
                # Wrap text to fit width
                chars_per_line = max(1, (image_size - 10) // (font_size // 2 + 1))
                wrapped = textwrap.fill(caption, width=chars_per_line)

                # Limit lines to fit caption height
                lines = wrapped.split('\n')
                max_lines = max(1, (caption_height - 10) // (font_size + 2))
                if len(lines) > max_lines:
                    lines = lines[:max_lines]
                    if lines[-1]:
                        lines[-1] = lines[-1][:-3] + "..."
                wrapped = '\n'.join(lines)

                draw.text(
                    (x + 5, y + image_size + 5),
                    wrapped,
                    font=font,
                    fill=text_color
                )

        # Convert back to tensor
        layout_np = np.array(layout).astype(np.float32) / 255.0
        layout_tensor = torch.from_numpy(layout_np).unsqueeze(0)

        return (layout_tensor,)


NODE_CLASS_MAPPINGS = {
    "ImageTextGrid": ImageTextGrid,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageTextGrid": "Image+Text Grid",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
