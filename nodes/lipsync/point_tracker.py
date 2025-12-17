"""
Lightweight point tracker for lip sync positioning.

Tracks a single point across video frames using optical flow,
outputting per-frame coordinates for mouth placement.
"""
import base64
from io import BytesIO
from typing import Dict, Any, Tuple, List

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from comfy.utils import ProgressBar


class PointTracker:
    """
    Track a point across video frames using optical flow.

    Given an initial (x, y) coordinate on the first frame,
    tracks that point through subsequent frames using
    iterative Lucas-Kanade optical flow with pyramids.
    """

    CATEGORY = "Trent/LipSync"
    DESCRIPTION = (
        "Track a point across video frames for mouth placement. "
        "Uses pyramidal Lucas-Kanade optical flow with sub-pixel accuracy."
    )

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "frames": ("IMAGE", {
                    "tooltip": "Video frames batch (B, H, W, C)"
                }),
                "start_x": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 8192,
                    "step": 1,
                    "tooltip": "Initial X coordinate to track"
                }),
                "start_y": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 8192,
                    "step": 1,
                    "tooltip": "Initial Y coordinate to track"
                }),
            },
            "optional": {
                "window_size": ("INT", {
                    "default": 31,
                    "min": 11,
                    "max": 1025,
                    "step": 2,
                    "tooltip": (
                        "Search window size in pixels. Use large values "
                        "(201+) for full-frame tracking of fast-moving objects"
                    )
                }),
                "pyramid_levels": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 8,
                    "step": 1,
                    "tooltip": (
                        "Pyramid levels (more = handles larger motion). "
                        "Use 6-8 for very large frame-to-frame motion"
                    )
                }),
                "iterations": ("INT", {
                    "default": 10,
                    "min": 3,
                    "max": 50,
                    "step": 1,
                    "tooltip": "Iterations per level (more = more accurate)"
                }),
                "smoothing": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 0.9,
                    "step": 0.05,
                    "tooltip": "Temporal smoothing (0=none, higher=smoother)"
                }),
                "search_radius_percent": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 5.0,
                    "tooltip": (
                        "Template match search as % of frame size. "
                        "0=use window_size, 50=search half the frame, "
                        "100=search entire frame"
                    )
                }),
            },
        }

    RETURN_TYPES = ("POINT_SEQUENCE", "MASK", "IMAGE")
    RETURN_NAMES = ("points", "tracking_masks", "preview")
    FUNCTION = "track_point"

    def track_point(
        self,
        frames: torch.Tensor,
        start_x: int,
        start_y: int,
        window_size: int = 31,
        pyramid_levels: int = 4,
        iterations: int = 10,
        smoothing: float = 0.3,
        search_radius_percent: float = 0.0
    ) -> Tuple[List[Tuple[int, int]], torch.Tensor, torch.Tensor]:
        """
        Track a point across video frames using pyramidal LK.
        """
        device = frames.device
        num_frames = frames.shape[0]
        H, W = frames.shape[1], frames.shape[2]

        # Ensure window size is odd and clamp to frame size
        window_size = window_size | 1
        max_window = min(H, W) - 2  # Leave margin
        window_size = min(window_size, max_window)
        half_win = window_size // 2

        # Calculate search radius for template matching fallback
        if search_radius_percent > 0:
            # Use percentage of frame diagonal
            frame_diag = (H ** 2 + W ** 2) ** 0.5
            search_radius = int(frame_diag * search_radius_percent / 100)
            search_radius = max(half_win, search_radius)
        else:
            search_radius = half_win

        # Convert to grayscale for tracking
        gray_frames = self._to_grayscale(frames)

        # Build pyramids for all frames upfront
        pyramids = [self._build_pyramid(gray_frames[i], pyramid_levels)
                    for i in range(num_frames)]

        # Initialize tracking with sub-pixel precision
        points = [(start_x, start_y)]
        current_x, current_y = float(start_x), float(start_y)

        # Store ORIGINAL template from first frame (never changes)
        original_template = self._extract_template(
            gray_frames[0], start_x, start_y, half_win
        )

        # Adaptive template that gets updated periodically
        adaptive_template = original_template.clone()
        last_good_x, last_good_y = current_x, current_y
        last_template_update = 0
        template_update_interval = max(5, num_frames // 20)  # Update ~20 times

        # Track consecutive low-confidence frames
        low_confidence_streak = 0

        pbar = ProgressBar(num_frames - 1)

        for i in range(1, num_frames):
            prev_pyr = pyramids[i - 1]
            curr_pyr = pyramids[i]

            # Compute flow using iterative pyramidal LK
            dx, dy, confidence = self._pyramidal_lk(
                prev_pyr, curr_pyr,
                current_x, current_y,
                half_win, iterations, device
            )

            # Update position
            new_x = current_x + dx
            new_y = current_y + dy

            # Clamp to frame bounds with margin
            margin = half_win
            new_x = max(margin, min(W - margin - 1, new_x))
            new_y = max(margin, min(H - margin - 1, new_y))

            # ROBUST RECOVERY: Multi-stage confidence handling
            used_fallback = False

            if confidence < 0.1:  # Raised threshold significantly
                low_confidence_streak += 1

                # Stage 1: Try template match with adaptive template first
                tx, ty = self._template_match(
                    gray_frames[i], adaptive_template,
                    int(new_x), int(new_y),
                    search_radius=search_radius
                )
                if tx is not None:
                    new_x, new_y = tx, ty
                    used_fallback = True

                # Stage 2: If still failing, try original template
                if tx is None or low_confidence_streak > 3:
                    tx2, ty2 = self._template_match(
                        gray_frames[i], original_template,
                        int(new_x), int(new_y),
                        search_radius=search_radius
                    )
                    if tx2 is not None:
                        new_x, new_y = tx2, ty2
                        used_fallback = True

                # Stage 3: If streak is long, do full-frame search
                if low_confidence_streak > 5:
                    # Full frame search with original template
                    full_search = max(W, H) // 2
                    tx3, ty3 = self._template_match(
                        gray_frames[i], original_template,
                        W // 2, H // 2,  # Start from center
                        search_radius=full_search
                    )
                    if tx3 is not None:
                        new_x, new_y = tx3, ty3
                        used_fallback = True
                        low_confidence_streak = 0  # Reset streak
            else:
                low_confidence_streak = 0  # Reset on good confidence

            # Validate against original template periodically
            if not used_fallback and i - last_template_update > template_update_interval:
                # Check if current position still matches original
                curr_patch = self._extract_template(
                    gray_frames[i], int(new_x), int(new_y), half_win
                )
                if curr_patch is not None and curr_patch.shape == original_template.shape:
                    ncc = self._compute_ncc(curr_patch, original_template)
                    if ncc < 0.4:  # Drifted too far from original
                        # Re-search with original template
                        tx, ty = self._template_match(
                            gray_frames[i], original_template,
                            int(new_x), int(new_y),
                            search_radius=search_radius
                        )
                        if tx is not None:
                            new_x, new_y = tx, ty
                    elif ncc > 0.6:
                        # Good match - update adaptive template
                        adaptive_template = curr_patch.clone()
                        last_good_x, last_good_y = new_x, new_y

                last_template_update = i

            # Apply temporal smoothing
            if smoothing > 0 and len(points) > 0:
                prev_x, prev_y = points[-1]
                new_x = prev_x + (1 - smoothing) * (new_x - prev_x)
                new_y = prev_y + (1 - smoothing) * (new_y - prev_y)

            current_x, current_y = new_x, new_y
            points.append((int(round(current_x)), int(round(current_y))))

            pbar.update(1)

        # Generate tracking masks
        masks = self._generate_point_masks(points, H, W, device)

        # Generate preview
        preview = self._draw_tracking_preview(frames, points)

        return (points, masks, preview)

    def _to_grayscale(self, frames: torch.Tensor) -> torch.Tensor:
        """Convert RGB frames to grayscale."""
        weights = torch.tensor(
            [0.299, 0.587, 0.114],
            device=frames.device,
            dtype=frames.dtype
        )
        return (frames[..., :3] * weights).sum(dim=-1)

    def _build_pyramid(
        self,
        img: torch.Tensor,
        levels: int
    ) -> List[torch.Tensor]:
        """Build Gaussian pyramid with proper smoothing."""
        pyramid = [img]

        for _ in range(1, levels):
            prev = pyramid[-1]
            if prev.shape[0] < 8 or prev.shape[1] < 8:
                break

            # Gaussian blur before downsampling (proper anti-aliasing)
            blurred = self._gaussian_blur(prev, sigma=1.0)

            # Downsample by 2
            downsampled = F.avg_pool2d(
                blurred.unsqueeze(0).unsqueeze(0),
                kernel_size=2,
                stride=2
            ).squeeze()
            pyramid.append(downsampled)

        return pyramid

    def _gaussian_blur(
        self,
        img: torch.Tensor,
        sigma: float = 1.0
    ) -> torch.Tensor:
        """Apply Gaussian blur."""
        ksize = int(4 * sigma + 1) | 1
        x = torch.arange(ksize, device=img.device) - ksize // 2
        kernel_1d = torch.exp(-x.float()**2 / (2 * sigma**2))
        kernel_1d = kernel_1d / kernel_1d.sum()

        # Separable convolution
        img_4d = img.unsqueeze(0).unsqueeze(0)
        padded = F.pad(img_4d, [ksize//2]*4, mode='reflect')

        # Horizontal pass
        kernel_h = kernel_1d.view(1, 1, 1, -1)
        blurred = F.conv2d(padded, kernel_h)

        # Vertical pass
        kernel_v = kernel_1d.view(1, 1, -1, 1)
        blurred = F.conv2d(blurred, kernel_v)

        return blurred.squeeze()

    def _pyramidal_lk(
        self,
        prev_pyr: List[torch.Tensor],
        curr_pyr: List[torch.Tensor],
        x: float,
        y: float,
        half_win: int,
        iterations: int,
        device: torch.device
    ) -> Tuple[float, float, float]:
        """
        Pyramidal Lucas-Kanade optical flow.
        Returns (dx, dy, confidence).
        """
        num_levels = min(len(prev_pyr), len(curr_pyr))

        # Start from coarsest level
        scale = 2 ** (num_levels - 1)
        px, py = x / scale, y / scale
        gx, gy = 0.0, 0.0  # Accumulated flow guess

        confidence = 0.0

        for level in range(num_levels - 1, -1, -1):
            prev_img = prev_pyr[level]
            curr_img = curr_pyr[level]
            lH, lW = prev_img.shape

            # Scale the guess from previous level
            if level < num_levels - 1:
                gx *= 2
                gy *= 2
                px = (x / (2 ** level))
                py = (y / (2 ** level))

            # Compute gradients using Scharr operator (more accurate than Sobel)
            Ix, Iy = self._scharr_gradients(prev_img)

            # Iterative refinement at this level
            vx, vy = 0.0, 0.0

            for _ in range(iterations):
                # Current estimate position in curr image
                cx = px + gx + vx
                cy = py + gy + vy

                # Extract windows with sub-pixel interpolation
                prev_win = self._extract_window_subpixel(
                    prev_img, px, py, half_win
                )
                curr_win = self._extract_window_subpixel(
                    curr_img, cx, cy, half_win
                )
                Ix_win = self._extract_window_subpixel(
                    Ix, px, py, half_win
                )
                Iy_win = self._extract_window_subpixel(
                    Iy, px, py, half_win
                )

                if prev_win is None or curr_win is None:
                    break

                # Temporal gradient
                It = curr_win - prev_win

                # Build structure tensor
                Ixx = (Ix_win * Ix_win).sum()
                Iyy = (Iy_win * Iy_win).sum()
                Ixy = (Ix_win * Iy_win).sum()
                Ixt = (Ix_win * It).sum()
                Iyt = (Iy_win * It).sum()

                # Check minimum eigenvalue (tracking quality)
                trace = Ixx + Iyy
                det = Ixx * Iyy - Ixy * Ixy

                # Eigenvalues: (trace +/- sqrt(trace^2 - 4*det)) / 2
                discriminant = trace * trace - 4 * det
                if discriminant < 0:
                    discriminant = torch.tensor(0.0, device=device)
                else:
                    discriminant = torch.sqrt(discriminant)

                min_eig = (trace - discriminant) / 2

                # Skip if texture is too weak
                if min_eig < 1e-4:
                    break

                confidence = min_eig.item()

                # Solve 2x2 system
                if abs(det.item()) < 1e-10:
                    break

                dvx = -(Iyy * Ixt - Ixy * Iyt) / det
                dvy = -(Ixx * Iyt - Ixy * Ixt) / det

                # Clamp update
                max_update = half_win / 2.0
                dvx = torch.clamp(dvx, -max_update, max_update)
                dvy = torch.clamp(dvy, -max_update, max_update)

                vx += dvx.item()
                vy += dvy.item()

                # Convergence check
                if abs(dvx.item()) < 0.01 and abs(dvy.item()) < 0.01:
                    break

            gx += vx
            gy += vy

        return gx, gy, confidence

    def _scharr_gradients(
        self,
        img: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute gradients using Scharr operator (more accurate)."""
        device = img.device
        dtype = img.dtype

        # Scharr kernels (more accurate than Sobel)
        kx = torch.tensor([
            [-3, 0, 3],
            [-10, 0, 10],
            [-3, 0, 3]
        ], device=device, dtype=dtype) / 32.0

        ky = torch.tensor([
            [-3, -10, -3],
            [0, 0, 0],
            [3, 10, 3]
        ], device=device, dtype=dtype) / 32.0

        img_4d = img.unsqueeze(0).unsqueeze(0)
        padded = F.pad(img_4d, [1, 1, 1, 1], mode='reflect')

        Ix = F.conv2d(padded, kx.view(1, 1, 3, 3)).squeeze()
        Iy = F.conv2d(padded, ky.view(1, 1, 3, 3)).squeeze()

        return Ix, Iy

    def _extract_window_subpixel(
        self,
        img: torch.Tensor,
        cx: float,
        cy: float,
        half_win: int
    ) -> torch.Tensor:
        """Extract window with bilinear interpolation for sub-pixel accuracy."""
        H, W = img.shape

        # Integer and fractional parts
        x0 = int(cx)
        y0 = int(cy)
        fx = cx - x0
        fy = cy - y0

        # Window bounds
        x1 = x0 - half_win
        y1 = y0 - half_win
        x2 = x0 + half_win + 2  # +2 for interpolation
        y2 = y0 + half_win + 2

        # Check bounds
        if x1 < 0 or y1 < 0 or x2 > W or y2 > H:
            return None

        # Extract larger window for interpolation
        window = img[y1:y2, x1:x2]

        # Bilinear interpolation weights
        w00 = (1 - fx) * (1 - fy)
        w01 = fx * (1 - fy)
        w10 = (1 - fx) * fy
        w11 = fx * fy

        # Interpolate
        h, w = window.shape
        if h < 2 or w < 2:
            return None

        result = (
            w00 * window[:-1, :-1] +
            w01 * window[:-1, 1:] +
            w10 * window[1:, :-1] +
            w11 * window[1:, 1:]
        )

        # Crop to exact window size
        win_size = 2 * half_win + 1
        if result.shape[0] >= win_size and result.shape[1] >= win_size:
            return result[:win_size, :win_size]

        return result

    def _extract_template(
        self,
        img: torch.Tensor,
        x: int,
        y: int,
        half_win: int
    ) -> torch.Tensor:
        """Extract template patch for matching."""
        H, W = img.shape
        x1 = max(0, x - half_win)
        y1 = max(0, y - half_win)
        x2 = min(W, x + half_win + 1)
        y2 = min(H, y + half_win + 1)
        return img[y1:y2, x1:x2].clone()

    def _compute_ncc(
        self,
        patch1: torch.Tensor,
        patch2: torch.Tensor
    ) -> float:
        """Compute normalized cross-correlation between two patches."""
        if patch1.shape != patch2.shape:
            return 0.0

        p1_mean = patch1.mean()
        p2_mean = patch2.mean()
        p1_std = patch1.std() + 1e-6
        p2_std = patch2.std() + 1e-6

        p1_norm = (patch1 - p1_mean) / p1_std
        p2_norm = (patch2 - p2_mean) / p2_std

        return (p1_norm * p2_norm).mean().item()

    def _template_match(
        self,
        img: torch.Tensor,
        template: torch.Tensor,
        cx: int,
        cy: int,
        search_radius: int
    ) -> Tuple[float, float]:
        """
        Template matching fallback using GPU-accelerated NCC.

        Uses convolution for large search radii, falls back to loop
        for small regions.
        """
        H, W = img.shape
        th, tw = template.shape
        device = img.device

        # Search region bounds
        x1 = max(tw // 2, cx - search_radius)
        y1 = max(th // 2, cy - search_radius)
        x2 = min(W - tw // 2, cx + search_radius + 1)
        y2 = min(H - th // 2, cy + search_radius + 1)

        if x2 <= x1 or y2 <= y1:
            return None, None

        search_h = y2 - y1
        search_w = x2 - x1

        # For large search regions, use GPU convolution
        if search_h * search_w > 2500:  # > 50x50 region
            return self._template_match_conv(
                img, template, x1, y1, x2, y2
            )

        # For small regions, use direct computation
        best_score = -float('inf')
        best_x, best_y = cx, cy

        # Normalize template once
        template_mean = template.mean()
        template_std = template.std() + 1e-6
        template_norm = (template - template_mean) / template_std
        template_energy = (template_norm * template_norm).sum()

        # Search in region
        for sy in range(y1, y2):
            for sx in range(x1, x2):
                px1 = sx - tw // 2
                py1 = sy - th // 2
                px2 = px1 + tw
                py2 = py1 + th

                if px1 < 0 or py1 < 0 or px2 > W or py2 > H:
                    continue

                patch = img[py1:py2, px1:px2]

                if patch.shape != template.shape:
                    continue

                patch_mean = patch.mean()
                patch_std = patch.std() + 1e-6
                patch_norm = (patch - patch_mean) / patch_std

                ncc = (template_norm * patch_norm).mean()

                if ncc > best_score:
                    best_score = ncc
                    best_x, best_y = sx, sy

        if best_score > 0.5:
            return float(best_x), float(best_y)

        return None, None

    def _template_match_conv(
        self,
        img: torch.Tensor,
        template: torch.Tensor,
        x1: int,
        y1: int,
        x2: int,
        y2: int
    ) -> Tuple[float, float]:
        """
        GPU-accelerated template matching using convolution.

        Uses normalized cross-correlation computed via convolution
        for efficient large-area search.
        """
        H, W = img.shape
        th, tw = template.shape
        device = img.device

        # Extract search region with padding for template
        pad_y1 = max(0, y1 - th // 2)
        pad_y2 = min(H, y2 + th // 2)
        pad_x1 = max(0, x1 - tw // 2)
        pad_x2 = min(W, x2 + tw // 2)

        search_region = img[pad_y1:pad_y2, pad_x1:pad_x2]

        # Normalize template
        template_mean = template.mean()
        template_std = template.std() + 1e-6
        template_norm = (template - template_mean) / template_std

        # Prepare for convolution: flip template for correlation
        kernel = template_norm.flip(0, 1).unsqueeze(0).unsqueeze(0)

        # Compute local means using box filter
        ones_kernel = torch.ones(1, 1, th, tw, device=device) / (th * tw)
        search_4d = search_region.unsqueeze(0).unsqueeze(0)

        # Local mean
        local_mean = F.conv2d(search_4d, ones_kernel, padding='valid')

        # Local squared mean for variance
        search_sq = search_4d ** 2
        local_sq_mean = F.conv2d(search_sq, ones_kernel, padding='valid')
        local_var = local_sq_mean - local_mean ** 2
        local_std = torch.sqrt(local_var.clamp(min=1e-10))

        # Cross-correlation
        cross_corr = F.conv2d(search_4d, kernel, padding='valid')

        # Normalized cross-correlation
        ncc = cross_corr / (local_std * th * tw + 1e-6)
        ncc = ncc.squeeze()

        if ncc.numel() == 0:
            return None, None

        # Find maximum
        max_idx = ncc.argmax()
        max_val = ncc.flatten()[max_idx]

        if max_val < 0.5:
            return None, None

        # Convert flat index to 2D coordinates
        ncc_h, ncc_w = ncc.shape
        max_y = (max_idx // ncc_w).item()
        max_x = (max_idx % ncc_w).item()

        # Convert to image coordinates (center of template)
        result_x = pad_x1 + max_x + tw // 2
        result_y = pad_y1 + max_y + th // 2

        return float(result_x), float(result_y)

    def _generate_point_masks(
        self,
        points: List[Tuple[int, int]],
        H: int,
        W: int,
        device: torch.device
    ) -> torch.Tensor:
        """Generate gaussian masks centered on each tracked point."""
        num_frames = len(points)
        masks = torch.zeros(num_frames, H, W, device=device)

        radius = 5
        sigma = 2.0
        size = radius * 2 + 1

        y_grid = torch.arange(size, device=device) - radius
        x_grid = torch.arange(size, device=device) - radius
        yy, xx = torch.meshgrid(y_grid, x_grid, indexing='ij')
        gaussian = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))

        for i, (px, py) in enumerate(points):
            y1 = py - radius
            y2 = py + radius + 1
            x1 = px - radius
            x2 = px + radius + 1

            sy1 = max(0, -y1)
            sy2 = size - max(0, y2 - H)
            sx1 = max(0, -x1)
            sx2 = size - max(0, x2 - W)

            dy1 = max(0, y1)
            dy2 = min(H, y2)
            dx1 = max(0, x1)
            dx2 = min(W, x2)

            if dy2 > dy1 and dx2 > dx1:
                masks[i, dy1:dy2, dx1:dx2] = gaussian[sy1:sy2, sx1:sx2]

        return masks

    def _draw_tracking_preview(
        self,
        frames: torch.Tensor,
        points: List[Tuple[int, int]]
    ) -> torch.Tensor:
        """Draw crosshairs on frames at tracked point locations."""
        preview = frames.clone()
        H, W = frames.shape[1], frames.shape[2]

        arm_length = 12
        thickness = 2

        green = torch.tensor([0.0, 1.0, 0.0], device=frames.device)
        red = torch.tensor([1.0, 0.0, 0.0], device=frames.device)
        white = torch.tensor([1.0, 1.0, 1.0], device=frames.device)

        for i, (px, py) in enumerate(points):
            # Draw white outline
            for dx in range(-arm_length - 1, arm_length + 2):
                for dy in range(-thickness // 2 - 1, thickness // 2 + 2):
                    x, y = px + dx, py + dy
                    if 0 <= x < W and 0 <= y < H:
                        preview[i, y, x, :3] = white

            for dy in range(-arm_length - 1, arm_length + 2):
                for dx in range(-thickness // 2 - 1, thickness // 2 + 2):
                    x, y = px + dx, py + dy
                    if 0 <= x < W and 0 <= y < H:
                        preview[i, y, x, :3] = white

            # Draw red crosshair
            for dx in range(-arm_length, arm_length + 1):
                for dy in range(-thickness // 2, thickness // 2 + 1):
                    x, y = px + dx, py + dy
                    if 0 <= x < W and 0 <= y < H:
                        preview[i, y, x, :3] = red

            for dy in range(-arm_length, arm_length + 1):
                for dx in range(-thickness // 2, thickness // 2 + 1):
                    x, y = px + dx, py + dy
                    if 0 <= x < W and 0 <= y < H:
                        preview[i, y, x, :3] = red

            # Green center dot
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    x, y = px + dx, py + dy
                    if 0 <= x < W and 0 <= y < H:
                        preview[i, y, x, :3] = green

        return preview


class PointPreview:
    """
    Preview a point location on an image with click-to-pick interface.

    Use this to visually select where start_x and start_y will be
    before running the full point tracker. Click on the image in the
    node to set coordinates.
    """

    CATEGORY = "Trent/LipSync"
    DESCRIPTION = (
        "Preview and pick a point location on an image. "
        "Click on the image to set coordinates for point tracker."
    )
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Image to preview point on (first frame)"
                }),
                "x": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 8192,
                    "step": 1,
                    "tooltip": "X coordinate (click image to set)"
                }),
                "y": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 8192,
                    "step": 1,
                    "tooltip": "Y coordinate (click image to set)"
                }),
            },
            "optional": {
                "crosshair_size": ("INT", {
                    "default": 15,
                    "min": 5,
                    "max": 50,
                    "step": 1,
                    "tooltip": "Size of crosshair arms"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("preview", "x", "y")
    FUNCTION = "preview_point"

    def preview_point(
        self,
        image: torch.Tensor,
        x: int,
        y: int,
        crosshair_size: int = 15
    ) -> Dict[str, Any]:
        """Draw crosshair on image at specified point."""
        if image.shape[0] > 1:
            frame = image[0:1].clone()
        else:
            frame = image.clone()

        H, W = frame.shape[1], frame.shape[2]
        thickness = 2

        green = torch.tensor([0.0, 1.0, 0.0], device=frame.device)
        red = torch.tensor([1.0, 0.0, 0.0], device=frame.device)
        white = torch.tensor([1.0, 1.0, 1.0], device=frame.device)

        # Draw outer white border
        for dx in range(-crosshair_size - 1, crosshair_size + 2):
            for dy in range(-thickness // 2 - 1, thickness // 2 + 2):
                px_x, px_y = x + dx, y + dy
                if 0 <= px_x < W and 0 <= px_y < H:
                    frame[0, px_y, px_x, :3] = white

        for dy in range(-crosshair_size - 1, crosshair_size + 2):
            for dx in range(-thickness // 2 - 1, thickness // 2 + 2):
                px_x, px_y = x + dx, y + dy
                if 0 <= px_x < W and 0 <= px_y < H:
                    frame[0, px_y, px_x, :3] = white

        # Draw red crosshair
        for dx in range(-crosshair_size, crosshair_size + 1):
            for dy in range(-thickness // 2, thickness // 2 + 1):
                px_x, px_y = x + dx, y + dy
                if 0 <= px_x < W and 0 <= px_y < H:
                    frame[0, px_y, px_x, :3] = red

        for dy in range(-crosshair_size, crosshair_size + 1):
            for dx in range(-thickness // 2, thickness // 2 + 1):
                px_x, px_y = x + dx, y + dy
                if 0 <= px_x < W and 0 <= px_y < H:
                    frame[0, px_y, px_x, :3] = red

        # Green center dot
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                px_x, px_y = x + dx, y + dy
                if 0 <= px_x < W and 0 <= px_y < H:
                    frame[0, px_y, px_x, :3] = green

        # Convert original image to base64 for JS widget
        orig_frame = image[0] if image.shape[0] > 0 else image
        img_np = (orig_frame.cpu().numpy() * 255).astype(np.uint8)

        if img_np.shape[-1] == 4:
            img_np = img_np[..., :3]

        pil_img = Image.fromarray(img_np)

        buffer = BytesIO()
        pil_img.save(buffer, format="JPEG", quality=75)
        img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return {
            "ui": {"preview_image": [img_base64]},
            "result": (frame, x, y)
        }


class PointsToMasks:
    """
    Convert point sequence to tracking masks.

    Utility node to convert POINT_SEQUENCE to MASK format
    for use with the tracked compositor.
    """

    CATEGORY = "Trent/LipSync"
    DESCRIPTION = "Convert point coordinates to tracking masks."

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "points": ("POINT_SEQUENCE", {
                    "tooltip": "Point sequence from tracker"
                }),
                "height": ("INT", {
                    "default": 512,
                    "min": 1,
                    "max": 8192,
                    "tooltip": "Output mask height"
                }),
                "width": ("INT", {
                    "default": 512,
                    "min": 1,
                    "max": 8192,
                    "tooltip": "Output mask width"
                }),
            },
            "optional": {
                "radius": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 50,
                    "tooltip": "Gaussian mask radius"
                }),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("masks",)
    FUNCTION = "convert"

    def convert(
        self,
        points: List[Tuple[int, int]],
        height: int,
        width: int,
        radius: int = 5
    ) -> Tuple[torch.Tensor]:
        """Convert points to gaussian masks."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_frames = len(points)
        masks = torch.zeros(num_frames, height, width, device=device)

        sigma = radius / 2.0
        size = radius * 2 + 1

        y_grid = torch.arange(size, device=device) - radius
        x_grid = torch.arange(size, device=device) - radius
        yy, xx = torch.meshgrid(y_grid, x_grid, indexing='ij')
        gaussian = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))

        for i, (px, py) in enumerate(points):
            y1 = py - radius
            y2 = py + radius + 1
            x1 = px - radius
            x2 = px + radius + 1

            sy1 = max(0, -y1)
            sy2 = size - max(0, y2 - height)
            sx1 = max(0, -x1)
            sx2 = size - max(0, x2 - width)

            dy1 = max(0, y1)
            dy2 = min(height, y2)
            dx1 = max(0, x1)
            dx2 = min(width, x2)

            if dy2 > dy1 and dx2 > dx1:
                masks[i, dy1:dy2, dx1:dx2] = gaussian[sy1:sy2, sx1:sx2]

        return (masks,)
