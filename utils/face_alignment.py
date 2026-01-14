"""
Face-based alignment utilities using MediaPipe landmarks.

Provides eye-based affine alignment for subject matching when faces
are detected, with fallback to centroid-based alignment otherwise.
"""

import numpy as np
import torch
import torch.nn.functional as F
import cv2


# MediaPipe landmark indices for key facial features
LANDMARK_LEFT_EYE = 33
LANDMARK_RIGHT_EYE = 263
LANDMARK_NOSE_TIP = 1
LANDMARK_MOUTH_LEFT = 61
LANDMARK_MOUTH_RIGHT = 291


def detect_eye_positions(image_np):
    """
    Detect eye positions using MediaPipe FaceMesh.

    Args:
        image_np: (H, W, 3) numpy array, uint8 RGB

    Returns:
        tuple: (left_eye, right_eye) as numpy arrays [x, y], or None if no face
    """
    try:
        import mediapipe as mp
    except ImportError:
        print("[FaceAlignment] MediaPipe not installed. "
              "Run: pip install mediapipe")
        return None

    try:
        mp_face_mesh = mp.solutions.face_mesh
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        ) as face_mesh:
            # MediaPipe expects RGB
            if image_np.shape[2] == 3:
                rgb_image = image_np
            else:
                rgb_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

            results = face_mesh.process(rgb_image)

            if not results.multi_face_landmarks:
                return None

            h, w = image_np.shape[:2]
            landmarks = results.multi_face_landmarks[0].landmark

            left_eye = np.array([
                landmarks[LANDMARK_LEFT_EYE].x * w,
                landmarks[LANDMARK_LEFT_EYE].y * h
            ], dtype=np.float32)

            right_eye = np.array([
                landmarks[LANDMARK_RIGHT_EYE].x * w,
                landmarks[LANDMARK_RIGHT_EYE].y * h
            ], dtype=np.float32)

            return left_eye, right_eye

    except Exception as e:
        print(f"[FaceAlignment] Error detecting face: {e}")
        return None


def compute_eye_affine_transform(src_eyes, dst_eyes):
    """
    Compute affine transform to align source eyes to destination eyes.

    The transform includes scale, rotation, and translation to map
    source eye positions to destination eye positions.

    Args:
        src_eyes: (left_eye, right_eye) from source image as numpy arrays
        dst_eyes: (left_eye, right_eye) from destination image

    Returns:
        tuple: (affine_matrix, scale, rotation_degrees)
            - affine_matrix: 2x3 numpy array
            - scale: float scale factor
            - rotation_degrees: float rotation in degrees
    """
    src_left, src_right = src_eyes
    dst_left, dst_right = dst_eyes

    # Compute scale from eye distance ratio
    src_dist = np.linalg.norm(src_right - src_left)
    dst_dist = np.linalg.norm(dst_right - dst_left)

    if src_dist < 1e-6:
        return None, 1.0, 0.0

    scale = dst_dist / src_dist

    # Compute rotation from eye vectors
    src_vec = src_right - src_left
    dst_vec = dst_right - dst_left

    src_angle = np.arctan2(src_vec[1], src_vec[0])
    dst_angle = np.arctan2(dst_vec[1], dst_vec[0])
    rotation = dst_angle - src_angle

    # Compute eye centers
    src_center = (src_left + src_right) / 2
    dst_center = (dst_left + dst_right) / 2

    # Build affine matrix
    cos_r = np.cos(rotation)
    sin_r = np.sin(rotation)

    # Rotation matrix
    R = np.array([
        [cos_r, -sin_r],
        [sin_r, cos_r]
    ], dtype=np.float32)

    # Translation: dst_center - scale * R @ src_center
    translation = dst_center - scale * R @ src_center

    affine = np.zeros((2, 3), dtype=np.float32)
    affine[:2, :2] = scale * R
    affine[:, 2] = translation

    return affine, scale, np.degrees(rotation)


def apply_affine_to_image(
    image: torch.Tensor,
    affine_matrix: np.ndarray,
    device: torch.device
) -> torch.Tensor:
    """
    Apply 2x3 affine transform to image tensor.

    Args:
        image: (B, H, W, C) tensor
        affine_matrix: 2x3 numpy affine matrix
        device: torch device

    Returns:
        Transformed image tensor (B, H, W, C)
    """
    B, H, W, C = image.shape

    # Convert affine matrix to torch theta format
    # PyTorch expects normalized coordinates [-1, 1]
    # theta transforms destination coords to source coords (inverse mapping)

    # Extract components
    a = affine_matrix[0, 0]  # scale * cos
    b = affine_matrix[0, 1]  # -scale * sin
    c = affine_matrix[1, 0]  # scale * sin
    d = affine_matrix[1, 1]  # scale * cos
    tx = affine_matrix[0, 2]  # translation x
    ty = affine_matrix[1, 2]  # translation y

    # Compute inverse affine for grid_sample (dst -> src mapping)
    det = a * d - b * c
    if abs(det) < 1e-6:
        return image

    inv_a = d / det
    inv_b = -b / det
    inv_c = -c / det
    inv_d = a / det
    # Fixed: correct matrix multiplication for inverse translation
    inv_tx = -(inv_a * tx + inv_c * ty)
    inv_ty = -(inv_b * tx + inv_d * ty)

    # Build theta for affine_grid (normalized coords)
    theta = torch.zeros(B, 2, 3, device=device, dtype=image.dtype)
    theta[:, 0, 0] = inv_a
    theta[:, 0, 1] = inv_b
    theta[:, 0, 2] = inv_tx / (W / 2)
    theta[:, 1, 0] = inv_c
    theta[:, 1, 1] = inv_d
    theta[:, 1, 2] = inv_ty / (H / 2)

    # Apply transform
    image_bchw = image.permute(0, 3, 1, 2).to(device)
    grid = F.affine_grid(theta, image_bchw.shape, align_corners=False)
    transformed = F.grid_sample(
        image_bchw, grid, mode='bilinear',
        padding_mode='border', align_corners=False
    )

    return transformed.permute(0, 2, 3, 1)


def apply_affine_to_mask(
    mask: torch.Tensor,
    affine_matrix: np.ndarray,
    device: torch.device
) -> torch.Tensor:
    """
    Apply 2x3 affine transform to mask tensor.

    Args:
        mask: (B, H, W) tensor
        affine_matrix: 2x3 numpy affine matrix
        device: torch device

    Returns:
        Transformed mask tensor (B, H, W)
    """
    B, H, W = mask.shape

    # Add channel dimension for grid_sample
    mask_4d = mask.unsqueeze(1).to(device)

    # Extract components
    a = affine_matrix[0, 0]
    b = affine_matrix[0, 1]
    c = affine_matrix[1, 0]
    d = affine_matrix[1, 1]
    tx = affine_matrix[0, 2]
    ty = affine_matrix[1, 2]

    # Compute inverse
    det = a * d - b * c
    if abs(det) < 1e-6:
        return mask

    inv_a = d / det
    inv_b = -b / det
    inv_c = -c / det
    inv_d = a / det
    # Fixed: correct matrix multiplication for inverse translation
    inv_tx = -(inv_a * tx + inv_c * ty)
    inv_ty = -(inv_b * tx + inv_d * ty)

    # Build theta
    theta = torch.zeros(B, 2, 3, device=device, dtype=mask.dtype)
    theta[:, 0, 0] = inv_a
    theta[:, 0, 1] = inv_b
    theta[:, 0, 2] = inv_tx / (W / 2)
    theta[:, 1, 0] = inv_c
    theta[:, 1, 1] = inv_d
    theta[:, 1, 2] = inv_ty / (H / 2)

    # Apply transform
    grid = F.affine_grid(theta, mask_4d.shape, align_corners=False)
    transformed = F.grid_sample(
        mask_4d, grid, mode='bilinear',
        padding_mode='zeros', align_corners=False
    )

    return transformed.squeeze(1)


def detect_eyes_in_masked_region(
    image: torch.Tensor,
    mask: torch.Tensor
) -> tuple:
    """
    Detect eye positions within a masked subject region.

    Extracts the bounding box of the mask, crops the image,
    detects eyes, and returns positions in original image coords.

    Args:
        image: (B, H, W, C) tensor in [0, 1] range
        mask: (B, H, W) tensor where 1 = subject region

    Returns:
        tuple: (left_eye, right_eye) in original image coordinates,
               or None if no face detected
    """
    # Get bounding box of mask
    mask_np = mask[0].cpu().numpy()
    ys, xs = np.where(mask_np > 0.5)

    if len(xs) == 0 or len(ys) == 0:
        return None

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    # Add padding
    pad = 20
    H, W = mask_np.shape
    x_min = max(0, x_min - pad)
    y_min = max(0, y_min - pad)
    x_max = min(W, x_max + pad)
    y_max = min(H, y_max + pad)

    # Crop image for detection
    crop = image[0, y_min:y_max, x_min:x_max, :].cpu().numpy()
    crop_uint8 = (crop * 255).astype(np.uint8)

    # Detect eyes in crop
    eyes = detect_eye_positions(crop_uint8)

    if eyes is None:
        return None

    left_eye, right_eye = eyes

    # Convert back to original image coordinates
    left_eye[0] += x_min
    left_eye[1] += y_min
    right_eye[0] += x_min
    right_eye[1] += y_min

    return left_eye, right_eye


def rotate_image(
    image: torch.Tensor,
    angle_degrees: float,
    device: torch.device
) -> torch.Tensor:
    """
    Rotate image by angle around its center.

    Args:
        image: (B, H, W, C) tensor
        angle_degrees: Rotation angle in degrees (positive = counter-clockwise)
        device: torch device

    Returns:
        Rotated image tensor (B, H, W, C)
    """
    B, H, W, C = image.shape
    angle_rad = np.radians(angle_degrees)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    # Rotation matrix (around center, no translation needed)
    theta = torch.zeros(B, 2, 3, device=device, dtype=image.dtype)
    theta[:, 0, 0] = cos_a
    theta[:, 0, 1] = -sin_a
    theta[:, 1, 0] = sin_a
    theta[:, 1, 1] = cos_a

    image_bchw = image.permute(0, 3, 1, 2).to(device)
    grid = F.affine_grid(theta, image_bchw.shape, align_corners=False)
    rotated = F.grid_sample(
        image_bchw, grid, mode='bilinear',
        padding_mode='border', align_corners=False
    )
    return rotated.permute(0, 2, 3, 1)


def rotate_mask(
    mask: torch.Tensor,
    angle_degrees: float,
    device: torch.device
) -> torch.Tensor:
    """
    Rotate mask by angle around its center.

    Args:
        mask: (B, H, W) tensor
        angle_degrees: Rotation angle in degrees (positive = counter-clockwise)
        device: torch device

    Returns:
        Rotated mask tensor (B, H, W)
    """
    B, H, W = mask.shape
    angle_rad = np.radians(angle_degrees)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    theta = torch.zeros(B, 2, 3, device=device, dtype=mask.dtype)
    theta[:, 0, 0] = cos_a
    theta[:, 0, 1] = -sin_a
    theta[:, 1, 0] = sin_a
    theta[:, 1, 1] = cos_a

    mask_4d = mask.unsqueeze(1).to(device)
    grid = F.affine_grid(theta, mask_4d.shape, align_corners=False)
    rotated = F.grid_sample(
        mask_4d, grid, mode='bilinear',
        padding_mode='zeros', align_corners=False
    )
    return rotated.squeeze(1)
