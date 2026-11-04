"""Preprocessing, bounding-box cropping, and batched MobileNetV2 inference.

Extracts left and right eye crops from video frames using facial landmarks,
resizes, applies ImageNet normalization, and infers eye closure probabilities.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import torch

from src.cnn_model import build_model, load_checkpoint
from src.features import LEFT_EYE_INDICES, RIGHT_EYE_INDICES

logger = logging.getLogger(__name__)

# ImageNet normalization statistics
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def crop_eye_region(
    frame: np.ndarray,
    landmarks: np.ndarray,
    eye_indices: Sequence[int],
    target_size: tuple[int, int] = (128, 128),
    margin_ratio: float = 0.35,
) -> np.ndarray | None:
    """Crops and pads the bounding rectangle of an eye from facial landmarks.

    Args:
        frame: BGR video frame of shape (H, W, 3).
        landmarks: Facial landmarks of shape (N, 2) or (N, 3).
        eye_indices: Sequence of landmark indices around the eye.
        target_size: Desired output dimensions (height, width).
        margin_ratio: Fractional padding around the minimum enclosing bounding box.

    Returns:
        Processed RGB eye patch normalized to target_size, or None on failure.
    """
    height, width = frame.shape[:2]
    pts = []
    for idx in eye_indices:
        pt = landmarks[idx]
        x = float(pt[0] * width) if pt[0] <= 1.0 else float(pt[0])
        y = float(pt[1] * height) if pt[1] <= 1.0 else float(pt[1])
        pts.append([x, y])

    pts_arr = np.array(pts, dtype=np.float32)
    min_x, min_y = np.min(pts_arr, axis=0)
    max_x, max_y = np.max(pts_arr, axis=0)

    box_w = max_x - min_x
    box_h = max_y - min_y

    pad_x = box_w * margin_ratio
    pad_y = box_h * margin_ratio

    x1 = max(int(min_x - pad_x), 0)
    y1 = max(int(min_y - pad_y), 0)
    x2 = min(int(max_x + pad_x), width)
    y2 = min(int(max_y + pad_y), height)

    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    crop_resized = cv2.resize(crop, target_size, interpolation=cv2.INTER_LINEAR)
    crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
    return crop_rgb


def preprocess_crops(crops: Sequence[np.ndarray]) -> torch.Tensor:
    """Converts a sequence of RGB crops to a normalized PyTorch NCHW tensor."""
    tensors = []
    for crop in crops:
        img_f = crop.astype(np.float32) / 255.0
        norm_img = (img_f - IMAGENET_MEAN) / IMAGENET_STD
        # Transpose from HWC to CHW
        tensors.append(torch.from_numpy(norm_img.transpose((2, 0, 1))))
    return torch.stack(tensors, dim=0)


class EyeStatePredictor:
    """Performs batched MobileNetV2 inference on eye regions."""

    def __init__(
        self,
        model_path: str | Path = "models/mobilenetv2_eye_state.pt",
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.is_available: bool = False
        self.model = None

        path_obj = Path(model_path)
        if path_obj.is_file():
            try:
                model_inst = build_model(num_classes=2, pretrained=False)
                self.model = load_checkpoint(model_inst, path_obj, device=device)
                self.model.eval()
                self.is_available = True
                logger.info("Successfully loaded MobileNetV2 from %s", path_obj)
            except (torch.jit.Error, RuntimeError, OSError, ValueError) as e:
                logger.warning("Failed to load MobileNetV2 from %s: %s", path_obj, e)
        else:
            logger.info(
                "MobileNetV2 weights not found at %s. Running in rule-based fallback.",
                path_obj,
            )

    def predict(
        self,
        frame: np.ndarray,
        landmarks: np.ndarray,
    ) -> float | None:
        """Predicts average eye closure probability for the driver.

        Args:
            frame: Raw BGR input frame.
            landmarks: Facial landmarks array.

        Returns:
            Probability in [0.0, 1.0] that eyes are closed, or None if model unavailable.
        """
        if not self.is_available or self.model is None:
            return None

        crop_left = crop_eye_region(frame, landmarks, LEFT_EYE_INDICES)
        crop_right = crop_eye_region(frame, landmarks, RIGHT_EYE_INDICES)

        valid_crops = [c for c in (crop_left, crop_right) if c is not None]
        if not valid_crops:
            return None

        batch_tensor = preprocess_crops(valid_crops).to(self.device)

        with torch.inference_mode():
            logits = self.model(batch_tensor)
            probs = torch.softmax(logits, dim=1)  # Class 0: open, Class 1: closed
            closed_probs = probs[:, 1].cpu().numpy()

        # Return mean closed probability across detected eyes
        return float(np.mean(closed_probs))
