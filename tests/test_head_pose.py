"""Unit tests for 3D head pose estimation and visual distraction detection."""

import numpy as np
import pytest

from src.head_pose import (
    GENERIC_3D_FACE_MODEL,
    HEAD_POSE_LANDMARK_INDICES,
    estimate_head_pose,
    is_head_distracted,
)


def test_head_pose_frontal_neutral() -> None:
    """Verifies that an idealized frontal face projects to approximately zero Euler angles."""
    # Create landmarks centered exactly in a 640x480 frame
    landmarks = np.zeros((478, 3), dtype=np.float64)
    w, h = 640.0, 480.0
    cx, cy = w / 2.0, h / 2.0
    scale = 0.5

    # Project the 6 3D model vertices to 2D
    for i, idx in enumerate(HEAD_POSE_LANDMARK_INDICES):
        x3d, y3d, _ = GENERIC_3D_FACE_MODEL[i]
        landmarks[idx, 0] = (cx + x3d * scale) / w
        landmarks[idx, 1] = (cy - y3d * scale) / h

    yaw, pitch, roll = estimate_head_pose(landmarks, image_shape=(480, 640))
    # Frontal projection should be close to 0 degrees within pinhole approximation
    assert abs(yaw) < 10.0
    assert abs(pitch) < 10.0
    assert abs(roll) < 10.0


@pytest.mark.parametrize(
    "yaw,pitch,expected_distracted",
    [
        (0.0, 0.0, False),
        (15.0, 5.0, False),
        (-25.0, -10.0, False),
        (32.0, 0.0, True),  # |yaw| > 30
        (-35.0, 0.0, True),  # |yaw| > 30
        (0.0, -25.0, True),  # pitch < -20 (looking down)
        (0.0, 25.0, True),  # pitch > 20 (looking up)
    ],
)
def test_is_head_distracted_boundaries(yaw: float, pitch: float, expected_distracted: bool) -> None:
    """Verifies distraction threshold classification across angular boundaries."""
    result = is_head_distracted(
        yaw=yaw,
        pitch=pitch,
        yaw_threshold_deg=30.0,
        pitch_min_deg=-20.0,
        pitch_max_deg=20.0,
    )
    assert result == expected_distracted
