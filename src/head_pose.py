"""Head pose estimation using Perspective-n-Point (solvePnP) and generic 3D face model.

Derives driver head Euler angles (yaw, pitch, roll) in degrees to identify
visual distraction (looking away, looking down at phone, looking up).
"""

from __future__ import annotations

import cv2
import numpy as np

# Standard 6-point anthropometric generic 3D face model in millimeters
# Reference: DeMenthon & Davis / POSIT facial model
GENERIC_3D_FACE_MODEL = np.array(
    [
        (0.0, 0.0, 0.0),  # Nose tip
        (0.0, -330.0, -65.0),  # Chin
        (-225.0, 170.0, -135.0),  # Left eye lateral corner
        (225.0, 170.0, -135.0),  # Right eye lateral corner
        (-150.0, -150.0, -125.0),  # Left mouth corner
        (150.0, -150.0, -125.0),  # Right mouth corner
    ],
    dtype=np.float64,
)

# MediaPipe Face Mesh landmark indices corresponding to the 3D model anchors
HEAD_POSE_LANDMARK_INDICES = (1, 152, 33, 263, 61, 291)


def estimate_head_pose(
    landmarks: np.ndarray,
    image_shape: tuple[int, int],
    model_3d_points: np.ndarray = GENERIC_3D_FACE_MODEL,
) -> tuple[float, float, float]:
    """Estimates head yaw, pitch, and roll in degrees using cv2.solvePnP.

    Args:
        landmarks: Facial landmarks of shape (N, 2) or (N, 3), normalized [0, 1] or in pixel coords.
        image_shape: Tuple of (height, width) of the input video frame in pixels.
        model_3d_points: Reference 3D model vertices of shape (6, 3).

    Returns:
        A tuple of (yaw, pitch, roll) angles in degrees.
        - Yaw: positive = turned right, negative = turned left.
        - Pitch: positive = looking up, negative = looking down.
        - Roll: positive = tilting right, negative = tilting left.
    """
    height, width = image_shape[:2]

    # Extract 2D pixel coordinates for the 6 anchor landmarks
    image_points = []
    for idx in HEAD_POSE_LANDMARK_INDICES:
        pt = landmarks[idx]
        # If coordinates are normalized in [0, 1], scale to pixel dimensions
        x = float(pt[0] * width) if pt[0] <= 1.0 else float(pt[0])
        y = float(pt[1] * height) if pt[1] <= 1.0 else float(pt[1])
        image_points.append([x, y])

    image_points_arr = np.array(image_points, dtype=np.float64)

    # Approximate camera intrinsic matrix (standard pinhole camera model)
    focal_length = float(width)
    center = (float(width) / 2.0, float(height) / 2.0)
    camera_matrix = np.array(
        [
            [focal_length, 0.0, center[0]],
            [0.0, focal_length, center[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)  # Assume zero lens distortion

    success, rvec, _tvec = cv2.solvePnP(
        model_3d_points,
        image_points_arr,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if not success:
        return 0.0, 0.0, 0.0

    # Convert rotation vector to 3x3 rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rvec)

    # Decompose rotation matrix into Euler angles using RQDecomp3x3
    angles, _mtxR, _mtxQ, _qxx, _qyy, _qzz = cv2.RQDecomp3x3(rotation_matrix)
    pitch = float(angles[0])
    yaw = float(angles[1])
    roll = float(angles[2])

    # Normalize pitch around 0 degrees for nominal forward gaze
    if pitch < -90.0:
        pitch = pitch + 180.0
    elif pitch > 90.0:
        pitch = pitch - 180.0

    return yaw, pitch, roll


def is_head_distracted(
    yaw: float,
    pitch: float,
    yaw_threshold_deg: float = 30.0,
    pitch_min_deg: float = -20.0,
    pitch_max_deg: float = 20.0,
) -> bool:
    """Checks whether current head pose angles exceed distraction boundaries.

    Args:
        yaw: Head yaw angle in degrees.
        pitch: Head pitch angle in degrees.
        yaw_threshold_deg: Maximum allowable absolute yaw deviation in degrees.
        pitch_min_deg: Minimum pitch angle (looking down threshold).
        pitch_max_deg: Maximum pitch angle (looking up threshold).

    Returns:
        True if head orientation constitutes visual distraction, False otherwise.
    """
    if abs(yaw) > yaw_threshold_deg:
        return True
    return bool(pitch < pitch_min_deg or pitch > pitch_max_deg)
