"""Pure mathematical facial feature extraction routines.

Implements Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), and blink duration
classification with zero OS or hardware dependencies.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

# MediaPipe Face Mesh standard landmark indices (6-point contours)
# Left eye: [lateral_corner, upper_1, upper_2, medial_corner, lower_2, lower_1]
LEFT_EYE_INDICES = (33, 160, 158, 133, 153, 144)

# Right eye: [lateral_corner, upper_1, upper_2, medial_corner, lower_2, lower_1]
RIGHT_EYE_INDICES = (362, 385, 387, 263, 373, 380)

# Mouth contour landmarks: [left_corner, right_corner, top_1, top_2, top_3, bot_1, bot_2, bot_3]
MOUTH_LEFT_CORNER = 61
MOUTH_RIGHT_CORNER = 291
MOUTH_TOP_INDICES = (81, 13, 311)
MOUTH_BOTTOM_INDICES = (178, 14, 402)


def compute_aspect_ratio_6pt(
    p1: Sequence[float],
    p2: Sequence[float],
    p3: Sequence[float],
    p4: Sequence[float],
    p5: Sequence[float],
    p6: Sequence[float],
    eps: float = 1e-6,
) -> float:
    """Computes the 6-point Euclidean aspect ratio (Soukupova & Cech).

    Formula: (|p2 - p6| + |p3 - p5|) / (2.0 * |p1 - p4|)

    Args:
        p1: Lateral horizontal corner coordinate [x, y] or [x, y, z].
        p2: Upper vertical point 1.
        p3: Upper vertical point 2.
        p4: Medial horizontal corner coordinate.
        p5: Lower vertical point 2.
        p6: Lower vertical point 1.
        eps: Small epsilon constant to prevent division by zero.

    Returns:
        Aspect ratio as a float.
    """
    v_p1 = np.asarray(p1, dtype=np.float64)[:2]
    v_p2 = np.asarray(p2, dtype=np.float64)[:2]
    v_p3 = np.asarray(p3, dtype=np.float64)[:2]
    v_p4 = np.asarray(p4, dtype=np.float64)[:2]
    v_p5 = np.asarray(p5, dtype=np.float64)[:2]
    v_p6 = np.asarray(p6, dtype=np.float64)[:2]

    # Vertical Euclidean distances
    dist_v1 = float(np.linalg.norm(v_p2 - v_p6))
    dist_v2 = float(np.linalg.norm(v_p3 - v_p5))

    # Horizontal Euclidean distance
    dist_h = float(np.linalg.norm(v_p1 - v_p4))

    return (dist_v1 + dist_v2) / (2.0 * max(dist_h, eps))


def calculate_ear(
    landmarks: np.ndarray,
    eye_indices: Sequence[int],
) -> float:
    """Calculates Eye Aspect Ratio (EAR) for a single eye from landmark array.

    Args:
        landmarks: Array of facial landmarks of shape (N, 2) or (N, 3).
        eye_indices: Sequence of 6 landmark indices defining the eye contour.

    Returns:
        Calculated EAR value.
    """
    if len(eye_indices) != 6:
        raise ValueError(f"eye_indices must have exactly 6 elements, got {len(eye_indices)}")

    pts = [landmarks[idx] for idx in eye_indices]
    return compute_aspect_ratio_6pt(pts[0], pts[1], pts[2], pts[3], pts[4], pts[5])


def calculate_average_ear(landmarks: np.ndarray) -> tuple[float, float, float]:
    """Calculates left EAR, right EAR, and their arithmetic mean.

    Args:
        landmarks: Facial landmark coordinate array of shape (N, 2) or (N, 3).

    Returns:
        A tuple of (left_ear, right_ear, avg_ear).
    """
    left_ear = calculate_ear(landmarks, LEFT_EYE_INDICES)
    right_ear = calculate_ear(landmarks, RIGHT_EYE_INDICES)
    avg_ear = (left_ear + right_ear) / 2.0
    return left_ear, right_ear, avg_ear


def calculate_mar(landmarks: np.ndarray, eps: float = 1e-6) -> float:
    """Calculates Mouth Aspect Ratio (MAR) for yawn detection.

    Formula: (|top1 - bot1| + |top2 - bot2| + |top3 - bot3|) / (2.0 * |left - right|)

    Args:
        landmarks: Facial landmark coordinate array of shape (N, 2) or (N, 3).
        eps: Epsilon to prevent division by zero.

    Returns:
        Calculated MAR value.
    """
    p_left = np.asarray(landmarks[MOUTH_LEFT_CORNER], dtype=np.float64)[:2]
    p_right = np.asarray(landmarks[MOUTH_RIGHT_CORNER], dtype=np.float64)[:2]

    horiz_dist = float(np.linalg.norm(p_left - p_right))

    vert_dists = [
        float(
            np.linalg.norm(
                np.asarray(landmarks[t], dtype=np.float64)[:2]
                - np.asarray(landmarks[b], dtype=np.float64)[:2]
            )
        )
        for t, b in zip(MOUTH_TOP_INDICES, MOUTH_BOTTOM_INDICES)
    ]

    return sum(vert_dists) / (2.0 * max(horiz_dist, eps))


def classify_closure_duration(
    closure_duration_sec: float,
    blink_max_sec: float = 0.40,
    microsleep_min_sec: float = 1.00,
) -> str:
    """Classifies an eye closure duration into a physiological category.

    Args:
        closure_duration_sec: Measured duration of continuous eye closure in seconds.
        blink_max_sec: Upper threshold for normal physiological blink.
        microsleep_min_sec: Lower threshold for dangerous microsleep event.

    Returns:
        One of 'NORMAL_BLINK', 'PROLONGED_CLOSURE', 'MICROSLEEP', or 'OPEN'.
    """
    if closure_duration_sec <= 0.0:
        return "OPEN"
    if closure_duration_sec <= blink_max_sec:
        return "NORMAL_BLINK"
    if closure_duration_sec >= microsleep_min_sec:
        return "MICROSLEEP"
    return "PROLONGED_CLOSURE"
