"""MediaPipe Face Mesh landmark extraction wrapper.

Tracks a single driver face with refined iris contours (468 mesh + 10 iris points).
Provides normalized 3D landmark arrays with zero network access.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class FaceLandmarkDetector:
    """Wrapper for MediaPipe Face Mesh model with iris refinement."""

    def __init__(
        self,
        max_num_faces: int = 1,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        import mediapipe as mp

        self._mp_face_mesh = mp.solutions.face_mesh
        self._face_mesh = self._mp_face_mesh.FaceMesh(
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, frame: np.ndarray) -> np.ndarray | None:
        """Extracts 3D facial landmarks from a BGR video frame.

        Args:
            frame: Raw BGR video frame from OpenCV (H, W, 3).

        Returns:
            NumPy array of shape (478, 3) containing normalized [x, y, z] coordinates,
            or None if no face is detected.
        """
        if frame is None or frame.size == 0:
            return None

        # Convert BGR to RGB for MediaPipe inference
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self._face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return None

        # Isolate the foreground driver face (first tracked face)
        driver_face = results.multi_face_landmarks[0]
        pts = np.empty((len(driver_face.landmark), 3), dtype=np.float64)
        for i, lm in enumerate(driver_face.landmark):
            pts[i, 0] = lm.x
            pts[i, 1] = lm.y
            pts[i, 2] = lm.z

        return pts

    def close(self) -> None:
        """Releases the MediaPipe Face Mesh graph resources."""
        if hasattr(self, "_face_mesh") and self._face_mesh:
            self._face_mesh.close()
