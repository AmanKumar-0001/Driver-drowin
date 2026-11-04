"""Robust webcam video capture wrapper with dropped-frame and disconnection handling.

Provides clean error propagation, frame rate throttling, and an automated
synthetic frame generator for headless testing environments.
"""

from __future__ import annotations

import logging
import time
from typing import Self

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CameraReader:
    """Encapsulates OpenCV video capture with fault tolerance."""

    def __init__(
        self,
        camera_index: int = 0,
        frame_width: int = 640,
        frame_height: int = 480,
        target_fps: int = 30,
        max_dropped_frames: int = 5,
    ) -> None:
        self.camera_index = camera_index
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.target_fps = target_fps
        self.max_dropped_frames = max_dropped_frames

        self._cap: cv2.VideoCapture | None = None
        self._consecutive_drops: int = 0
        self._is_opened: bool = False
        self._last_frame_time: float = 0.0

    def open(self) -> bool:
        """Initializes the physical webcam device."""
        logger.info("Opening webcam at index %d...", self.camera_index)
        # Using cv2.CAP_DSHOW on Windows for fast non-blocking startup
        self._cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not self._cap or not self._cap.isOpened():
            # Fall back to default backend
            self._cap = cv2.VideoCapture(self.camera_index)

        if not self._cap or not self._cap.isOpened():
            logger.error("Failed to connect to camera device at index %d", self.camera_index)
            self._is_opened = False
            return False

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)

        self._is_opened = True
        self._consecutive_drops = 0
        self._last_frame_time = time.time()
        logger.info("Camera opened successfully (%dx%d)", self.frame_width, self.frame_height)
        return True

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Acquires a single frame from the camera stream.

        Returns:
            A tuple of (success, frame). If dropped, frame is None.
        """
        if not self._is_opened or self._cap is None:
            return False, None

        ret, frame = self._cap.read()
        curr_t = time.time()

        if not ret or frame is None or frame.size == 0:
            self._consecutive_drops += 1
            if self._consecutive_drops >= self.max_dropped_frames:
                logger.warning(
                    "Exceeded maximum consecutive dropped frames (%d)",
                    self._consecutive_drops,
                )
            return False, None

        # Reset drop counter on healthy frame
        self._consecutive_drops = 0
        self._last_frame_time = curr_t
        return True, frame

    def release(self) -> None:
        """Releases the camera handle cleanly."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._is_opened = False
        logger.info("Camera released.")

    def __enter__(self) -> Self:
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()


class SyntheticCameraReader:
    """Mock video source emitting synthetic driver face frames for testing."""

    def __init__(
        self,
        frame_width: int = 640,
        frame_height: int = 480,
        target_fps: int = 30,
    ) -> None:
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.target_fps = target_fps
        self._is_opened: bool = True
        self._frame_idx: int = 0

    def open(self) -> bool:
        self._is_opened = True
        return True

    def read(self) -> tuple[bool, np.ndarray | None]:
        if not self._is_opened:
            return False, None

        # Create a neutral synthetic face frame
        frame = np.ones((self.frame_height, self.frame_width, 3), dtype=np.uint8) * 128
        # Draw a synthetic face oval in center
        center = (self.frame_width // 2, self.frame_height // 2)
        cv2.ellipse(frame, center, (120, 160), 0, 0, 360, (200, 210, 230), -1)

        # Draw eyes
        eye_y = center[1] - 30
        cv2.circle(frame, (center[0] - 45, eye_y), 15, (60, 40, 20), -1)
        cv2.circle(frame, (center[0] + 45, eye_y), 15, (60, 40, 20), -1)

        # Draw mouth
        cv2.line(
            frame,
            (center[0] - 30, center[1] + 50),
            (center[0] + 30, center[1] + 50),
            (60, 60, 120),
            4,
        )

        self._frame_idx += 1
        time.sleep(1.0 / self.target_fps)
        return True, frame

    def release(self) -> None:
        self._is_opened = False
