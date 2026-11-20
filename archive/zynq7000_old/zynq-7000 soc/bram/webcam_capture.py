#!/usr/bin/env python3
"""
Webcam Capture Module
AI-Enabled Smart Waste Segregation & Monitoring System

Manages USB webcam connected to NVIDIA Jetson Nano.
Provides thread-safe frame capture with configurable resolution and FPS.
"""

import cv2
import sys
import time
import logging
import threading
from typing import Optional, Tuple
import numpy as np

from config import CameraConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


class WebcamCapture:
    """
    Thread-safe USB webcam capture for Jetson Nano.
    Runs capture in a background thread for non-blocking frame access.
    """

    def __init__(self, config: CameraConfig = None):
        self.config = config or DEFAULT_CONFIG.camera
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame: Optional[np.ndarray] = None
        self.frame_id: int = 0
        self.timestamp: float = 0.0
        self.running: bool = False
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

        # Performance tracking
        self._fps_times = []
        self._frames_captured = 0

    def open(self) -> bool:
        """Open the webcam device"""
        try:
            # On Windows, try DirectShow (CAP_DSHOW) first for fast, reliable hardware grab
            if sys.platform.startswith("win"):
                self.cap = cv2.VideoCapture(self.config.device_index, cv2.CAP_DSHOW)

            if not self.cap or not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.config.device_index)

            if not self.cap.isOpened():
                logger.error(f"❌ Cannot open webcam at index {self.config.device_index}")
                return False

            # Set resolution and FPS
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.config.fps)

            # Verify settings
            actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)

            logger.info(f"✅ Webcam opened: {actual_w}x{actual_h}@{actual_fps:.0f}fps")

            # Warm up: skip initial frames (keep to max 3 so window opens immediately)
            warmup = min(self.config.warmup_frames, 3)
            for _ in range(warmup):
                self.cap.read()

            return True

        except Exception as e:
            logger.error(f"❌ Webcam open error: {e}")
            return False

    def start_capture(self) -> bool:
        """Start background capture thread"""
        if not self.cap or not self.cap.isOpened():
            if not self.open():
                return False

        self.running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info("📹 Background capture started")
        return True

    def _capture_loop(self):
        """Background thread: continuously capture frames"""
        while self.running:
            try:
                ret, frame = self.cap.read()
                if ret:
                    with self._lock:
                        self.frame = frame
                        self.frame_id += 1
                        self.timestamp = time.time()
                        self._frames_captured += 1

                    # Track FPS
                    self._fps_times.append(time.time())
                    if len(self._fps_times) > 60:
                        self._fps_times.pop(0)
                else:
                    logger.warning("Frame capture failed")
                    time.sleep(0.01)

            except Exception as e:
                logger.error(f"Capture error: {e}")
                time.sleep(0.1)

    def get_frame(self) -> Tuple[Optional[np.ndarray], int, float]:
        """
        Get the latest frame (thread-safe).

        Returns:
            (frame, frame_id, timestamp) or (None, 0, 0.0) if no frame
        """
        with self._lock:
            if self.frame is not None:
                return self.frame.copy(), self.frame_id, self.timestamp
            return None, 0, 0.0

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a single frame directly (blocking, no thread).

        Returns:
            (success, frame)
        """
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.frame_id += 1
                self.timestamp = time.time()
                self._frames_captured += 1
            return ret, frame
        return False, None

    def get_fps(self) -> float:
        """Calculate current capture FPS"""
        if len(self._fps_times) < 2:
            return 0.0
        elapsed = self._fps_times[-1] - self._fps_times[0]
        return (len(self._fps_times) - 1) / elapsed if elapsed > 0 else 0.0

    def get_resolution(self) -> Tuple[int, int]:
        """Get current capture resolution"""
        if self.cap and self.cap.isOpened():
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return w, h
        return 0, 0

    def stop(self):
        """Stop capture and release resources"""
        self.running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

        if self.cap:
            self.cap.release()
            self.cap = None

        logger.info(f"📹 Webcam stopped — {self._frames_captured} frames captured")

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


# ─── Standalone Test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("\n📹 WEBCAM CAPTURE TEST")
    print("=" * 40)

    cam = WebcamCapture()
    if cam.open():
        print(f"Resolution: {cam.get_resolution()}")
        print("Press 'q' to quit\n")

        while True:
            ret, frame = cam.read_frame()
            if not ret:
                break

            cv2.putText(frame, f"Frame: {cam.frame_id}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("Webcam Test", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cam.stop()
        cv2.destroyAllWindows()
    else:
        print("❌ Could not open webcam")
