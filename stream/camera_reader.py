"""
stream/camera_reader.py — Reads frames from RTSP / USB / video file.

Correctly handles two different sources:
  - Video file (.mp4, .avi, etc): stops cleanly when the file ends
  - Live stream (RTSP, USB camera): reconnects automatically on drop
"""

import logging
import time
from pathlib import Path

import cv2

from config import CAMERA_SOURCE, FRAME_SKIP

logger = logging.getLogger(__name__)


def _is_video_file(source) -> bool:
    """Return True if source is a file path (not a live stream or device)."""
    if isinstance(source, int):
        return False   # USB camera index like 0, 1, 2
    path = str(source)
    # RTSP / HTTP streams
    if path.lower().startswith(("rtsp://", "http://", "https://")):
        return False
    # Everything else we treat as a file
    return True


class CameraReader:
    """
    Wraps OpenCV VideoCapture.

    Video file  → reads until end then stops cleanly.
    Live stream → reconnects automatically if connection drops.

    Args:
        source:          RTSP URL, device index (int), or video file path.
        frame_skip:      Yield 1 frame every N frames.
        reconnect_delay: Seconds to wait before reconnecting a live stream.
    """

    def __init__(
        self,
        source          = CAMERA_SOURCE,
        frame_skip: int = FRAME_SKIP,
        reconnect_delay: float = 5.0,
    ):
        self.source          = source
        self.frame_skip      = frame_skip
        self.reconnect_delay = reconnect_delay
        self._cap            = None
        self._is_file        = _is_video_file(source)

        if self._is_file:
            logger.info("Source is a VIDEO FILE — will stop when file ends")
        else:
            logger.info("Source is a LIVE STREAM — will reconnect on drop")

    # ── Public ─────────────────────────────────────────────────────────────────

    def read_frames(self):
        """
        Generator — yields (frame_index, np.ndarray) for every Nth frame.

        Stops automatically when:
          - A video file finishes playing
          - main.py sets _running = False (Ctrl-C / SIGTERM)
        """
        frame_index = 0
        self._open()

        while True:
            if self._cap is None or not self._cap.isOpened():
                if self._is_file:
                    # File failed to open — nothing to retry, stop cleanly
                    logger.error("Could not open video file — stopping pipeline")
                    return
                # Live stream — keep retrying
                logger.warning("Stream not open — retrying in %.1fs …",
                               self.reconnect_delay)
                time.sleep(self.reconnect_delay)
                self._open()
                continue

            ret, frame = self._cap.read()

            if not ret:
                if self._is_file:
                    # Video file finished — this is normal, just stop
                    logger.info(
                        "Video file finished after %d frames — pipeline complete",
                        frame_index,
                    )
                    return   # exits the generator cleanly

                else:
                    # Live stream dropped — reconnect
                    logger.warning(
                        "Live stream lost — reconnecting in %.1fs …",
                        self.reconnect_delay,
                    )
                    self._release()
                    time.sleep(self.reconnect_delay)
                    continue

            frame_index += 1

            if frame_index % self.frame_skip == 0:
                logger.debug("Yielding frame %d", frame_index)
                yield frame_index, frame

    def release(self):
        self._release()

    # ── Private ────────────────────────────────────────────────────────────────

    def _open(self):
        logger.info("Opening source: %s", self.source)
        self._cap = cv2.VideoCapture(str(self.source), cv2.CAP_FFMPEG)
        if not self._cap.isOpened():
            logger.error("Could not open source: %s", self.source)
            self._release()

    def _release(self):
        if self._cap:
            self._cap.release()
            self._cap = None
