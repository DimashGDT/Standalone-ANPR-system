"""
pipeline/frame_selector.py — Chooses the sharpest / most useful frame.

Currently: returns the frame as-is (single-frame pipeline).
Later:     collect a short burst, score by Laplacian variance (sharpness),
           return the best one. Swap out this class without touching anything else.
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class FrameSelector:
    """
    Accepts frames and decides which one to forward for detection.

    Simple mode: pass-through (every frame is forwarded).
    Burst mode:  collect `burst_size` frames, return the sharpest.
    """

    def __init__(self, burst_size: int = 1):
        """
        Args:
            burst_size: 1 = pass-through, N>1 = collect N then pick best.
        """
        self.burst_size = burst_size
        self._buffer: list[tuple[int, np.ndarray]] = []

    def add(self, frame_index: int, frame: np.ndarray):
        """
        Add a frame to the selector.

        Returns:
            (frame_index, frame) if a frame is ready to be processed, else None.
        """
        if self.burst_size == 1:
            return frame_index, frame

        self._buffer.append((frame_index, frame))

        if len(self._buffer) >= self.burst_size:
            best = self._pick_sharpest()
            self._buffer.clear()
            return best

        return None

    # ── Private ────────────────────────────────────────────────────────────────

    def _pick_sharpest(self):
        """Return the (index, frame) with highest Laplacian variance."""
        best_score = -1
        best       = self._buffer[0]

        for idx, frame in self._buffer:
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            score = cv2.Laplacian(gray, cv2.CV_64F).var()
            if score > best_score:
                best_score = score
                best       = (idx, frame)

        logger.debug("Frame selector: chose index %d (score=%.1f)", best[0], best_score)
        return best
