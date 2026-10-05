"""
pipeline/event_processor.py

Pipeline per frame:
  full frame (middle ROI only)
    → vehicle model   → vehicle crop
    → plate model     → plate crop (in memory only, NOT saved)
        → upscale to 640x192
        → sharpen
        → OCR
    → save vehicle crop to disk
    → write to DB
    → push to Google Sheets: timestamp + plate + link to vehicle photo
"""

import logging
import os
import time
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from config import (
    SNAPSHOT_DIR,
    ROI_ENABLED, ROI_X1, ROI_X2, ROI_Y1, ROI_Y2,
    DUPLICATE_WINDOW_SECONDS,
    REQUIRE_PLATE, SOURCE_FPS,
)
from models.vehicle_detector import VehicleDetector
from models.plate_reader     import PlateReader
from db.database             import Database, VehicleEvent
from db.sheets_reporter      import SheetsReporter
from notifications.telegram  import TelegramNotifier

logger = logging.getLogger(__name__)


class EventProcessor:

    def __init__(
        self,
        vehicle_detector: VehicleDetector,
        plate_reader:     PlateReader,
        database:         Database,
        snapshot_dir:     Path = SNAPSHOT_DIR,
    ):
        self.vehicle_detector = vehicle_detector
        self.plate_reader     = plate_reader
        self.db               = database
        self.snapshot_dir     = Path(snapshot_dir).resolve()
        self.sheets           = SheetsReporter()
        self.telegram         = TelegramNotifier()

        self._ensure_snapshot_dir()
        self._seen: dict[str, float] = {}

    def _ensure_snapshot_dir(self):
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        test = self.snapshot_dir / "_write_test.tmp"
        try:
            test.write_text("ok")
            test.unlink()
            logger.info("Snapshot dir ready: %s", self.snapshot_dir)
        except Exception as exc:
            logger.critical("Cannot write to snapshot dir: %s", exc)

    # ── Public ─────────────────────────────────────────────────────────────────

    def process(self, frame_index: int, frame: np.ndarray) -> list:
        h, w     = frame.shape[:2]
        vehicles = self.vehicle_detector.detect(frame)

        if not vehicles:
            logger.debug("Frame %d: no vehicles", frame_index)
            return []

        logger.info("Frame %d: %d vehicle(s) detected", frame_index, len(vehicles))
        events = []

        for v in vehicles:
            # ROI — only process vehicles in the middle zone of the frame
            if ROI_ENABLED and not self._in_roi(v.bbox, w, h):
                logger.debug("Vehicle #%d outside ROI — skipped", v.vehicle_id)
                continue
            event = self._process_vehicle(frame, v, frame_index)
            if event:
                events.append(event)

        return events

    # ── Preview drawing ─────────────────────────────────────────────────────────

    def annotate(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw detection boxes, ROI zone, and labels on a copy of the frame.
        Used for the live preview window.
        """
        out  = frame.copy()
        h, w = out.shape[:2]

        # Draw ROI rectangle (orange)
        if ROI_ENABLED:
            rx1, ry1 = int(ROI_X1 * w), int(ROI_Y1 * h)
            rx2, ry2 = int(ROI_X2 * w), int(ROI_Y2 * h)
            cv2.rectangle(out, (rx1, ry1), (rx2, ry2), (0, 165, 255), 2)
            cv2.putText(out, "ROI", (rx1 + 5, ry1 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        # Detect and draw vehicles
        for v in self.vehicle_detector.detect(frame):
            x1, y1, x2, y2 = v.bbox
            in_roi = (not ROI_ENABLED) or self._in_roi(v.bbox, w, h)
            colour = (0, 255, 0) if in_roi else (128, 128, 128)
            cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
            label = f"{v.class_name} {v.confidence:.2f}"
            cv2.putText(out, label, (x1, max(y1 - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)

        return out

    # ── ROI ────────────────────────────────────────────────────────────────────

    @staticmethod
    def _in_roi(bbox, fw, fh) -> bool:
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2 / fw
        cy = (y1 + y2) / 2 / fh
        return ROI_X1 <= cx <= ROI_X2 and ROI_Y1 <= cy <= ROI_Y2

    # ── Duplicate suppression ──────────────────────────────────────────────────

    def _dedup_key(self, plate, bbox):
        if plate:
            return f"plate:{plate}"
        x1, y1, x2, y2 = bbox
        return f"pos:{round((x1+x2)/2/50)}:{round((y1+y2)/2/50)}"

    def _is_duplicate(self, key, frame_index) -> bool:
        # Session-wide mode: a plate is logged only ONCE per run, ever.
        if DUPLICATE_WINDOW_SECONDS <= 0:
            return key in self._seen

        # Windowed mode: based on FRAME distance, not wall-clock time.
        # This works correctly on video files regardless of playback speed.
        window_frames = DUPLICATE_WINDOW_SECONDS * SOURCE_FPS
        last = self._seen.get(key)
        if last is not None and (frame_index - last) < window_frames:
            logger.info("Duplicate suppressed: '%s'", key)
            return True
        return False

    def _mark_seen(self, key, frame_index):
        self._seen[key] = frame_index

    # ── Core pipeline ──────────────────────────────────────────────────────────

    def _process_vehicle(self, frame, detection, frame_index):
        vehicle_crop = detection.crop

        # 1. Plate model → upscale → OCR (plate crop stays in memory only)
        plate_result = self.plate_reader.read(vehicle_crop)
        plate        = plate_result.plate_number or ""

        logger.info("Vehicle #%d | plate='%s' (conf=%.2f) | raw: %s",
                    detection.vehicle_id, plate,
                    plate_result.confidence, plate_result.raw_texts)

        # 1b. If configured, skip vehicles with no readable plate entirely.
        #     No DB record, no image upload, no sheet row.
        if REQUIRE_PLATE and not plate:
            logger.debug("Vehicle #%d has no plate — skipped (REQUIRE_PLATE=True)",
                         detection.vehicle_id)
            return None

        # 2. Duplicate check
        key = self._dedup_key(plate, detection.bbox)
        if self._is_duplicate(key, frame_index):
            return None

        # 3. Save vehicle crop image (this one goes to Sheets)
        ts       = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        uid      = uuid.uuid4().hex[:6]
        filename = f"{ts}_v{detection.vehicle_id}_{uid}.jpg"
        img_path = self._save_image(vehicle_crop, filename)

        # 3b. Save the cropped PLATE image locally (data/plate_crops/)
        #     for inspection — NOT uploaded to Google Sheets.
        #     Filename includes the OCR guess so you can compare with reality.
        if plate_result.plate_crop is not None:
            plate_dir = self.snapshot_dir.parent / "plate_crops"
            plate_dir.mkdir(parents=True, exist_ok=True)
            label = plate or "UNREAD"
            plate_filename = f"{ts}_v{detection.vehicle_id}_{label}_{uid}.jpg"
            try:
                cv2.imwrite(str(plate_dir / plate_filename), plate_result.plate_crop)
            except Exception as exc:
                logger.warning("Could not save plate crop: %s", exc)

        # 4. Write to local DB
        event = VehicleEvent(
            vehicle_id   = detection.vehicle_id,
            timestamp    = datetime.utcnow(),
            plate_number = plate or None,
            vehicle_type = detection.class_name,
            image_path   = str(img_path) if img_path else None,
            event_type   = "unknown",
        )
        self.db.save_event(event)
        self._mark_seen(key, frame_index)

        # 5. Push to Google Sheets: timestamp + plate + vehicle photo link
        self.sheets.append_row(
            timestamp  = event.timestamp,
            plate      = plate or "UNREAD",
            image_path = img_path,
        )

        # 6. Telegram alert (new vehicle). Checks its own toggles internally.
        self.telegram.notify(
            plate        = plate,
            vehicle_type = detection.class_name,
            timestamp    = event.timestamp,
            image_path   = img_path,
        )

        logger.info("Saved — plate='%s' image=%s", plate, img_path)
        return event

    # ── Image saving ───────────────────────────────────────────────────────────

    def _save_image(self, img: np.ndarray, filename: str):
        """Save vehicle crop. Returns path on success, None on failure."""
        try:
            path    = self.snapshot_dir / filename
            success = cv2.imwrite(str(path), img)
            if success:
                return path
            logger.error("cv2.imwrite failed for %s", path)
            return None
        except Exception as exc:
            logger.exception("Error saving image: %s", exc)
            return None
