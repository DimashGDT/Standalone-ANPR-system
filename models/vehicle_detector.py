"""
models/vehicle_detector.py — Detects vehicles in a full camera frame.

- Assigns a simple incremental tracking ID to each unique vehicle
- Skips persons entirely (no photo, no DB record)
- Returns cropped vehicle images + bounding boxes
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from config import VEHICLE_MODEL, VEHICLE_CONF, VEHICLE_CLASSES, PERSON_CLASS_IDS

logger = logging.getLogger(__name__)

# Global counter — gives each vehicle detection a unique ID this session
_vehicle_counter = 0


@dataclass
class VehicleDetection:
    """One detected vehicle."""
    vehicle_id:  int           # Unique ID for this detection
    crop:        np.ndarray    # Cropped BGR image
    bbox:        tuple         # (x1, y1, x2, y2) in original frame coords
    confidence:  float
    class_id:    int
    class_name:  str


class VehicleDetector:
    def __init__(
        self,
        model_path: Path = VEHICLE_MODEL,
        conf: float      = VEHICLE_CONF,
        classes: list    = None,
    ):
        logger.info("Loading vehicle model from %s", model_path)
        self.model        = YOLO(str(model_path))
        self.conf         = conf
        # None = detect all classes, then filter manually so we can log skips
        self.person_ids   = set(PERSON_CLASS_IDS)
        self.vehicle_ids  = set(VEHICLE_CLASSES)

    def detect(self, frame: np.ndarray) -> list[VehicleDetection]:
        global _vehicle_counter

        results    = self.model(frame, conf=self.conf, verbose=False)
        detections = []

        for r in results:
            for box in r.boxes:
                cls_id   = int(box.cls[0])
                cls_name = self.model.names.get(cls_id, str(cls_id))

                # ── Person? Skip entirely — no crop, no record ──────────────
                if cls_id in self.person_ids:
                    logger.debug("Person detected (class %d) — skipped", cls_id)
                    continue

                # ── Not a vehicle class we care about? Skip ─────────────────
                if self.vehicle_ids and cls_id not in self.vehicle_ids:
                    logger.debug("Class '%s' not in VEHICLE_CLASSES — skipped", cls_name)
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                x1, y1 = max(x1, 0), max(y1, 0)
                x2, y2 = min(x2, frame.shape[1]), min(y2, frame.shape[0])

                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                _vehicle_counter += 1
                detections.append(
                    VehicleDetection(
                        vehicle_id  = _vehicle_counter,
                        crop        = crop,
                        bbox        = (x1, y1, x2, y2),
                        confidence  = float(box.conf[0]),
                        class_id    = cls_id,
                        class_name  = cls_name,
                    )
                )

        logger.debug("Vehicle detector: %d detections (persons skipped)", len(detections))
        return detections
