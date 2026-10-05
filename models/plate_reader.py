"""
models/plate_reader.py

Pipeline:
  vehicle crop
    → YOLO plate model   → plate crop (in memory only, NOT saved)
    → upscale + sharpen
    → fast-plate-ocr     → reads the plate text
    → KZ format cleanup  → final plate number

fast-plate-ocr is a CRNN/transformer OCR built specifically for license
plates. Much better than EasyOCR on plate-style text.

If fast-plate-ocr is not installed or fails to load, it automatically
falls back to EasyOCR so the pipeline never crashes.
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from config import PLATE_MODEL, PLATE_CONF

logger = logging.getLogger(__name__)

# Which fast-plate-ocr model to use. 's' = small (more accurate),
# 'xs' = extra-small (faster). v2 models also predict region.
FAST_OCR_MODEL = "cct-s-v2-global-model"

OCR_TARGET_W = 1280
OCR_TARGET_H = 384

# Plates blurrier than this are skipped (no OCR, no save).
# Higher = stricter (only very sharp plates). Lower = allow blurrier.
# Typical range: 50 (lenient) to 200 (strict). Start at 100.
BLUR_THRESHOLD = 100.0

_KZ_RE        = re.compile(r"\d{3}[A-Z]{2,3}\d{2}")
_LOOSE_RE     = re.compile(r"[A-Z0-9]{4,10}")
_CYRILLIC_MAP = str.maketrans("АВЕКМНОРСТУХ", "ABEKMHOPCTYX")


@dataclass
class PlateResult:
    plate_number: str
    confidence:   float
    plate_bbox:   tuple | None
    plate_crop:   np.ndarray | None = None   # the cropped plate image, for saving
    raw_texts:    list = field(default_factory=list)


class PlateReader:

    def __init__(
        self,
        model_path: Path = PLATE_MODEL,
        conf: float       = PLATE_CONF,
    ):
        logger.info("Loading plate model from %s", model_path)
        self.model = YOLO(str(model_path))
        self.conf  = conf

        # ── Try to load fast-plate-ocr, fall back to EasyOCR ──────────────────
        self.fast_ocr = None
        self.easy_ocr = None

        try:
            from fast_plate_ocr import LicensePlateRecognizer
            logger.info("Loading fast-plate-ocr model: %s", FAST_OCR_MODEL)
            self.fast_ocr = LicensePlateRecognizer(FAST_OCR_MODEL)
            logger.info("fast-plate-ocr ready")
        except Exception as exc:
            logger.warning("fast-plate-ocr unavailable (%s) — falling back to EasyOCR", exc)
            import easyocr
            self.easy_ocr = easyocr.Reader(["en"], gpu=False)

    # ── Public ─────────────────────────────────────────────────────────────────

    def read(self, vehicle_crop: np.ndarray) -> PlateResult:
        plate_crop, plate_bbox = self._find_plate(vehicle_crop)
        if plate_crop is None:
            return PlateResult("", 0.0, None)

        # upscale + sharpen the plate crop
        prepared = self._prepare(plate_crop)

        # OCR
        if self.fast_ocr is not None:
            plate, conf, raw = self._ocr_fast(prepared)
        else:
            plate, conf, raw = self._ocr_easy(prepared)

        logger.info("Plate result: '%s' (conf=%.2f) raw=%s", plate, conf, raw)
        # Return the prepared (upscaled+sharpened) plate crop so it can be saved
        return PlateResult(plate, conf, plate_bbox, plate_crop=prepared, raw_texts=raw)

    # ── Plate detection ────────────────────────────────────────────────────────

    def _find_plate(self, vehicle_crop):
        results   = self.model(vehicle_crop, conf=self.conf, verbose=False)
        best      = None
        best_conf = 0.0
        for r in results:
            for box in r.boxes:
                c = float(box.conf[0])
                if c > best_conf:
                    best_conf = c
                    best = box

        if best is None or best_conf < 0.5:
            return None, None

        x1, y1, x2, y2 = map(int, best.xyxy[0].tolist())
        x1 = max(x1, 0); y1 = max(y1, 0)
        x2 = min(x2, vehicle_crop.shape[1])
        y2 = min(y2, vehicle_crop.shape[0])
        crop = vehicle_crop[y1:y2, x1:x2]

        if crop.size == 0:
            return None, None

        # ── Blur check ────────────────────────────────────────────────────────
        # Skip plates that are too blurry to read. Laplacian variance measures
        # sharpness — low value = blurry. Threshold tuned for plate crops.
        # Raise BLUR_THRESHOLD to be stricter, lower it to allow blurrier plates.
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        if sharpness < BLUR_THRESHOLD:
            logger.debug("Plate too blurry (sharpness=%.1f < %.1f) — skipped",
                         sharpness, BLUR_THRESHOLD)
            return None, None

        return crop, (x1, y1, x2, y2)

    # ── Prepare (upscale + sharpen) ───────────────────────────────────────────

    def _prepare(self, img: np.ndarray) -> np.ndarray:
        # upscale to fixed size
        img = cv2.resize(img, (OCR_TARGET_W, OCR_TARGET_H), interpolation=cv2.INTER_CUBIC)
        # sharpen
        smoothed  = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
        blurred   = cv2.GaussianBlur(smoothed, (0, 0), sigmaX=3)
        sharpened = cv2.addWeighted(smoothed, 1.5, blurred, -0.5, 0)
        return sharpened

    # ── OCR: fast-plate-ocr ───────────────────────────────────────────────────

    def _ocr_fast(self, img: np.ndarray):
        try:
            # fast-plate-ocr accepts a numpy array directly
            result = self.fast_ocr.run(img)

            # DEBUG: log the raw structure so we can see what fast-plate-ocr returns
            logger.debug("fast-ocr raw type=%s value=%r", type(result), result)

            # The result can be:
            #   - a list of strings
            #   - a list of PlatePrediction objects (newer versions)
            #   - a single string or object
            text = ""
            if isinstance(result, (list, tuple)) and result:
                first = result[0]
            else:
                first = result

            # Extract the plate string from whatever type we got
            if isinstance(first, str):
                text = first
            elif hasattr(first, "plate"):          # PlatePrediction.plate
                text = first.plate
            elif hasattr(first, "text"):           # some versions use .text
                text = first.text
            else:
                text = str(first)

            cleaned = self._clean(text)
            raw  = [("fast-ocr", text)]
            conf = 1.0 if cleaned else 0.0
            return cleaned, conf, raw

        except Exception as exc:
            logger.error("fast-plate-ocr error: %s", exc)
            return "", 0.0, []

    # ── OCR: EasyOCR fallback ─────────────────────────────────────────────────

    def _ocr_easy(self, img: np.ndarray):
        try:
            hits = self.easy_ocr.readtext(img)
            best_text = ""
            best_conf = 0.0
            raw = []
            for (_, text, conf) in hits:
                raw.append((text, round(conf, 3)))
                cleaned = self._clean(text)
                if cleaned and conf > best_conf:
                    best_text = cleaned
                    best_conf = conf
            return best_text, best_conf, raw
        except Exception as exc:
            logger.error("EasyOCR error: %s", exc)
            return "", 0.0, []

    # ── Cleanup ────────────────────────────────────────────────────────────────

    @staticmethod
    def _clean(text: str) -> str:
        text = text.upper().translate(_CYRILLIC_MAP)
        text = re.sub(r"[^A-Z0-9]", "", text)
        m = _KZ_RE.search(text)
        if m:
            return m.group(0)
        m = _LOOSE_RE.search(text)
        return m.group(0) if m else ""
