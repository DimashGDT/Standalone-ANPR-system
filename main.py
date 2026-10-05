"""
main.py — Boras plate detection pipeline.

camera → vehicle model → plate model → sharpen → OCR → DB + Excel

Run:   python main.py
Stop:  Ctrl-C
"""

import logging
import signal
import sys
import time

from config import LOG_LEVEL
from stream.camera_reader     import CameraReader
from models.vehicle_detector  import VehicleDetector
from models.plate_reader      import PlateReader
from pipeline.frame_selector  import FrameSelector
from pipeline.event_processor import EventProcessor
from db.database              import Database

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("boras.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

_running = True

def _handle_signal(sig, frame):
    global _running
    logger.info("Shutdown signal received — stopping …")
    _running = False

signal.signal(signal.SIGINT,  _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


def main():
    logger.info("=" * 60)
    logger.info("Boras Starting")
    logger.info("=" * 60)

    # Database
    try:
        db = Database()
        db.init()
        if not db.health_check():
            logger.critical("Database not reachable — aborting")
            sys.exit(1)
    except Exception:
        logger.exception("Failed to initialise database")
        sys.exit(1)

    # Models — vehicle + plate only (no load model)
    try:
        logger.info("Loading vehicle model …")
        vehicle_detector = VehicleDetector()

        logger.info("Loading plate model + OCR …")
        plate_reader = PlateReader()

    except Exception:
        logger.exception("Failed to load models — check weights/ folder")
        sys.exit(1)

    # Pipeline
    frame_selector = FrameSelector(burst_size=1)
    processor = EventProcessor(
        vehicle_detector=vehicle_detector,
        plate_reader=plate_reader,
        database=db,
    )

    # Single camera instance
    camera = CameraReader()
    logger.info("Boras ready — starting camera")

    consecutive_errors = 0

    try:
        for frame_index, frame in camera.read_frames():
            if not _running:
                break

            selected = frame_selector.add(frame_index, frame)
            if selected is None:
                continue

            idx, f = selected
            try:
                processor.process(idx, f)
                consecutive_errors = 0
            except Exception:
                consecutive_errors += 1
                logger.exception("Error on frame %d (%d consecutive)", idx, consecutive_errors)
                if consecutive_errors >= 10:
                    logger.critical("10 consecutive errors — waiting 10s …")
                    time.sleep(10)
                    consecutive_errors = 0

    except Exception:
        logger.exception("Fatal error in main loop")
    finally:
        camera.release()
        logger.info("=" * 60)
        logger.info("Boras Stopped")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
