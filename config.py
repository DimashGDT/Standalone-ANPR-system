"""
config.py — Central configuration for the Boras pipeline.
Edit this file to match your environment.
"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
WEIGHTS_DIR   = BASE_DIR / "weights"
SNAPSHOT_DIR  = BASE_DIR / "data" / "snapshots"
DB_PATH       = BASE_DIR / "data" / "boras.db"

VEHICLE_MODEL = WEIGHTS_DIR / "vehicle.pt"
PLATE_MODEL   = WEIGHTS_DIR / "plate.pt"

# ── Camera ─────────────────────────────────────────────────────────────────────
# Use an RTSP URL like: "rtsp://user:pass@192.168.1.100:554/stream"
# Or an integer (0, 1, …) for a local USB camera.
# Or a path to a video file for testing.
CAMERA_SOURCE = os.getenv("CAMERA_SOURCE", "0")
# How often to process — in SECONDS.
# The pipeline picks one frame every PROCESS_INTERVAL_SECONDS.
# Set to 3 = process every 3 seconds, 5 = every 5 seconds, etc.
# Lower = catch more vehicles but more CPU. Higher = lighter but may miss fast cars.
PROCESS_INTERVAL_SECONDS = float(os.getenv("PROCESS_INTERVAL_SECONDS", 3.0))

# Video/camera frame rate. Used to convert the seconds interval into frames.
# Most CCTV and videos are 25 or 30 fps. Check your source if unsure.
SOURCE_FPS = float(os.getenv("SOURCE_FPS", 30.0))

# Computed: how many frames to skip between processing.
FRAME_SKIP = max(1, int(PROCESS_INTERVAL_SECONDS * SOURCE_FPS))

# ── Detection thresholds ───────────────────────────────────────────────────────
VEHICLE_CONF   = float(os.getenv("VEHICLE_CONF",  0.5))
PLATE_CONF     = float(os.getenv("PLATE_CONF",    0.4))

# YOLO class IDs that count as "vehicles" in your vehicle model.
# Adjust to match your model's class list.
VEHICLE_CLASSES = [0, 1, 2, 3]   # e.g. car, truck, bus, motorbike

# YOLO class IDs that are PERSONS — these are silently skipped.
# No photo is taken, no DB record is created.
# Standard COCO class for person = 0. If your custom model uses a different
# ID for person, change it here. Add multiple IDs if needed: [0, 5]
PERSON_CLASS_IDS = [0]

# ── OCR ────────────────────────────────────────────────────────────────────────
# EasyOCR language list — add more if needed
OCR_LANGUAGES = ["en"]

# Minimum OCR confidence to accept a plate reading.
# 0.1 is intentionally low — the _clean() regex acts as the real filter.
# Raise it once your setup is working reliably.
OCR_MIN_CONF = 0.1

# ── Database ───────────────────────────────────────────────────────────────────
DATABASE_URL = f"sqlite:///{DB_PATH}"

# ── Logging ────────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ── ROI — Region of Interest ───────────────────────────────────────────────────
# Only vehicles whose bounding box CENTER falls inside this zone are processed.
# All values are FRACTIONS of the frame (0.0 = left/top, 1.0 = right/bottom).
#
# Example — 4-lane road, you want only the middle two lanes:
#   ROI_X1, ROI_X2 = 0.25, 0.75   (middle 50% horizontally)
#   ROI_Y1, ROI_Y2 = 0.0,  1.0    (full height)
#
# Example — single lane, top-right quadrant:
#   ROI_X1, ROI_X2 = 0.5,  1.0
#   ROI_Y1, ROI_Y2 = 0.0,  0.5
#
# To DISABLE the ROI and process the full frame, set:
#   ROI_ENABLED = False
#
ROI_ENABLED = True

ROI_X1 = float(os.getenv("ROI_X1", 0.33))   # left edge   — start at 1/3
ROI_X2 = float(os.getenv("ROI_X2", 0.67))   # right edge  — end at 2/3
ROI_Y1 = float(os.getenv("ROI_Y1", 0.0))    # top edge    — full height
ROI_Y2 = float(os.getenv("ROI_Y2", 1.0))    # bottom edge — full height

# ── Duplicate suppression ──────────────────────────────────────────────────────
# A plate is considered a duplicate if it was already seen within this many
# seconds. The same truck passing through will only create ONE database record.
#
# Set to 0 to disable duplicate suppression entirely.
#
# Recommended values:
#   30  — very busy site (trucks every 30s)
#   120 — normal site
#   300 — slow site / you want to catch the same truck on re-entry after 5 min
#
DUPLICATE_WINDOW_SECONDS = int(os.getenv("DUPLICATE_WINDOW_SECONDS", 300))

# ── Plate requirement ──────────────────────────────────────────────────────────
# If True, vehicles with no readable plate are skipped entirely:
#   no database record, no image upload, no Google Sheet row.
# If False, they are saved with plate marked as "UNREAD".
# Set True for clean data, False if you want to log every vehicle seen.
REQUIRE_PLATE = os.getenv("REQUIRE_PLATE", "true").lower() == "true"

# ── Telegram notifications ─────────────────────────────────────────────────────
# Turn alerts on/off
TELEGRAM_ENABLED = os.getenv("TELEGRAM_ENABLED", "true").lower() == "true"

# From @BotFather and getUpdates (see notifications/telegram.py header)
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
# Include the vehicle photo in the alert
TELEGRAM_SEND_PHOTO = os.getenv("TELEGRAM_SEND_PHOTO", "true").lower() == "true"

# Set True later to alert ONLY for trucks (skip cars)
TELEGRAM_TRUCKS_ONLY = os.getenv("TELEGRAM_TRUCKS_ONLY", "false").lower() == "true"

# Which vehicle_type names count as trucks (match your model's class names)
TELEGRAM_TRUCK_CLASSES = ["truck", "lorry", "freight"]

# ── Live preview window ────────────────────────────────────────────────────────
# Show a window with the camera feed + detection boxes while running.
# Set False for headless/server use (no screen).
SHOW_PREVIEW = False
