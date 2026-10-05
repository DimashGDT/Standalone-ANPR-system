"""
notifications/telegram.py

Sends a Telegram alert when a new vehicle is detected.
Message contains: plate number, time, vehicle type — plus the vehicle photo.

Setup:
  1. Create a bot via @BotFather in Telegram → get the TOKEN
  2. Message your bot, then visit
     https://api.telegram.org/bot<TOKEN>/getUpdates  → get your CHAT_ID
  3. Put both into config.py (TELEGRAM_TOKEN, TELEGRAM_CHAT_ID)

Uses plain HTTP requests — no extra library needed beyond 'requests'.
Never crashes the pipeline: any send failure is logged and swallowed.
"""

import logging
from datetime import datetime
from pathlib import Path

from config import (
    TELEGRAM_ENABLED, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID,
    TELEGRAM_SEND_PHOTO, TELEGRAM_TRUCKS_ONLY, TELEGRAM_TRUCK_CLASSES,
)

logger = logging.getLogger(__name__)


class TelegramNotifier:

    def __init__(self):
        self.enabled = TELEGRAM_ENABLED
        if self.enabled and (not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID):
            logger.warning("Telegram enabled but TOKEN/CHAT_ID missing — disabling")
            self.enabled = False
        if self.enabled:
            logger.info("Telegram notifications enabled")

    # ── Public ─────────────────────────────────────────────────────────────────

    def notify(
        self,
        plate:        str,
        vehicle_type: str,
        timestamp:    datetime,
        image_path:   Path | None = None,
    ):
        """Send an alert for a new vehicle. Safe to call always — checks toggles."""
        if not self.enabled:
            return

        # Trucks-only filter (off by default; flip TELEGRAM_TRUCKS_ONLY later)
        if TELEGRAM_TRUCKS_ONLY:
            if vehicle_type.lower() not in [c.lower() for c in TELEGRAM_TRUCK_CLASSES]:
                logger.debug("Telegram: '%s' is not a truck — no alert", vehicle_type)
                return

        plate_text = plate if plate else "UNREAD"
        message = (
            f"🚗 New vehicle detected\n"
            f"Plate: {plate_text}\n"
            f"Type: {vehicle_type}\n"
            f"Time: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
        )

        try:
            if TELEGRAM_SEND_PHOTO and image_path and Path(image_path).exists():
                self._send_photo(image_path, message)
            else:
                self._send_text(message)
        except Exception as exc:
            logger.warning("Telegram send failed: %s", exc)

    # ── Private ────────────────────────────────────────────────────────────────

    def _send_text(self, text: str):
        import requests
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        resp = requests.post(url, data={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": text,
        }, timeout=10)
        if resp.status_code == 200:
            logger.info("Telegram text alert sent")
        else:
            logger.warning("Telegram text failed: %s %s", resp.status_code, resp.text)

    def _send_photo(self, image_path: Path, caption: str):
        import requests
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
        with open(str(image_path), "rb") as photo:
            resp = requests.post(
                url,
                data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption},
                files={"photo": photo},
                timeout=20,
            )
        if resp.status_code == 200:
            logger.info("Telegram photo alert sent")
        else:
            logger.warning("Telegram photo failed: %s %s", resp.status_code, resp.text)
