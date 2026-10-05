"""
db/sheets_reporter.py — OAuth version

Logs in as YOU (your Google account) via browser one time.
After that, a token.json is saved and it works forever without asking again.

Because it logs in as you (not a service account), image uploads to your
Google Drive work normally — no storageQuotaExceeded error.

ONE-TIME SETUP:
  1. Go to console.cloud.google.com
  2. APIs & Services → Credentials → Create Credentials → OAuth client ID
  3. Application type: Desktop app
  4. Download the JSON, rename it to:  oauth_credentials.json
  5. Put it in the project root (next to main.py)
  6. Run the pipeline — a browser opens, log in, click Allow
  7. token.json is saved automatically. Done forever.
"""

import logging
from datetime import datetime
from pathlib import Path

from config import BASE_DIR

logger = logging.getLogger(__name__)

OAUTH_CREDENTIALS = BASE_DIR / "oauth_credentials.json"
TOKEN_FILE        = BASE_DIR / "token.json"

SHEET_ID        = "1RFQrqFEPD9tpaFS25XOktGaMIXDZKOBeoh5SDL35_0g"
DRIVE_FOLDER_ID = "1L-ZwCwhRZ0aOytG17QT4UHk-zlOmPblo"

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]


class SheetsReporter:

    def __init__(self):
        self._gc    = None
        self._ws    = None
        self._drive = None
        self._row   = 1
        self._init()

    def _init(self):
        try:
            import gspread
            from google.auth.transport.requests import Request
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
            from googleapiclient.discovery import build

            creds = None

            # Load saved token if it exists
            if TOKEN_FILE.exists():
                creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)

            # If no valid token, log in via browser
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    if not OAUTH_CREDENTIALS.exists():
                        logger.critical(
                            "oauth_credentials.json not found at %s\n"
                            "Download it from Google Cloud Console "
                            "(OAuth client ID → Desktop app).",
                            OAUTH_CREDENTIALS,
                        )
                        raise FileNotFoundError(OAUTH_CREDENTIALS)

                    flow  = InstalledAppFlow.from_client_secrets_file(
                        str(OAUTH_CREDENTIALS), SCOPES
                    )
                    creds = flow.run_local_server(port=0)

                # Save token for next time
                TOKEN_FILE.write_text(creds.to_json())
                logger.info("Login successful — token saved to %s", TOKEN_FILE)

            # Connect
            self._gc    = gspread.authorize(creds)
            sh          = self._gc.open_by_key(SHEET_ID)
            self._ws    = sh.get_worksheet(0)
            self._drive = build("drive", "v3", credentials=creds)

            self._ensure_header()

            all_vals  = self._ws.get_all_values()
            filled    = [r for r in all_vals if any(c.strip() for c in r)]
            self._row = len(filled) + 1

            logger.info("Google Sheets ready (OAuth) — next_row=%d", self._row)

        except ImportError:
            logger.critical(
                "Run: pip install gspread google-auth google-auth-oauthlib "
                "google-api-python-client"
            )
            raise
        except Exception as exc:
            logger.critical("Failed to connect to Google Sheets: %s", exc)
            raise

    def _ensure_header(self):
        first = self._ws.row_values(1)
        if not first:
            self._ws.update("A1:D1", [["#", "Timestamp", "Plate Number", "Vehicle Image"]])
            self._ws.format("A1:D1", {
                "textFormat":          {"bold": True, "fontSize": 12},
                "backgroundColor":     {"red": 0.18, "green": 0.25, "blue": 0.34},
                "horizontalAlignment": "CENTER",
            })
            logger.info("Header row written")

    def append_row(self, timestamp: datetime, plate: str, image_path: Path | None):
        try:
            image_url = ""
            if image_path and Path(image_path).exists():
                image_url = self._upload_image(image_path, plate, timestamp)

            row_num = self._row - 1
            self._ws.append_row(
                [row_num, timestamp.strftime("%Y-%m-%d %H:%M:%S"), plate, image_url],
                value_input_option="USER_ENTERED",
            )

            if image_url:
                # Russian/EU locale sheets use ';' as the formula argument
                # separator instead of ','. Using ';' works in both.
                self._ws.update(
                    f"D{self._row}",
                    [[f'=HYPERLINK("{image_url}";"📷 View Photo")']],
                    value_input_option="USER_ENTERED",
                )

            self._ws.format(f"C{self._row}", {
                "textFormat":          {"bold": True, "fontSize": 13},
                "horizontalAlignment": "CENTER",
            })

            logger.info("Sheets updated — row=%d plate='%s' image=%s",
                        self._row, plate, image_url or "none")
            self._row += 1

        except Exception as exc:
            logger.error("Failed to update Google Sheets: %s", exc)

    def _upload_image(self, image_path: Path, plate: str, ts: datetime) -> str:
        try:
            from googleapiclient.http import MediaFileUpload

            filename      = f"{ts.strftime('%Y%m%d_%H%M%S')}_{plate}.jpg"
            file_metadata = {"name": filename, "parents": [DRIVE_FOLDER_ID]}

            media    = MediaFileUpload(str(image_path), mimetype="image/jpeg")
            uploaded = self._drive.files().create(
                body=file_metadata,
                media_body=media,
                fields="id",
            ).execute()

            file_id = uploaded.get("id")

            self._drive.permissions().create(
                fileId=file_id,
                body={"type": "anyone", "role": "reader"},
            ).execute()

            # Direct-view URL that works with =IMAGE() in Sheets
            url = f"https://drive.google.com/uc?export=view&id={file_id}"
            logger.info("Image uploaded to Drive: %s", url)
            return url

        except Exception as exc:
            logger.warning("Drive upload failed: %s", exc)
            return ""
