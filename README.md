# Boras — Vehicle Plate Detection

Detects vehicles from a camera, reads license plates, logs to a Google Sheet.

## Pipeline

```
Camera / video file
   ↓
Read every 25th frame
   ↓
Vehicle model → crop each vehicle
   ↓
ROI filter → keep only vehicles in the middle zone
   ↓
Plate model → crop the plate out of the vehicle
   ↓
Upscale plate to 1280x384 + sharpen
   ↓
EasyOCR reads the plate (votes across 4 image variants)
   ↓
Save: vehicle photo + DB record + Google Sheet row
```

## Setup

### 1. Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Add your models
```
weights/vehicle.pt   ← vehicle detection model
weights/plate.pt     ← plate detection model
```

### 3. Google OAuth (one time)
1. Go to console.cloud.google.com → APIs & Services → Credentials
2. Create Credentials → OAuth client ID → Desktop app
3. Download JSON, rename to `oauth_credentials.json`, put in project root
4. First run opens a browser → log in → Allow. Saves `token.json` forever.

### 4. Set your Sheet + Drive folder IDs
In `db/sheets_reporter.py`:
```python
SHEET_ID        = "your_sheet_id"
DRIVE_FOLDER_ID = "your_drive_folder_id"
```

### 5. Set camera source
Copy .env.example to .env and fill in:
CAMERA_SOURCE=rtsp://user:pass@your-ip/stream
TELEGRAM_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_chat_id

### 6. Run
```bash
python main.py
```

## Tuning

All in `config.py`:

| Setting | What it does |
|---|---|
| `FRAME_SKIP` | Process 1 of every N frames (default 25) |
| `ROI_X1 / ROI_X2` | Left/right edge of the zone (0.33–0.67 = middle third) |
| `VEHICLE_CONF` | Vehicle detection threshold |
| `PLATE_CONF` | Plate detection threshold |
| `DUPLICATE_WINDOW_SECONDS` | Ignore same vehicle for N seconds |

## Output

- `data/snapshots/` — saved vehicle photos
- `data/boras.db` — local database
- Google Sheet — timestamp + plate + photo, updated live
