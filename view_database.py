"""
view_database.py — Quick way to see everything in your local database.

Usage:
    python view_database.py            → prints all records to the screen
    python view_database.py --csv      → also exports to data/boras_export.csv
                                          (open this in Excel / Google Sheets)

This reads the same data/boras.db that the pipeline writes to.
"""

import argparse
import csv
from pathlib import Path

from config import DB_PATH
from db.database import Database, VehicleEvent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", action="store_true",
                        help="Export to data/boras_export.csv")
    args = parser.parse_args()

    db = Database()
    db.init()

    rows = db.query_all(limit=100000)   # get everything

    if not rows:
        print("Database is empty — no detections yet.")
        return

    # Print to screen
    print(f"\n{'ID':<5} {'Timestamp':<22} {'Plate':<12} {'Type':<10} Image")
    print("-" * 80)
    for r in rows:
        print(f"{r.id:<5} "
              f"{str(r.timestamp):<22} "
              f"{(r.plate_number or '—'):<12} "
              f"{(r.vehicle_type or '—'):<10} "
              f"{r.image_path or '—'}")

    print(f"\nTotal records: {len(rows)}")
    print(f"Database file: {DB_PATH}")

    # Optional CSV export
    if args.csv:
        export_path = Path(DB_PATH).parent / "boras_export.csv"
        with open(export_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["ID", "Timestamp", "Plate Number",
                             "Vehicle Type", "Image Path", "Event Type"])
            for r in rows:
                writer.writerow([
                    r.id,
                    r.timestamp,
                    r.plate_number or "",
                    r.vehicle_type or "",
                    r.image_path or "",
                    r.event_type or "",
                ])
        print(f"\n✅ Exported to: {export_path}")
        print("   Open this file in Excel or Google Sheets.")


if __name__ == "__main__":
    main()
