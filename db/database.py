"""
db/database.py — SQLite database via SQLAlchemy.

Table: vehicle_events
Columns match the spec in the project brief.
"""

import logging
from datetime import datetime

from sqlalchemy import (
    Column, DateTime, Integer, String,
    create_engine, text,
)
from sqlalchemy.orm import DeclarativeBase, Session

from config import DATABASE_URL

logger = logging.getLogger(__name__)


# ── ORM model ─────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


class VehicleEvent(Base):
    """One row per detected vehicle."""

    __tablename__ = "vehicle_events"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    vehicle_id      = Column(Integer,    nullable=True)   # session tracking ID
    timestamp       = Column(DateTime,  nullable=False, default=datetime.utcnow)
    plate_number    = Column(String(20), nullable=True)   # NULL if unreadable
    vehicle_type    = Column(String(50), nullable=True)
    image_path      = Column(String(255), nullable=True)
    event_type      = Column(String(20),  nullable=False, default="unknown")
    # ↑ "unknown" now; "entrance"/"exit" later

    def __repr__(self):
        return (
            f"<VehicleEvent id={self.id} "
            f"plate={self.plate_number!r} "
            f"ts={self.timestamp} "
            f"type={self.event_type!r}>"
        )


# ── Database wrapper ───────────────────────────────────────────────────────────

class Database:
    """
    Thin wrapper around SQLAlchemy engine + session.

    Usage:
        db = Database()
        db.init()          # creates tables if missing
        db.save_event(evt)
        rows = db.query_all()
    """

    def __init__(self, url: str = DATABASE_URL):
        self.engine = create_engine(url, echo=False, future=True)

    def init(self):
        """Create all tables (idempotent — safe to call on every startup)."""
        Base.metadata.create_all(self.engine)
        logger.info("Database ready at %s", self.engine.url)

    def save_event(self, event: VehicleEvent) -> VehicleEvent:
        """Insert a VehicleEvent row and return it with id populated."""
        with Session(self.engine) as session:
            session.add(event)
            session.commit()
            session.refresh(event)
            logger.debug("Saved event id=%d", event.id)
        return event

    def query_all(self, limit: int = 100) -> list[VehicleEvent]:
        """Return the most recent `limit` events (newest first)."""
        with Session(self.engine) as session:
            rows = (
                session.query(VehicleEvent)
                .order_by(VehicleEvent.timestamp.desc())
                .limit(limit)
                .all()
            )
        return rows

    def query_by_plate(self, plate: str) -> list[VehicleEvent]:
        """Return all events for a given plate number."""
        with Session(self.engine) as session:
            rows = (
                session.query(VehicleEvent)
                .filter(VehicleEvent.plate_number == plate)
                .order_by(VehicleEvent.timestamp.desc())
                .all()
            )
        return rows

    def health_check(self) -> bool:
        """Return True if the database is reachable."""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception as exc:
            logger.error("DB health check failed: %s", exc)
            return False
