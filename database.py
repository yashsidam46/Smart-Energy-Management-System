# sems/database.py

"""Database connection and schema setup."""

import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import Column, Integer, DateTime, Float, String, Boolean
import pandas as pd
from config import Settings

Base = declarative_base()

# ===============================
# ORM Model
# ===============================

class EnergyReading(Base):
    __tablename__ = "energy_readings"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, nullable=False)
    power_kwh = Column(Float, nullable=False)
    voltage = Column(Float, nullable=True)
    current = Column(Float, nullable=True)
    temperature = Column(Float, nullable=True)
    humidity = Column(Float, nullable=True)
    occupancy = Column(Boolean, nullable=True)
    appliance_id = Column(String(50), nullable=True)
    room_id = Column(String(50), nullable=True)


# ===============================
# Engine (cached)
# ===============================

_engine = None


def get_db_url() -> str:
    """Build SQLAlchemy URL from Settings."""
    s = Settings

    if s.DB_DRIVER == "postgresql":
        return f"postgresql://{s.DB_USER}:{s.DB_PASSWORD}@{s.DB_HOST}:{s.DB_PORT}/{s.DB_NAME}"
    elif s.DB_DRIVER == "mysql":
        return f"mysql+pymysql://{s.DB_USER}:{s.DB_PASSWORD}@{s.DB_HOST}:{s.DB_PORT}/{s.DB_NAME}"
    elif s.DB_DRIVER == "sqlite":
        db_path = os.path.abspath(s.SQLITE_PATH)
        return f"sqlite:///{db_path}"
    else:
        raise ValueError(f"Unsupported DB_DRIVER: {s.DB_DRIVER}")


def get_engine():
    """Create (or reuse) SQLAlchemy engine."""
    global _engine
    if _engine is None:
        _engine = create_engine(get_db_url(), echo=False)
    return _engine


def get_session():
    """Create session."""
    SessionLocal = sessionmaker(bind=get_engine())
    return SessionLocal()


# ===============================
# Table Setup
# ===============================

def create_tables():
    """Create all tables if they don’t exist."""
    Base.metadata.create_all(bind=get_engine())


def table_exists(session, tablename: str) -> bool:
    """Check if table exists."""
    if Settings.DB_DRIVER == "sqlite":
        result = session.execute(
            text("SELECT name FROM sqlite_master WHERE type='table' AND name=:tname;"),
            {"tname": tablename}
        ).fetchone()
        return result is not None
    else:
        result = session.execute(
            text("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables
                    WHERE table_name = :tname
                )
            """),
            {"tname": tablename},
        ).fetchone()
        return result[0]


# ===============================
# Data Access
# ===============================

def fetch_all_data() -> pd.DataFrame:
    """Load all energy data into a DataFrame."""
    query = "SELECT * FROM energy_readings ORDER BY timestamp"
    return pd.read_sql_query(query, con=get_engine())


def load_sample_data(csv_path: str):
    """Load sample CSV data if table is empty."""
    import logging

    logger = logging.getLogger(__name__)
    session = get_session()

    try:
        # Check if table has data (safe for SQLite)
        exists = session.execute(
            text("SELECT COUNT(1) FROM energy_readings")
        ).fetchone()

        if exists and exists[0] > 0:
            logger.info("Table already has data. Skipping sample load.")
            return

        logger.info(f"Loading sample data from {csv_path}...")

        df = pd.read_csv(csv_path)

        # ✅ Convert timestamp (IMPORTANT)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Required columns
        required_cols = [
            "timestamp",
            "power_kwh",
            "voltage",
            "current",
            "temperature",
            "humidity",
            "occupancy",
            "appliance_id",
            "room_id",
        ]

        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"CSV must contain columns: {required_cols}")

        # Insert into DB
        df.to_sql("energy_readings", con=get_engine(), if_exists="append", index=False)

        logger.info(f"Loaded {len(df)} rows into energy_readings.")

    finally:
        session.close()