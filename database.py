# sems/database.py

"""Database connection and schema setup."""
import os
import sqlite3
from typing import Optional
from sqlalchemy import create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import Column, Integer, DateTime, Float, String, Boolean
import pandas as pd
from config import Settings


Base = declarative_base()


class EnergyReading(Base):
    __tablename__ = "energy_readings"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, nullable=False)
    power_kwh = Column(Float, nullable=False)
    voltage = Column(Float, nullable=True)
    current = Column(Float, nullable=True)
    temperature = Column(Float, nullable=True)
    humidity = Column(Float, nullable=True)
    occupancy = Column(Boolean, nullable=True)  # 0/1 as bool
    appliance_id = Column(String(50), nullable=True)
    room_id = Column(String(50), nullable=True)


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
    """Create SQLAlchemy engine."""
    db_url = get_db_url()
    return create_engine(db_url, echo=False)


def get_session():
    """Create scoped session factory."""
    engine = get_engine()
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal()


def create_tables():
    """Create all tables if they don’t exist."""
    engine = get_engine()
    Base.metadata.create_all(bind=engine)


def table_exists(session, tablename: str) -> bool:
    """Check if table exists in DB."""
    result = session.execute(
        text(
            """
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_name = :tname
            )
            """
        ),
        {"tname": tablename},
    ).fetchone()
    if result is None:
        return False
    return result[0]


def fetch_all_data() -> pd.DataFrame:
    """Load entire energy_readings table into a DataFrame."""
    engine = get_engine()
    query = "SELECT * FROM energy_readings ORDER BY timestamp"
    df = pd.read_sql_query(query, con=engine)
    return df


def load_sample_data(csv_path: str):
    """Seed energy_readings from a CSV if table is empty."""
    import logging

    logger = logging.getLogger(__name__)
    session = get_session()

    # Use raw SQL so we can handle any DB
    exists = session.execute(
        text(
            "SELECT EXISTS (SELECT 1 FROM energy_readings LIMIT 1) AS has_data;"
        )
    ).fetchone()

    if exists and exists[0]:
        logger.info("energy_readings table already has data; skipping sample load.")
        return

    logger.info(f"Loading sample data from {csv_path} ...")

    df = pd.read_csv(csv_path)

    # Ensure required columns
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
    if not all(c in df.columns for c in required_cols):
        raise ValueError(f"CSV must have columns: {required_cols}")

    engine = get_engine()
    df.to_sql("energy_readings", con=engine, if_exists="append", index=False)
    logger.info(f"Sample data ({len(df)} rows) loaded into energy_readings.")