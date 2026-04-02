# sems/config.py

"""Configuration loader from .env."""
import os
from dotenv import load_dotenv


load_dotenv()


class Settings:
    DB_DRIVER = os.getenv("DB_DRIVER", "sqlite")  # "postgresql", "mysql", "sqlite"
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = int(os.getenv("DB_PORT", 5432))
    DB_NAME = os.getenv("DB_NAME", "sems")
    DB_USER = os.getenv("DB_USER", "sems_user")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "sems_pass")

    SQLITE_PATH = os.getenv("SQLITE_PATH", "sems.db")

    TARIFF_RATE = float(os.getenv("TARIFF_RATE", 8.5))

    LSTM_LOOKBACK_HOURS = int(os.getenv("LSTM_LOOKBACK_HOURS", 72))
    LSTM_BATCH_SIZE = int(os.getenv("LSTM_BATCH_SIZE", 32))
    LSTM_EPOCHS = int(os.getenv("LSTM_EPOCHS", 50))

    PROPHET_RESOLUTION_H = int(os.getenv("PROPHET_RESOLUTION_H", 1))

    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")