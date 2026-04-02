# sems/prediction.py

"""
Prediction utilities for SEMS.
Handles LSTM and Prophet forecasting.
"""

import os
import logging
import pandas as pd
from typing import Optional

from models import train_lstm, predict_lstm_forecast
from config import Settings

logger = logging.getLogger(__name__)

# ===============================
# Paths (use config)
# ===============================

LSTM_MODEL_PATH = Settings.LSTM_MODEL_PATH
LSTM_SCALER_PATH = Settings.LSTM_SCALER_PATH


# ===============================
# Ensure model directory exists
# ===============================

def ensure_model_dir():
    os.makedirs(os.path.dirname(LSTM_MODEL_PATH), exist_ok=True)


# ===============================
# Train LSTM Model
# ===============================

def train_lstm_model(df: pd.DataFrame) -> Optional[dict]:
    """Train and save LSTM model."""
    if df.empty:
        logger.warning("Empty dataframe. Cannot train LSTM.")
        return None

    ensure_model_dir()

    logger.info("Training LSTM model...")

    metrics = train_lstm(
        df=df,
        model_path=LSTM_MODEL_PATH,
        scaler_path=LSTM_SCALER_PATH,
        lookback_hours=Settings.LSTM_LOOKBACK_HOURS,
        batch_size=Settings.LSTM_BATCH_SIZE,
        epochs=Settings.LSTM_EPOCHS,
    )

    return metrics


# ===============================
# Predict using LSTM
# ===============================

def get_lstm_forecast(df: pd.DataFrame, hours: int = 24) -> Optional[pd.DataFrame]:
    """Get LSTM forecast."""
    if df.empty:
        logger.warning("Empty dataframe for LSTM forecast.")
        return None

    return predict_lstm_forecast(
        df=df,
        model_path=LSTM_MODEL_PATH,
        scaler_path=LSTM_SCALER_PATH,
        forecast_hours=hours,
        lookback_hours=Settings.LSTM_LOOKBACK_HOURS,
    )


# ===============================
# Prophet Forecast
# ===============================

def get_prophet_forecast(df: pd.DataFrame, days: int = 7) -> Optional[pd.DataFrame]:
    """Generate forecast using Prophet."""
    try:
        from prophet import Prophet
    except ImportError:
        logger.warning("Prophet not installed.")
        return None

    if df.empty:
        logger.warning("Empty dataframe for Prophet.")
        return None

    if "timestamp" not in df.columns or "power_kwh" not in df.columns:
        logger.warning("Required columns missing for Prophet.")
        return None

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Prepare data
    df_p = df[["timestamp", "power_kwh"]].rename(
        columns={"timestamp": "ds", "power_kwh": "y"}
    )

    model = Prophet()
    model.fit(df_p)

    future = model.make_future_dataframe(periods=days * 24, freq="H")
    forecast = model.predict(future)

    result = forecast[["ds", "yhat"]].tail(days * 24)
    result = result.rename(
        columns={"ds": "timestamp", "yhat": "predicted_power_kwh"}
    )

    return result


# ===============================
# Combined Forecast
# ===============================

def get_combined_forecast(df: pd.DataFrame) -> dict:
    """Get both LSTM and Prophet forecasts."""
    return {
        "lstm": get_lstm_forecast(df),
        "prophet": get_prophet_forecast(df),
    }