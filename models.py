# sems/models.py

import os
import logging
import joblib
import numpy as np
import pandas as pd
from datetime import timedelta
from typing import Optional
from tensorflow.keras.models import load_model

logger = logging.getLogger(__name__)


def predict_lstm_forecast(
    df: pd.DataFrame,
    model_path: str,
    scaler_path: str,
    forecast_hours: int = 24,
    lookback_hours: int = 72,
) -> Optional[pd.DataFrame]:
    """Predict next hours using trained LSTM."""

    # ===============================
    # Check model files
    # ===============================
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        logger.warning("Model or scaler not found.")
        return None

    # ===============================
    # Load model + scaler
    # ===============================
    try:
        model = load_model(model_path)
        info = joblib.load(scaler_path)
        scaler = info["scaler"]
        feat_cols = info["feat_cols"]
    except Exception as e:
        logger.warning(f"Error loading model: {e}")
        return None

    # ===============================
    # Prepare data
    # ===============================
    if df.empty:
        logger.warning("Input dataframe is empty.")
        return None

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    df = df.sort_values("timestamp").reset_index(drop=True)

    dfi = df.set_index("timestamp").resample("H").sum(numeric_only=True)
    dfi = dfi.fillna(method="ffill")

    if len(dfi) < lookback_hours:
        logger.warning("Not enough data for prediction.")
        return None

    # ===============================
    # Prepare window
    # ===============================
    last_window = dfi[feat_cols].iloc[-lookback_hours:]
    scaled_window = scaler.transform(last_window)

    preds = []
    current_window = scaled_window.copy()

    target_index = 0  # assuming target column is first

    # ===============================
    # Prediction loop
    # ===============================
    for _ in range(forecast_hours):
        X = np.expand_dims(current_window, axis=0)
        pred = model.predict(X, verbose=0)[0][0]

        preds.append(pred)

        new_row = current_window[-1].copy()
        new_row[target_index] = pred

        current_window = np.vstack([current_window[1:], new_row])

    # ===============================
    # Inverse scaling
    # ===============================
    dummy = np.zeros((len(preds), len(feat_cols)))
    dummy[:, target_index] = preds

    inv = scaler.inverse_transform(dummy)[:, target_index]

    # ===============================
    # Create timestamps
    # ===============================
    last_time = dfi.index[-1]
    future_times = [
        last_time + timedelta(hours=i + 1) for i in range(forecast_hours)
    ]

    result = pd.DataFrame({
        "timestamp": future_times,
        "predicted_power_kwh": inv
    })

    return result