# sems/models.py

"""LSTM and Prophet energy prediction models."""
import os
import logging
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from prophet import Prophet
from config import Settings


logger = logging.getLogger(__name__)


def prepare_lstm_data(
    df: pd.DataFrame, target_col: str = "power_kwh", lookback_hours: int = 72
) -> tuple:
    """Prepare supervised LSTM data (X, y) from time‑series df."""
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Resample to hourly if needed
    if len(df) > 0:
        dfi = df.set_index("timestamp").resample("H").sum(numeric_only=True)
        dfi = dfi.fillna(method="ffill")
    else:
        dfi = df.set_index("timestamp")

    # Use only numeric columns
    cols = dfi.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    scaled = scaler.fit_transform(dfi[cols])

    X, y = [], []
    for i in range(lookback_hours, len(scaled)):
        X.append(scaled[i - lookback_hours : i])
        y.append(scaled[i, dfi.columns.get_loc(target_col)])

    X = np.array(X)
    y = np.array(y)

    logger.info(
        f"LSTM data prepared: X shape {X.shape}, y shape {y.shape}, lookback={lookback_hours}h"
    )
    return X, y, scaler, cols, dfi.index


def build_lstm_model(input_shape, units: int = 50, lr: float = 0.001):
    """Build a simple LSTM model."""
    model = Sequential()
    model.add(
        LSTM(
            units=units,
            return_sequences=True,
            input_shape=input_shape,
        )
    )
    model.add(LSTM(units=units // 2, return_sequences=False))
    model.add(Dense(1))

    optimizer = Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss="mse")
    return model


def train_lstm(
    df: pd.DataFrame,
    model_path: str,
    scaler_path: str,
    lookback_hours: int = 72,
    batch_size: int = 32,
    epochs: int = 50,
) -> Optional[dict]:
    """Train LSTM on energy data and save model + scaler."""
    logger.info("Preparing LSTM training data...")
    try:
        X, y, scaler, feat_cols, timestamps = prepare_lstm_data(
            df, lookback_hours=lookback_hours
        )
    except Exception as e:
        logger.warning(f"LSTM data prep failed: {str(e)}")
        return None

    if len(X) == 0 or len(y) == 0:
        logger.warning("Insufficient data for LSTM training.")
        return None

    model = build_lstm_model((X.shape[1], X.shape[2]))
    logger.info(f"Training LSTM for {epochs} epochs, batch_size={batch_size}...")

    history = model.fit(X, y, batch_size=batch_size, epochs=epochs, verbose=0)

    # Save model and scaler
    model.save(model_path)
    joblib.dump(
        {"scaler": scaler, "feat_cols": feat_cols, "timestamps": timestamps},
        scaler_path,
    )
    logger.info(f"LSTM model saved to {model_path} and scaler to {scaler_path}.")

    # Metrics on last 20% as simple validation
    split = int(0.8 * len(X))
    X_val, y_val = X[split:], y[split:]

    if len(X_val) > 0:
        y_pred = model.predict(X_val, verbose=0).flatten()
        mae = np.mean(np.abs(y_val - y_pred))
        rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))
        mape = (
            np.mean(np.abs((y_val - y_pred) / (y_val + 1e‑8))) * 100
        )  # +1e‑8 to avoid div0

        logger.info(
            f"LSTM metrics on last 20%: MAE={mae:.4f}, RMSE={rmse:.4f}, MAPE={mape:.2f}%"
        )
        return {"mae": mae, "rmse": rmse, "mape": mape}
    return {}


def predict_lstm_forecast(
    df: pd.DataFrame,
    model_path: str,
    scaler_path: str,
    forecast_hours: int = 24,
    lookback_hours: int = 72,
) -> Optional[pd.DataFrame]:
    """Use trained LSTM to predict next `forecast_hours` hours."""
    import os
    from tensorflow.keras.models import load_model

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        logger.warning("LSTM model or scaler not found; cannot forecast.")
        return None

    try:
        model = load_model(model_path)
        info = joblib.load(scaler_path)
        scaler = info["scaler"]
        feat_cols = info["feat_cols"]
    except Exception as e:
        logger.warning(f"Failed to load LSTM model/scaler: {str(e)}")
        return None

    # Get last window from data
    df = df.sort_values("timestamp").reset_index(drop=True)
    dfi = df.set_index("timestamp").resample("H").sum(numeric_only=True)
    dfi = dfi.fillna(method="ffill")
    last_row = dfi[feat_cols].iloc[-lookback_hours:]
    scaled_last = scaler.transform(last_row)

    preds_scaled = []
    current_window = scaled_last.copy()

   