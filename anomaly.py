# sems/anomaly.py

"""
Anomaly detection on energy readings using Isolation Forest.
Adds an 'is_anomaly' boolean column: -1 = anomaly, 1 = normal (scikit‑learn labels).
"""

import logging
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from datetime import datetime


logger = logging.getLogger(__name__)


def detect_anomalies_isolation_forest(
    df: pd.DataFrame,
    numeric_cols: list = None,
    contamination: float = 0.05,
) -> pd.DataFrame:
    """
    Detect anomalies in energy data using Isolation Forest.

    Args:
        df: DataFrame with energy readings.
        numeric_cols: list of numeric columns to use (if None, all numeric cols).
        contamination: fraction of anomalies expected (default 5%).

    Returns:
        DataFrame with new columns:
        - is_anomaly (int): -1 for anomaly, 1 for normal.
        - anomaly_score (float): anomaly score; lower = more anomalous.
    """
    df = df.copy()

    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) == 0:
        logger.warning("No numeric columns for anomaly detection; returning original df.")
        df["is_anomaly"] = 1
        df["anomaly_score"] = 0.0
        return df

    # Drop rows with NaN in numeric_cols
    mask = df[numeric_cols].notna().all(axis=1)
    subset = df.loc[mask, numeric_cols].copy()

    if len(subset) == 0:
        logger.warning("No complete rows for numeric columns; no anomaly detection.")
        df["is_anomaly"] = 1
        df["anomaly_score"] = 0.0
        return df

    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(subset)

    # Fit Isolation Forest
    logger.info(
        f"Running Isolation Forest with contamination={contamination}, "
        f"n_features={X_scaled.shape[1]}, n_samples={X_scaled.shape[0]}"
    )
    model = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    preds = model.fit_predict(X_scaled)  # -1 = anomaly, 1 = normal
    scores = model.decision_function(X_scaled)

    # Join back to df
    result = df.copy()
    result.loc[mask, "is_anomaly"] = preds
    result.loc[mask, "anomaly_score"] = scores

    # Fill missing rows as "normal" to avoid NaNs
    result["is_anomaly"] = result["is_anomaly"].fillna(1).astype(int)
    result["anomaly_score"] = result["anomaly_score"].fillna(0.0)

    logger.info(
        f"Anomaly detection done: {result['is_anomaly'].eq(-1).sum()} anomalies flagged."
    )
    return result


def get_top_anomalies(
    df: pd.DataFrame,
    n: int = 10,
    sort_by: str = "anomaly_score",
) -> pd.DataFrame:
    """
    Get top N most anomalous rows.

    Args:
        df: DataFrame with at least 'is_anomaly' and 'anomaly_score'.
        n: number of top anomalies to return.
        sort_by: column to sort by (default 'anomaly_score', lower is more anomalous).

    Returns:
        Top N anomalous rows.
    """
    if "is_anomaly" not in df.columns:
        logger.warning("No 'is_anomaly' column; returning empty DataFrame.")
        return pd.DataFrame()

    anomalies = df[df["is_anomaly"] == -1].copy()
    if len(anomalies) == 0:
        logger.info("No anomalies found.")
        return anomalies

    ascending = True if sort_by == "anomaly_score" else True
    top = anomalies.nlargest(n, sort_by) if "decrease" in "decrease" else anomalies.nsmallest(n, sort_by)
    return top


def flag_anomalies_power_kwh(
    df: pd.DataFrame,
    power_col: str = "power_kwh",
    contamination: float = 0.05,
) -> pd.DataFrame:
    """
    Convenience wrapper: detect anomalies based mainly on power_kwh,
    using a few related features if present.

    Args:
        df: energy DataFrame.
        power_col: power column to emphasize.
        contamination: fraction of anomalies.

    Returns:
        df with anomaly flags and scores.
    """
    cols = [power_col]
    for c in ["voltage", "current", "temperature", "humidity"]:
        if c in df.columns:
            cols.append(c)

    return detect_anomalies_isolation_forest(
        df,
        numeric_cols=cols,
        contamination=contamination,
    )