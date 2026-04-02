# sems/insights.py

"""
Insights and analytics functions for SEMS.
"""

import pandas as pd
from typing import Dict
from config import Settings


# ===============================
# Peak Hour Detection
# ===============================

def get_peak_hour(df: pd.DataFrame) -> int:
    if df.empty:
        return -1

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    df["hour"] = df["timestamp"].dt.hour
    by_hour = df.groupby("hour")["power_kwh"].sum()

    if by_hour.empty:
        return -1

    return int(by_hour.idxmax())


# ===============================
# Hourly Consumption
# ===============================

def get_hourly_consumption(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series()

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    df["hour"] = df["timestamp"].dt.hour
    return df.groupby("hour")["power_kwh"].sum()


# ===============================
# Correlation Matrix
# ===============================

def get_correlation(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    cols = ["power_kwh", "voltage", "current", "temperature", "humidity", "occupancy"]
    cols = [c for c in cols if c in df.columns]

    if len(cols) < 2:
        return pd.DataFrame()

    return df[cols].corr()


# ===============================
# Appliance Consumption
# ===============================

def get_appliance_consumption(df: pd.DataFrame) -> pd.Series:
    if df.empty or "appliance_id" not in df.columns:
        return pd.Series()

    return df.groupby("appliance_id")["power_kwh"].sum()


# ===============================
# Room Consumption
# ===============================

def get_room_consumption(df: pd.DataFrame) -> pd.Series:
    if df.empty or "room_id" not in df.columns:
        return pd.Series()

    return df.groupby("room_id")["power_kwh"].sum()


# ===============================
# Cost Calculation
# ===============================

def calculate_cost(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0

    total_kwh = df["power_kwh"].sum()
    return float(total_kwh * Settings.TARIFF_RATE)


# ===============================
# Generate Smart Insights
# ===============================

def generate_insights(df: pd.DataFrame) -> Dict:
    if df.empty:
        return {
            "total_energy_kwh": 0,
            "total_cost": 0,
            "peak_hour": -1,
            "recommendation": "No data available."
        }

    peak_hour = get_peak_hour(df)
    total_kwh = df["power_kwh"].sum()
    cost = calculate_cost(df)

    insights = {
        "total_energy_kwh": round(total_kwh, 2),
        "total_cost": round(cost, 2),
        "peak_hour": peak_hour,
        "recommendation": ""
    }

    if peak_hour == -1:
        insights["recommendation"] = "Not enough data to generate insights."
    elif 18 <= peak_hour <= 23:
        insights["recommendation"] = "High evening usage detected. Shift heavy appliances to daytime."
    elif 0 <= peak_hour <= 6:
        insights["recommendation"] = "Night usage is high. Check for unnecessary loads."
    else:
        insights["recommendation"] = "Energy usage is relatively balanced."

    return insights