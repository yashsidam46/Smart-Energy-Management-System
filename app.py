# sems/app.py

"""
Smart Energy Management System (SEMS) - Streamlit Dashboard

Tabs:
- Overview          → daily trend + weekly heatmap
- Predictions       → next 24h + 7‑day forecast (LSTM / Prophet)
- Insights          → peak hours, correlations, cost breakdown
- Anomalies         → table of top anomalies
- Recommendations   → rule‑based optimization tips
"""

import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path

from config import Settings
from database import (
    create_tables,
    load_sample_data,
    fetch_all_data,
)
from anomaly import flag_anomalies_power_kwh, get_top_anomalies

# ===============================
# Setup logging
# ===============================

logging.basicConfig(
    level=getattr(logging, Settings.LOG_LEVEL),
    format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ===============================
# Load data and flag anomalies
# ===============================

@st.cache_data(ttl=300)  # cache for 5 minutes
def load_energy_data() -> pd.DataFrame:
    logger.info("Loading energy data from database...")
    df = fetch_all_data()

    if df.empty:
        st.warning("No data found in `energy_readings`; using sample CSV only.")
        sample_path = Path(__file__).parent / "sems_sample.csv"
        if sample_path.exists():
            df = pd.read_csv(sample_path)
        else:
            st.error("Sample CSV `sems_sample.csv` not found.")
            return pd.DataFrame()

    # Ensure timestamp is datetime
    if df["timestamp"].dtype == object:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Sort and reset index
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Add date and hour for analysis
    df["date"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.hour

    # Detect anomalies
    logger.info("Running anomaly detection...")
    try:
        df = flag_anomalies_power_kwh(df, contamination=0.05)
    except Exception as e:
        st.warning(f"Anomaly detection failed: {str(e)}")
        df["is_anomaly"] = 1
        df["anomaly_score"] = 0.0

    return df


# ===============================
# Sidebar: filters
# ===============================

st.set_page_config(
    layout="wide",
    page_title="SEMS · Smart Energy Management",
    page_icon="⚡",
)

st.sidebar.title("📊 Filters")

df = load_energy_data()

if df.empty:
    st.error("No valid data loaded. Check DB connection and sample CSV.")
    st.stop()

min_dt = df["timestamp"].min()
max_dt = df["timestamp"].max()

st.sidebar.write(f"Available data: {min_dt.date()} → {max_dt.date()}")

date_range = st.sidebar.date_input(
    "Date range",
    value=[min_dt.date(), max_dt.date()],
    min_value=min_dt.date(),
    max_value=max_dt.date(),
)

selected_rooms = st.sidebar.multiselect(
    "Room ID",
    options=sorted(df["room_id"].dropna().unique()) if "room_id" in df else [],
    default=None,
)
selected_appliances = st.sidebar.multiselect(
    "Appliance ID",
    options=sorted(df["appliance_id"].dropna().unique()) if "appliance_id" in df else [],
    default=None,
)


# Apply filters
t0 = pd.Timestamp(date_range[0]) if len(date_range) >= 1 else min_dt
t1 = (
    pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)
    if len(date_range) >= 2
    else max_dt
)
mask = (df["timestamp"] >= t0) & (df["timestamp"] < t1)

if selected_rooms and "room_id" in df:
    mask = mask & df["room_id"].isin(selected_rooms)
if selected_appliances and "appliance_id" in df:
    mask = mask & df["appliance_id"].isin(selected_appliances)

df_f = df.loc[mask].copy()

if df_f.empty:
    st.warning("No data matches the selected filters.")
    st.stop()

# ===============================
# Top metrics cards
# ===============================

today = datetime.now().date()
today_mask = df_f["date"] == today
today_cons = df_f[today_mask]["power_kwh"].sum() if today in df_f["date"].values else 0.0

# Dummy "tomorrow prediction" for now (placeholder)
tomorrow_pred = today_cons * 1.1  # simplistic 10% growth

total_anomalies = df_f["is_anomaly"].eq(-1).sum()
total_cost = df_f["power_kwh"].sum() * Settings.TARIFF_RATE

cols = st.columns(4)
with cols[0]:
    st.metric("Today’s consumption (kWh)", f"{today_cons:.2f}")
with cols[1]:
    st.metric("Tomorrow’s predicted (kWh)", f"{tomorrow_pred:.2f}")
with cols[2]:
    st.metric("Anomalies detected", total_anomalies)
with cols[3]:
    st.metric("Estimated cost (₹)", f"{total_cost:.2f}")


# ===============================
# Tab layout
# ===============================

tabs = st.tabs([
    "Overview",
    "Predictions",
    "Insights",
    "Anomalies",
    "Recommendations",
])


# ===============================
# Tab 1: Overview
# ===============================

with tabs[0]:
    st.subheader("Overview")

    # Daily trend
    df_daily = (
        df_f.set_index("timestamp")
        .resample("D")["power_kwh"]
        .sum()
        .reset_index()
    )

    fig_daily = px.line(
        df_daily,
        x="timestamp",
        y="power_kwh",
        title="Daily energy consumption",
        labels={"power_kwh": "Energy (kWh)", "timestamp": "Date"},
        markers=True,
    )
    st.plotly_chart(fig_daily, use_container_width=True)

    # Weekly heatmap (day of week vs hour)
    df_h = df_f.copy()
    df_h["day_of_week"] = df_f["timestamp"].dt.dayofweek
    df_h["day_name"] = df_f["timestamp"].dt.strftime("%A")
    df_h["hour"] = df_f["timestamp"].dt.hour

    wb = df_h.groupby(["day_of_week", "hour"])["power_kwh"].sum().reset_index()
    wb = wb.pivot(index="day_name", columns="hour", values="power_kwh").fillna(0)

    wb = wb.reindex(
        index=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    )

    fig_heat = px.imshow(
        wb,
        title="Weekly heatmap (kWh)",
        labels={"x": "Hour of day", "y": "Day of week", "color": "kWh"},
    )
    st.plotly_chart(fig_heat, use_container_width=True)


# ===============================
# Tab 2: Predictions (stub)
# ===============================

with tabs[1]:
    st.subheader("Predictions")

    st.write("Placeholder: LSTM / Prophet model not yet implemented in this snippet.")

    # Example fake 24h forecast
    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    h = pd.date_range(start=now, periods=24, freq="H")
    vals = np.random.uniform(0.1, 0.5, size=24) * 1.5
    vals[:8] *= 0.7  # lower nights
    vals[-4:] *= 1.3  # higher evening

    fig_pred = go.Figure(
        data=go.Bar(x=h.strftime("%H:%M"), y=vals, name="Predicted kWh"),
    )
    fig_pred.update_layout(
        title="Next 24‑hour forecast (stub data)",
        xaxis_title="Time",
        yaxis_title="Energy (kWh)",
        xaxis_tickangle=-45,
    )
    st.plotly_chart(fig_pred, use_container_width=True)

    # 7‑day line (fake)
    d7 = pd.date_range(start=now.date(), periods=7)
    levels = np.random.uniform(0.8, 1.2, 7) * today_cons
    fig_7d = px.line(
        x=d7,
        y=levels,
        labels={"x": "Date", "y": "Total kWh"},
        title="7‑day forecast (stub)",
        markers=True,
    )
    st.plotly_chart(fig_7d, use_container_width=True)


# ===============================
# Tab 3: Insights
# ===============================

with tabs[2]:
    st.subheader("Insights")

    # Peak hours
    df_h = df_f.copy()
    df_h["hour"] = df_f["timestamp"].dt.hour
    by_hour = df_h.groupby("hour")["power_kwh"].sum()
    peak_hour = by_hour.idxmax()
    st.write("### Peak hours")
    st.write(f"**Highest total consumption hour: {peak_hour:02d}:00 – {peak_hour+1:02d}:00**")

    fig_bar = px.bar(
        by_hour,
        x=by_hour.index,
        y="power_kwh",
        title="Hourly consumption",
        labels={"x": "Hour", "y": "kWh"},
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Correlations
    cols_corr = ["power_kwh", "voltage", "current", "temperature", "humidity", "occupancy"]
    cols_exist = [c for c in cols_corr if c in df_f.columns]
    if len(cols_exist) >= 2:
        corr = df_f[cols_exist].corr()
        fig_corr = px.imshow(
            corr,
            title="Correlation matrix",
            labels={"color": "Correlation"},
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    # Appliance / room breakdown (if columns exist)
    if "appliance_id" in df_f.columns:
        by_appliance = df_f.groupby("appliance_id")["power_kwh"].sum()
        fig_appl = px.pie(
            values=by_appliance.values,
            names=by_appliance.index,
            title="Appliance‑wise consumption",
        )
        st.plotly_chart(fig_appl, use_container_width=True)

    if "room_id" in df_f.columns:
        by_room = df_f.groupby("room_id")["power_kwh"].sum()
        fig_room = px.pie(
            values=by_room.values,
            names=by_room.index,
            title="Room‑wise consumption",
        )
        st.plotly_chart(fig_room, use_container_width=True)

    # Cost breakdown
    st.write("### Cost breakdown")
    total_kwh = df_f["power_kwh"].sum()
    cost = total_kwh * Settings.TARIFF_RATE
    st.write(f"**Total kWh:** {total_kwh:.2f}")
    st.write(f"**Total cost (₹):** {cost:.2f}")


# ===============================
# Tab 4: Anomalies
# ===============================

with tabs[3]:
    st.subheader("Anomalies")

    n_anom = df_f["is_anomaly"].eq(-1).sum()
    st.write(f"Detected anomalies: **{n_anom}** records")

    top_anom = get_top_anomalies(df_f, n=10, sort_by="anomaly_score")

    if top_anom.empty:
        st.info("No strong anomalies found in this time range.")
    else:
        st.dataframe(
            top_anom[
                [
                    "timestamp",
                    "power_kwh",
                    "temperature",
                    "humidity",
                    "room_id",
                    "appliance_id",
                    "anomaly_score",
                ]
            ],
            use_container_width=True,
        )


# ===============================
# Tab 5: Recommendations
# ===============================

with tabs[4]:
    st.subheader("Recommendations")

    st.write("""
    **Rule‑based energy optimization suggestions:**
    - **Shift AC usage to off‑peak hours** (e.g., outside 18:00–23:00) to reduce strain on the grid and potentially save ~15–25% in off‑peak‑discount plans.
    - **Turn off lighting** in unoccupied rooms; occupancy‑based control can cut lighting load by 10–30%.
    - **Group appliance usage** (e.g., laundry, dishwasher, EV charging) into fewer high‑consumption windows so the system can better manage peaks.
    - **Invest in occupancy / motion sensors** to automatically switch off HVAC and lights when rooms are empty.
    - **Monitor high‑anomaly timestamps** above and investigate possible equipment faults or phantom loads.
    """)