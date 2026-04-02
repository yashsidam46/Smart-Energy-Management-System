import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
from pathlib import Path

# ===============================
# CONFIG
# ===============================

APP_NAME = "Smart Energy Management System ⚡"
TARIFF_RATE = 8  # ₹ per kWh

st.set_page_config(layout="wide", page_title=APP_NAME)

# ===============================
# LOAD DATA
# ===============================

@st.cache_data
def load_data():
    file_path = Path("energy_data.csv")

    df = pd.read_csv(file_path)

    df.columns = df.columns.str.strip()

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    df["timestamp"] = df["timestamp"].astype("datetime64[ns]")

    df = df.sort_values("timestamp")

    df["date"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.hour

    return df


df = load_data()

if df.empty:
    st.stop()

# ===============================
# SIDEBAR
# ===============================

st.sidebar.title("Filters")

min_date = df["timestamp"].min().date()
max_date = df["timestamp"].max().date()

date_range = st.sidebar.date_input(
    "Select Date Range",
    [min_date, max_date]
)

# Safety if only one date selected
if len(date_range) == 1:
    start_date = date_range[0]
    end_date = date_range[0]
else:
    start_date, end_date = date_range

df_f = df[
    (df["timestamp"] >= pd.Timestamp(start_date)) &
    (df["timestamp"] <= pd.Timestamp(end_date) + pd.Timedelta(days=1))
]

# ===============================
# METRICS
# ===============================

today = datetime.now().date()
today_consumption = df_f[df_f["date"] == today]["power_kwh"].sum()

total_energy = df_f["power_kwh"].sum()
total_cost = total_energy * TARIFF_RATE

col1, col2, col3 = st.columns(3)

col1.metric("Today's Consumption", f"{today_consumption:.2f} kWh")
col2.metric("Total Energy", f"{total_energy:.2f} kWh")
col3.metric("Total Cost", f"₹{total_cost:.2f}")

# ===============================
# TABS
# ===============================

tabs = st.tabs(["Overview", "Insights", "Appliances", "Rooms"])

# ===============================
# OVERVIEW
# ===============================

with tabs[0]:
    st.subheader("Energy Usage Over Time")

    fig = px.line(df_f, x="timestamp", y="power_kwh")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Daily Consumption")

    daily = df_f.set_index("timestamp").resample("D")["power_kwh"].sum()
    st.line_chart(daily)

# ===============================
# INSIGHTS
# ===============================

with tabs[1]:
    st.subheader("Insights")

    hourly = df_f.groupby("hour")["power_kwh"].sum()

    if not hourly.empty:
        peak_hour = hourly.idxmax()
        st.write(f"🔝 Peak Usage Hour: **{peak_hour}:00**")
        st.bar_chart(hourly)
    else:
        st.warning("No data available for insights.")

    st.subheader("Correlation")

    corr = df_f.select_dtypes(include="number").corr()

    if not corr.empty:
        st.dataframe(corr)
    else:
        st.warning("Not enough data for correlation.")

# ===============================
# APPLIANCES
# ===============================

with tabs[2]:
    st.subheader("Appliance Consumption")

    if "appliance_id" in df_f.columns:
        app_usage = df_f.groupby("appliance_id")["power_kwh"].sum().sort_values()

        if not app_usage.empty:
            fig = px.bar(app_usage, orientation="h")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No appliance data available.")
    else:
        st.warning("Column 'appliance_id' not found.")

# ===============================
# ROOMS
# ===============================

with tabs[3]:
    st.subheader("Room Consumption")

    if "room_id" in df_f.columns:
        room_usage = df_f.groupby("room_id")["power_kwh"].sum()

        if not room_usage.empty:
            fig = px.pie(values=room_usage.values, names=room_usage.index)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No room data available.")
    else:
        st.warning("Column 'room_id' not found.")

# ===============================
# FOOTER
# ===============================

st.markdown("---")
st.caption("SEMS Project | Built with Streamlit ⚡")