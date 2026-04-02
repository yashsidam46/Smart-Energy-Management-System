# ⚡ Smart Energy Management System (SEMS)

## 📌 Overview

The **Smart Energy Management System (SEMS)** is a data-driven web application that monitors, analyzes, and predicts energy consumption. It helps users understand usage patterns, detect anomalies, and optimize energy efficiency.

This project integrates:

* 📊 Data Visualization (Streamlit)
* 🗄️ Database Management (SQLAlchemy)
* 🤖 Machine Learning (Isolation Forest, LSTM)
* 📈 Forecasting (LSTM & Prophet)

---

## 🚀 Features

### 🔹 1. Dashboard

* Daily energy consumption trends
* Weekly heatmap visualization
* Real-time metrics (consumption, cost, anomalies)

### 🔹 2. Anomaly Detection

* Detects unusual energy spikes
* Uses **Isolation Forest algorithm**
* Highlights abnormal readings

### 🔹 3. Insights & Analytics

* Peak hour detection
* Appliance-wise and room-wise usage
* Correlation analysis
* Cost estimation

### 🔹 4. Energy Forecasting

* 🔮 LSTM-based time series prediction
* 📊 Prophet-based forecasting
* Next 24 hours and 7-day predictions

### 🔹 5. Smart Recommendations

* Suggests energy-saving strategies
* Identifies inefficient usage patterns

---

## 🏗️ Project Structure

```
SEMS/
│
├── app.py                # Main Streamlit dashboard
├── config.py             # Configuration settings
├── database.py           # Database management (SQLAlchemy)
├── anomaly.py            # Anomaly detection (Isolation Forest)
├── models.py             # LSTM model
├── prediction.py         # Forecasting logic
├── insights.py           # Data analysis functions
│
├── sems_sample.csv       # Sample dataset
├── requirements.txt      # Dependencies
├── .env                  # Environment variables
```

---



## ▶️ Run the Application

```
streamlit run app.py
```

Open in browser:

```
http://localhost:8501
```

---

## 🧠 Technologies Used

* **Frontend:** Streamlit
* **Backend:** Python
* **Database:** SQLite / PostgreSQL
* **Libraries:**

  * Pandas, NumPy
  * Plotly
  * Scikit-learn
  * TensorFlow (LSTM)
  * Prophet

---

## 📊 Machine Learning Models

### 🔹 Isolation Forest

* Used for anomaly detection
* Identifies abnormal energy consumption

### 🔹 LSTM (Long Short-Term Memory)

* Time-series forecasting
* Predicts future energy usage

### 🔹 Prophet

* Statistical forecasting model
* Provides trend and seasonal predictions

---

## 💡 Future Improvements

* IoT sensor integration
* Real-time data streaming
* Email/SMS alerts for anomalies
* Cloud deployment (AWS / Render)
* Mobile app integration

---

## 👨‍💻 Author

**Yash Sidam**
Artificial Intelligence & Data Science Student

---

## 📌 Conclusion

SEMS demonstrates how **AI + Data Analytics** can be used to optimize energy usage, reduce costs, and build smarter systems.

---
