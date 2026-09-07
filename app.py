import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from datetime import datetime

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Real-Time Air Quality Monitoring",
    page_icon="🌍",
    layout="wide"
)

# ---------------- TITLE ----------------
st.title("🌍 Real-Time Air Quality Monitoring and AQI Prediction")
st.markdown(
    "**Real-Time Air Quality Data Acquisition, ML-Based AQI Prediction "
    "and Health Risk Assessment**"
)

# ---------------- TAMIL NADU LOCATIONS ----------------
cities = {
    "Chennai": (13.0827, 80.2707),
    "Coimbatore": (11.0168, 76.9558),
    "Madurai": (9.9252, 78.1198),
    "Tiruchirappalli": (10.7905, 78.7047),
    "Salem": (11.6643, 78.1460),
    "Tirunelveli": (8.7139, 77.7567),
    "Erode": (11.3410, 77.7172),
    "Vellore": (12.9165, 79.1325),
    "Thoothukudi": (8.7642, 78.1348),
    "Dindigul": (10.3673, 77.9803),
    "Thanjavur": (10.7870, 79.1378),
    "Kanchipuram": (12.8342, 79.7036),
    "Tiruppur": (11.1085, 77.3411),
    "Nagercoil": (8.1833, 77.4119),
    "Cuddalore": (11.7480, 79.7714)
}

# ---------------- AQI CATEGORY ----------------
def aqi_category(aqi):
    if aqi <= 50:
        return "Good", "Low"
    elif aqi <= 100:
        return "Moderate", "Moderate"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "High"
    elif aqi <= 200:
        return "Unhealthy", "High"
    elif aqi <= 300:
        return "Very Unhealthy", "Very High"
    else:
        return "Hazardous", "Severe"

# ---------------- HEALTH ADVICE ----------------
def health_advice(aqi):
    if aqi <= 50:
        return "Air quality is good. Normal outdoor activities are safe."
    elif aqi <= 100:
        return "Sensitive people should consider reducing prolonged outdoor activity."
    elif aqi <= 150:
        return "Sensitive groups should reduce prolonged outdoor activity."
    elif aqi <= 200:
        return "Everyone should reduce prolonged outdoor activity. Sensitive groups should stay indoors when possible."
    elif aqi <= 300:
        return "Avoid prolonged outdoor activity. Wear a suitable mask when going outside."
    else:
        return "Avoid outdoor activity and remain indoors as much as possible."

# ---------------- ML MODEL ----------------
@st.cache_resource
def train_model():

    np.random.seed(42)

    n = 1500

    pm25 = np.random.uniform(5, 150, n)
    pm10 = np.random.uniform(10, 250, n)
    co = np.random.uniform(100, 1500, n)
    no2 = np.random.uniform(5, 100, n)
    so2 = np.random.uniform(2, 80, n)
    o3 = np.random.uniform(10, 220, n)

    # EPA-style pollutant influence approximation
    aqi = (
        pm25 * 1.45
        + pm10 * 0.18
        + co * 0.025
        + no2 * 0.55
        + so2 * 0.18
        + o3 * 0.30
    )

    aqi = np.clip(aqi + np.random.normal(0, 4, n), 0, 500)

    X = pd.DataFrame({
        "PM2.5": pm25,
        "PM10": pm10,
        "CO": co,
        "NO2": no2,
        "SO2": so2,
        "O3": o3
    })

    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    )

    model.fit(X, aqi)

    return model

model = train_model()

# ---------------- API FUNCTION ----------------
def get_air_quality(city):

    lat, lon = cities[city]

    url = (
        "https://air-quality-api.open-meteo.com/v1/air-quality"
        f"?latitude={lat}&longitude={lon}"
        "&current=us_aqi,pm2_5,pm10,carbon_monoxide,"
        "nitrogen_dioxide,sulphur_dioxide,ozone"
        "&hourly=us_aqi,pm2_5,pm10"
        "&timezone=auto"
    )

    response = requests.get(url, timeout=15)

    if response.status_code != 200:
        return None

    data = response.json()

    current = data.get("current", {})
    hourly = data.get("hourly", {})

    return {
        "aqi": current.get("us_aqi", 0),
        "pm25": current.get("pm2_5", 0),
        "pm10": current.get("pm10", 0),
        "co": current.get("carbon_monoxide", 0),
        "no2": current.get("nitrogen_dioxide", 0),
        "so2": current.get("sulphur_dioxide", 0),
        "o3": current.get("ozone", 0),
        "time": current.get("time", ""),
        "hourly_time": hourly.get("time", []),
        "hourly_aqi": hourly.get("us_aqi", []),
        "hourly_pm25": hourly.get("pm2_5", []),
        "hourly_pm10": hourly.get("pm10", [])
    }

# ---------------- SIDEBAR ----------------
st.sidebar.header("📍 Location")

city = st.sidebar.selectbox(
    "Select Tamil Nadu City",
    list(cities.keys())
)

st.sidebar.info(
    "Live air-quality data is obtained using the Open-Meteo Air Quality API."
)

# ---------------- FETCH DATA ----------------
if st.sidebar.button("🔄 Get Live Air Quality", use_container_width=True):

    st.session_state["data"] = get_air_quality(city)
    st.session_state["city"] = city

if "data" not in st.session_state:
    st.info("👈 Select a Tamil Nadu city and click **Get Live Air Quality**.")
    st.stop()

data = st.session_state["data"]
city = st.session_state["city"]

# ---------------- MODULE 1 ----------------
st.header("📡 Module 1: Real-Time Air Quality Monitoring")

actual_aqi = data["aqi"]

category, risk = aqi_category(actual_aqi)

st.subheader(f"📍 {city}, Tamil Nadu")

col1, col2, col3, col4 = st.columns(4)

col1.metric("US AQI", round(actual_aqi))
col2.metric("PM2.5", f"{data['pm25']:.1f} µg/m³")
col3.metric("PM10", f"{data['pm10']:.1f} µg/m³")
col4.metric("CO", f"{data['co']:.1f} µg/m³")

col5, col6, col7, col8 = st.columns(4)

col5.metric("NO₂", f"{data['no2']:.1f} µg/m³")
col6.metric("SO₂", f"{data['so2']:.1f} µg/m³")
col7.metric("O₃", f"{data['o3']:.1f} µg/m³")
col8.metric("Risk Level", risk)

st.success(f"### AQI Category: {category}")

# ---------------- MODULE 1 CHART ----------------
st.subheader("📊 Historical Air Quality Trend")

if len(data["hourly_time"]) > 0:

    chart_df = pd.DataFrame({
        "Time": data["hourly_time"],
        "AQI": data["hourly_aqi"],
        "PM2.5": data["hourly_pm25"],
        "PM10": data["hourly_pm10"]
    })

    chart_df["Time"] = pd.to_datetime(chart_df["Time"])
    chart_df = chart_df.tail(24)

    st.line_chart(
        chart_df.set_index("Time")[["AQI", "PM2.5", "PM10"]]
    )

# ---------------- MODULE 2 ----------------
st.header("🤖 Module 2: ML-Based AQI Prediction")

input_data = pd.DataFrame([{
    "PM2.5": data["pm25"],
    "PM10": data["pm10"],
    "CO": data["co"],
    "NO2": data["no2"],
    "SO2": data["so2"],
    "O3": data["o3"]
}])

predicted_aqi = float(model.predict(input_data)[0])
predicted_aqi = max(0, min(500, predicted_aqi))

pred_category, predicted_risk = aqi_category(predicted_aqi)

difference = abs(actual_aqi - predicted_aqi)

c1, c2, c3 = st.columns(3)

c1.metric(
    "Actual AQI",
    f"{actual_aqi:.0f}"
)

c2.metric(
    "Predicted AQI",
    f"{predicted_aqi:.0f}"
)

c3.metric(
    "Prediction Difference",
    f"{difference:.1f}"
)

st.success(
    f"### Predicted AQI Category: {pred_category}"
)

# ---------------- HEALTH RISK ----------------
st.header("🏥 Health Risk Assessment")

h1, h2 = st.columns(2)

h1.metric(
    "Health Risk",
    predicted_risk
)

if predicted_aqi <= 50:
    population = "General population"
elif predicted_aqi <= 100:
    population = "Sensitive individuals should take care"
elif predicted_aqi <= 150:
    population = "Children, elderly and sensitive groups"
elif predicted_aqi <= 200:
    population = "Everyone may experience health effects"
else:
    population = "Entire population is susceptible"

h2.metric(
    "Affected Population",
    population
)

st.warning(
    f"**Recommended Precaution:** {health_advice(predicted_aqi)}"
)

# ---------------- COMPARISON ----------------
st.header("📈 Actual vs ML Predicted AQI")

comparison = pd.DataFrame({
    "AQI Type": ["Actual AQI", "ML Predicted AQI"],
    "AQI": [actual_aqi, predicted_aqi]
})

st.bar_chart(
    comparison.set_index("AQI Type")
)

# ---------------- PROJECT INFO ----------------
st.divider()

st.subheader("⚙️ System Workflow")

st.markdown("""
**Air Quality API → Live Pollutant Data → AQI Monitoring → 
Machine Learning Prediction → Health Risk Assessment → 
Health Recommendations**
""")

st.caption(
    f"Last updated: {data['time']} | Model: Gradient Boosting Regressor"
)
