import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from zoneinfo import ZoneInfo
from sklearn.ensemble import GradientBoostingRegressor
from streamlit_autorefresh import st_autorefresh

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Real-Time Air Quality Monitoring",
    page_icon="🌍",
    layout="wide"
)

# =========================================================
# AUTO REFRESH - EVERY 10 MINUTES
# =========================================================

st_autorefresh(
    interval=10 * 60 * 1000,
    key="aqi_auto_refresh"
)

# =========================================================
# TAMIL NADU CITIES
# =========================================================

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

# =========================================================
# AQI CATEGORY
# =========================================================

def get_aqi_category(aqi):

    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"


def get_health_risk(aqi):

    if aqi <= 50:
        return "Low"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "High"
    elif aqi <= 200:
        return "Very High"
    else:
        return "Severe"


def get_population(aqi):

    if aqi <= 50:
        return "General population"
    elif aqi <= 100:
        return "Sensitive groups"
    elif aqi <= 150:
        return "Children, elderly and sensitive groups"
    elif aqi <= 200:
        return "Entire population may be affected"
    else:
        return "Entire population is at serious risk"


def get_precaution(aqi):

    if aqi <= 50:
        return "Air quality is satisfactory. Normal outdoor activities are safe."

    elif aqi <= 100:
        return "Sensitive people should consider reducing prolonged outdoor activities."

    elif aqi <= 150:
        return "Sensitive groups should reduce prolonged outdoor exertion."

    elif aqi <= 200:
        return "Everyone should reduce prolonged outdoor activities. Sensitive groups should stay indoors when possible."

    else:
        return "Avoid outdoor activities. Keep windows closed and use protective measures when necessary."


# =========================================================
# GET CURRENT IST TIME
# =========================================================

def current_ist_time():

    return datetime.now(
        ZoneInfo("Asia/Kolkata")
    ).strftime("%d %b %Y, %I:%M:%S %p IST")


# =========================================================
# FETCH REAL-TIME AIR QUALITY
# =========================================================

@st.cache_data(ttl=600)
def get_air_quality(city):

    latitude, longitude = cities[city]

    url = "https://air-quality-api.open-meteo.com/v1/air-quality"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": (
            "us_aqi,"
            "pm2_5,"
            "pm10,"
            "carbon_monoxide,"
            "nitrogen_dioxide,"
            "sulphur_dioxide,"
            "ozone"
        ),
        "hourly": "us_aqi,pm2_5,pm10",
        "timezone": "Asia/Kolkata",
        "past_days": 1,
        "forecast_days": 1
    }

    response = requests.get(
        url,
        params=params,
        timeout=20
    )

    response.raise_for_status()

    return response.json()


# =========================================================
# TRAIN GRADIENT BOOSTING MODEL
# =========================================================

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

    aqi = (
        pm25 * 1.45
        + pm10 * 0.18
        + co * 0.025
        + no2 * 0.55
        + so2 * 0.18
        + o3 * 0.30
    )

    aqi = np.clip(
        aqi + np.random.normal(0, 4, n),
        0,
        500
    )

    X = pd.DataFrame({
        "PM2.5": pm25,
        "PM10": pm10,
        "CO": co,
        "NO2": no2,
        "SO2": so2,
        "O3": o3
    })

    y = aqi

    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    )

    model.fit(X, y)

    return model


# =========================================================
# HEADER
# =========================================================

st.title("🌍 Real-Time Air Quality Monitoring and AQI Prediction")

st.markdown(
    "### Tamil Nadu Air Quality Monitoring System"
)

st.info(
    "🔄 Live data automatically refreshes every 10 minutes."
)

st.caption(
    f"Last application update: {current_ist_time()}"
)

# =========================================================
# CITY SELECTION
# =========================================================

st.sidebar.header("📍 Location")

selected_city = st.sidebar.selectbox(
    "Select Tamil Nadu City",
    list(cities.keys())
)

st.sidebar.success(
    "Data Source: Open-Meteo Air Quality API"
)

# =========================================================
# FETCH DATA
# =========================================================

try:

    data = get_air_quality(selected_city)

    current = data["current"]

    aqi = float(current.get("us_aqi", 0))
    pm25 = float(current.get("pm2_5", 0))
    pm10 = float(current.get("pm10", 0))
    co = float(current.get("carbon_monoxide", 0))
    no2 = float(current.get("nitrogen_dioxide", 0))
    so2 = float(current.get("sulphur_dioxide", 0))
    o3 = float(current.get("ozone", 0))

    category = get_aqi_category(aqi)

except Exception as e:

    st.error(
        f"Unable to fetch live air-quality data: {e}"
    )

    st.stop()


# =========================================================
# MODULE 1
# =========================================================

st.header("Module 1: Real-Time Air Quality Data Acquisition and Monitoring")

st.write(
    f"Currently monitoring **{selected_city}, Tamil Nadu**."
)

# ---------------------------------------------------------
# AQI SUMMARY
# ---------------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Current AQI",
        f"{aqi:.0f}"
    )

with col2:

    st.metric(
        "AQI Category",
        category
    )

with col3:

    st.metric(
        "PM2.5",
        f"{pm25:.1f} µg/m³"
    )

with col4:

    st.metric(
        "PM10",
        f"{pm10:.1f} µg/m³"
    )

# ---------------------------------------------------------
# POLLUTANTS
# ---------------------------------------------------------

st.subheader("Air Quality Parameters")

p1, p2, p3, p4, p5, p6 = st.columns(6)

with p1:
    st.metric(
        "PM2.5",
        f"{pm25:.1f}"
    )

with p2:
    st.metric(
        "PM10",
        f"{pm10:.1f}"
    )

with p3:
    st.metric(
        "CO",
        f"{co:.1f}"
    )

with p4:
    st.metric(
        "NO₂",
        f"{no2:.1f}"
    )

with p5:
    st.metric(
        "SO₂",
        f"{so2:.1f}"
    )

with p6:
    st.metric(
        "O₃",
        f"{o3:.1f}"
    )

st.caption(
    "Pollutant concentrations are provided by the Open-Meteo Air Quality API."
)

# =========================================================
# HISTORICAL DATA
# =========================================================

st.subheader("📊 Historical Air Quality")

hourly = data.get("hourly", {})

if hourly:

    times = hourly.get("time", [])
    aqi_values = hourly.get("us_aqi", [])
    pm25_values = hourly.get("pm2_5", [])
    pm10_values = hourly.get("pm10", [])

    history = pd.DataFrame({
        "Time": times,
        "AQI": aqi_values,
        "PM2.5": pm25_values,
        "PM10": pm10_values
    })

    history["Time"] = pd.to_datetime(
        history["Time"]
    )

    history = history.dropna()

    if not history.empty:

        st.line_chart(
            history.set_index("Time")[
                ["AQI", "PM2.5", "PM10"]
            ]
        )

else:

    st.warning(
        "Historical data is not available."
    )


# =========================================================
# MODULE 2
# =========================================================

st.header(
    "Module 2: Machine Learning-Based AQI Prediction and Health Risk Assessment"
)

st.write(
    "The live pollutant values from Module 1 are passed to the machine-learning model."
)

# =========================================================
# MODEL PREDICTION
# =========================================================

model = train_model()

input_data = pd.DataFrame({

    "PM2.5": [pm25],
    "PM10": [pm10],
    "CO": [co],
    "NO2": [no2],
    "SO2": [so2],
    "O3": [o3]

})

predicted_aqi = float(
    model.predict(input_data)[0]
)

predicted_aqi = np.clip(
    predicted_aqi,
    0,
    500
)

prediction_category = get_aqi_category(
    predicted_aqi
)

difference = predicted_aqi - aqi

# =========================================================
# PREDICTION RESULTS
# =========================================================

st.subheader("🤖 AQI Prediction")

m1, m2, m3 = st.columns(3)

with m1:

    st.metric(
        "Actual AQI",
        f"{aqi:.0f}"
    )

with m2:

    st.metric(
        "Predicted AQI",
        f"{predicted_aqi:.0f}"
    )

with m3:

    st.metric(
        "Prediction Difference",
        f"{difference:+.1f}"
    )

# =========================================================
# HEALTH RISK
# =========================================================

risk = get_health_risk(
    predicted_aqi
)

population = get_population(
    predicted_aqi
)

precaution = get_precaution(
    predicted_aqi
)

st.subheader("🏥 Health Risk Assessment")

h1, h2 = st.columns(2)

with h1:

    st.metric(
        "Health Risk",
        risk
    )

with h2:

    st.metric(
        "Predicted AQI Category",
        prediction_category
    )

st.write(
    f"**Affected Population:** {population}"
)

st.write(
    f"**Health Precaution:** {precaution}"
)

# =========================================================
# ACTUAL VS PREDICTED CHART
# =========================================================

st.subheader("📈 Actual AQI vs Predicted AQI")

comparison = pd.DataFrame({

    "AQI Type": [
        "Actual AQI",
        "Predicted AQI"
    ],

    "AQI": [
        aqi,
        predicted_aqi
    ]

})

st.bar_chart(
    comparison.set_index("AQI Type")
)

# =========================================================
# SYSTEM WORKFLOW
# =========================================================

st.header("⚙️ System Workflow")

st.markdown("""
**Tamil Nadu City Selection**

↓

**Open-Meteo Air Quality API**

↓

**Real-Time Pollutant Data Acquisition**

↓

**AQI and Pollutant Monitoring**

↓

**PM2.5, PM10, CO, NO₂, SO₂, O₃**

↓

**Machine Learning Model**

↓

**AQI Prediction**

↓

**Health Risk Assessment**

↓

**Affected Population and Precautions**
""")

# =========================================================
# MODEL INFORMATION
# =========================================================

st.header("🧠 Machine Learning Model")

st.write(
    "**Algorithm:** Gradient Boosting Regressor"
)

st.write(
    "**Estimators:** 100"
)

st.write(
    "**Learning Rate:** 0.1"
)

st.write(
    "**Maximum Depth:** 6"
)

st.write(
    "**Random State:** 42"
)

# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Real-Time Air Quality Monitoring and AQI Prediction using Machine Learning"
)

st.caption(
    "Data source: Open-Meteo Air Quality API | Location scope: Tamil Nadu"
)

st.caption(
    f"Current IST Time: {current_ist_time()}"
)
