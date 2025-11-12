import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
from sklearn.preprocessing import LabelEncoder

# -------------------------------------------------------
# Page Setup
st.set_page_config(page_title="AirVision - AQI Prediction", layout="wide")
st.title("AirVision: Interactive AQI Prediction & Analysis Dashboard")
st.markdown("#### Powered by Data Mining + Machine Learning + Streamlit ")

# -------------------------------------------------------
# Custom CSS for DARK MODE
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to right, #2f4f4f, #00008b);
        color: white;
    }
    .sidebar .sidebar-content {
        background-color: rgba(0, 0, 0, 0.6);
        color: white;
    }
    h1, h2, h3, h4, h5, h6, p, div, label {
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# Data Loading
st.sidebar.header("Dataset")
st.sidebar.info("Using default dataset: `city_day.csv`")

try:
    data = pd.read_csv("city_day.csv")
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
except FileNotFoundError:
    st.error("`city_day.csv` not found! Place it in the project folder.")
    st.stop()

# -------------------------------------------------------
# Load Model, Scaler, Encoder
try:
    model = pickle.load(open("model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    try:
        le = pickle.load(open("encoder.pkl", "rb"))
    except:
        le = LabelEncoder()
        le.fit(data["City"])
except FileNotFoundError:
    st.error("model.pkl or scaler.pkl not found! Run main.py to generate them.")
    st.stop()

# -------------------------------------------------------
# Sidebar Navigation
st.sidebar.header("Explore & Predict")
page = st.sidebar.radio("Choose Section:", ["Data Exploration", "AQI Prediction"])

# -------------------------------------------------------
if page == "Data Exploration":
    st.subheader("Explore Air Quality Data")
    tabs = st.tabs(["Trends & Charts", "Heatmap & Distributions", "Top Polluted Cities", "Map View"])

    # ----- TRENDS & CHARTS -----
    with tabs[0]:
        selected_city = st.selectbox("Select City:", sorted(data["City"].unique()))
        city_data = data[data["City"] == selected_city].sort_values("Date")

        fig_line = px.line(city_data, x="Date", y="AQI", title=f"AQI Trend for {selected_city}", markers=True)
        st.plotly_chart(fig_line, use_container_width=True)

        pollutants = ["PM2.5", "PM10", "NO", "NO2", "SO2", "CO", "O3"]
        avg_pollution = city_data[pollutants].mean().reset_index()
        avg_pollution.columns = ["Pollutant", "Average Concentration"]
        fig_bar = px.bar(avg_pollution, x="Pollutant", y="Average Concentration",
                         title=f"Average Pollutant Levels in {selected_city}")
        st.plotly_chart(fig_bar, use_container_width=True)

    # ----- HEATMAP -----
    with tabs[1]:
        st.subheader("Correlation Heatmap")
        corr = data.select_dtypes(include=np.number).corr()
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.8)
        st.pyplot(fig)

    # ----- TOP CITIES -----
    with tabs[2]:
        latest_date = data["Date"].max()
        latest_data = data[data["Date"] == latest_date].sort_values("AQI", ascending=False).head(5)
        st.table(latest_data[["City", "AQI", "Date"]])
        st.markdown("### Insight: Highest AQI indicates worst air quality.")

    # ----- MAP VIEW -----
    with tabs[3]:
        city_coords = {
            "Gurugram": {"lat": 28.4595,"lon": 77.0266},
            "Ahmedabad": {"lat": 23.0225, "lon": 72.5714},
            "Delhi": {"lat": 28.6139, "lon": 77.2090},
            "Mumbai": {"lat": 19.0760, "lon": 72.8777},
            "Ernakulam": {"lat": 9.9816, "lon": 76.2999}
        }
        map_data = pd.DataFrame([
            {"City": city, "lat": coords["lat"], "lon": coords["lon"],
             "AQI": data[data["City"] == city]["AQI"].mean()}
            for city, coords in city_coords.items()
        ])
        st.map(map_data)

    st.markdown("### Raw Data Sample")
    st.dataframe(data.sample(n=20, random_state=None).reset_index(drop=True))

# -------------------------------------------------------
elif page == "AQI Prediction":
    st.subheader("Predict Air Quality Index (AQI)")
    tabs = st.tabs(["Input & Predict", "Predicted vs Actual"])

    # ----- TAB 1: Input & Predict -----
    with tabs[0]:
        city = st.selectbox("Select City:", sorted(data["City"].unique()))
        pm25 = st.slider("PM2.5", 0.0, 500.0, 28.0)
        pm10 = st.slider("PM10", 0.0, 500.0, 60.0)
        no = st.slider("NO", 0.0, 100.0, 10.0)
        no2 = st.slider("NO2", 0.0, 100.0, 20.0)
        nox = st.slider("NOx", 0.0, 200.0, 32.0)
        nh3 = st.slider("NH3", 0.0, 100.0, 23.0)
        co = st.slider("CO", 0.0, 10.0, 1.0)
        so2 = st.slider("SO2", 0.0, 100.0, 15.0)
        o3 = st.slider("O3", 0.0, 200.0, 50.0)
        benzene = st.slider("Benzene", 0.0, 50.0, 3.0)
        toluene = st.slider("Toluene", 0.0, 100.0, 10.0)

        city_encoded = le.transform([city])[0]

        # exact same features as training
        feature_names = ['City', 'PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene']
        input_data = pd.DataFrame([[city_encoded, pm25, pm10, no, no2, nox, nh3, co, so2, o3, benzene, toluene]],
                                  columns=feature_names)

        input_scaled = scaler.transform(input_data)
        pred_aqi = model.predict(input_scaled)[0]

        st.metric("Predicted AQI", f"{pred_aqi:.2f}")

        if pred_aqi <= 50:
            st.success("Air Quality: Good")
        elif pred_aqi <= 100:
            st.info("Air Quality: Moderate")
        elif pred_aqi <= 200:
            st.warning("Air Quality: Unhealthy for Sensitive Groups")
        else:
            st.error("Air Quality: Hazardous")

    # ----- TAB 2: Predicted vs Actual -----
    with tabs[1]:
        st.markdown("### Model Performance: Predicted vs Actual AQI")
        sample = data.sample(30, random_state=42)
        sample["City"] = le.transform(sample["City"])

        X_sample = sample[['City', 'PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene']]
        X_sample_scaled = scaler.transform(X_sample)
        y_true = sample['AQI']
        y_pred = model.predict(X_sample_scaled)

        perf_df = pd.DataFrame({'Actual AQI': y_true, 'Predicted AQI': y_pred})
        st.line_chart(perf_df)

        st.markdown(f"**Model R² (from training): ~0.91** - strong prediction consistency.")
