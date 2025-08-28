import streamlit as st
import requests

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(page_title="Smart Farming Disease Risk", layout="centered")

st.title(" Smart Farming Disease Risk Prediction")
st.write("Enter soil and weather data to predict the **crop disease risk level**.")

# -----------------------------
# Input Fields (UI)
# -----------------------------
soil_pH = st.number_input("Soil pH", min_value=4.99, max_value=7.96, value=6.5, step=0.1)
soil_moisture = st.number_input("Soil Moisture (fraction)", min_value=0.056, max_value=0.676, value=0.3, step=0.01)
soil_temp = st.number_input("Soil Temperature (°C)", min_value=13.45, max_value=35.36, value=20.0, step=0.1)
nitrogen = st.number_input("Nitrogen (mg/kg)", min_value=10.6, max_value=105.29, value=50.0, step=0.5)
rainfall = st.number_input("Rainfall (mm)", min_value=-3.02, max_value=211.97, value=100.0, step=1.0)
humidity = st.number_input("Humidity (%)", min_value=21.04, max_value=105.61, value=60.0, step=1.0)
air_temp = st.number_input("Air Temperature (°C)", min_value=10.28, max_value=38.78, value=25.0, step=0.5)
wind = st.number_input("Wind Speed (km/h)", min_value=-3.76, max_value=24.35, value=5.0, step=0.5)

# -----------------------------
# API Call
# -----------------------------
if st.button(" Predict Disease Risk"):
    # Prepare input JSON
    input_data = {
        "soil_pH": soil_pH,
        "soil_moisture": soil_moisture,
        "soil_temp": soil_temp,
        "nitrogen": nitrogen,
        "rainfall": rainfall,
        "humidity": humidity,
        "air_temp": air_temp,
        "wind": wind
    }

    try:
        # Call FastAPI endpoint
        response = requests.post(" http://127.0.0.1:8000/predict", json=input_data)

        if response.status_code == 200:
            result = response.json()
            st.success(f"Prediction: **{result['prediction_label']}**")
            st.write(f"Prediction Code: {result['prediction_code']}")
            st.write("Probabilities:")
            st.json(result["probabilities"])
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        st.error(f"Could not connect to API. Error: {e}")
