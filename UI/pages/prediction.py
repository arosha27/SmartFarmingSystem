import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt

API_URL = "http://127.0.0.1:8000"  # FastAPI backend URL

st.title("Crop Disease Risk Prediction")

st.markdown("Enter soil and weather parameters to predict crop disease risk.")

# Input form
with st.form("prediction_form"):
    soil_pH = st.number_input("Soil pH (4.99 - 7.96)", 4.99, 7.96, step=0.01)
    soil_moisture = st.number_input("Soil Moisture (0.056 - 0.676)", 0.056, 0.676, step=0.01)
    soil_temp = st.number_input("Soil Temp (13.45 - 35.36 °C)", 13.45, 35.36, step=0.1)
    nitrogen = st.number_input("Nitrogen (10.6 - 105.29)", 10.6, 105.29, step=0.1)
    rainfall = st.number_input("Rainfall (-3.02 - 211.97 mm)", -3.02, 211.97, step=0.1)
    humidity = st.number_input("Humidity (21.04 - 105.61 %)", 21.04, 105.61, step=0.1)
    air_temp = st.number_input("Air Temp (10.28 - 38.78 °C)", 10.28, 38.78, step=0.1)
    wind = st.number_input("Wind (-3.76 - 24.35 km/h)", -3.76, 24.35, step=0.1)

    submitted = st.form_submit_button("Predict")

if submitted:
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

    # Call FastAPI predict endpoint
    response = requests.post(f"{API_URL}/predict", json=input_data)
    explain_response = requests.post(f"{API_URL}/explain", json=input_data)

    if response.status_code == 200 and explain_response.status_code == 200:
        pred_data = response.json()
        exp_data = explain_response.json()

        st.subheader("Prediction Result")
        st.success(f"**Risk Level:** {pred_data['prediction_label']}")

        #Show probabilities
        st.subheader("Prediction Probabilities")
        probs = pred_data["probabilities"]
        labels = ["Low Risk", "Medium Risk", "High Risk"]
        df_probs = pd.DataFrame({"Risk Level": labels, "Probability": probs})
        # st.bar_chart(df_probs.set_index("Risk Level") , colors=["green" , "yellow" , "red"])
        
        plt.figure(figsize=(6,4))
        plt.bar(df_probs["Risk Level"], df_probs["Probability"], color=["green", "yellow", "red"])
        plt.xlabel("Risk Level")
        plt.ylabel("Probability")
        plt.title("Prediction Probabilities")
        st.pyplot(plt)


        # SHAP feature contributions
        st.subheader("Feature Contributions")
        feature_importance = pd.DataFrame(exp_data["feature_importance"])
        feature_importance = feature_importance.sort_values("shap_value", ascending=True)

        st.dataframe(feature_importance)

        # Plot feature contributions
        plt.figure(figsize=(8,5))
        colors = feature_importance["shap_value"].apply(lambda x: "green" if x > 0 else "red")
        plt.barh(feature_importance["feature"], feature_importance["shap_value"], color=colors)
        plt.xlabel("SHAP Value (Impact on Prediction)")
        plt.ylabel("Feature")
        plt.title("Feature Importance for Prediction")
        st.pyplot(plt)
        
        
        

    else:
        st.error("❌ Error: Could not get prediction from API")
