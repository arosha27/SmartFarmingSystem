from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Annotated

import pickle
import numpy as np

############# Load the saved model and scaler ################
with open("E:/SmartFarmingSystem/models/xgboost_classifier.pickle", "rb") as f:
    model = pickle.load(f)

with open("E:/SmartFarmingSystem/models/scaler.pickle", "rb") as f:
    scaler = pickle.load(f)

############### Create the FastAPI object ##############
app = FastAPI(
    title="Smart Farming Disease Risk API",
    description="Predicts crop disease risk based on weather and soil data",
    version="1.0"
)

############ Create the Object of the FastAPI ###################

app = FastAPI()


############### Pydantic class For Data Validation (Type and Range) ######################
################### Define input schema with validation #################
class CropFeatures(BaseModel):
    soil_pH: Annotated[
        float,
        Field(..., ge=4.99, le=7.96, description="Soil pH level (Range: 4.99 - 7.96)")
    ]
    soil_moisture: Annotated[
        float,
        Field(..., ge=0.056, le=0.676, description="Soil moisture fraction (Range: 0.056 - 0.676)")
    ]
    soil_temp: Annotated[
        float,
        Field(..., ge=13.45, le=35.36, description="Soil temperature in °C (Range: 13.45 - 35.36)")
    ]
    nitrogen: Annotated[
        float,
        Field(..., ge=10.60, le=105.29, description="Nitrogen content mg/kg (Range: 10.60 - 105.29)")
    ]
    rainfall: Annotated[
        float,
        Field(..., ge=-3.02, le=211.97, description="Rainfall in mm (Range: -3.02 - 211.97)")
    ]
    humidity: Annotated[
        float,
        Field(..., ge=21.04, le=105.61, description="Humidity (%) (Range: 21.04 - 105.61)")
    ]
    air_temp: Annotated[
        float,
        Field(..., ge=10.28, le=38.78, description="Air temperature °C (Range: 10.28 - 38.78)")
    ]
    wind: Annotated[
        float,
        Field(..., ge=-3.76, le=24.35, description="Wind speed km/h (Range: -3.76 - 24.35)")
    ]


################### Root Endpoint ########################
@app.get("/")
def home():
    return {"message": "Welcome to the Smart Farming System API"}



################### Prediction Endpoint ##################
@app.post("/predict")
def predict_risk(features: CropFeatures):
    try:
        # Convert input to numpy array
        data = np.array([[features.soil_pH,
                          features.soil_moisture,
                          features.soil_temp,
                          features.nitrogen,
                          features.rainfall,
                          features.humidity,
                          features.air_temp,
                          features.wind]])
        
        # Scale input
        data_scaled = scaler.transform(data)

        # Predict
        prediction = model.predict(data_scaled)[0]
        probabilities = model.predict_proba(data_scaled)[0].tolist()

        # Map prediction to human-readable risk levels
        risk_mapping = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
        
        return {
            "prediction_code": int(prediction),
            "prediction_label": risk_mapping[int(prediction)],
            "probabilities": probabilities
        }

    except Exception as e:
        return {"error": str(e)}

