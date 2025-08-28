import pandas as pd
import numpy as np

np.random.seed(42)

# Samples
n_samples = 1000

# Classes
classes = ["Low", "Moderate", "High"]

# Balance: ~equal samples
samples_per_class = n_samples // len(classes)

data = []

for risk in classes:
    for _ in range(samples_per_class):
        # Soil & weather ranges with overlap + noise
        soil_pH = np.random.normal(6.5, 0.5)  # Normal around 6.5
        soil_moisture = np.random.normal(0.25 if risk=="Low" else 0.35 if risk=="Moderate" else 0.45, 0.08)
        soil_temp = np.random.normal(20 if risk=="Low" else 25 if risk=="Moderate" else 28, 2.5)
        nitrogen = np.random.normal(40 if risk=="Low" else 55 if risk=="Moderate" else 70, 10)

        rainfall = np.random.normal(50 if risk=="Low" else 100 if risk=="Moderate" else 150, 20)
        humidity = np.random.normal(50 if risk=="Low" else 65 if risk=="Moderate" else 80, 10)
        air_temp = np.random.normal(20 if risk=="Low" else 25 if risk=="Moderate" else 30, 3)
        wind = np.random.normal(5 if risk=="Low" else 10 if risk=="Moderate" else 15, 3)

        data.append([soil_pH, soil_moisture, soil_temp, nitrogen,
                     rainfall, humidity, air_temp, wind, risk])

df = pd.DataFrame(data, columns=[
    "soil_pH", "soil_moisture", "soil_temp", "nitrogen",
    "rainfall", "humidity", "air_temp", "wind", "crop_disease_risk"
])

# Shuffle rows
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save
df.to_csv("Data/raw/synthetic_data.csv", index=False)

print(df.head())
print(df["crop_disease_risk"].value_counts())
