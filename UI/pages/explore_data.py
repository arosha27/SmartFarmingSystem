import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load synthetic dataset
@st.cache_data
def load_data():
    return pd.read_csv("E:/SmartFarmingSystem/UI/data/processed_farming_data.csv")

df = load_data()

st.title(" Dataset Preview & Visualizations")

st.subheader("Dataset Overview")
st.write(df.head())

st.subheader("Statistical Summary")
st.write(df.describe())

# Correlation Heatmap
st.subheader("Correlation Heatmap")
plt.figure(figsize=(10,6))
sns.heatmap(df.corr( numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
st.pyplot(plt)

# Pairplot for selected features
st.subheader("Pairplot of Features")
plt.figure(figsize=(10,6))
sns.pairplot(df[["soil_pH","soil_moisture","soil_temp","nitrogen","humidity","air_temp","wind","crop_disease_risk_code"]],
             hue="crop_disease_risk_code", palette="Set2")
st.pyplot(plt)


