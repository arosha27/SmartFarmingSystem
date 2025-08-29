
import streamlit as st

st.set_page_config(
    page_title="Smart Farming • Disease Risk",
    page_icon="🌱",
    layout="wide",
    menu_items={"About": "Smart Farming System — Crop Disease Risk"}
)

st.title("🌱 Smart Farming — Crop Disease Risk")
st.markdown(
    """
Welcome! Use the pages on the left:
- **📊 Explore Data**: upload & preview your dataset, with colorful visuals.
- **🤖 Predict & Explain**: enter soil & weather features to predict risk and see **feature contributions**.
"""
)

