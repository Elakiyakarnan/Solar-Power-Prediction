import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("solar_power_prediction_model.pkl")

# Page configuration
st.set_page_config(
    page_title="Solar Power Prediction",
    page_icon="☀️",
    layout="centered"
)

# Title
st.title("☀️ Solar Power Generation Prediction")
st.write("Predict solar AC output using Machine Learning.")

st.divider()

st.subheader("Enter Weather and Solar Details")

# Input fields
GHI = st.number_input("Global Horizontal Irradiance (GHI)", 0.0, 1200.0, 200.0)
Temperature = st.number_input("Temperature", -10.0, 50.0, 20.0)
Relative_Humidity = st.number_input("Relative Humidity", 0.0, 100.0, 75.0)
Wind_Speed = st.number_input("Wind Speed", 0.0, 20.0, 3.0)
Wind_Direction = st.number_input("Wind Direction", 0.0, 360.0, 140.0)

Hour = st.slider("Hour", 0, 23, 12)
Day = st.slider("Day", 1, 31, 16)
Month = st.slider("Month", 1, 12, 10)

SZ_Angle = st.number_input("SZ Angle", 0.0, 180.0, 90.0)
SA_Angle = st.number_input("SA Angle", 0.0, 360.0, 180.0)

# Prediction
if st.button("Predict Solar Power"):

    input_data = pd.DataFrame([{
        "GHI": GHI,
        "Temperature": Temperature,
        "Relative Humidity": Relative_Humidity,
        "Wind Speed": Wind_Speed,
        "Wind Direction": Wind_Direction,
        "Hour": Hour,
        "Day": Day,
        "Month": Month,
        "SZ Angle": SZ_Angle,
        "SA Angle": SA_Angle
    }])

    prediction = model.predict(input_data)[0]

    st.success(f"Predicted AC Output: {prediction:.2f}")

    st.info("Prediction generated using the trained Random Forest model.")