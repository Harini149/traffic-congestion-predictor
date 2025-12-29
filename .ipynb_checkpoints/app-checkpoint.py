import streamlit as st
import pandas as pd
import numpy as np
import pickle

# -------------------------------
# Load model and scaler
# -------------------------------
with open("traffic_congestion_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# -------------------------------
# Helper functions
# -------------------------------
def congestion_label(pred):
    if pred == 0:
        return "Low Congestion 🟢"
    elif pred == 1:
        return "Medium Congestion 🟡"
    else:
        return "High Congestion 🔴"

def generate_alert(pred):
    if pred == 0:
        return "Traffic is smooth. You can proceed normally."
    elif pred == 1:
        return "Moderate traffic detected. Plan your journey."
    else:
        return "Heavy traffic ahead! Consider alternative routes."

# -------------------------------
# Initialize session state
# -------------------------------
if "prediction" not in st.session_state:
    st.session_state.prediction = None

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(page_title="Traffic Congestion Predictor", layout="centered")

st.title("🚦Traffic Congestion Prediction")
st.write("Predict traffic congestion using ML and weather-aware inputs")

# -------------------------------
# User inputs
# -------------------------------
junction = st.selectbox("Select Junction", [0, 1, 2, 3, 4])
hour = st.slider("Hour of Day", 0, 23, 18)
is_weekend = st.selectbox("Is it a weekend?", [0, 1])
is_peak_hour = st.selectbox("Is it peak hour?", [0, 1])
vehicle_count = st.slider("Estimated Vehicle Count (system input)", 0, 120, 60)
is_rain = st.selectbox("Rainy Weather?", [0, 1])
is_fog = st.selectbox("Foggy Weather?", [0, 1])
temperature = st.slider("Temperature (°C)", 10, 45, 28)

# -------------------------------
# Predict button
# -------------------------------
if st.button("Predict Traffic"):
    input_df = pd.DataFrame([{
        'junction_encoded': junction,
        'hour': hour,
        'is_weekend': is_weekend,
        'is_peak_hour': is_peak_hour,
        'vehicle_count': vehicle_count,
        'is_rain': is_rain,
        'is_fog': is_fog,
        'temperature': temperature
    }])

    input_scaled = scaler.transform(input_df)
    st.session_state.prediction = model.predict(input_scaled)[0]

# -------------------------------
# Display results (SAFE)
# -------------------------------
if st.session_state.prediction is not None:
    pred = st.session_state.prediction

    st.subheader(congestion_label(pred))
    st.write(generate_alert(pred))

    # Junction graph (prototype routing)
    junction_graph = {
        0: [1, 2],
        1: [0, 3],
        2: [0, 3],
        3: [1, 2, 4],
        4: [3]
    }

    if pred == 2:
        st.warning("🚦 High congestion detected")
        st.write("🛣 Suggested alternative junctions:")
        st.write(junction_graph.get(junction, []))
