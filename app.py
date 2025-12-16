import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# ==========================
# LOAD MODEL & DATA
# ==========================

@st.cache_resource
def load_artifacts():
    model = load_model("weather_models/best_weather_model.h5")
    with open("weather_models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    last_seq = np.load("weather_models/last_sequence.npy")
    return model, scaler, last_seq

model, scaler, last_seq = load_artifacts()

# ==========================
# APP CONFIG
# ==========================

st.set_page_config(
    page_title="Weather Forecasting",
    page_icon="🌦️",
    layout="centered"
)

st.title("🌦️ Weather Forecasting System")
st.write("Dự báo **nhiệt độ – độ ẩm – lượng mưa** bằng Deep Learning (RNN/LSTM)")

st.warning("⚠️ Dự báo dài hạn có độ không chắc chắn cao, chỉ mang tính tham khảo.")

# ==========================
# INPUT
# ==========================

days = st.selectbox(
    "📅 Chọn số ngày dự báo",
    options=[7, 30],
    index=0
)

# ==========================
# FORECAST FUNCTION
# ==========================

def forecast_iterative(model, last_sequence, n_steps):
    forecasts = []
    current_seq = last_sequence.copy()

    for _ in range(n_steps):
        pred = model.predict(current_seq, verbose=0)
        next_temp = pred[0, 0]

        last_vals = current_seq[0, -1, :]
        new_row = np.array([next_temp, last_vals[1], last_vals[2]]).reshape(1, 1, 3)
        current_seq = np.concatenate([current_seq[:, 1:, :], new_row], axis=1)

        forecasts.append([next_temp, last_vals[1], last_vals[2]])

    return np.array(forecasts)

# ==========================
# PREDICT
# ==========================

if st.button("🔮 Dự báo"):
    forecast = forecast_iterative(model, last_seq, days)
    forecast_real = scaler.inverse_transform(forecast)

    st.subheader(f"📊 Kết quả dự báo {days} ngày")

    st.line_chart({
        "Temperature (°C)": forecast_real[:, 0],
        "Humidity (%)": forecast_real[:, 1],
        "Rainfall (mm)": forecast_real[:, 2]
    })

    st.success("✅ Dự báo hoàn tất")
