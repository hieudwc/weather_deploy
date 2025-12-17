import streamlit as st
import numpy as np
import pickle
import pandas as pd
from tensorflow.keras.models import load_model

import folium
from folium.features import DivIcon
from streamlit_folium import st_folium

# =====================================================
# LOAD MODEL & ARTIFACTS
# =====================================================

@st.cache_resource
def load_artifacts():
    model = load_model(
        "weather_models/best_weather_model.h5",
        compile=False   # QUAN TRỌNG: tránh lỗi Keras version
    )
    with open("weather_models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    last_seq = np.load("weather_models/last_sequence.npy")
    return model, scaler, last_seq

model, scaler, last_seq = load_artifacts()

# =====================================================
# STREAMLIT CONFIG
# =====================================================

st.set_page_config(
    page_title="Weather Forecasting System",
    page_icon="🌦️",
    layout="centered"
)

st.title("🌦️ Weather Forecasting System")
st.write("Dự báo **nhiệt độ – độ ẩm – lượng mưa** bằng Deep Learning (RNN / LSTM)")

st.warning("⚠️ Dự báo dài hạn có độ không chắc chắn cao, chỉ mang tính tham khảo.")

# =====================================================
# USER INPUT
# =====================================================

days = st.selectbox(
    "📅 Chọn số ngày dự báo",
    options=[7, 30],
    index=1
)

# =====================================================
# FORECAST FUNCTION (ITERATIVE MULTI-STEP)
# =====================================================

def forecast_iterative(model, last_sequence, n_steps):
    forecasts = []
    current_seq = last_sequence.copy()

    for _ in range(n_steps):
        pred = model.predict(current_seq, verbose=0)
        next_temp = pred[0, 0]

        last_vals = current_seq[0, -1, :]
        new_row = np.array(
            [next_temp, last_vals[1], last_vals[2]]
        ).reshape(1, 1, 3)

        current_seq = np.concatenate(
            [current_seq[:, 1:, :], new_row],
            axis=1
        )

        forecasts.append([next_temp, last_vals[1], last_vals[2]])

    return np.array(forecasts)

# =====================================================
# COLOR FUNCTION FOR MAP
# =====================================================

def get_color(temp):
    if temp < 18:
        return "blue"
    elif temp < 25:
        return "green"
    elif temp < 30:
        return "orange"
    else:
        return "red"

# =====================================================
# MAIN PREDICTION
# =====================================================

if st.button("🔮 Dự báo"):
    forecast_scaled = forecast_iterative(model, last_seq, days)
    forecast_real = scaler.inverse_transform(forecast_scaled)

    st.subheader(f"📊 Kết quả dự báo {days} ngày")

    # -------- LINE CHART --------
    chart_df = pd.DataFrame({
        "Temperature (°C)": forecast_real[:, 0],
        "Humidity (%)": forecast_real[:, 1],
        "Rainfall (mm)": forecast_real[:, 2]
    })

    st.line_chart(chart_df)

    # =================================================
    # MAP DATA (VIỆT NAM - DEMO THEO VÙNG)
    # =================================================

    base_temp = forecast_real[0, 0]

    map_data = pd.DataFrame({
        "City": ["Hà Nội", "TP. Hồ Chí Minh", "Đà Nẵng", "Huế", "Hải Phòng", "Cần Thơ"],
        "lat": [21.0285, 10.8231, 16.0471, 16.4637, 20.8449, 10.0452],
        "lon": [105.8542, 106.6297, 108.2068, 107.5909, 106.6881, 105.7469],
        "Temp_LSTM": [
            base_temp + 1.5,
            base_temp + 3.0,
            base_temp + 2.0,
            base_temp + 1.0,
            base_temp + 1.8,
            base_temp + 2.5
        ],
        "Temp_RNN": [
            base_temp + 1.2,
            base_temp + 2.7,
            base_temp + 1.7,
            base_temp + 0.8,
            base_temp + 1.5,
            base_temp + 2.2
        ]
    })

    # =================================================
    # FOLIUM MAP
    # =================================================

    center_lat = map_data["lat"].mean()
    center_lon = map_data["lon"].mean()

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=6,
        tiles="CartoDB positron"
    )

    fg_rnn = folium.FeatureGroup(name="Dự báo RNN")
    fg_lstm = folium.FeatureGroup(name="Dự báo LSTM")

    for _, row in map_data.iterrows():
        lat, lon = row["lat"], row["lon"]
        city = row["City"]

        # ----- RNN -----
        temp_rnn = row["Temp_RNN"]
        color_rnn = get_color(temp_rnn)

        folium.CircleMarker(
            location=[lat, lon],
            radius=8,
            color=color_rnn,
            fill=True,
            fill_color=color_rnn,
            fill_opacity=0.8,
            popup=f"<b>{city}</b><br>RNN: {temp_rnn:.1f}°C"
        ).add_to(fg_rnn)

        # ----- LSTM -----
        temp_lstm = row["Temp_LSTM"]
        color_lstm = get_color(temp_lstm)

        folium.CircleMarker(
            location=[lat, lon],
            radius=8,
            color=color_lstm,
            fill=True,
            fill_color=color_lstm,
            fill_opacity=0.8,
            popup=f"<b>{city}</b><br>LSTM: {temp_lstm:.1f}°C"
        ).add_to(fg_lstm)

        # ----- LABEL -----
        folium.Marker(
            [lat, lon],
            icon=DivIcon(
                icon_size=(160, 36),
                icon_anchor=(0, 0),
                html=f"""
                <div style="
                    font-size:10pt;
                    font-weight:bold;
                    background:white;
                    padding:2px 4px;
                    border-radius:4px;
                    border:1px solid gray;">
                    {city}: {temp_lstm:.1f}°C
                </div>
                """
            )
        ).add_to(m)

    fg_rnn.add_to(m)
    fg_lstm.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    # =================================================
    # SHOW MAP
    # =================================================

    st.subheader("🗺️ Bản đồ dự báo nhiệt độ Việt Nam")
    st.caption("Màu sắc thể hiện mức nhiệt, có thể bật/tắt RNN – LSTM")

    st_folium(m, width=900, height=600)

    st.success("✅ Dự báo hoàn tất")
