import streamlit as st
import tensorflow as tf
import numpy as np
import joblib
import time

# --- DESAIN HALAMAN ---
st.set_page_config(page_title="Turbine AI Predictor", page_icon="⚡", layout="wide")

# Custom CSS untuk mempercantik tampilan
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    # Ubah baris ini di dalam file app.py kamu
    model = tf.keras.models.load_model('model_turbine.h5', compile=False)
    scaler_X = joblib.load('scaler_X.pkl')
    scaler_y = joblib.load('scaler_y.pkl')
    return model, scaler_X, scaler_y

try:
    model, scaler_X, scaler_y = load_assets()

    # --- SIDEBAR ---
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/1085/1085739.png", width=100)
        st.title("Konfigurasi Sistem")
        st.info("Model: LSTM v3\nWindow Size: 30\nTarget: Electrical Power (Watt)")
        st.write("---")
        st.write("Dibuat untuk Tugas UAS Deep Learning")

    # --- MAIN CONTENT ---
    st.title("⚡ Gas Turbine Power Forecasting")
    st.write("Sistem cerdas berbasis Deep Learning untuk memprediksi keluaran daya turbin berdasarkan tegangan.")
    
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📥 Input Data Sensor")
        voltage = st.number_input("Voltage (Volt)", min_value=0.0, max_value=50.0, value=10.0, step=0.1, help="Masukkan nilai tegangan input dari turbin")
        
        predict_btn = st.button("Mulai Prediksi")

    with col2:
        st.subheader("📊 Hasil Analisis")
        if predict_btn:
            with st.spinner('Menghitung prediksi...'):
                time.sleep(0.5) # Simulasi loading biar keren
                
                # Logika Model 3 (Window Size 30)
                data_input = np.zeros((30, 2))
                data_input[:, 0] = voltage
                
                scaled_input = scaler_X.transform(data_input)
                final_input = np.reshape(scaled_input, (1, 30, 2))
                
                # Prediksi
                pred_scaled = model.predict(final_input)
                pred_watt = scaler_y.inverse_transform(pred_scaled)
                
                result = pred_watt[0][0]
                
                # Tampilan Gauge atau Metric
                st.metric(label="Predicted Power Output", value=f"{result:.2f} Watt", delta=f"{result*0.01:.2f} % stability")
                
                if result > 0:
                    st.success("Sistem Berjalan Normal")
                else:
                    st.warning("Peringatan: Daya Terdeteksi Rendah")

    st.write("---")
    st.subheader("📈 Visualisasi Prediksi")
    # Membuat grafik sederhana dummy untuk mempercantik desain
    chart_data = np.random.randn(20, 1) * 10 + result
    st.line_chart(chart_data)

except Exception as e:
    st.error(f"⚠️ Terjadi kesalahan: Pastikan file .h5 dan .pkl ada di folder yang sama. Error: {e}")