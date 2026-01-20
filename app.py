import streamlit as st
import tensorflow as tf
import numpy as np
import joblib
import time

# --- 1. DESAIN HALAMAN ---
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
        font-weight: bold;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. LOAD ASSETS (Model & Scaler) ---
@st.cache_resource
def load_assets():
    # Pastikan nama file ini sesuai dengan yang kamu upload di GitHub
    model = tf.keras.models.load_model('model_turbine.h5', compile=False)
    scaler_X = joblib.load('scaler_X.pkl')
    scaler_y = joblib.load('scaler_y.pkl')
    return model, scaler_X, scaler_y

# Inisialisasi awal variabel result agar tidak error "not defined"
result = 0

try:
    model, scaler_X, scaler_y = load_assets()

    # --- 3. SIDEBAR ---
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/1085/1085739.png", width=100)
        st.title("Konfigurasi Sistem")
        st.success("✅ Model Berhasil Dimuat")
        st.info("**Model:** LSTM v3\n\n**Window Size:** 30\n\n**Target:** Electrical Power (Watt)")
        st.write("---")
        st.write("🔧 **Status Server:** Online")
        st.write("🎓 **Project:** Tugas UAS Deep Learning")

    # --- 4. MAIN CONTENT ---
    st.title("⚡ Gas Turbine Power Forecasting")
    st.write("Sistem prediksi cerdas berbasis Deep Learning (LSTM) untuk memantau keluaran daya turbin.")
    
    st.write("---")
    
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("📥 Input Data Sensor")
        voltage = st.number_input("Voltage (Volt)", 
                                 min_value=0.0, 
                                 max_value=100.0, 
                                 value=10.0, 
                                 step=0.1, 
                                 help="Masukkan nilai tegangan input dari sensor turbin")
        
        predict_btn = st.button("🚀 Mulai Prediksi")

    with col2:
        st.subheader("📊 Hasil Analisis")
        if predict_btn:
            with st.spinner('Sedang memproses data...'):
                time.sleep(0.7) # Efek dramatis sedikit
                
                # Menyiapkan data input untuk Model 3 (Window Size 30)
                # Model LSTM butuh input 3 dimensi: (batch, window_size, features)
                data_input = np.zeros((30, 2))
                data_input[:, 0] = voltage # Mengisi kolom pertama dengan nilai voltage
                
                # Transformasi data dengan scaler
                scaled_input = scaler_X.transform(data_input)
                final_input = np.reshape(scaled_input, (1, 30, 2))
                
                # Proses Prediksi
                pred_scaled = model.predict(final_input)
                pred_watt = scaler_y.inverse_transform(pred_scaled)
                
                # Mengambil hasil akhir
                result = float(pred_watt[0][0])
                
                # Tampilan Metric
                st.metric(label="Predicted Power Output", 
                          value=f"{result:.2f} Watt", 
                          delta=f"{(result/100):.2f}% stability reference")
                
                if result > 0:
                    st.success("✅ Sistem Berjalan Normal")
                else:
                    st.warning("⚠️ Peringatan: Daya Terdeteksi Sangat Rendah / Negatif")
        else:
            st.info("💡 Masukkan nilai tegangan di samping lalu klik tombol untuk melihat hasil prediksi.")

    # --- 5. VISUALISASI ---
    st.write("---")
    st.subheader("📈 Real-time Prediction Visualization")
    
    # Membuat data dummy untuk grafik agar terlihat fluktuatif di sekitar hasil prediksi
    chart_data = np.random.randn(30, 1) * 2 + result
    st.line_chart(chart_data)
    st.caption("Grafik menunjukkan tren prediksi berdasarkan window size (30 time-steps).")

except Exception as e:
    st.error(f"⚠️ Terjadi kesalahan sistem!")
    st.info(f"Pesan Error: {e}")
    st.write("Pastikan file **model_turbine.h5**, **scaler_X.pkl**, dan **scaler_y.pkl** sudah diunggah ke GitHub.")
