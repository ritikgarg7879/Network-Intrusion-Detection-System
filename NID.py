# app.py
import streamlit as st
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Network Security Anomaly Detection", layout="wide")
st.title("🔐 Network Security Anomaly Detection with Explainable AI")
st.write("Enter network traffic details to check if it is **Anomaly or Normal**.")

# --- Input Fields ---
packet_size = st.number_input("Packet Size", value=0.9677)
inter_arrival_time = st.number_input("Inter Arrival Time", value=0.4934)
src_port = st.number_input("Source Port", value=28474)
dst_port = st.number_input("Destination Port", value=443)
packet_count_5s = st.number_input("Packet Count (5s)", value=0.0714)
mean_packet_size = st.number_input("Mean Packet Size", value=0.0)
spectral_entropy = st.number_input("Spectral Entropy", value=0.3306)
frequency_band_energy = st.number_input("Frequency Band Energy", value=0.0239)

# Protocol flags
st.subheader("Protocol & Flags")
protocol_type_TCP = st.selectbox("Protocol Type TCP", [0, 1], index=0)
protocol_type_UDP = st.selectbox("Protocol Type UDP", [0, 1], index=0)
src_ip_192_168_1_2 = st.selectbox("Source IP 192.168.1.2", [0, 1], index=0)
src_ip_192_168_1_3 = st.selectbox("Source IP 192.168.1.3", [0, 1], index=0)
dst_ip_192_168_1_5 = st.selectbox("Destination IP 192.168.1.5", [0, 1], index=0)
dst_ip_192_168_1_6 = st.selectbox("Destination IP 192.168.1.6", [0, 1], index=0)
tcp_flags_FIN = st.selectbox("TCP Flag FIN", [0, 1], index=0)
tcp_flags_SYN = st.selectbox("TCP Flag SYN", [0, 1], index=1)
tcp_flags_SYNACK = st.selectbox("TCP Flag SYN-ACK", [0, 1], index=0)

# --- Prepare Input Data ---
input_data = pd.DataFrame([[
    packet_size, inter_arrival_time, src_port, dst_port, packet_count_5s,
    mean_packet_size, spectral_entropy, frequency_band_energy,
    protocol_type_TCP, protocol_type_UDP,
    src_ip_192_168_1_2, src_ip_192_168_1_3,
    dst_ip_192_168_1_5, dst_ip_192_168_1_6,
    tcp_flags_FIN, tcp_flags_SYN, tcp_flags_SYNACK
]], columns=[
    'packet_size', 'inter_arrival_time', 'src_port', 'dst_port',
    'packet_count_5s', 'mean_packet_size', 'spectral_entropy',
    'frequency_band_energy', 'protocol_type_TCP', 'protocol_type_UDP',
    'src_ip_192.168.1.2', 'src_ip_192.168.1.3',
    'dst_ip_192.168.1.5', 'dst_ip_192.168.1.6',
    'tcp_flags_FIN', 'tcp_flags_SYN', 'tcp_flags_SYN-ACK'
])

# --- Load Model ---
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.load_model("SecurityDatasetXgBoost.json")

# --- Predict & Explain ---
if st.button("🔍 Predict"):
    prediction = model.predict(input_data)[0]
    pred_prob = model.predict_proba(input_data)[0]

    if prediction == 1:
        st.error(f"⚠️ The network traffic is **Anomaly** (Probability: {pred_prob[1]:.2f})")
    else:
        st.success(f"✅ The network traffic is **Normal** (Probability: {pred_prob[0]:.2f})")

    # SHAP Explainer
    explainer = shap.Explainer(model, input_data)  # Recreate explainer for single input
    shap_values = explainer(input_data)

    st.subheader("📊 SHAP Feature Contribution")
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)

