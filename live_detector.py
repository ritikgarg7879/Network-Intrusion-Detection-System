import pyshark
import pandas as pd
import joblib
import datetime

# Load your trained model
with open("voting_classifier.pkl", "rb") as file:
    model = joblib.load(file)

# Define basic flag mapping (simplified)
flag_mapping = {
    '0x000': 0,
    '0x004': 1,
    '0x010': 2,
    '0x012': 3,
    '0x018': 4,
    '0x02': 5,
    '0x01': 6
}

# Function to extract features from packet
def extract_features(packet):
    try:
        proto = packet.transport_layer
        ip_layer = packet.ip
        tcp_layer = packet.tcp

        features = {
            'src_bytes': int(packet.length),
            'dst_bytes': int(packet.length),  # Placeholder; both same for now
            'logged_in': 0,                   # Assume not logged in
            'same_srv_rate': 0,
            'diff_srv_rate': 0,
            'dst_host_srv_count': 0,
            'dst_host_same_srv_rate': 0,
            'dst_host_diff_srv_rate': 0,
            'service': 22,                    # Placeholder (22 = http)
            'flag': flag_mapping.get(tcp_layer.flags, 5)
        }
        return features
    except Exception:
        return None

# Start live capture (change interface name if needed)
capture = pyshark.LiveCapture(interface='Wi-Fi')  # Or 'Ethernet', 'eth0', etc.

print("🔍 Real-Time Network Intrusion Detection Started...\n")

# Optional: Save anomaly log
log_file = "anomaly_log.txt"

for pkt in capture.sniff_continuously():
    features = extract_features(pkt)
    if features:
        df = pd.DataFrame([features])
        prediction = model.predict(df)[0]
        result = "🔴 Anomaly" if prediction == 0 else "🟢 Normal"

        print(f"{datetime.datetime.now()} | {result} | {features}")

        if prediction == 0:
          with open(log_file, "a", encoding="utf-8") as log:
                log.write(f"{datetime.datetime.now()} | {result} | {features}\n")

