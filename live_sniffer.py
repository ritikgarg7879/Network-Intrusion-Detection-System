# live_sniffer.py
from scapy.all import sniff
import pandas as pd
import pickle

# Load model
with open("voting_classifier.pkl", "rb") as f:
    model = pickle.load(f)

# Feature extraction from packet
def extract_features(pkt):
    try:
        return {
            'src_bytes': len(pkt),
            'dst_bytes': len(pkt),
            'logged_in': 0,
            'same_srv_rate': 0,
            'diff_srv_rate': 0,
            'dst_host_srv_count': 0,
            'dst_host_same_srv_rate': 0,
            'dst_host_diff_srv_rate': 0,
            'service': 22,  # Placeholder
            'flag': 5       # Placeholder
        }
    except:
        return None

def classify_packet(pkt):
    features = extract_features(pkt)
    if features:
        df = pd.DataFrame([features])
        prediction = model.predict(df)
        result = "🔴 Anomaly" if prediction[0] == 0 else "🟢 Normal"
        print(result, features)

print("[INFO] Starting packet capture...")
sniff(prn=classify_packet, count=50)  # Capture 50 packets and stop
