# pcap_processor.py
import pyshark
import pandas as pd
import pickle

with open("voting_classifier.pkl", "rb") as f:
    model = pickle.load(f)

cap = pyshark.FileCapture('sample.pcap', only_summaries=True)

for pkt in cap:
    try:
        features = {
            'src_bytes': int(pkt.length),
            'dst_bytes': int(pkt.length),
            'logged_in': 0,
            'same_srv_rate': 0,
            'diff_srv_rate': 0,
            'dst_host_srv_count': 0,
            'dst_host_same_srv_rate': 0,
            'dst_host_diff_srv_rate': 0,
            'service': 22,
            'flag': 5
        }
        df = pd.DataFrame([features])
        prediction = model.predict(df)
        result = "🔴 Anomaly" if prediction[0] == 0 else "🟢 Normal"
        print(result, features)
    except Exception as e:
        continue
