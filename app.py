import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Network Intrusion Detection", layout="wide")

st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        color: #003366;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Load the model
with open("voting_classifier.pkl", "rb") as file:
    model = pickle.load(file)

#Define the mapping for 'service' and 'flag'
service_mapping = {'IRC': 0, 'X11': 1, 'Z39_50': 2, 'auth': 3, 'bgp': 4, 'courier': 5, 'csnet_ns': 6, 'ctf': 7, 
                   'daytime': 8, 'discard': 9, 'domain': 10, 'domain_u': 11, 'echo': 12, 'eco_i': 13, 'ecr_i': 14, 
                   'efs': 15, 'exec': 16, 'finger': 17, 'ftp': 18, 'ftp_data': 19, 'gopher': 20, 'hostnames': 21, 
                   'http': 22, 'http_443': 23, 'http_8001': 24, 'imap4': 25, 'iso_tsap': 26, 'klogin': 27, 
                   'kshell': 28, 'ldap': 29, 'link': 30, 'login': 31, 'mtp': 32, 'name': 33, 'netbios_dgm': 34, 
                   'netbios_ns': 35, 'netbios_ssn': 36, 'netstat': 37, 'nnsp': 38, 'nntp': 39, 'ntp_u': 40, 
                   'other': 41, 'pm_dump': 42, 'pop_2': 43, 'pop_3': 44, 'printer': 45, 'private': 46, 
                   'red_i': 47, 'remote_job': 48, 'rje': 49, 'shell': 50, 'smtp': 51, 'sql_net': 52, 'ssh': 53, 
                   'sunrpc': 54, 'supdup': 55, 'systat': 56, 'telnet': 57, 'tim_i': 58, 'time': 59, 'urh_i': 60, 
                   'urp_i': 61, 'uucp': 62, 'uucp_path': 63, 'vmnet': 64, 'whois': 65}
flag_mapping = {'OTH': 0, 'REJ': 1, 'RSTO': 2, 'RSTOS0': 3, 'RSTR': 4, 'S0': 5, 'S1': 6, 'S2': 7, 
                'S3': 8, 'SF': 9, 'SH': 10}

# Load dataset for visualization
reference_csv_path = r"D:\\VIT 2022-26\\FINAL YEAR PROJECTS\\NETWORK INTRUSION DETECTION\\Train_data.csv"
data = pd.read_csv(reference_csv_path)

st.title("🚨 Network Intrusion Detection System")
st.markdown("An AI-powered dashboard to predict and visualize network intrusions.")

with st.expander("📂 Show Sample Dataset"):
    st.dataframe(data.head(10), use_container_width=True)

st.subheader("🔧 Input Parameters")
fields = ['service', 'flag', 'src_bytes', 'dst_bytes', 'logged_in',
          'same_srv_rate', 'diff_srv_rate', 'dst_host_srv_count',
          'dst_host_same_srv_rate', 'dst_host_diff_srv_rate']

input_data = {}
col1, col2 = st.columns(2)

for i, field in enumerate(fields):
    with (col1 if i % 2 == 0 else col2):
        if field == 'service':
            input_data[field] = service_mapping[st.selectbox(f"Select {field}:", list(service_mapping.keys()))]
        elif field == 'flag':
            input_data[field] = flag_mapping[st.selectbox(f"Select {field}:", list(flag_mapping.keys()))]
        else:
            input_data[field] = st.number_input(f"Enter {field}:", min_value=0.0, step=1.0)

# Prediction
if st.button("🚀 Predict Intrusion"):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)
    result = "🔴 Anomaly Detected" if prediction[0] == 0 else "🟢 Normal Traffic"
    st.success(f"**Prediction Result:** {result}")

# Visualizations
if st.checkbox("📊 Show Visual Insights"):
    st.subheader("📈 Anomaly vs Normal Count")
    if 'class' in data.columns:
        class_data = data['class'].map(lambda x: 'Normal' if x == 'normal' else 'Anomaly')
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        sns.countplot(x=class_data, palette=['green', 'red'], ax=ax1)
        ax1.set_title("Anomaly vs Normal Count")
        st.pyplot(fig1)

    st.subheader("📉 Traffic Volume Distribution")
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    sns.histplot(data['src_bytes'], bins=50, kde=True, color='blue', ax=ax2)
    ax2.set_title("Distribution of Source Bytes")
    st.pyplot(fig2)

    st.subheader("🧾 Most Frequent Flags")
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    sns.countplot(data=data, x='flag', order=data['flag'].value_counts().index[:10], palette='magma', ax=ax3)
    ax3.set_title("Top 10 Flags")
    st.pyplot(fig3)

    st.subheader("🌐 Most Frequent Services")
    fig4, ax4 = plt.subplots(figsize=(8, 4))
    sns.countplot(data=data, x='service', order=data['service'].value_counts().index[:10], palette='viridis', ax=ax4)
    ax4.set_title("Top 10 Services")
    st.pyplot(fig4)

st.markdown("""
<hr style="border: 0.5px solid #f0f0f0" />
<div style='text-align: center; padding-top: 10px; font-size: 16px; color: gray;'>
    🚀 Built with ❤️ by <b style='color: #1f77b4;'>Ritik Garg</b><br>
    📚 Final Year Project | Network Intrusion Detection System
</div>
""", unsafe_allow_html=True)


# import streamlit as st
# import pandas as pd
# import pickle

# # Load the trained Voting Classifier model
# with open("voting_classifier.pkl", "rb") as file:
#     scaler, model = pickle.load(file)  # Assuming scaler is saved with the model

# Define the mapping for 'service' and 'flag'
# service_mapping = {'IRC': 0, 'X11': 1, 'Z39_50': 2, 'auth': 3, 'bgp': 4, 'courier': 5, 'csnet_ns': 6, 'ctf': 7, 
#                    'daytime': 8, 'discard': 9, 'domain': 10, 'domain_u': 11, 'echo': 12, 'eco_i': 13, 'ecr_i': 14, 
#                    'efs': 15, 'exec': 16, 'finger': 17, 'ftp': 18, 'ftp_data': 19, 'gopher': 20, 'hostnames': 21, 
#                    'http': 22, 'http_443': 23, 'http_8001': 24, 'imap4': 25, 'iso_tsap': 26, 'klogin': 27, 
#                    'kshell': 28, 'ldap': 29, 'link': 30, 'login': 31, 'mtp': 32, 'name': 33, 'netbios_dgm': 34, 
#                    'netbios_ns': 35, 'netbios_ssn': 36, 'netstat': 37, 'nnsp': 38, 'nntp': 39, 'ntp_u': 40, 
#                    'other': 41, 'pm_dump': 42, 'pop_2': 43, 'pop_3': 44, 'printer': 45, 'private': 46, 
#                    'red_i': 47, 'remote_job': 48, 'rje': 49, 'shell': 50, 'smtp': 51, 'sql_net': 52, 'ssh': 53, 
#                    'sunrpc': 54, 'supdup': 55, 'systat': 56, 'telnet': 57, 'tim_i': 58, 'time': 59, 'urh_i': 60, 
#                    'urp_i': 61, 'uucp': 62, 'uucp_path': 63, 'vmnet': 64, 'whois': 65}
# flag_mapping = {'OTH': 0, 'REJ': 1, 'RSTO': 2, 'RSTOS0': 3, 'RSTR': 4, 'S0': 5, 'S1': 6, 'S2': 7, 
#                 'S3': 8, 'SF': 9, 'SH': 10}

# # Load the CSV file
# reference_csv_path = "/Users/sheenkocher/Desktop/projects/network_intrusion_detection/kaggle_intrusion_data/Train_data.csv" # Replace with the path to your CSV file
# data = pd.read_csv(reference_csv_path)

# # Streamlit app title
# st.title("Intrusion Detection System")
# st.write("This application predicts whether a network activity is an intrusion using a trained Voting Classifier.")

# # Show the reference CSV
# if st.checkbox("Show Reference Data"):
#     st.write(data.head())

# # Define the input fields
# st.header("Enter Input for the Following Fields:")
# fields = ['service', 'flag', 'src_bytes', 'dst_bytes', 'logged_in',
#           'same_srv_rate', 'diff_srv_rate', 'dst_host_srv_count',
#           'dst_host_same_srv_rate', 'dst_host_diff_srv_rate']

# # Create inputs for each field
# input_data = {}
# for field in fields:
#     if field == 'service':  # Use service mapping
#         input_data[field] = service_mapping[st.selectbox(f"Select {field}:", list(service_mapping.keys()))]
#     elif field == 'flag':  # Use flag mapping
#         input_data[field] = flag_mapping[st.selectbox(f"Select {field}:", list(flag_mapping.keys()))]
#     else:  # Numeric fields
#         input_data[field] = st.number_input(f"Enter {field}:", value=0.0)

# # When the user clicks "Predict", send the input to the model
# if st.button("Predict"):
#     # Convert the input_data to a DataFrame
#     input_df = pd.DataFrame([input_data])
    
#     # Standardize the input
#     input_scaled = scaler.transform(input_df)
    
#     # Make the prediction
#     prediction = model.predict(input_scaled)
    
#     # Display the prediction
#     st.write(f"Prediction: {prediction[0]}")
