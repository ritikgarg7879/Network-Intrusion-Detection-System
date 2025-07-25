# 🔐 Network Anomaly Intrusion Detection System

A Machine Learning-based system to detect and classify anomalies in network traffic, with a user-friendly Streamlit interface.

---

## 🚀 Key Features
- Real-world simulation using the **KDD Cup 1999** dataset
- **Feature selection** using Mutual Information
- **Exploratory Data Analysis (EDA)** with Plotly
- **Multiple ML Models**: XGBoost, LightGBM, SVM, Random Forest, Logistic Regression, Naïve Bayes
- **Ensemble Voting Classifier** for best results
- **Optuna** for hyperparameter tuning
- **Streamlit App** for real-time prediction

---

## 📁 Dataset
- Military-like simulated dataset with 41 features (38 numeric, 3 categorical)
- Each connection labeled as **Normal** or **Anomalous**

🔗 [KDD Cup 1999 Dataset on Kaggle](https://www.kaggle.com/datasets/sampadab17/network-intrusion-detection/data)

---

## 🏆 Best Performing Model
> A Voting Classifier combining XGBoost, LightGBM, and SVM achieved the highest precision, recall, and F1-score.

---

## 🖥️ Streamlit Web App
Interact with the model via a simple UI:
- 📤 Upload custom network samples
- 📈 View predictions in real-time
- 📊 Visual insights (coming soon)

---

## ⚙️ Getting Started

### 📦 Prerequisites
Make sure Python 3.7+ is installed.

Install required libraries:
```bash
pip install -r requirements.txt
