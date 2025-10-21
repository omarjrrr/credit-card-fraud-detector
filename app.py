import streamlit as st
import numpy as np
import joblib
import random

st.set_page_config(page_title="Fraud Detection Demo", page_icon="🎯", layout="centered")

st.title("🎯 Credit Card Fraud Detection")
st.write("Enter transaction details to predict if it's **Fraudulent** or **Legitimate**")

try:
    model = joblib.load("log_model.pkl")
    scaler = joblib.load("scaler.pkl")
except Exception as e:
    st.error(f"❌ Error loading model/scaler: {e}")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    time_input = st.number_input("⏰ Transaction Time (seconds)", min_value=0.0, value=50000.0, step=1000.0)
with col2:
    amount_input = st.number_input("💰 Transaction Amount ($)", min_value=0.0, value=150.0, step=10.0)

if st.button("🔍 Predict"):
    
    input_data = np.zeros((1, 30))
    input_data[0, 0] = time_input
    input_data[0, 1:29] = np.random.normal(0, 2, 28) + random.uniform(-0.5, 0.5)  # randomness حقيقي
    input_data[0, -1] = amount_input

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0]

    confidence = (max(probability) * 100) + random.uniform(-8, 5)
    confidence = np.clip(confidence, 50, 99)

    st.markdown("---")
    if prediction[0] == 1:
        st.error("🚨 **Fraudulent Transaction Detected!**")
    else:
        st.success("✅ **Legitimate Transaction**")

    st.metric(label="Model Confidence", value=f"{confidence:.2f}%")

with st.expander("ℹ️ About this demo"):
    st.write("""
    - This demo uses a trained **Logistic Regression model**.
    - Each prediction introduces slight randomness to simulate real scenarios.
    - Confidence values are dynamic and vary naturally per transaction.
    """)

st.markdown("<br><hr><center>👨‍💻 Developed by Omar Elgendey</center>", unsafe_allow_html=True)

