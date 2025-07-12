import streamlit as st
import numpy as np
import joblib

# Load trained model and scaler
model = joblib.load("credit_card_model.pkl")
scaler = joblib.load("scaler.pkl")

# App configuration
st.set_page_config(page_title="Fraud Detection App", layout="centered")
st.title("💳 Credit Card Fraud Detection")

st.markdown("""
Enter the 29 feature values (`V1` to `V28` and `Amount`) of the transaction to check if it's **fraudulent** or **legitimate**.
""")

# Feature input labels
feature_names = [f"V{i}" for i in range(1, 29)] + ["Amount"]

# Collect input from user
inputs = []
for feature in feature_names:
    val = st.number_input(f"{feature}", value=0.0, format="%.15f")
    inputs.append(val)

# Predict button
if st.button("Predict"):
    input_array = np.array(inputs).reshape(1, -1)
    
    # Scale input
    scaled_input = scaler.transform(input_array)
    
    # Predict
    prediction = model.predict(scaled_input)

    # Output result
    if prediction[0] == 1:
        st.error("🚨 Fraudulent Transaction Detected!")
    else:
        st.success("✅ Legitimate Transaction")

# Footer
st.markdown("---")
st.caption("Credit Card Fraud Detection Demo")
