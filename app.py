import streamlit as st
import pandas as pd
import joblib
import time

# ----------------------
# Load model, scaler, columns
# ----------------------
model = joblib.load("LR_heart.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

# Numeric columns scaled during training
scaled_cols = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]

# ----------------------
# Page config
# ----------------------
st.set_page_config(page_title="Heart Disease Predictor", layout="centered")
st.title("❤️ Heart Disease Prediction App")
st.markdown("### Fill in the patient information below:")

# ----------------------
# Inputs Top-to-Bottom
# ----------------------
age = st.slider("Age", 20, 100, 50)
restingbp = st.slider("Resting Blood Pressure", 80, 200, 120)
chol = st.slider("Cholesterol", 100, 600, 200)
fastingbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1])
maxhr = st.slider("Max Heart Rate", 60, 220, 150)
oldpeak = st.slider("Oldpeak", 0.0, 6.0, 1.0, step=0.1)
sex = st.selectbox("Sex", ["M", "F"])
cp = st.selectbox("Chest Pain Type", ["ASY", "ATA", "NAP", "TA"])
ecg = st.selectbox("Resting ECG", ["Normal", "LVH", "ST"])
angina = st.selectbox("Exercise Angina", ["N", "Y"])
slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# ----------------------
# Prediction
# ----------------------
if st.button("Predict Heart Disease Risk"):

    # Start with zeros for all columns
    input_dict = {col: 0 for col in columns}

    # Fill numeric values
    input_dict.update({
        "Age": age,
        "RestingBP": restingbp,
        "Cholesterol": chol,
        "FastingBS": fastingbs,
        "MaxHR": maxhr,
        "Oldpeak": oldpeak
    })

    # Fill one-hot encoded values
    input_dict[f"Sex_{sex}"] = 1
    input_dict[f"ChestPainType_{cp}"] = 1
    input_dict[f"RestingECG_{ecg}"] = 1
    input_dict[f"ExerciseAngina_{angina}"] = 1
    input_dict[f"ST_Slope_{slope}"] = 1

    # Create DataFrame
    df_input = pd.DataFrame([input_dict])

    # Scale numeric columns only
    df_input[scaled_cols] = scaler.transform(df_input[scaled_cols])

    # ----------------------
    # Flashing tension effect
    # ----------------------
    placeholder = st.empty()
    for i in range(6):  # flash 6 times
        color = "#B22222" if i % 2 == 0 else "#228B22"
        placeholder.markdown(f'''
            <div style="background-color:{color};color:white;padding:25px;border-radius:15px;text-align:center">
            <h2>Processing Prediction...</h2>
            </div>
        ''', unsafe_allow_html=True)
        time.sleep(0.5)

    # ----------------------
    # Predict
    # ----------------------
    pred = model.predict(df_input)
    pred_proba = model.predict_proba(df_input)[0][1]

    # Clear placeholder and show final prediction
    placeholder.empty()
    if pred[0] == 1:
        st.markdown(f'''
            <div style="background-color:#B22222;color:white;padding:25px;border-radius:15px;text-align:center">
            <h2>⚠️ High Risk of Heart Disease</h2>
            <p style="font-size:20px;">Probability: {pred_proba*100:.2f}%</p>
            </div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown(f'''
            <div style="background-color:#228B22;color:white;padding:25px;border-radius:15px;text-align:center">
            <h2>✅ Low Risk of Heart Disease</h2>
            <p style="font-size:20px;">Probability: {pred_proba*100:.2f}%</p>
            </div>
        ''', unsafe_allow_html=True)