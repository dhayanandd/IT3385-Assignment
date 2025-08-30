import streamlit as st
import pandas as pd
from pycaret.classification import load_model, predict_model

# Load the exported PyCaret model
model = load_model(r'WheatAppClassifier/wheat_classifier')
st.title("🌾 Wheat Type Classification App")

# Mode selection
mode = st.radio("Choose input mode:", ["Enter all features", "Use Length + Width + Groove calculator"])

if mode == "Enter all features":
    area = st.number_input("Area", min_value=0.0)
    perimeter = st.number_input("Perimeter", min_value=0.0)
    compactness = st.number_input("Compactness", min_value=0.0)
    asymmetry = st.number_input("Asymmetry Coefficient", min_value=0.0)
    groove = st.number_input("Groove Length", min_value=0.0)

else:
    # Calculator mode
    length = st.number_input("Length", min_value=0.0)
    width = st.number_input("Width", min_value=0.0)
    groove = st.number_input("Groove Length", min_value=0.0)
    area = st.number_input("Area", min_value=0.0)
    asymmetry = st.number_input("Asymmetry Coefficient", min_value=0.0)

    # Compute perimeter and compactness
    perimeter = length + width + groove
    compactness = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

    st.write(f"📏 Calculated Perimeter: **{perimeter:.3f}**")
    st.write(f"⚙️ Calculated Compactness: **{compactness:.3f}**")

    # Disclaimer
    st.warning(
        "⚠️ Disclaimer: Perimeter and compactness are estimated using simplified formulas. "
        "Actual seed geometry may differ, so results may not be exact."
    )

# Predict button
if st.button("Predict Wheat Type"):
    input_data = pd.DataFrame([{
        "Area": area,
        "Perimeter": perimeter,
        "Compactness": compactness,
        "AsymmetryCoeff": asymmetry,
        "Groove": groove
    }])
    prediction = predict_model(model, data=input_data)
    st.write(prediction)
    predicted_class = prediction.loc[0, 'prediction_label']
    st.success(f"Predicted Wheat Type: {predicted_class}")




