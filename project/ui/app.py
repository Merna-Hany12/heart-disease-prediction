import streamlit as st
import pandas as pd
import joblib
import os

# Load trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "final_model.pkl")
model = joblib.load(MODEL_PATH)

# Streamlit UI
st.set_page_config(page_title="Heart Disease Prediction", layout="centered")

st.title("❤️ Heart Disease Prediction Web App")
st.write("This app predicts the likelihood of **heart disease** based on user input.")

# Sidebar for user input
st.sidebar.header("Enter Patient Data")

def user_input_features():
    # Numeric inputs
    age = st.sidebar.number_input("Age", 20, 100, 50)
    trestbps = st.sidebar.number_input("Resting Blood Pressure", 80, 200, 120)
    chol = st.sidebar.number_input("Serum Cholestoral (mg/dl)", 100, 600, 200)
    thalach = st.sidebar.number_input("Maximum Heart Rate Achieved", 60, 220, 150)
    oldpeak = st.sidebar.number_input("ST depression (oldpeak)", 0.0, 10.0, 1.0)
    ca = st.sidebar.number_input("Number of major vessels (0-3)", 0, 3, 0)

    # Categorical inputs
    sex = st.sidebar.selectbox("Sex", ("Male", "Female"))
    cp = st.sidebar.selectbox("Chest Pain Type (cp)", [1, 3, 4])   # only ones in your data
    exang = st.sidebar.selectbox("Exercise Induced Angina (exang)", [0, 1])
    restecg = st.sidebar.selectbox("Resting ECG (restecg)", [0, 1, 2])
    slope = st.sidebar.selectbox("Slope of the peak exercise ST segment", [1, 2])
    thal = st.sidebar.selectbox("Thalassemia (thal)", [3, 7])

    data = {
        "num__age": age,
        "num__trestbps": trestbps,
        "num__chol": chol,
        "num__thalach": thalach,
        "num__oldpeak": oldpeak,
        "num__ca": ca,

        # one-hot encoded categorical features
        "cat__sex_0.0": 1 if sex == "Female" else 0,

        "cat__cp_1.0": 1 if cp == 1 else 0,
        "cat__cp_3.0": 1 if cp == 3 else 0,
        "cat__cp_4.0": 1 if cp == 4 else 0,

        "cat__exang_1.0": 1 if exang == 1 else 0,

        "cat__restecg_0.0": 1 if restecg == 0 else 0,  # adjust if model used this
        "cat__slope_1.0": 1 if slope == 1 else 0,
        "cat__slope_2.0": 1 if slope == 2 else 0,

        "cat__thal_3.0": 1 if thal == 3 else 0,
        "cat__thal_7.0": 1 if thal == 7 else 0,

        # target placeholder
        "num": 0
    }

    features = pd.DataFrame(data, index=[0])
    return features

# Get input
input_df = user_input_features()

st.subheader("Processed Patient Data (for model)")
st.write(input_df)

# Prediction only when button clicked
if st.button("Predict"):
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    st.subheader("Prediction Result")
    if prediction[0] == 0:
        st.success("✅ No Heart Disease")
    else:
        st.error("⚠️ Likely Heart Disease")

    st.subheader("Prediction Probability")
    st.write(f"No Disease: {prediction_proba[0][0]*100:.2f}%")
    st.write(f"Disease: {prediction_proba[0][1]*100:.2f}%")
