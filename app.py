import streamlit as st
import numpy as np
import pickle


# ------------------------------------------------------
import joblib

# ------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------
st.set_page_config(
    page_title="Liver Disease Stage Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    /* MAIN CONTENT AREA (RIGHT SIDE) */
    section.main > div {
        background-image:
        linear-gradient(rgba(255,255,255,0.90), rgba(255,255,255,0.90)),
        url("https://cdn.pixabay.com/photo/2017/01/31/13/14/anatomy-2020090_1280.png");
        background-repeat: no-repeat;
        background-position: right center;
        background-size: 420px;
    }

    /* Sidebar untouched */
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------------------------------------
# BACKGROUND IMAGE + OVERLAY (PROFESSIONAL)
# ------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background-image:
        linear-gradient(rgba(255,255,255,0.88), rgba(255,255,255,0.88)),
        url("https://images.unsplash.com/photo-1581090700227-1e37b190418e");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }

    .main-title {
        background: linear-gradient(to right, #4b79a1, #283e51);
        padding: 20px;
        border-radius: 12px;
        color: white;
        text-align: center;
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 25px;
    }

    .prediction-box {
        padding: 25px;
        border-radius: 12px;
        font-size: 22px;
        font-weight: 600;
        text-align: center;
        margin-top: 25px;
    }

    div[data-testid="stNumberInput"] {
        background: white;
        padding: 10px;
        border-radius: 8px;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.05);
    }

    .footer {
        margin-top: 60px;
        text-align: center;
        font-size: 14px;
        color: gray;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------------------------------------
# LOAD MODEL FILES
# ------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("best_liver_model.pkl")
    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    return model, scaler, label_encoder

model, scaler, label_encoder = load_artifacts()
st.write("Scaler expects these features:", scaler.feature_names_in_)


# ------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------
st.sidebar.title("🩺 Liver Disease Predictor")
st.sidebar.info(
    "This AI-powered system predicts **five stages** of liver disease:\n\n"
    "• No Disease\n"
    "• Suspect Disease\n"
    "• Hepatitis\n"
    "• Fibrosis\n"
    "• Cirrhosis"
)

st.sidebar.write("---")
st.sidebar.subheader("📊 Normal Clinical Ranges")

st.sidebar.markdown("""
**Age:** 19 – 77  
**Albumin:** 14.9 – 82.2  
**Alkaline Phosphatase:** 11.3 – 416.6  
**ALT:** 0.9 – 325.3  
**AST:** 10.6 – 324.0  
**Bilirubin:** 0.8 – 254.0  
**Cholinesterase:** 1.42 – 16.41  
**Cholesterol:** 1.43 – 9.67  
**Creatinine:** 8.0 – 1079.1  
**Gamma GT:** 4.5 – 650.9  
""")

st.sidebar.write("---")
st.sidebar.write("Made with ❤️ by **Ujjwal Tomar**")

# ------------------------------------------------------
# MAIN TITLE
# ------------------------------------------------------
st.markdown("<div class='main-title'>Liver Disease Stage Prediction</div>", unsafe_allow_html=True)
st.markdown(
    "<h3 style='color:#2c3e50; font-weight:600;'>Enter patient test values below:</h3>",
    unsafe_allow_html=True
)


# ------------------------------------------------------
# INPUT FIELDS
# ------------------------------------------------------
col1, col2 = st.columns(2)

label_style = "color:#1f4fd8; font-weight:600; margin-bottom:4px;"

with col1:
    st.markdown(f"<div style='{label_style}'>Age</div>", unsafe_allow_html=True)
    age = st.number_input("", 1, 100, 30, label_visibility="collapsed")
    
    st.markdown(f"<div style='{label_style}'>Sex</div>", unsafe_allow_html=True)
    sex = st.selectbox("", ["Male", "Female"], label_visibility="collapsed")
    sex_encoded = 1 if sex == "Male" else 0
   
    st.markdown(f"<div style='{label_style}'>Albumin</div>", unsafe_allow_html=True)
    albumin = st.number_input("", 0.0, 100.0, 45.0, label_visibility="collapsed")

    st.markdown(f"<div style='{label_style}'>Alkaline Phosphatase</div>", unsafe_allow_html=True)
    alkaline_phosphatase = st.number_input("", 0.0, 500.0, 200.0, label_visibility="collapsed")

    st.markdown(f"<div style='{label_style}'>ALT (SGPT)</div>", unsafe_allow_html=True)
    alt = st.number_input("", 0.0, 400.0, 30.0, label_visibility="collapsed")

    st.markdown(f"<div style='{label_style}'>AST (SGOT)</div>", unsafe_allow_html=True)
    ast = st.number_input("", 0.0, 400.0, 40.0, label_visibility="collapsed")

with col2:
    st.markdown(f"<div style='{label_style}'>Bilirubin</div>", unsafe_allow_html=True)
    bilirubin = st.number_input("", 0.0, 300.0, 1.0, label_visibility="collapsed")

    st.markdown(f"<div style='{label_style}'>Cholinesterase</div>", unsafe_allow_html=True)
    cholinesterase = st.number_input("", 0.0, 20.0, 8.0, label_visibility="collapsed")

    st.markdown(f"<div style='{label_style}'>Cholesterol</div>", unsafe_allow_html=True)
    cholesterol = st.number_input("", 0.0, 15.0, 4.5, label_visibility="collapsed")

    st.markdown(f"<div style='{label_style}'>Creatinine</div>", unsafe_allow_html=True)
    creatinine = st.number_input("", 0.0, 1200.0, 90.0, label_visibility="collapsed")

    st.markdown(f"<div style='{label_style}'>Gamma GT</div>", unsafe_allow_html=True)
    gamma_gt = st.number_input("", 0.0, 700.0, 35.0, label_visibility="collapsed")



# ------------------------------------------------------
# PREDICTION
# ------------------------------------------------------
import pandas as pd

if st.button("🔍 Predict Liver Disease Stage"):

    # Create DataFrame with feature names
    input_df = pd.DataFrame([{
        "Age": age,
        "Sex": sex_encoded,
        "Albumin": albumin,
        "Alkaline Phosphatase": alkaline_phosphatase,
        "ALT": alt,
        "AST": ast,
        "Bilirubin": bilirubin,
        "Cholinesterase": cholinesterase,
        "Cholesterol": cholesterol,
        "Creatinine": creatinine,
        "Gamma GT": gamma_gt
    }])

    # Align EXACT feature order used during training
    input_df = input_df[scaler.feature_names_in_]

    # Scale and predict
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    stage = label_encoder.inverse_transform(prediction)[0]

    st.markdown(
        f"<div class='prediction-box'>🧾 Predicted Stage: <b>{stage}</b></div>",
        unsafe_allow_html=True
    )


# ------------------------------------------------------
# FOOTER
# ------------------------------------------------------
st.markdown(
    "<div class='footer'>This tool is for educational purposes only and not a medical diagnosis.</div>",
    unsafe_allow_html=True
)







