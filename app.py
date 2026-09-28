import streamlit as st
import pandas as pd
import joblib


# Load trained model
model = joblib.load("ckd_model.pkl")
model_columns = joblib.load("model_columns.pkl")


# Page settings
st.set_page_config(
    page_title="CKD Prediction",
    page_icon="🩺",
    layout="centered"
)


# Title
st.title("Early Chronic Kidney Disease Prediction")

st.write(
    "Enter the patient's medical information below "
    "to predict whether the patient is likely to have CKD."
)


# --------------------------------------------------
# Random Forest Feature Importance Dashboard
# --------------------------------------------------

st.subheader("📊 Random Forest Model Insights")

try:
    feature_importance = pd.read_csv("feature_importance.csv")

    # Get top 10 features
    top_features = feature_importance.head(10).copy()

    st.write("### Top 10 Clinical Features Influencing CKD Prediction")

    # Display horizontal bar chart
    chart_data = top_features.set_index("Feature")["Importance"]

    st.bar_chart(chart_data)

    st.caption(
        "Higher feature-importance values indicate that the feature "
        "has a greater influence on the Random Forest model's predictions."
    )

except FileNotFoundError:

    st.warning(
        "Feature importance file not found. "
        "Run train_model.py first to generate feature_importance.csv."
    )


# --------------------------------------------------
# Patient information
# --------------------------------------------------

st.subheader("Patient Information")

age = st.number_input(
    "Age",
    min_value=1,
    max_value=120,
    value=50
)

bp = st.number_input(
    "Blood Pressure",
    min_value=40,
    max_value=200,
    value=80
)

sg = st.number_input(
    "Specific Gravity",
    min_value=1.000,
    max_value=1.030,
    value=1.020,
    format="%.3f"
)

al = st.number_input(
    "Albumin",
    min_value=0,
    max_value=5,
    value=0
)

su = st.number_input(
    "Sugar",
    min_value=0,
    max_value=5,
    value=0
)

bgr = st.number_input(
    "Blood Glucose Random",
    min_value=0,
    max_value=500,
    value=120
)

bu = st.number_input(
    "Blood Urea",
    min_value=0.0,
    max_value=300.0,
    value=40.0
)

sc = st.number_input(
    "Serum Creatinine",
    min_value=0.0,
    max_value=20.0,
    value=1.2
)

sod = st.number_input(
    "Sodium",
    min_value=80.0,
    max_value=200.0,
    value=140.0
)

pot = st.number_input(
    "Potassium",
    min_value=1.0,
    max_value=15.0,
    value=4.5
)

hemo = st.number_input(
    "Hemoglobin",
    min_value=1.0,
    max_value=25.0,
    value=13.0
)

pcv = st.number_input(
    "Packed Cell Volume",
    min_value=10,
    max_value=70,
    value=40
)

wbcc = st.number_input(
    "White Blood Cell Count",
    min_value=1000,
    max_value=30000,
    value=8000
)

rbcc = st.number_input(
    "Red Blood Cell Count",
    min_value=1.0,
    max_value=10.0,
    value=5.0
)


# --------------------------------------------------
# Categorical information
# --------------------------------------------------

st.subheader("Medical Conditions")

rbc = st.selectbox(
    "Red Blood Cells",
    ["normal", "abnormal"]
)

pc = st.selectbox(
    "Pus Cell",
    ["normal", "abnormal"]
)

pcc = st.selectbox(
    "Pus Cell Clumps",
    ["notpresent", "present"]
)

ba = st.selectbox(
    "Bacteria",
    ["notpresent", "present"]
)

htn = st.selectbox(
    "Hypertension",
    ["no", "yes"]
)

dm = st.selectbox(
    "Diabetes Mellitus",
    ["no", "yes"]
)

cad = st.selectbox(
    "Coronary Artery Disease",
    ["no", "yes"]
)

appet = st.selectbox(
    "Appetite",
    ["good", "poor"]
)

pe = st.selectbox(
    "Pedal Edema",
    ["no", "yes"]
)

ane = st.selectbox(
    "Anemia",
    ["no", "yes"]
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("Predict CKD"):

    # Create patient dataframe
    patient = pd.DataFrame([{
        "age": age,
        "bp": bp,
        "sg": sg,
        "al": al,
        "su": su,
        "rbc": rbc,
        "pc": pc,
        "pcc": pcc,
        "ba": ba,
        "bgr": bgr,
        "bu": bu,
        "sc": sc,
        "sod": sod,
        "pot": pot,
        "hemo": hemo,
        "pcv": pcv,
        "wbcc": wbcc,
        "rbcc": rbcc,
        "htn": htn,
        "dm": dm,
        "cad": cad,
        "appet": appet,
        "pe": pe,
        "ane": ane
    }])


    # Convert categorical variables
    patient = pd.get_dummies(
        patient,
        columns=[
            "rbc",
            "pc",
            "pcc",
            "ba",
            "htn",
            "dm",
            "cad",
            "appet",
            "pe",
            "ane"
        ]
    )


    # Make sure patient columns match model columns
    patient = patient.reindex(
        columns=model_columns,
        fill_value=False
    )


    # Prediction
    prediction = model.predict(patient)[0]

    probability = model.predict_proba(patient)[0][1]


    # Display result
    if prediction == 1:

        st.error("⚠️ Prediction: CKD")

        st.write(
            f"Model probability of CKD: "
            f"{probability * 100:.2f}%"
        )

    else:

        st.success("✅ Prediction: NOT CKD")

        st.write(
            f"Model probability of CKD: "
            f"{probability * 100:.2f}%"
        )

    st.info(
        "This application is an academic machine-learning project "
        "and should not be used as a medical diagnosis."
    )