import streamlit as st
import pandas as pd
import joblib
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ==================================================
# PAGE SETTINGS
# ==================================================

st.set_page_config(
    page_title="CKD Health Support",
    page_icon="🩺",
    layout="centered"
)


# ==================================================
# LOAD CKD MODEL
# ==================================================

model = joblib.load("ckd_model.pkl")
model_columns = joblib.load("model_columns.pkl")


# ==================================================
# TRANSLATION MODEL
# ==================================================

@st.cache_resource
def load_translation_model():

    name = "facebook/nllb-200-distilled-600M"

    tokenizer = AutoTokenizer.from_pretrained(name)

    translator = AutoModelForSeq2SeqLM.from_pretrained(name)

    return tokenizer, translator


@st.cache_data
def T(text, language):

    if language == "English":
        return text

    language_codes = {
        "Telugu": "tel_Telu",
        "Hindi": "hin_Deva"
    }

    tokenizer, translator = load_translation_model()

    tokenizer.src_lang = "eng_Latn"

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True
    )

    output = translator.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(
            language_codes[language]
        ),
        max_length=256
    )

    translated_text = tokenizer.batch_decode(
        output,
        skip_special_tokens=True
    )[0]

    return translated_text


# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.title("🩺 CKD Health Support")


language = st.sidebar.selectbox(
    "🌐 Language",
    [
        "English",
        "Telugu",
        "Hindi"
    ]
)


font_size = st.sidebar.selectbox(
    "🔠 Text Size",
    [
        "Normal",
        "Large"
    ]
)


high_contrast = st.sidebar.checkbox(
    "👁️ High Contrast Mode"
)


# ==================================================
# ACCESSIBILITY SETTINGS
# ==================================================

if font_size == "Large":

    st.markdown(
        """
        <style>
        .stApp {
            font-size: 20px;
        }

        p, label, .stMarkdown {
            font-size: 20px !important;
        }

        button {
            font-size: 18px !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


if high_contrast:

    st.markdown(
        """
        <style>
        .stApp {
            filter: contrast(1.2);
        }
        </style>
        """,
        unsafe_allow_html=True
    )


# ==================================================
# NAVIGATION
# ==================================================

page = st.sidebar.radio(
    "Go to",
    [
        "🏠 CKD Prediction",
        "📝 Self-Assessment",
        "📋 Action Plan",
        "📄 Health Report",
        "🌐 Language & Accessibility",
        "📍 Find Kidney Care"
    ]
)


# ==================================================
# CKD PREDICTION
# ==================================================

if page == "🏠 CKD Prediction":

    st.title(
        T(
            "Early Chronic Kidney Disease Prediction",
            language
        )
    )

    st.write(
        T(
            "Enter the patient's medical information below to predict whether the patient is likely to have CKD.",
            language
        )
    )


    # ----------------------------------------------
    # FEATURE IMPORTANCE GRAPH
    # ----------------------------------------------

    st.subheader(
        T(
            "📊 Random Forest Model Insights",
            language
        )
    )

    try:

        data = pd.read_csv(
            "feature_importance.csv"
        )

        data = data.head(10)

        chart = data.set_index(
            "Feature"
        )["Importance"]

        st.bar_chart(chart)

        st.caption(
            T(
                "Higher values indicate greater influence on the model prediction.",
                language
            )
        )

    except FileNotFoundError:

        st.info(
            T(
                "Feature importance graph is not available.",
                language
            )
        )


    # ----------------------------------------------
    # PATIENT INFORMATION
    # ----------------------------------------------

    st.subheader(
        T(
            "Patient Information",
            language
        )
    )


    age = st.number_input(
        T("Age", language),
        1,
        120,
        50
    )


    bp = st.number_input(
        T("Blood Pressure", language),
        40,
        200,
        80
    )


    sg = st.number_input(
        T("Specific Gravity", language),
        1.000,
        1.030,
        1.020,
        format="%.3f"
    )


    al = st.number_input(
        T("Albumin", language),
        0,
        5,
        0
    )


    su = st.number_input(
        T("Sugar", language),
        0,
        5,
        0
    )


    bgr = st.number_input(
        T("Blood Glucose Random", language),
        0,
        500,
        120
    )


    bu = st.number_input(
        T("Blood Urea", language),
        0.0,
        300.0,
        40.0
    )


    sc = st.number_input(
        T("Serum Creatinine", language),
        0.0,
        20.0,
        1.2
    )


    sod = st.number_input(
        T("Sodium", language),
        80.0,
        200.0,
        140.0
    )


    pot = st.number_input(
        T("Potassium", language),
        1.0,
        15.0,
        4.5
    )


    hemo = st.number_input(
        T("Hemoglobin", language),
        1.0,
        25.0,
        13.0
    )


    pcv = st.number_input(
        T("Packed Cell Volume", language),
        10,
        70,
        40
    )


    wbcc = st.number_input(
        T("White Blood Cell Count", language),
        1000,
        30000,
        8000
    )


    rbcc = st.number_input(
        T("Red Blood Cell Count", language),
        1.0,
        10.0,
        5.0
    )


    # ----------------------------------------------
    # MEDICAL CONDITIONS
    # ----------------------------------------------

    st.subheader(
        T(
            "Medical Conditions",
            language
        )
    )


    rbc = st.selectbox(
        T("Red Blood Cells", language),
        [
            "normal",
            "abnormal"
        ]
    )


    pc = st.selectbox(
        T("Pus Cell", language),
        [
            "normal",
            "abnormal"
        ]
    )


    pcc = st.selectbox(
        T("Pus Cell Clumps", language),
        [
            "notpresent",
            "present"
        ]
    )


    ba = st.selectbox(
        T("Bacteria", language),
        [
            "notpresent",
            "present"
        ]
    )


    htn = st.selectbox(
        T("Hypertension", language),
        [
            "no",
            "yes"
        ]
    )


    dm = st.selectbox(
        T("Diabetes Mellitus", language),
        [
            "no",
            "yes"
        ]
    )


    cad = st.selectbox(
        T("Coronary Artery Disease", language),
        [
            "no",
            "yes"
        ]
    )


    appet = st.selectbox(
        T("Appetite", language),
        [
            "good",
            "poor"
        ]
    )


    pe = st.selectbox(
        T("Pedal Edema", language),
        [
            "no",
            "yes"
        ]
    )


    ane = st.selectbox(
        T("Anemia", language),
        [
            "no",
            "yes"
        ]
    )


    # ----------------------------------------------
    # PREDICT BUTTON
    # ----------------------------------------------

    if st.button(
        T(
            "Predict CKD",
            language
        )
    ):

        patient = pd.DataFrame(
            [
                {
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
                }
            ]
        )


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


        patient = patient.reindex(
            columns=model_columns,
            fill_value=False
        )


        prediction = model.predict(
            patient
        )[0]


        probability = model.predict_proba(
            patient
        )[0][1]


        # Save prediction for other pages
        st.session_state["prediction"] = int(
            prediction
        )

        st.session_state["probability"] = float(
            probability
        )


        # ------------------------------------------
        # DISPLAY RESULT
        # ------------------------------------------

        if prediction == 1:

            st.error(
                T(
                    "⚠️ Prediction: CKD",
                    language
                )
            )

        else:

            st.success(
                T(
                    "✅ Prediction: NOT CKD",
                    language
                )
            )


        st.write(
            T(
                "Model probability of CKD",
                language
            )
            + f": {probability * 100:.2f}%"
        )


        st.info(
            T(
                "This application is an academic machine-learning project and should not be used as a medical diagnosis.",
                language
            )
        )


# ==================================================
# SELF-ASSESSMENT
# ==================================================

elif page == "📝 Self-Assessment":

    st.header(
        T(
            "📝 Pre-Test Kidney Health Self-Assessment",
            language
        )
    )


    st.write(
        T(
            "Answer a few questions about your general health. This questionnaire is for awareness only and does not diagnose CKD.",
            language
        )
    )


    questions = [
        "Have you been diagnosed with high blood pressure?",
        "Do you have diabetes?",
        "Do you experience swelling in your feet, ankles, or legs?",
        "Do you frequently feel unusually tired or weak?",
        "Do you have a family history of kidney disease?",
        "Have you previously been told that you have abnormal kidney-related test results?",
        "Do you have difficulty controlling your blood pressure or blood sugar?"
    ]


    st.subheader(
        T(
            "Health Questions",
            language
        )
    )


    answers = []


    for i, question in enumerate(questions):

        answer = st.radio(
            f"{i + 1}. {T(question, language)}",
            [
                T("No", language),
                T("Yes", language)
            ],
            key=f"question_{i}"
        )

        answers.append(answer)


    if st.button(
        T(
            "Assess My Responses",
            language
        )
    ):

        score = answers.count(
            T(
                "Yes",
                language
            )
        )


        # Save score for Health Report
        st.session_state[
            "self_assessment_score"
        ] = score


        st.subheader(
            T(
                "Self-Assessment Result",
                language
            )
        )


        if score <= 1:

            st.success(
                T(
                    "Your responses indicate relatively few reported kidney-health risk factors.",
                    language
                )
            )


        elif score <= 3:

            st.warning(
                T(
                    "Your responses indicate some kidney-health risk factors. Consider discussing your health with a healthcare professional.",
                    language
                )
            )


        else:

            st.error(
                T(
                    "Your responses indicate several kidney-health risk factors. Consider discussing your symptoms and health history with a healthcare professional.",
                    language
                )
            )


        st.write(
            T(
                "Self-assessment score",
                language
            )
            + f": **{score}/7**"
        )


        st.info(
            T(
                "This self-assessment is not a medical diagnosis and should not replace professional medical advice.",
                language
            )
        )



# ==================================================
# ACTION PLAN
# ==================================================

elif page == "📋 Action Plan":

    st.header(
        T(
            "📋 Patient Action Plan",
            language
        )
    )


    st.write(
        T(
            "This action plan provides general next steps based on the latest CKD prediction.",
            language
        )
    )


    if "prediction" not in st.session_state:

        st.warning(
            T(
                "Please complete the CKD prediction first to generate your action plan.",
                language
            )
        )


    else:

        prediction = st.session_state[
            "prediction"
        ]

        probability = st.session_state.get(
            "probability",
            0
        )


        st.subheader(
            T(
                "Latest Prediction",
                language
            )
        )


        if prediction == 1:

            st.error(
                T(
                    "⚠️ Latest prediction: CKD",
                    language
                )
            )


            st.write(
                T(
                    "Model probability of CKD",
                    language
                )
                + f": {probability * 100:.2f}%"
            )


            st.subheader(
                T(
                    "Recommended Next Steps",
                    language
                )
            )


            steps = [
                "Consult a qualified healthcare professional for further evaluation.",
                "Discuss appropriate kidney-function tests with your healthcare professional.",
                "Monitor blood pressure and blood sugar regularly.",
                "Keep a record of important medical reports and test results.",
                "Do not start or stop medicines or supplements without professional advice."
            ]


        else:

            st.success(
                T(
                    "✅ Latest prediction: NOT CKD",
                    language
                )
            )


            st.write(
                T(
                    "Model probability of CKD",
                    language
                )
                + f": {probability * 100:.2f}%"
            )


            st.subheader(
                T(
                    "Recommended Health Practices",
                    language
                )
            )


            steps = [
                "Continue regular health monitoring.",
                "Monitor blood pressure and blood sugar regularly.",
                "Maintain a balanced and healthy lifestyle.",
                "Keep regular medical check-ups when recommended.",
                "Seek professional medical advice if symptoms or risk factors develop."
            ]


        for i, step in enumerate(
            steps,
            1
        ):

            st.write(
                f"**{i}.** {T(step, language)}"
            )


        st.info(
            T(
                "This action plan provides general health information and does not replace professional medical advice.",
                language
            )
        )


# ==================================================
# HEALTH REPORT
# ==================================================

elif page == "📄 Health Report":

    st.header(
        T(
            "📄 Patient Health Report",
            language
        )
    )


    st.write(
        T(
            "This report summarizes the latest CKD prediction and self-assessment information.",
            language
        )
    )


    if "prediction" not in st.session_state:

        st.warning(
            T(
                "Please complete the CKD prediction first to generate the health report.",
                language
            )
        )


    else:

        prediction = st.session_state[
            "prediction"
        ]

        probability = st.session_state.get(
            "probability",
            0
        )


        # ------------------------------------------
        # CKD PREDICTION
        # ------------------------------------------

        st.subheader(
            T(
                "🩺 CKD Prediction",
                language
            )
        )


        if prediction == 1:

            prediction_text = T(
                "CKD",
                language
            )


            st.error(
                T(
                    "⚠️ Prediction: CKD",
                    language
                )
            )


        else:

            prediction_text = T(
                "NOT CKD",
                language
            )


            st.success(
                T(
                    "✅ Prediction: NOT CKD",
                    language
                )
            )


        st.write(
            T(
                "Model probability of CKD",
                language
            )
            + f": {probability * 100:.2f}%"
        )


        # ------------------------------------------
        # SELF-ASSESSMENT
        # ------------------------------------------

        st.subheader(
            T(
                "📝 Self-Assessment",
                language
            )
        )


        if "self_assessment_score" in st.session_state:

            score = st.session_state[
                "self_assessment_score"
            ]


            st.write(
                T(
                    "Self-assessment score",
                    language
                )
                + f": **{score}/7**"
            )


        else:

            st.info(
                T(
                    "Self-assessment has not been completed yet.",
                    language
                )
            )


        # ------------------------------------------
        # RECOMMENDED ACTION
        # ------------------------------------------

        st.subheader(
            T(
                "📋 Recommended Action",
                language
            )
        )


        if prediction == 1:

            report_action = (
                "Consult a qualified healthcare professional "
                "for further evaluation and appropriate testing."
            )


        else:

            report_action = (
                "Continue regular health monitoring and "
                "maintain a healthy lifestyle."
            )


        st.write(
            T(
                report_action,
                language
            )
        )


        # ------------------------------------------
        # DISCLAIMER
        # ------------------------------------------

        st.warning(
            T(
                "This report is generated from an academic machine-learning project. "
                "It is not a medical diagnosis and does not replace professional medical advice.",
                language
            )
        )


        # ------------------------------------------
        # DOWNLOAD REPORT
        # ------------------------------------------

        report = (
            T(
                "CKD HEALTH REPORT",
                language
            )
            + "\n\n"
        )


        report += (
            T(
                "Prediction",
                language
            )
            + f": {prediction_text}\n\n"
        )


        report += (
            T(
                "CKD Probability",
                language
            )
            + f": {probability * 100:.2f}%\n\n"
        )


        if "self_assessment_score" in st.session_state:

            report += (
                T(
                    "Self-Assessment Score",
                    language
                )
                + f": {st.session_state['self_assessment_score']}/7\n\n"
            )


        else:

            report += (
                T(
                    "Self-Assessment Score",
                    language
                )
                + ": "
                + T(
                    "Not completed",
                    language
                )
                + "\n\n"
            )


        report += (
            T(
                "Recommended Action",
                language
            )
            + ":\n"
            + T(
                report_action,
                language
            )
            + "\n\n"
        )


        report += (
            T(
                "Disclaimer",
                language
            )
            + ":\n"
            + T(
                "This report is generated from an academic machine-learning project. "
                "It is not a medical diagnosis and does not replace professional medical advice.",
                language
            )
        )


        st.download_button(
            label=T(
                "📥 Download Health Report",
                language
            ),
            data=report,
            file_name="CKD_Health_Report.txt",
            mime="text/plain"
        )


# ==================================================
# LANGUAGE & ACCESSIBILITY
# ==================================================

elif page == "🌐 Language & Accessibility":

    st.header(
        T(
            "🌐 Language & Accessibility",
            language
        )
    )


    st.write(
        T(
            "Customize the application according to your language and accessibility preferences.",
            language
        )
    )


    st.success(
        T(
            "Current Language",
            language
        )
        + f": {language}"
    )


    st.success(
        T(
            "Current Text Size",
            language
        )
        + f": {font_size}"
    )


    if high_contrast:

        st.success(
            T(
                "High Contrast Mode is enabled.",
                language
            )
        )


    else:

        st.info(
            T(
                "High Contrast Mode is disabled.",
                language
            )
        )


# ==================================================
# FIND KIDNEY CARE
# ==================================================
# ==================================================
# FIND KIDNEY CARE
# ==================================================

elif page == "📍 Find Kidney Care":

    st.header(
        T("📍 Find Kidney Care", language)
    )

    st.write(
        T(
            "Find nearby hospitals, nephrology clinics, and kidney care services using your location.",
            language
        )
    )

    st.subheader(
        T("Search Location", language)
    )

    city = st.text_input(
        T(
            "Enter your city or area",
            language
        ),
        placeholder="Example: Hyderabad"
    )

    care_type = st.selectbox(
        T("What type of care are you looking for?", language),
        [
            T("Kidney Care / Nephrology", language),
            T("Dialysis Center", language),
            T("Kidney Hospital", language),
            T("Nephrologist", language)
        ]
    )

    if st.button(
        T("🔎 Find Nearby Care", language)
    ):

        if city.strip() == "":
            
            st.warning(
                T(
                    "Please enter your city or area first.",
                    language
                )
            )

        else:

            # Search terms
            search_terms = {
                "Kidney Care / Nephrology":
                    "kidney care nephrology",
                "Dialysis Center":
                    "dialysis center",
                "Kidney Hospital":
                    "kidney hospital",
                "Nephrologist":
                    "nephrologist"
            }

            # Get English search term
            selected_type = care_type

            if language != "English":

                if "Dialysis" in care_type:
                    selected_type = "Dialysis Center"

                elif "Nephrologist" in care_type:
                    selected_type = "Nephrologist"

                elif "Hospital" in care_type:
                    selected_type = "Kidney Hospital"

                else:
                    selected_type = "Kidney Care / Nephrology"

            search_term = search_terms[selected_type]

            query = f"{search_term} near {city}"

            import urllib.parse

            maps_url = (
                "https://www.google.com/maps/search/?api=1&query="
                + urllib.parse.quote_plus(query)
            )

            st.success(
                T(
                    "Care options found for your selected location.",
                    language
                )
            )

            st.write(
                T(
                    "Search for nearby healthcare services using the button below.",
                    language
                )
            )

            st.link_button(
                T("📍 Open Nearby Kidney Care in Maps", language),
                maps_url
            )

            st.info(
                T(
                    "Please verify the hospital or clinic details, availability, and services before visiting.",
                    language
                )
            )

            st.warning(
                T(
                    "This feature is provided for locating healthcare services and does not recommend or endorse any particular hospital or doctor.",
                    language
                )
            )

    st.divider()

    st.subheader(
        T("🚨 When to Seek Medical Help", language)
    )

    st.write(
        T(
            "If you have concerning symptoms or abnormal kidney-related test results, consult a qualified healthcare professional.",
            language
        )
    )