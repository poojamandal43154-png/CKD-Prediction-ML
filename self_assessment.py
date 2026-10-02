import streamlit as st


def show_self_assessment(T, language):

    st.header(
        T(
            "📝 Pre-Test Kidney Health Self-Assessment",
            language
        )
    )

    st.write(
        T(
            "Answer a few questions about your general health. "
            "This questionnaire is for awareness only and does not diagnose CKD.",
            language
        )
    )

    st.subheader(
        T("Health Questions", language)
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

    answers = []

    for i, question in enumerate(questions):

        answer = st.radio(
            f"{i + 1}. {T(question, language)}",
            [
                T("No", language),
                T("Yes", language)
            ],
            key=f"self_q{i}"
        )

        answers.append(answer)

    if st.button(
        T("Assess My Responses", language)
    ):

        score = answers.count(
            T("Yes", language)
        )

        st.subheader(
            T("Self-Assessment Result", language)
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
                    "Your responses indicate some kidney-health risk factors. "
                    "Consider discussing your health with a healthcare professional.",
                    language
                )
            )

        else:

            st.error(
                T(
                    "Your responses indicate several kidney-health risk factors. "
                    "Consider discussing your symptoms and health history with a healthcare professional.",
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
                "This self-assessment is not a medical diagnosis "
                "and should not replace professional medical advice.",
                language
            )
        )